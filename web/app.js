(function () {
  "use strict";

  if (window.__VC_APP_INITIALIZED__) {
    return;
  }
  window.__VC_APP_INITIALIZED__ = true;

  const SAMPLE_RATE = 24000;
  const CHUNK_SAMPLES = 960; // 40ms @ 24kHz
  const CHUNK_FRAMES = 4;
  const N_MELS = 80;
  const REF_SECONDS = 3.0;
  const REF_SAMPLES = Math.floor(SAMPLE_RATE * REF_SECONDS); // 72000

  const CACHE_SHAPES = {
    stft: [1, 1, 720],
    content: [3, 1, 256, 6],
    f0: [2, 1, 128, 6],
    flow: [2, 6, 1, 256, 6],
    vocoder: [4, 1, 256, 6],
    ola: [1, 720],
  };

  function numElements(shape) {
    return shape.reduce((a, b) => a * b, 1);
  }

  function fillGaussianNoise(outArray) {
    // Box-Muller transform for standard normal noise (mean=0, std=1)
    for (let i = 0; i < outArray.length; i += 2) {
      let u1 = Math.random();
      if (u1 < 1e-7) u1 = 1e-7;
      const u2 = Math.random();
      const mag = Math.sqrt(-2.0 * Math.log(u1));
      const z0 = mag * Math.cos(2.0 * Math.PI * u2);
      const z1 = mag * Math.sin(2.0 * Math.PI * u2);
      outArray[i] = z0;
      if (i + 1 < outArray.length) {
        outArray[i + 1] = z1;
      }
    }
  }

  function computeRms(arr) {
    let sumSq = 0;
    for (let i = 0; i < arr.length; i++) {
      sumSq += arr[i] * arr[i];
    }
    return Math.sqrt(sumSq / Math.max(1, arr.length));
  }

  function generateSyntheticReferenceAudio(seconds = REF_SECONDS, baseHz = 220.0) {
    const len = Math.floor(SAMPLE_RATE * seconds);
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      const t = i / SAMPLE_RATE;
      const vibrato = 1.0 + 0.015 * Math.sin(2 * Math.PI * 5.2 * t);
      const f = baseHz * vibrato;
      const env = 0.5 + 0.5 * Math.sin(2 * Math.PI * 1.4 * t);
      // Fundamental + harmonics mimicking vocal formants
      const wave =
        0.55 * Math.sin(2 * Math.PI * f * t) +
        0.28 * Math.sin(2 * Math.PI * 2 * f * t) +
        0.16 * Math.sin(2 * Math.PI * 3 * f * t) +
        0.09 * Math.sin(2 * Math.PI * 4 * f * t);
      out[i] = 0.35 * env * wave;
    }
    return out;
  }

  class StreamingCacheManager {
    constructor() {
      this.reset();
    }

    reset() {
      this.dispose();
      this.stft = new ort.Tensor("float32", new Float32Array(numElements(CACHE_SHAPES.stft)), CACHE_SHAPES.stft);
      this.content = new ort.Tensor("float32", new Float32Array(numElements(CACHE_SHAPES.content)), CACHE_SHAPES.content);
      this.f0 = new ort.Tensor("float32", new Float32Array(numElements(CACHE_SHAPES.f0)), CACHE_SHAPES.f0);
      this.flow = new ort.Tensor("float32", new Float32Array(numElements(CACHE_SHAPES.flow)), CACHE_SHAPES.flow);
      this.vocoder = new ort.Tensor("float32", new Float32Array(numElements(CACHE_SHAPES.vocoder)), CACHE_SHAPES.vocoder);
      this.ola = new ort.Tensor("float32", new Float32Array(numElements(CACHE_SHAPES.ola)), CACHE_SHAPES.ola);
    }

    dispose() {
      for (const key of ["stft", "content", "f0", "flow", "vocoder", "ola"]) {
        if (this[key] && typeof this[key].dispose === "function") {
          try {
            this[key].dispose();
          } catch (_) {
            // Ignore already disposed tensors
          }
        }
        this[key] = null;
      }
    }

    async updateFromOutputs(outputs) {
      const mapping = [
        ["stft", outputs.stft_cache_out, CACHE_SHAPES.stft],
        ["content", outputs.content_cache_out, CACHE_SHAPES.content],
        ["f0", outputs.f0_cache_out, CACHE_SHAPES.f0],
        ["flow", outputs.flow_cache_out, CACHE_SHAPES.flow],
        ["vocoder", outputs.vocoder_cache_out, CACHE_SHAPES.vocoder],
        ["ola", outputs.ola_cache_out, CACHE_SHAPES.ola],
      ];

      for (const [key, outTensor, shape] of mapping) {
        const raw = typeof outTensor.getData === "function" ? await outTensor.getData() : outTensor.data;
        const copy = new Float32Array(raw);
        if (this[key] && typeof this[key].dispose === "function") {
          this[key].dispose();
        }
        this[key] = new ort.Tensor("float32", copy, shape);
        if (typeof outTensor.dispose === "function") {
          outTensor.dispose();
        }
      }
    }
  }

  const state = {
    engine: "seedvc",
    knnEngine: null,
    knnReady: false,
    seedvcEngine: null,
    seedvcReady: false,
    backendName: "unknown",
    speakerSession: null,
    coreSession: null,
    modelsReady: false,
    spkEmbData: null,
    cacheManager: null,
    pitchShift: 0.0,
    noiseBuffer: new Float32Array(N_MELS * CHUNK_FRAMES),
    inferenceBusy: false,
    pendingChunk: null,
    isStreaming: false,
    bypass: false,
    audioCtx: null,
    micStream: null,
    micSource: null,
    workletNode: null,
    totalChunks: 0,
    isRecordingRef: false,
    isConvertingFile: false,
    fileAbort: false,
    filePlayCtx: null,
    fileDryGain: null,
    fileWetGain: null,
    fileOutputUrl: null,
    fileInputUrl: null,
    fileLoop: true,
    isFileLoopback: false,
    lastRefAudio: null,
    lastRefLabel: null,
    lastRefSampleRate: SAMPLE_RATE,
    extractedRefByEngine: {},
    wakeSeedvcStream: null,
  };

  // DOM Elements
  const el = {
    engineSelect: document.getElementById("engine-select"),
    dotBackend: document.getElementById("dot-backend"),
    statusBackend: document.getElementById("status-backend"),
    dotModel: document.getElementById("dot-model"),
    statusModel: document.getElementById("status-model"),
    refFileInput: document.getElementById("ref-file-input"),
    btnRecordRef: document.getElementById("btn-record-ref"),
    btnTestRef: document.getElementById("btn-test-ref"),
    refStatus: document.getElementById("ref-status"),
    refTime: document.getElementById("ref-time"),
    refNorm: document.getElementById("ref-norm"),
    canvasSpk: document.getElementById("canvas-spk-emb"),
    pitchSlider: document.getElementById("pitch-slider"),
    pitchValue: document.getElementById("pitch-value"),
    pitchPresets: document.querySelectorAll(".pitch-preset"),
    btnToggleVc: document.getElementById("btn-toggle-vc"),
    btnBypass: document.getElementById("btn-bypass"),
    btnResetCache: document.getElementById("btn-reset-cache"),
    srcFileInput: document.getElementById("src-file-input"),
    btnStopFileVc: document.getElementById("btn-stop-file-vc"),
    fileVcStatus: document.getElementById("file-vc-status"),
    fileLoopbackBar: document.getElementById("file-loopback-bar"),
    fileLoopbackAudio: document.getElementById("file-loopback-audio"),
    btnFilePlayPause: document.getElementById("btn-file-play-pause"),
    btnFileLoop: document.getElementById("btn-file-loop"),
    fileVcOutput: document.getElementById("file-vc-output"),
    fileVcPlayer: document.getElementById("file-vc-player"),
    fileVcDownload: document.getElementById("file-vc-download"),
    meterIn: document.getElementById("meter-in"),
    meterOut: document.getElementById("meter-out"),
    canvasWave: document.getElementById("canvas-waveform"),
    btnRunBench: document.getElementById("btn-run-benchmark"),
    benchStatus: document.getElementById("bench-status"),
    perfAvgMs: document.getElementById("perf-avg-ms"),
    perfMinMaxMs: document.getElementById("perf-minmax-ms"),
    perfRtf: document.getElementById("perf-rtf"),
    perfChunks: document.getElementById("perf-chunks"),
    appLog: document.getElementById("app-log"),
  };

  function log(msg) {
    const ts = new Date().toLocaleTimeString("ja-JP", { hour12: false });
    el.appLog.textContent += `[${ts}] ${msg}\n`;
    el.appLog.scrollTop = el.appLog.scrollHeight;
  }

  function drawSpeakerEmbedding(vec) {
    const canvas = el.canvasSpk;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "#1e293b";
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    const barW = w / vec.length;
    let maxAbs = 0.05;
    for (let i = 0; i < vec.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(vec[i]));
    }

    for (let i = 0; i < vec.length; i++) {
      const val = vec[i] / maxAbs;
      const barH = Math.abs(val) * (h * 0.44);
      ctx.fillStyle = val >= 0 ? "#38bdf8" : "#818cf8";
      const y = val >= 0 ? h / 2 - barH : h / 2;
      ctx.fillRect(i * barW, y, Math.max(1, barW - 0.5), barH);
    }
  }

  function drawWaveforms(inArr, outArr) {
    const canvas = el.canvasWave;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "#1e293b";
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    const plotLine = (arr, color, yCenter, ampScale) => {
      if (!arr || arr.length === 0) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const step = Math.max(1, Math.floor(arr.length / w));
      for (let x = 0; x < w; x++) {
        const idx = Math.min(arr.length - 1, Math.floor((x / w) * arr.length));
        let val = arr[idx];
        if (step > 1) {
          val = Math.max(-1, Math.min(1, val));
        }
        const y = yCenter - val * ampScale;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    plotLine(inArr, "#38bdf8", h * 0.32, h * 0.25);
    plotLine(outArr, "#10b981", h * 0.72, h * 0.25);
  }

  function activeModelsReady() {
    if (state.engine === "knn") return state.knnReady;
    if (state.engine === "seedvc") return state.seedvcReady;
    return state.modelsReady;
  }

  function hasReferenceReady() {
    if (state.engine === "knn") {
      return Boolean(state.knnEngine && state.knnEngine.nRef > 0);
    }
    if (state.engine === "seedvc") {
      return Boolean(state.seedvcEngine && state.seedvcEngine.ref);
    }
    return Boolean(state.spkEmbData);
  }

  async function extractSpeakerEmbedding(float32Audio, label = "参照音声", updateUi = true, audioSampleRate = SAMPLE_RATE) {
    if (updateUi) {
      state.lastRefAudio = float32Audio;
      state.lastRefLabel = label;
      state.lastRefSampleRate = audioSampleRate;
    }
    if (state.engine === "seedvc") {
      if (!state.seedvcReady || !state.seedvcEngine) {
        throw new Error("Seed-VC エンジンがまだ初期化されていません。");
      }
      if (updateUi) el.refStatus.textContent = "Seed-VC 参照抽出中...";
      const res = await state.seedvcEngine.setReference(float32Audio, audioSampleRate, (msg) => {
        if (updateUi) el.refStatus.textContent = msg;
      });
      if (updateUi) {
        state.extractedRefByEngine.seedvc = float32Audio;
        drawSpeakerEmbedding(res.embedding);
        el.refStatus.textContent = `${label} (${res.seconds.toFixed(1)}秒 / ${res.frames}fr)`;
        el.refTime.textContent = `${res.buildMs.toFixed(1)} ms`;
        el.refNorm.textContent = `${res.frames} fr`;
        log(`Seed-VC 参照抽出完了 [${label}]: ${res.frames}フレーム (${res.seconds.toFixed(1)}秒, ${res.buildMs.toFixed(0)} ms)`);
      }
      return null;
    }

    if (state.engine === "knn") {
      if (!state.knnReady || !state.knnEngine) {
        throw new Error("kNN-VC モデルがまだロードされていません。");
      }
      if (updateUi) el.refStatus.textContent = "kNN 特徴抽出中...";
      const res = await state.knnEngine.setReference(float32Audio, audioSampleRate, (w, tot) => {
        if (updateUi) el.refStatus.textContent = `kNN 特徴抽出中 (${w}/${tot})...`;
      });
      if (updateUi) {
        state.extractedRefByEngine.knn = float32Audio;
        el.refStatus.textContent = `${label} (${res.seconds.toFixed(1)}秒 / ${res.frames}fr)`;
        el.refTime.textContent = `${res.buildMs.toFixed(1)} ms`;
        el.refNorm.textContent = `${res.frames} fr`;
        log(`kNN参照抽出完了 [${label}]: ${res.frames}フレーム (${res.seconds.toFixed(1)}秒, ${res.buildMs.toFixed(0)} ms)`);
      }
      return null;
    }

    if (!state.modelsReady || !state.speakerSession) {
      throw new Error("モデルがまだロードされていません。");
    }

    const flowAudio =
      audioSampleRate === SAMPLE_RATE
        ? float32Audio
        : await resampleFloat32Mono(float32Audio, audioSampleRate, SAMPLE_RATE);

    // Ensure at least 240 samples and trim/pad cleanly around 3 seconds
    const usableLen = Math.max(240, Math.min(flowAudio.length, SAMPLE_RATE * 10));
    const inputAudio = flowAudio.subarray(0, usableLen);
    const refTensor = new ort.Tensor("float32", inputAudio, [1, usableLen]);

    const t0 = performance.now();
    const outputs = await state.speakerSession.run({ ref_audio: refTensor });
    const spkOut = outputs.spk_emb;
    const raw = typeof spkOut.getData === "function" ? await spkOut.getData() : spkOut.data;
    const emb = new Float32Array(raw);
    const elapsedMs = performance.now() - t0;

    refTensor.dispose();
    if (typeof spkOut.dispose === "function") spkOut.dispose();

    let normSq = 0;
    for (let i = 0; i < emb.length; i++) normSq += emb[i] * emb[i];
    const l2Norm = Math.sqrt(normSq);

    state.spkEmbData = emb;
    if (updateUi) {
      state.extractedRefByEngine.flow = float32Audio;
      el.refStatus.textContent = `${label} (${(usableLen / SAMPLE_RATE).toFixed(1)}秒)`;
      el.refTime.textContent = `${elapsedMs.toFixed(1)} ms`;
      el.refNorm.textContent = l2Norm.toFixed(4);
      drawSpeakerEmbedding(emb);
      log(`話者抽出完了 [${label}]: ${elapsedMs.toFixed(2)} ms (L2=${l2Norm.toFixed(4)})`);
    }
    return emb;
  }

  async function runStreamingChunk(inputChunk) {
    if (!state.spkEmbData) {
      const synth = generateSyntheticReferenceAudio(REF_SECONDS, 220.0);
      await extractSpeakerEmbedding(synth, "自動テスト音声");
    }

    fillGaussianNoise(state.noiseBuffer);

    const audioTensor = new ort.Tensor("float32", inputChunk, [1, CHUNK_SAMPLES]);
    const spkTensor = new ort.Tensor("float32", state.spkEmbData, [1, 256]);
    const pitchTensor = new ort.Tensor("float32", new Float32Array([state.pitchShift]), [1, 1]);
    const noiseTensor = new ort.Tensor("float32", state.noiseBuffer, [1, N_MELS, CHUNK_FRAMES]);

    const feeds = {
      audio_chunk: audioTensor,
      spk_emb: spkTensor,
      pitch_shift_semitones: pitchTensor,
      noise_seed: noiseTensor,
      stft_cache_in: state.cacheManager.stft,
      content_cache_in: state.cacheManager.content,
      f0_cache_in: state.cacheManager.f0,
      flow_cache_in: state.cacheManager.flow,
      vocoder_cache_in: state.cacheManager.vocoder,
      ola_cache_in: state.cacheManager.ola,
    };

    const t0 = performance.now();
    const outputs = await state.coreSession.run(feeds);
    const audioOutTensor = outputs.audio_out;
    const rawOut =
      typeof audioOutTensor.getData === "function"
        ? await audioOutTensor.getData()
        : audioOutTensor.data;
    const outChunk = new Float32Array(rawOut);
    const inferMs = performance.now() - t0;

    // Dispose input tensors & update causal caches
    audioTensor.dispose();
    spkTensor.dispose();
    pitchTensor.dispose();
    noiseTensor.dispose();
    if (typeof audioOutTensor.dispose === "function") audioOutTensor.dispose();

    await state.cacheManager.updateFromOutputs(outputs);

    state.totalChunks += 1;
    el.perfChunks.textContent = String(state.totalChunks);

    return { outChunk, inferMs };
  }

  async function detectAndCreateSessions() {
    if (typeof window.ort === "undefined") {
      throw new Error("ONNX Runtime Web (ort) が読み込まれていません。");
    }

    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = window.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 2)
      : 1;

    let webgpuAvailable = false;
    if (typeof navigator !== "undefined" && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        webgpuAvailable = Boolean(adapter);
      } catch (_) {
        webgpuAvailable = false;
      }
    }

    const getFlowModelUrl = (filename) =>
      (window.VC_CONFIG && window.VC_CONFIG.getModelUrl(filename)) || `/models/${filename}`;

    const createWithProvider = async (providers) => {
      const spk = await ort.InferenceSession.create(getFlowModelUrl("speaker_encoder.onnx"), {
        executionProviders: providers,
      });
      const core = await ort.InferenceSession.create(getFlowModelUrl("streaming_vc_core.onnx"), {
        executionProviders: providers,
      });
      return { spk, core };
    };

    if (webgpuAvailable) {
      try {
        log("WebGPU アダプタ検出。WebGPU バックエンドでセッション初期化を試行します...");
        const { spk, core } = await createWithProvider(["webgpu", "wasm"]);
        state.speakerSession = spk;
        state.coreSession = core;
        state.backendName = "WebGPU";
        state.cacheManager = new StreamingCacheManager();
        state.modelsReady = true;

        // Warmup run to verify WebGPU kernels execute cleanly
        const warmChunk = new Float32Array(CHUNK_SAMPLES);
        const warmRef = generateSyntheticReferenceAudio(1.0, 220.0);
        await extractSpeakerEmbedding(warmRef, "ウォームアップ", false);
        await runStreamingChunk(warmChunk);
        state.cacheManager.reset();
        state.spkEmbData = null;
        state.totalChunks = 0;
        el.perfChunks.textContent = "0";
        return;
      } catch (err) {
        log(`WebGPU 実行検証でフォールバック発生: ${err.message}`);
        if (state.cacheManager) state.cacheManager.dispose();
        state.speakerSession = null;
        state.coreSession = null;
        state.spkEmbData = null;
        state.totalChunks = 0;
        el.perfChunks.textContent = "0";
        state.modelsReady = false;
      }
    }

    log("WASM (CPU) バックエンドでセッションを初期化します...");
    const { spk, core } = await createWithProvider(["wasm"]);
    state.speakerSession = spk;
    state.coreSession = core;
    state.backendName = "WASM (CPU)";
    state.cacheManager = new StreamingCacheManager();
    state.modelsReady = true;
  }

  function configureOrtEnv() {
    if (typeof window.ort === "undefined") {
      throw new Error("ONNX Runtime Web (ort) が読み込まれていません。");
    }
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = window.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 2)
      : 1;
  }

  async function initKnnModels() {
    configureOrtEnv();
    if (!state.knnEngine) {
      state.knnEngine = new window.KnnVcEngine();
    }
    if (!state.knnReady) {
      await state.knnEngine.load(ort, (msg) => {
        el.statusModel.textContent = msg;
        log(`[kNN-VC] ${msg}`);
      });
      state.knnReady = true;
    }
    state.backendName = `${state.knnEngine.backendLabel} (kNN-VC)`;
  }

  function setControlsEnabled(enabled) {
    el.btnRecordRef.disabled = !enabled;
    el.btnTestRef.disabled = !enabled;
    el.btnToggleVc.disabled = !enabled;
    el.btnBypass.disabled = !enabled;
    el.btnResetCache.disabled = !enabled;
    el.srcFileInput.disabled = !enabled;
    el.btnRunBench.disabled = !enabled;
  }

  async function initModels() {
    setControlsEnabled(false);
    try {
      el.dotBackend.className = "dot";
      el.dotModel.className = "dot";
      el.statusBackend.textContent = "バックエンド検証中...";
      el.statusModel.textContent = "ONNXモデル読込中...";

      if (state.engine === "seedvc") {
        if (!state.seedvcEngine) {
          state.seedvcEngine = new window.SeedVcEngine();
        }
        if (!state.seedvcReady) {
          await state.seedvcEngine.load(ort, (msg) => {
            el.statusModel.textContent = msg;
            log(`[Seed-VC] ${msg}`);
          });
          state.seedvcReady = true;
        }
        state.backendName = `${state.seedvcEngine.backendLabel} (Seed-VC)`;
        el.statusModel.textContent = "準備完了 (Seed-VC 44.1kHz)";
        log(`Seed-VC 初期化完了 (${state.backendName})`);
      } else if (state.engine === "knn") {
        await initKnnModels();
        el.statusModel.textContent = "準備完了 (kNN-VC 16kHz)";
        log(`kNN-VC 初期化完了 (${state.backendName})`);
      } else {
        await detectAndCreateSessions();
        el.statusModel.textContent = "準備完了 (FlowVC 24kHz)";
        log(`FlowVC 初期化完了 (${state.backendName}) — Total ONNX ~39MB`);
      }

      el.dotBackend.className = "dot ok";
      el.statusBackend.textContent = `Backend: ${state.backendName}`;
      el.dotModel.className = "dot ok";

      setControlsEnabled(true);
      el.fileVcStatus.textContent = "音声ファイル (.mp3 / .wav 等) を選択してリアルタイム再生";
      el.benchStatus.textContent = `${state.backendName} でベンチマーク実行可能です。`;
    } catch (err) {
      el.dotBackend.className = "dot err";
      el.dotModel.className = "dot err";
      el.statusBackend.textContent = "初期化エラー";
      el.statusModel.textContent = "読込失敗";
      log(`エラー: ${err.message}`);
      console.error(err);
    }
  }

  async function resampleBufferToMono(audioBuffer, targetRate) {
    if (audioBuffer.sampleRate === targetRate && audioBuffer.numberOfChannels === 1) {
      return audioBuffer.getChannelData(0);
    }
    const targetLen = Math.max(240, Math.ceil(audioBuffer.duration * targetRate));
    const offline = new OfflineAudioContext(1, targetLen, targetRate);
    const src = offline.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(offline.destination);
    src.start(0);
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0);
  }

  async function resampleFloat32Mono(audio, srcRate, dstRate) {
    if (srcRate === dstRate) return audio;
    const dstLen = Math.max(240, Math.round((audio.length * dstRate) / srcRate));
    const offline = new OfflineAudioContext(1, dstLen, dstRate);
    const buf = offline.createBuffer(1, audio.length, srcRate);
    buf.copyToChannel(audio, 0);
    const src = offline.createBufferSource();
    src.buffer = buf;
    src.connect(offline.destination);
    src.start(0);
    const rendered = await offline.startRendering();
    return new Float32Array(rendered.getChannelData(0));
  }

  async function decodeAudioFileToMono(file, targetRate = SAMPLE_RATE) {
    const arrayBuf = await file.arrayBuffer();
    const decodeCtx = new OfflineAudioContext(1, 1, targetRate);
    const decoded = await decodeCtx.decodeAudioData(arrayBuf);
    return await resampleBufferToMono(decoded, targetRate);
  }

  async function handleFileUpload(file) {
    if (!file) return;
    try {
      // Decode directly at the active engine's native sample rate to preserve 12-22kHz highs for Seed-VC
      const targetRate =
        state.engine === "seedvc" && window.SeedVcEngine
          ? window.SeedVcEngine.SAMPLE_RATE
          : state.engine === "knn" && window.KnnVcEngine
            ? window.KnnVcEngine.SAMPLE_RATE
            : SAMPLE_RATE;
      log(`参照音声をデコード中: ${file.name} (${targetRate} Hz)`);
      const mono = await decodeAudioFileToMono(file, targetRate);
      await extractSpeakerEmbedding(mono, file.name, true, targetRate);
    } catch (err) {
      el.refStatus.textContent = `抽出エラー: ${err.message}`;
      log(`ファイル読込エラー: ${err.message}`);
    }
  }

  function encodeWavBlob(samples, sampleRate) {
    const numSamples = samples.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);
    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + numSamples * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, numSamples * 2, true);
    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
    return new Blob([buffer], { type: "audio/wav" });
  }

  function syncFilePlayPauseBtn() {
    if (!el.btnFilePlayPause || !el.fileLoopbackAudio) return;
    el.btnFilePlayPause.textContent = el.fileLoopbackAudio.paused ? "▶️ 再生" : "⏸️ 一時停止";
  }

  async function stopFilePlayback() {
    state.fileAbort = true;
    state.fileDryGain = null;
    state.fileWetGain = null;
    if (state.filePlayCtx) {
      try {
        await state.filePlayCtx.close();
      } catch (_) {}
      state.filePlayCtx = null;
    }
    if (state.isStreaming && state.isFileLoopback) {
      await stopRealtimeVC();
      return;
    }
    if (el.fileLoopbackAudio && !el.fileLoopbackAudio.paused) {
      el.fileLoopbackAudio.pause();
    }
    if (el.fileLoopbackBar) el.fileLoopbackBar.hidden = true;
    el.btnStopFileVc.hidden = true;
  }

  async function handleSourceFileConvert(file) {
    if (!file || !activeModelsReady() || state.isConvertingFile) return;
    if (state.isStreaming) {
      await stopRealtimeVC();
    }

    state.isConvertingFile = true;
    state.fileAbort = false;
    el.srcFileInput.disabled = true;
    el.fileVcStatus.textContent = `ループバック準備中: ${file.name}...`;

    try {
      if (!hasReferenceReady()) {
        el.fileVcStatus.textContent = "自動参照音声を準備中...";
        await extractSpeakerEmbedding(generateSyntheticReferenceAudio(), "自動テスト音声");
      }

      if (state.fileInputUrl) {
        URL.revokeObjectURL(state.fileInputUrl);
      }
      state.fileInputUrl = URL.createObjectURL(file);

      // Replace <audio> element so createMediaElementSource can bind cleanly to a new AudioContext
      const freshAudio = el.fileLoopbackAudio.cloneNode(false);
      freshAudio.src = state.fileInputUrl;
      freshAudio.loop = state.fileLoop;
      freshAudio.addEventListener("play", syncFilePlayPauseBtn);
      freshAudio.addEventListener("pause", syncFilePlayPauseBtn);
      freshAudio.addEventListener("ended", syncFilePlayPauseBtn);
      el.fileLoopbackAudio.replaceWith(freshAudio);
      el.fileLoopbackAudio = freshAudio;

      el.fileLoopbackBar.hidden = false;
      el.btnStopFileVc.hidden = false;
      el.fileVcOutput.hidden = true;

      await startRealtimeVC({ mediaElement: freshAudio, fileName: file.name });
      await freshAudio.play();
      syncFilePlayPauseBtn();
      el.fileVcStatus.textContent = `🔁 ループバック変換中: ${file.name}`;
      log(`ファイル入力ループバック開始 [${file.name}] (ループ: ${state.fileLoop ? "ON" : "OFF"})`);
    } catch (err) {
      el.fileVcStatus.textContent = `再生エラー: ${err.message}`;
      log(`ファイルループバックエラー: ${err.message}`);
      await stopFilePlayback();
    } finally {
      state.isConvertingFile = false;
      el.srcFileInput.disabled = false;
      el.srcFileInput.value = "";
    }
  }

  async function record3SecReference() {
    if (state.isRecordingRef) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      log("この環境ではマイク録音 API (getUserMedia) が利用できません（HTTPSまたはlocalhostが必要）。");
      return;
    }

    state.isRecordingRef = true;
    el.btnRecordRef.disabled = true;
    el.btnRecordRef.textContent = "🔴 録音中 (3秒)...";

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, echoCancellation: false, noiseSuppression: false },
      });
      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
      const source = ctx.createMediaStreamSource(stream);
      const recorder = ctx.createScriptProcessor(2048, 1, 1);
      const chunks = [];
      let totalCollected = 0;

      await new Promise((resolve) => {
        recorder.onaudioprocess = (ev) => {
          const input = ev.inputBuffer.getChannelData(0);
          chunks.push(new Float32Array(input));
          totalCollected += input.length;
          if (totalCollected >= REF_SAMPLES) {
            resolve();
          }
        };
        source.connect(recorder);
        recorder.connect(ctx.destination);
      });

      recorder.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      await ctx.close();

      const merged = new Float32Array(totalCollected);
      let offset = 0;
      for (const c of chunks) {
        merged.set(c, offset);
        offset += c.length;
      }
      await extractSpeakerEmbedding(merged.subarray(0, REF_SAMPLES), "マイク録音");
    } catch (err) {
      log(`録音エラー: ${err.message}`);
    } finally {
      state.isRecordingRef = false;
      el.btnRecordRef.disabled = false;
      el.btnRecordRef.textContent = "🎙️ 3秒マイク録音";
    }
  }

  function processAudioChunkQueued(chunk, inputRms) {
    if (state.inferenceBusy) {
      // Keep only the freshest chunk to bound real-time latency
      state.pendingChunk = { chunk, inputRms };
      return;
    }

    state.inferenceBusy = true;
    const isKnn = state.engine === "knn";
    const runPromise = isKnn
      ? (async () => {
          const t0 = performance.now();
          const outChunk = await state.knnEngine.processChunk(chunk);
          state.totalChunks += 1;
          el.perfChunks.textContent = String(state.totalChunks);
          return { outChunk, inferMs: performance.now() - t0 };
        })()
      : runStreamingChunk(chunk);

    runPromise
      .then(({ outChunk, inferMs }) => {
        const outRms = computeRms(outChunk);
        el.meterIn.style.width = `${Math.min(100, inputRms * 350)}%`;
        el.meterOut.style.width = `${Math.min(100, outRms * 350)}%`;
        el.perfAvgMs.textContent = `${inferMs.toFixed(2)} ms`;
        el.perfRtf.textContent = (inferMs / (isKnn ? 200.0 : 40.0)).toFixed(3);
        drawWaveforms(chunk, outChunk);

        if (state.workletNode && state.isStreaming) {
          state.workletNode.port.postMessage(
            { type: "output_chunk", chunk: outChunk },
            [outChunk.buffer]
          );
        }
      })
      .catch((err) => {
        log(`推論エラー: ${err.message}`);
      })
      .finally(() => {
        state.inferenceBusy = false;
        if (state.pendingChunk && state.isStreaming) {
          const next = state.pendingChunk;
          state.pendingChunk = null;
          processAudioChunkQueued(next.chunk, next.inputRms);
        }
      });
  }

  async function startRealtimeVC(options = {}) {
    if (state.isStreaming) return;
    const mediaEl = options.mediaElement || null;
    const isFileMode = Boolean(mediaEl);
    if (!isFileMode && (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)) {
      log("マイク入力がサポートされていません（HTTPSまたはlocalhost環境が必要です）。");
      return;
    }

    try {
      if (!hasReferenceReady()) {
        await extractSpeakerEmbedding(generateSyntheticReferenceAudio(), "自動テスト音声");
      }
      if (state.engine === "seedvc") {
        const ctxRate = window.SeedVcEngine.SAMPLE_RATE;
        const stream = isFileMode
          ? null
          : await navigator.mediaDevices.getUserMedia({
              audio: {
                channelCount: 1,
                sampleRate: ctxRate,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
              },
            });
        const ctx = new AudioContext({ sampleRate: ctxRate, latencyHint: "interactive" });
        if (ctx.state === "suspended") await ctx.resume();
        const source = isFileMode
          ? ctx.createMediaElementSource(mediaEl)
          : ctx.createMediaStreamSource(stream);
        const proc = ctx.createScriptProcessor(1024, 1, 1);
        const silentGain = ctx.createGain();
        silentGain.gain.value = 0;

        state.seedvcEngine.resetStream();
        const chunkSamples = state.seedvcEngine.streamChunkSamples; // 40 * 512 = 20,480 (~0.464s)
        const rightMargin = state.seedvcEngine.streamRightSamples;  // 8 * 512 = 4,096 (~93ms)
        const needSamples = chunkSamples + rightMargin;
        // Zero-allocation circular ring buffer (avoids Array push/shift & per-callback Float32Array GC)
        const maxQueued = isFileMode
          ? chunkSamples * 8 + rightMargin
          : chunkSamples * 2 + rightMargin;
        const ringCap = maxQueued + 4096;
        const ringBuf = new Float32Array(ringCap);
        let ringRead = 0;
        let ringWrite = 0;
        let ringCount = 0;
        let wakeConsumer = null;
        let lastMeterInMs = 0;

        const wakeIfWaiting = () => {
          if (wakeConsumer) {
            const w = wakeConsumer;
            wakeConsumer = null;
            w();
          }
        };
        state.wakeSeedvcStream = wakeIfWaiting;

        proc.onaudioprocess = (ev) => {
          if (!state.isStreaming) return;
          const inp = ev.inputBuffer.getChannelData(0);
          const n = inp.length;
          const first = Math.min(n, ringCap - ringWrite);
          ringBuf.set(inp.subarray(0, first), ringWrite);
          if (first < n) {
            ringBuf.set(inp.subarray(first), 0);
          }
          ringWrite = (ringWrite + n) % ringCap;
          ringCount += n;
          // Drop oldest excess in O(1) so live latency stays bounded
          if (ringCount > maxQueued) {
            const excess = ringCount - maxQueued;
            ringRead = (ringRead + excess) % ringCap;
            ringCount = maxQueued;
          }
          if (ringCount >= needSamples) {
            wakeIfWaiting();
          }
          // Throttle input meter DOM updates to ~15Hz instead of 43Hz audio callback rate
          const nowMs = performance.now();
          if (nowMs - lastMeterInMs >= 66) {
            lastMeterInMs = nowMs;
            const rms = computeRms(inp);
            el.meterIn.style.width = `${Math.min(100, rms * 350)}%`;
          }
        };
        source.connect(proc);
        proc.connect(silentGain);
        silentGain.connect(ctx.destination);

        state.audioCtx = ctx;
        state.micStream = stream;
        state.micSource = source;
        state.workletNode = proc;
        state.isStreaming = true;
        state.isFileLoopback = isFileMode;
        el.btnToggleVc.textContent = "⏹️ リアルタイム変換停止";
        el.btnToggleVc.classList.add("active-stop");
        if (!isFileMode) {
          el.fileVcStatus.textContent = "🎙️ Seed-VC リアルタイム変換中...";
          log(`Seed-VC リアルタイムストリーミング開始 (SR: ${ctx.sampleRate} Hz, チャンク: 0.46s / 左文脈0.56s / 3sプロンプト)`);
        }

        const readRing = (dst, n, consume) => {
          const first = Math.min(n, ringCap - ringRead);
          dst.set(ringBuf.subarray(ringRead, ringRead + first), 0);
          if (first < n) {
            dst.set(ringBuf.subarray(0, n - first), first);
          }
          if (consume) {
            ringRead = (ringRead + n) % ringCap;
            ringCount -= n;
          }
        };

        const pieceBuf = new Float32Array(chunkSamples);
        const rightBuf = new Float32Array(rightMargin);
        let nextPlayTime = 0;
        (async () => {
          while (state.isStreaming && state.audioCtx === ctx) {
            if (ringCount < needSamples) {
              await new Promise((r) => {
                wakeConsumer = r;
              });
              continue;
            }
            readRing(pieceBuf, chunkSamples, true);
            readRing(rightBuf, rightMargin, false);
            const piece = pieceBuf;
            const rightCtx = rightBuf;
            try {
              let outAudio;
              let outRms;
              let elapsedMs = 0;
              if (state.bypass) {
                outAudio = piece;
                outRms = computeRms(piece);
              } else {
                const conv = await state.seedvcEngine.convertChunkStream(
                  piece,
                  { pitchShift: state.pitchShift, steps: 1, cfgRate: 0.0, rightCtx }
                );
                outAudio = conv.audio;
                outRms = conv.rms;
                elapsedMs = conv.elapsedMs;
              }
              if (!state.isStreaming || state.audioCtx !== ctx) break;
              const now = ctx.currentTime;
              let playAudio = outAudio;
              if (nextPlayTime === 0) {
                nextPlayTime = now + (isFileMode ? 0.15 : 0.04);
              } else if (nextPlayTime < now + 0.015) {
                nextPlayTime = now + (isFileMode ? 0.08 : 0.03);
                const fadeLen = Math.min(Math.round(ctxRate * 0.015), outAudio.length);
                playAudio = new Float32Array(outAudio);
                for (let i = 0; i < fadeLen; i++) {
                  playAudio[i] *= i / fadeLen;
                }
              }
              const buf = ctx.createBuffer(1, playAudio.length, ctxRate);
              buf.copyToChannel(playAudio, 0);
              const srcNode = ctx.createBufferSource();
              srcNode.buffer = buf;
              srcNode.connect(ctx.destination);
              srcNode.start(nextPlayTime);
              nextPlayTime += buf.duration;

              state.totalChunks += 1;
              el.perfChunks.textContent = String(state.totalChunks);
              el.meterOut.style.width = `${Math.min(100, outRms * 350)}%`;
              drawWaveforms(piece, outAudio);
              const durSec = outAudio.length / ctxRate;
              el.perfAvgMs.textContent = `${elapsedMs.toFixed(0)} ms`;
              el.perfRtf.textContent = (elapsedMs / 1000 / Math.max(0.01, durSec)).toFixed(3);
            } catch (err) {
              log(`Seed-VC ライブ変換エラー: ${err.message}`);
              break;
            }
          }
        })();
        return;
      }

      const isKnn = state.engine === "knn";
      if (isKnn) {
        state.knnEngine.reset();
      } else {
        state.cacheManager.reset();
      }
      const ctxRate = isKnn ? window.KnnVcEngine.SAMPLE_RATE : SAMPLE_RATE;

      const stream = isFileMode
        ? null
        : await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              sampleRate: ctxRate,
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false,
            },
          });

      const ctx = new AudioContext({ sampleRate: ctxRate, latencyHint: "interactive" });
      await ctx.audioWorklet.addModule(isKnn ? "/web/knn-audio-worklet.js" : "/web/vc-audio-worklet.js");

      const source = isFileMode
        ? ctx.createMediaElementSource(mediaEl)
        : ctx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(ctx, isKnn ? "knn-stream-processor" : "vc-stream-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });

      workletNode.port.onmessage = (event) => {
        const msg = event.data;
        if (msg && msg.type === "input_chunk") {
          processAudioChunkQueued(msg.chunk, msg.inputRms);
        }
      };

      workletNode.port.postMessage({ type: "set_bypass", value: state.bypass });
      source.connect(workletNode);
      workletNode.connect(ctx.destination);

      if (ctx.state === "suspended") {
        await ctx.resume();
      }

      state.audioCtx = ctx;
      state.micStream = stream;
      state.micSource = source;
      state.workletNode = workletNode;
      state.isStreaming = true;
      state.isFileLoopback = isFileMode;

      el.btnToggleVc.textContent = "⏹️ リアルタイム変換停止";
      el.btnToggleVc.classList.add("active-stop");
      log(`リアルタイムVC開始 (AudioContext SR: ${ctx.sampleRate} Hz)`);
    } catch (err) {
      log(`VC開始エラー: ${err.message}`);
      await stopRealtimeVC();
    }
  }

  async function stopRealtimeVC() {
    const wasFileLoopback = state.isFileLoopback;
    state.isStreaming = false;
    state.isFileLoopback = false;
    state.pendingChunk = null;
    if (typeof state.wakeSeedvcStream === "function") {
      state.wakeSeedvcStream();
      state.wakeSeedvcStream = null;
    }

    if (wasFileLoopback && el.fileLoopbackAudio && !el.fileLoopbackAudio.paused) {
      el.fileLoopbackAudio.pause();
    }
    if (state.workletNode) {
      state.workletNode.disconnect();
      state.workletNode = null;
    }
    if (state.micSource) {
      state.micSource.disconnect();
      state.micSource = null;
    }
    if (state.micStream) {
      state.micStream.getTracks().forEach((t) => t.stop());
      state.micStream = null;
    }
    if (state.audioCtx) {
      await state.audioCtx.close();
      state.audioCtx = null;
    }

    if (wasFileLoopback) {
      if (el.fileLoopbackBar) el.fileLoopbackBar.hidden = true;
      el.btnStopFileVc.hidden = true;
      el.fileVcStatus.textContent = "音声ファイル (.mp3 / .wav 等) を選択してリアルタイム再生";
    }
    el.btnToggleVc.textContent = "▶️ リアルタイム変換開始";
    el.btnToggleVc.classList.remove("active-stop");
    el.meterIn.style.width = "0%";
    el.meterOut.style.width = "0%";
    log(wasFileLoopback ? "ファイルループバックVC停止" : "リアルタイムVC停止");
  }

  async function runBenchmark() {
    if (!activeModelsReady()) return;
    el.btnRunBench.disabled = true;
    if (state.engine === "seedvc") {
      el.benchStatus.textContent = "Seed-VC (1秒音声) ベンチマーク実行中...";
      try {
        if (!hasReferenceReady()) {
          const ref = generateSyntheticReferenceAudio(REF_SECONDS, 220.0);
          await extractSpeakerEmbedding(ref, "ベンチマークテスト話者");
        }
        const sr = window.SeedVcEngine.SAMPLE_RATE;
        const src = new Float32Array(sr);
        for (let i = 0; i < sr; i++) src[i] = 0.25 * Math.sin((2 * Math.PI * 330 * i) / sr);
        const conv = await state.seedvcEngine.convertAudio(src, sr, { pitchShift: 0, steps: 1, cfgRate: 0.0, maxPromptFrames: 258 }, (m) => {
          el.benchStatus.textContent = m;
        });
        const rtf = conv.elapsedMs / 1000.0;
        const tm = conv.timings
          ? ` [W:${conv.timings.tWhisperMs.toFixed(0)}ms Y:${conv.timings.tYinMs.toFixed(0)}ms D:${conv.timings.tDitMs.toFixed(0)}ms BV:${conv.timings.tBigvganMs.toFixed(0)}ms]`
          : "";
        el.perfAvgMs.textContent = `${conv.elapsedMs.toFixed(0)} ms`;
        el.perfRtf.textContent = rtf.toFixed(3);
        el.benchStatus.textContent = `Seed-VC 完了: 1.0秒音声 -> ${(conv.elapsedMs / 1000).toFixed(2)}s (RTF ${rtf.toFixed(3)}, rms=${conv.rms.toFixed(3)}, peak=${conv.peak.toFixed(3)})${tm}`;
        log(`Seed-VC ベンチマーク完了: ${(conv.elapsedMs / 1000).toFixed(2)}s (RTF ${rtf.toFixed(3)}, rms=${conv.rms.toFixed(3)}, peak=${conv.peak.toFixed(3)})${tm}`);
        if (conv.dbg) log(`[Seed-VC DBG] ${conv.dbg}`);
      } catch (err) {
        el.benchStatus.textContent = `ベンチマークエラー: ${err.message}`;
        log(`ベンチマークエラー: ${err.message}`);
      } finally {
        el.btnRunBench.disabled = false;
      }
      return;
    }
    const isKnn = state.engine === "knn";
    const totalChunks = isKnn ? 5 : 20;
    const chunkLen = isKnn ? window.KnnVcEngine.CHUNK : CHUNK_SAMPLES;
    const benchRate = isKnn ? window.KnnVcEngine.SAMPLE_RATE : SAMPLE_RATE;
    const chunkDurMs = isKnn ? 200.0 : 40.0;
    el.benchStatus.textContent = `${totalChunks}チャンク連続ベンチマーク実行中...`;

    try {
      if (!hasReferenceReady()) {
        const ref = generateSyntheticReferenceAudio(REF_SECONDS, 220.0);
        await extractSpeakerEmbedding(ref, "ベンチマークテスト話者");
      }
      if (isKnn) {
        state.knnEngine.reset();
      } else {
        state.cacheManager.reset();
      }

      const times = [];
      let lastIn = null;
      let lastOut = null;

      for (let c = 0; c < totalChunks; c++) {
        const chunk = new Float32Array(chunkLen);
        const phaseOffset = c * chunkLen;
        for (let i = 0; i < chunkLen; i++) {
          const t = (phaseOffset + i) / benchRate;
          chunk[i] = 0.3 * Math.sin(2 * Math.PI * 190.0 * t);
        }
        let outChunk;
        let inferMs;
        if (isKnn) {
          const t0 = performance.now();
          outChunk = await state.knnEngine.processChunk(chunk);
          inferMs = performance.now() - t0;
        } else {
          const res = await runStreamingChunk(chunk);
          outChunk = res.outChunk;
          inferMs = res.inferMs;
        }
        times.push(inferMs);
        lastIn = chunk;
        lastOut = outChunk;
      }

      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      const minT = Math.min(...times);
      const maxT = Math.max(...times);
      const rtf = avg / chunkDurMs;

      el.perfAvgMs.textContent = `${avg.toFixed(2)} ms`;
      el.perfMinMaxMs.textContent = `${minT.toFixed(1)} / ${maxT.toFixed(1)} ms`;
      el.perfRtf.textContent = rtf.toFixed(3);
      el.benchStatus.textContent = `完了 (${state.backendName}): 平均 ${avg.toFixed(2)} ms / chunk (RTF=${rtf.toFixed(3)})`;

      if (lastIn && lastOut) {
        el.meterIn.style.width = `${Math.min(100, computeRms(lastIn) * 350)}%`;
        el.meterOut.style.width = `${Math.min(100, computeRms(lastOut) * 350)}%`;
        drawWaveforms(lastIn, lastOut);
      }
      log(`ベンチマーク完了 [${state.backendName}]: avg=${avg.toFixed(2)}ms, min=${minT.toFixed(2)}ms, max=${maxT.toFixed(2)}ms, RTF=${rtf.toFixed(3)}`);
    } catch (err) {
      el.benchStatus.textContent = `エラー: ${err.message}`;
      log(`ベンチマークエラー: ${err.message}`);
    } finally {
      el.btnRunBench.disabled = false;
    }
  }

  function updatePitch(val) {
    state.pitchShift = parseFloat(val);
    if (state.knnEngine && typeof state.knnEngine.setPitchShift === "function") {
      state.knnEngine.setPitchShift(state.pitchShift);
    }
    el.pitchSlider.value = String(state.pitchShift);
    const sign = state.pitchShift >= 0 ? "+" : "";
    el.pitchValue.textContent = `${sign}${state.pitchShift.toFixed(1)} st`;
    el.pitchPresets.forEach((btn) => {
      btn.classList.toggle("active", parseFloat(btn.dataset.pitch) === state.pitchShift);
    });
  }

  // Event Listeners
  el.refFileInput.addEventListener("change", (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (file) handleFileUpload(file);
  });

  el.srcFileInput.addEventListener("change", (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (file) handleSourceFileConvert(file);
  });

  el.btnStopFileVc.addEventListener("click", () => {
    stopFilePlayback();
  });

  el.btnFilePlayPause.addEventListener("click", () => {
    if (!el.fileLoopbackAudio) return;
    if (el.fileLoopbackAudio.paused) {
      el.fileLoopbackAudio.play().catch((err) => log(`再生エラー: ${err.message}`));
    } else {
      el.fileLoopbackAudio.pause();
    }
  });

  el.btnFileLoop.addEventListener("click", () => {
    state.fileLoop = !state.fileLoop;
    if (el.fileLoopbackAudio) {
      el.fileLoopbackAudio.loop = state.fileLoop;
    }
    el.btnFileLoop.textContent = `🔁 ループ: ${state.fileLoop ? "ON" : "OFF"}`;
    el.btnFileLoop.classList.toggle("active", state.fileLoop);
  });

  el.btnRecordRef.addEventListener("click", () => {
    record3SecReference();
  });

  el.btnTestRef.addEventListener("click", async () => {
    try {
      const synth = generateSyntheticReferenceAudio(REF_SECONDS, 220.0);
      await extractSpeakerEmbedding(synth, "内蔵テスト音声 (220Hz)");
    } catch (err) {
      el.refStatus.textContent = `抽出エラー: ${err.message}`;
      log(`テスト参照抽出エラー: ${err.message}`);
    }
  });

  el.pitchSlider.addEventListener("input", (ev) => {
    updatePitch(ev.target.value);
  });

  el.pitchPresets.forEach((btn) => {
    btn.addEventListener("click", () => {
      updatePitch(btn.dataset.pitch);
    });
  });

  el.btnToggleVc.addEventListener("click", () => {
    if (state.isStreaming) stopRealtimeVC();
    else startRealtimeVC();
  });

  el.btnBypass.addEventListener("click", () => {
    state.bypass = !state.bypass;
    el.btnBypass.textContent = `🔀 バイパス: ${state.bypass ? "ON" : "OFF"}`;
    if (state.workletNode) {
      state.workletNode.port.postMessage({ type: "set_bypass", value: state.bypass });
    }
    if (state.fileDryGain && state.fileWetGain && state.filePlayCtx) {
      const t = state.filePlayCtx.currentTime;
      state.fileDryGain.gain.setTargetAtTime(state.bypass ? 1.0 : 0.0, t, 0.01);
      state.fileWetGain.gain.setTargetAtTime(state.bypass ? 0.0 : 1.0, t, 0.01);
    }
  });

  el.fileVcPlayer.addEventListener("play", () => {
    if (state.isConvertingFile && state.filePlayCtx) {
      stopFilePlayback();
    }
  });

  el.btnResetCache.addEventListener("click", () => {
    if (state.knnEngine) state.knnEngine.reset();
    if (state.cacheManager) state.cacheManager.reset();
    if (state.workletNode) state.workletNode.port.postMessage({ type: "reset" });
    log("因果ストリーミングキャッシュをリセットしました。");
  });

  el.engineSelect.addEventListener("change", async () => {
    if (state.isStreaming) await stopRealtimeVC();
    if (state.isConvertingFile) await stopFilePlayback();
    const prevEngine = state.engine;
    state.engine = el.engineSelect.value || "knn";
    if (prevEngine === "seedvc" && state.engine !== "seedvc" && state.seedvcEngine && state.seedvcEngine.lowMemoryMode) {
      await state.seedvcEngine.dispose();
    }
    const labels = {
      seedvc: "Seed-VC (WebGPU 44.1kHz)",
      knn: "kNN-VC (WavLM+HiFi-GAN)",
      flow: "Causal FlowVC",
    };
    log(`エンジン切替: ${labels[state.engine] || state.engine}`);
    await initModels();
    if (state.lastRefAudio && state.extractedRefByEngine[state.engine] !== state.lastRefAudio) {
      try {
        log(`切替先エンジンで参照音声 [${state.lastRefLabel || "参照音声"}] を自動抽出中...`);
        await extractSpeakerEmbedding(
          state.lastRefAudio,
          state.lastRefLabel || "参照音声",
          true,
          state.lastRefSampleRate || SAMPLE_RATE
        );
      } catch (err) {
        log(`参照自動抽出エラー: ${err.message}`);
      }
    }
  });

  el.btnRunBench.addEventListener("click", () => {
    runBenchmark();
  });

  // Boot
  if (location.search.includes("test=chunk_stream")) {
    state.engine = "seedvc";
    if (el.engineSelect) el.engineSelect.value = "seedvc";
    initModels().then(async () => {
      try {
        const sr = window.SeedVcEngine.SAMPLE_RATE;
        const ref = generateSyntheticReferenceAudio(8.0, 220.0);
        await state.seedvcEngine.setReference(ref, sr);
        state.seedvcEngine.resetStream();
        const src = generateSyntheticReferenceAudio(12.0, 330.0);
        const { condFull, T_full, srcEnv } = await state.seedvcEngine.precomputeSourceCond(src, sr);
        const chunkFrames = 516;
        const totalChunks = Math.ceil(T_full / chunkFrames);
        let prevLast = 0;
        for (let c = 0; c < totalChunks; c++) {
          const conv = await state.seedvcEngine.convertCondWindow(condFull, T_full, c * chunkFrames, chunkFrames, 16, 8, { steps: 1, cfgRate: 0.0, srcEnv });
          const a = conv.audio;
          let maxAmp1 = 0, sum1 = 0, maxJump1 = 0, maxAmpAll = 0;
          for (let i = 0; i < a.length; i++) {
            const av = Math.abs(a[i]);
            if (av > maxAmpAll) maxAmpAll = av;
            if (i < 2048) {
              sum1 += a[i] * a[i];
              if (av > maxAmp1) maxAmp1 = av;
              if (i > 0) maxJump1 = Math.max(maxJump1, Math.abs(a[i] - a[i - 1]));
            }
          }
          const bJump = c > 0 ? Math.abs(a[0] - prevLast) : 0;
          prevLast = a[a.length - 1];
          // Quick band energy check (<400Hz vs 1-3kHz) on first 4096 samples around middle
          let eLow = 0, eMid = 0, eTot = 1e-12;
          const midOff = Math.min(a.length - 1024, Math.floor(a.length / 2));
          if (midOff >= 0) {
            const N = 1024;
            for (let k = 1; k < N / 2; k++) {
              const freq = (k * sr) / N;
              let re = 0, im = 0;
              for (let n = 0; n < N; n++) {
                const ang = (-2 * Math.PI * k * n) / N;
                re += a[midOff + n] * Math.cos(ang);
                im += a[midOff + n] * Math.sin(ang);
              }
              const p = re * re + im * im;
              eTot += p;
              if (freq < 400) eLow += p;
              else if (freq >= 1000 && freq <= 3000) eMid += p;
            }
          }
          log(
            `[CHUNK_TEST c=${c + 1}] ms=${conv.elapsedMs.toFixed(0)} dur=${(a.length / sr).toFixed(2)}s rms=${conv.rms.toFixed(3)} pk=${maxAmpAll.toFixed(3)} ` +
            `<400=${((eLow / eTot) * 100).toFixed(1)}% 1-3k=${((eMid / eTot) * 100).toFixed(1)}% bJump=${bJump.toFixed(4)}`
          );
        }
        log("[CHUNK_TEST DONE]");
      } catch (e) {
        log(`[CHUNK_TEST ERR] ${e.message}`);
      }
    });
  } else if (location.search.includes("test=live_stream")) {
    state.engine = "seedvc";
    if (el.engineSelect) el.engineSelect.value = "seedvc";
    initModels().then(async () => {
      try {
        const sr = window.SeedVcEngine.SAMPLE_RATE;
        const ref = generateSyntheticReferenceAudio(8.0, 220.0);
        await state.seedvcEngine.setReference(ref, sr);
        state.seedvcEngine.resetStream();
        const chunkLen = state.seedvcEngine.streamChunkSamples; // 40 * 512 = 20480 (464.4ms)
        const rightLen = state.seedvcEngine.streamRightSamples; // 8 * 512 = 4096
        const src = generateSyntheticReferenceAudio(4.0, 330.0);
        for (let c = 0; c < 5; c++) {
          const piece = src.subarray(c * chunkLen, (c + 1) * chunkLen);
          const rightCtx = src.subarray((c + 1) * chunkLen, (c + 1) * chunkLen + rightLen);
          const conv = await state.seedvcEngine.convertChunkStream(piece, { pitchShift: 0, steps: 1, cfgRate: 0.0, rightCtx });
          const durMs = (piece.length / sr) * 1000;
          const rtf = conv.elapsedMs / durMs;
          const tm = conv.timings
            ? `[W:${conv.timings.tWhisperMs.toFixed(0)}ms Y:${conv.timings.tYinMs.toFixed(0)}ms D:${conv.timings.tDitMs.toFixed(0)}ms BV:${conv.timings.tBigvganMs.toFixed(0)}ms]`
            : "";
          log(`[LIVE_TEST c=${c + 1}] ms=${conv.elapsedMs.toFixed(0)}/${durMs.toFixed(0)} rtf=${rtf.toFixed(3)} pk=${conv.peak.toFixed(3)} rms=${conv.rms.toFixed(3)} ${tm}`);
        }
        log("[LIVE_TEST DONE]");
      } catch (e) {
        log(`[LIVE_TEST ERR] ${e.message}`);
      }
    });
  } else {
    state.engine = (el.engineSelect && el.engineSelect.value) || "seedvc";
    initModels();
  }
})();
