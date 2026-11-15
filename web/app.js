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
    engine: "knn",
    knnEngine: null,
    knnReady: false,
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
    fileOutputUrl: null,
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
    return state.engine === "knn" ? state.knnReady : state.modelsReady;
  }

  function hasReferenceReady() {
    if (state.engine === "knn") {
      return Boolean(state.knnEngine && state.knnEngine.nRef > 0);
    }
    return Boolean(state.spkEmbData);
  }

  async function extractSpeakerEmbedding(float32Audio, label = "参照音声", updateUi = true) {
    if (state.engine === "knn") {
      if (!state.knnReady || !state.knnEngine) {
        throw new Error("kNN-VC モデルがまだロードされていません。");
      }
      if (updateUi) el.refStatus.textContent = "kNN 特徴抽出中...";
      const res = await state.knnEngine.setReference(float32Audio, SAMPLE_RATE, (w, tot) => {
        if (updateUi) el.refStatus.textContent = `kNN 特徴抽出中 (${w}/${tot})...`;
      });
      if (updateUi) {
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

    // Ensure at least 240 samples and trim/pad cleanly around 3 seconds
    const usableLen = Math.max(240, Math.min(float32Audio.length, SAMPLE_RATE * 10));
    const inputAudio = float32Audio.subarray(0, usableLen);
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

    const createWithProvider = async (providers) => {
      const spk = await ort.InferenceSession.create("/models/speaker_encoder.onnx", {
        executionProviders: providers,
      });
      const core = await ort.InferenceSession.create("/models/streaming_vc_core.onnx", {
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
    state.backendName = `WASM ${ort.env.wasm.numThreads}T (kNN-VC)`;
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

      if (state.engine === "knn") {
        await initKnnModels();
        el.statusModel.textContent = "準備完了 (kNN-VC 16kHz)";
        log("kNN-VC 初期化完了 (WavLM-int8 127MB + HiFi-GAN 66MB)");
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

  async function resampleTo24kMono(audioBuffer) {
    const targetLen = Math.max(240, Math.ceil(audioBuffer.duration * SAMPLE_RATE));
    const offline = new OfflineAudioContext(1, targetLen, SAMPLE_RATE);
    const src = offline.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(offline.destination);
    src.start(0);
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0);
  }

  async function decodeAudioFileTo24kMono(file) {
    const arrayBuf = await file.arrayBuffer();
    const tempCtx = new AudioContext();
    try {
      const decoded = await tempCtx.decodeAudioData(arrayBuf);
      return await resampleTo24kMono(decoded);
    } finally {
      await tempCtx.close();
    }
  }

  async function handleFileUpload(file) {
    if (!file) return;
    try {
      log(`参照音声をデコード中: ${file.name}`);
      const mono24k = await decodeAudioFileTo24kMono(file);
      await extractSpeakerEmbedding(mono24k, file.name);
    } catch (err) {
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

  async function stopFilePlayback() {
    state.fileAbort = true;
    if (state.filePlayCtx) {
      try {
        await state.filePlayCtx.close();
      } catch (_) {}
      state.filePlayCtx = null;
    }
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
    el.btnStopFileVc.hidden = false;
    el.fileVcStatus.textContent = `デコード中: ${file.name}...`;

    try {
      log(`変換元ファイルをデコード中: ${file.name}`);
      const mono24k = await decodeAudioFileTo24kMono(file);
      if (!hasReferenceReady()) {
        await extractSpeakerEmbedding(generateSyntheticReferenceAudio(), "自動テスト音声");
      }

      const isKnn = state.engine === "knn";
      const playRate = isKnn ? window.KnnVcEngine.SAMPLE_RATE : SAMPLE_RATE;
      const chunkLen = isKnn ? window.KnnVcEngine.CHUNK : CHUNK_SAMPLES;
      const alignTrim = isKnn ? window.KnnVcEngine.ALIGN_TRIM_SAMPLES : 720;
      const srcAudio = isKnn
        ? await window.KnnVcEngine.resampleMono(mono24k, SAMPLE_RATE, playRate)
        : mono24k;

      if (isKnn) {
        state.knnEngine.reset();
      } else {
        state.cacheManager.reset();
      }

      const playCtx = new AudioContext({ sampleRate: playRate, latencyHint: "interactive" });
      state.filePlayCtx = playCtx;
      if (playCtx.state === "suspended") {
        await playCtx.resume();
      }

      const totalChunks = Math.ceil(srcAudio.length / chunkLen);
      const outRaw = new Float32Array(totalChunks * chunkLen);
      const chunkBuf = new Float32Array(chunkLen);
      let inferTotalMs = 0;
      let playedChunks = 0;
      // Pre-buffer initial chunks before starting audio clock so inference latency never causes initial underrun
      const prebufferCount = Math.min(totalChunks, isKnn ? 3 : 2);
      const prebuffered = [];
      let nextPlayTime = 0;

      const schedulePlay = (samples) => {
        const audioBuf = playCtx.createBuffer(1, chunkLen, playRate);
        audioBuf.copyToChannel(samples, 0);
        const srcNode = playCtx.createBufferSource();
        srcNode.buffer = audioBuf;
        srcNode.connect(playCtx.destination);
        const now = playCtx.currentTime;
        if (nextPlayTime === 0) {
          nextPlayTime = now + 0.05;
        } else if (nextPlayTime < now + 0.005) {
          nextPlayTime = now + (isKnn ? 0.15 : 0.05);
        }
        srcNode.start(nextPlayTime);
        nextPlayTime += chunkLen / playRate;
      };

      for (let idx = 0; idx < totalChunks; idx++) {
        if (state.fileAbort) break;

        const start = idx * chunkLen;
        const end = Math.min(srcAudio.length, start + chunkLen);
        chunkBuf.fill(0);
        chunkBuf.set(srcAudio.subarray(start, end));

        const inputRms = computeRms(chunkBuf);
        let playSamples;
        if (state.bypass) {
          playSamples = new Float32Array(chunkBuf);
          outRaw.set(playSamples, start);
          el.meterIn.style.width = `${Math.min(100, inputRms * 350)}%`;
          el.meterOut.style.width = `${Math.min(100, inputRms * 350)}%`;
          drawWaveforms(chunkBuf, playSamples);
        } else {
          let outChunk;
          let inferMs;
          if (isKnn) {
            const t0 = performance.now();
            outChunk = await state.knnEngine.processChunk(chunkBuf);
            inferMs = performance.now() - t0;
            state.totalChunks += 1;
            el.perfChunks.textContent = String(state.totalChunks);
          } else {
            const res = await runStreamingChunk(chunkBuf);
            outChunk = res.outChunk;
            inferMs = res.inferMs;
          }
          playSamples = outChunk;
          outRaw.set(outChunk, start);
          inferTotalMs += inferMs;
          const outRms = computeRms(outChunk);
          el.meterIn.style.width = `${Math.min(100, inputRms * 350)}%`;
          el.meterOut.style.width = `${Math.min(100, outRms * 350)}%`;
          el.perfAvgMs.textContent = `${inferMs.toFixed(2)} ms`;
          el.perfRtf.textContent = (inferMs / (isKnn ? 200.0 : 40.0)).toFixed(3);
          drawWaveforms(chunkBuf, outChunk);
        }

        if (state.fileAbort) break;
        playedChunks = idx + 1;
        if (idx < prebufferCount - 1) {
          prebuffered.push(playSamples);
          el.fileVcStatus.textContent = `先読みバッファ中 (${playedChunks}/${prebufferCount})...`;
          continue;
        }
        if (prebuffered.length > 0) {
          for (const buf of prebuffered) schedulePlay(buf);
          prebuffered.length = 0;
        }
        schedulePlay(playSamples);

        el.fileVcStatus.textContent = `リアルタイム再生中 (${playedChunks}/${totalChunks})...`;

        const maxAhead = isKnn ? 1.20 : 0.30;
        while (!state.fileAbort && nextPlayTime - playCtx.currentTime > maxAhead) {
          await new Promise((r) => setTimeout(r, 25));
        }
      }

      if (!state.fileAbort && prebuffered.length > 0) {
        for (const buf of prebuffered) schedulePlay(buf);
        prebuffered.length = 0;
      }

      if (!state.fileAbort) {
        const remainMs = Math.max(0, (nextPlayTime - playCtx.currentTime) * 1000);
        if (remainMs > 0) {
          await new Promise((r) => setTimeout(r, Math.min(remainMs + 40, 2000)));
        }
      }

      if (state.filePlayCtx === playCtx) {
        try {
          await playCtx.close();
        } catch (_) {}
        state.filePlayCtx = null;
      }

      if (playedChunks > 0) {
        const delay = state.bypass ? 0 : Math.min(alignTrim, playedChunks * chunkLen);
        const validEnd = Math.min(playedChunks * chunkLen, delay + srcAudio.length);
        const trimmed = outRaw.subarray(delay, Math.max(delay, validEnd));
        const wavBlob = encodeWavBlob(trimmed, playRate);

        if (state.fileOutputUrl) {
          URL.revokeObjectURL(state.fileOutputUrl);
        }
        state.fileOutputUrl = URL.createObjectURL(wavBlob);
        el.fileVcPlayer.src = state.fileOutputUrl;
        el.fileVcDownload.href = state.fileOutputUrl;
        const baseName = file.name.replace(/\.[^.]+$/, "");
        el.fileVcDownload.download = `${baseName}_converted.wav`;
        el.fileVcOutput.hidden = false;
      }

      const avgMs = playedChunks > 0 ? inferTotalMs / playedChunks : 0;
      const durSec = srcAudio.length / playRate;
      el.fileVcStatus.textContent = state.fileAbort
        ? `停止: ${file.name} (${playedChunks}/${totalChunks}チャンク)`
        : `再生完了: ${file.name} (${durSec.toFixed(2)}秒 / 平均 ${avgMs.toFixed(1)} ms)`;
      log(`ファイルリアルタイム再生${state.fileAbort ? "停止" : "完了"} [${file.name}]: ${playedChunks}チャンク, 平均 ${avgMs.toFixed(2)} ms`);
    } catch (err) {
      el.fileVcStatus.textContent = `再生エラー: ${err.message}`;
      log(`ファイル再生エラー: ${err.message}`);
    } finally {
      state.isConvertingFile = false;
      state.fileAbort = false;
      el.btnStopFileVc.hidden = true;
      el.srcFileInput.disabled = false;
      el.srcFileInput.value = "";
      el.meterIn.style.width = "0%";
      el.meterOut.style.width = "0%";
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

  async function startRealtimeVC() {
    if (state.isStreaming) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      log("マイク入力がサポートされていません（HTTPSまたはlocalhost環境が必要です）。");
      return;
    }

    try {
      if (!hasReferenceReady()) {
        await extractSpeakerEmbedding(generateSyntheticReferenceAudio(), "自動テスト音声");
      }
      const isKnn = state.engine === "knn";
      if (isKnn) {
        state.knnEngine.reset();
      } else {
        state.cacheManager.reset();
      }
      const ctxRate = isKnn ? window.KnnVcEngine.SAMPLE_RATE : SAMPLE_RATE;

      const stream = await navigator.mediaDevices.getUserMedia({
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

      const source = ctx.createMediaStreamSource(stream);
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

      el.btnToggleVc.textContent = "⏹️ リアルタイム変換停止";
      el.btnToggleVc.classList.add("active-stop");
      log(`リアルタイムVC開始 (AudioContext SR: ${ctx.sampleRate} Hz)`);
    } catch (err) {
      log(`VC開始エラー: ${err.message}`);
      await stopRealtimeVC();
    }
  }

  async function stopRealtimeVC() {
    state.isStreaming = false;
    state.pendingChunk = null;

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

    el.btnToggleVc.textContent = "▶️ リアルタイム変換開始";
    el.btnToggleVc.classList.remove("active-stop");
    el.meterIn.style.width = "0%";
    el.meterOut.style.width = "0%";
    log("リアルタイムVC停止");
  }

  async function runBenchmark() {
    if (!activeModelsReady()) return;
    el.btnRunBench.disabled = true;
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

  el.btnRecordRef.addEventListener("click", () => {
    record3SecReference();
  });

  el.btnTestRef.addEventListener("click", async () => {
    const synth = generateSyntheticReferenceAudio(REF_SECONDS, 220.0);
    await extractSpeakerEmbedding(synth, "内蔵テスト音声 (220Hz)");
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
    state.engine = el.engineSelect.value === "flow" ? "flow" : "knn";
    log(`エンジン切替: ${state.engine === "knn" ? "kNN-VC (WavLM+HiFi-GAN)" : "Causal FlowVC"}`);
    await initModels();
  });

  el.btnRunBench.addEventListener("click", () => {
    runBenchmark();
  });

  // Boot
  state.engine = el.engineSelect && el.engineSelect.value === "flow" ? "flow" : "knn";
  initModels();
})();
