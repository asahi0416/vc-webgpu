/**
 * Seed-VC (f0_condition=True, 44.1kHz BigVGAN) WebGPU / WASM ONNX Engine
 * Sequential load -> infer -> release per stage to stay within browser VRAM/RAM limits.
 */
(function () {
  const SR_OUT = 44100;
  const SR_16K = 16000;
  function getModelBase() {
    return (window.VC_CONFIG && window.VC_CONFIG.getModelUrl("seedvc")) || "/models/seedvc";
  }

  // ── Pure-JS Radix-2 FFT (in-place) ─────────────────────────────────────────
  function makeFft(n) {
    const bits = Math.round(Math.log2(n));
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let x = i;
      let y = 0;
      for (let b = 0; b < bits; b++) {
        y = (y << 1) | (x & 1);
        x >>= 1;
      }
      rev[i] = y;
    }
    const cosTab = new Float32Array(n / 2);
    const sinTab = new Float32Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      const ang = (-2 * Math.PI * i) / n;
      cosTab[i] = Math.cos(ang);
      sinTab[i] = Math.sin(ang);
    }
    return function fft(re, im) {
      for (let i = 0; i < n; i++) {
        const j = rev[i];
        if (i < j) {
          let t = re[i]; re[i] = re[j]; re[j] = t;
          t = im[i]; im[i] = im[j]; im[j] = t;
        }
      }
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1;
        const step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = 0; j < half; j++) {
            const k = j * step;
            const wr = cosTab[k];
            const wi = sinTab[k];
            const idx1 = i + j;
            const idx2 = idx1 + half;
            const tr = wr * re[idx2] - wi * im[idx2];
            const ti = wr * im[idx2] + wi * re[idx2];
            re[idx2] = re[idx1] - tr;
            im[idx2] = im[idx1] - ti;
            re[idx1] += tr;
            im[idx1] += ti;
          }
        }
      }
    };
  }

  function reflectPad(sig, padLeft, padRight) {
    const n = sig.length;
    const out = new Float32Array(n + padLeft + padRight);
    for (let i = 0; i < padLeft; i++) {
      let idx = padLeft - i;
      while (idx < 0 || idx >= n) {
        if (idx < 0) idx = -idx;
        if (idx >= n) idx = 2 * (n - 1) - idx;
      }
      out[i] = sig[idx];
    }
    out.set(sig, padLeft);
    for (let i = 0; i < padRight; i++) {
      let idx = n - 2 - i;
      while (idx < 0 || idx >= n) {
        if (idx < 0) idx = -idx;
        if (idx >= n) idx = 2 * (n - 1) - idx;
      }
      out[padLeft + n + i] = sig[idx];
    }
    return out;
  }

  function periodicHann(n) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      w[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / n));
    }
    return w;
  }

  function resample44to16Fast(src44) {
    const n = src44.length;
    const dstLen = Math.max(1, Math.round((n * SR_16K) / SR_OUT));
    const out = new Float32Array(dstLen);
    const ratio = SR_OUT / SR_16K;
    for (let i = 0; i < dstLen; i++) {
      const center = i * ratio;
      const idx = Math.floor(center);
      const frac = center - idx;
      const s0 = src44[Math.max(0, idx - 1)];
      const s1 = src44[Math.min(n - 1, idx)];
      const s2 = src44[Math.min(n - 1, idx + 1)];
      const s3 = src44[Math.min(n - 1, idx + 2)];
      const a = 0.25 * s0 + 0.5 * s1 + 0.25 * s2;
      const b = 0.25 * s1 + 0.5 * s2 + 0.25 * s3;
      out[i] = a + (b - a) * frac;
    }
    return out;
  }

  async function resampleAudio(audio, srcRate, dstRate) {
    if (srcRate === dstRate) return audio;
    const dstLen = Math.max(1, Math.round((audio.length * dstRate) / srcRate));
    if (typeof OfflineAudioContext !== "undefined") {
      const ctx = new OfflineAudioContext(1, dstLen, dstRate);
      const buf = ctx.createBuffer(1, audio.length, srcRate);
      buf.copyToChannel(audio, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.start(0);
      const rendered = await ctx.startRendering();
      return new Float32Array(rendered.getChannelData(0));
    }
    // Linear fallback
    const out = new Float32Array(dstLen);
    const ratio = srcRate / dstRate;
    for (let i = 0; i < dstLen; i++) {
      const pos = i * ratio;
      const i0 = Math.floor(pos);
      const i1 = Math.min(audio.length - 1, i0 + 1);
      const frac = pos - i0;
      out[i] = audio[i0] * (1 - frac) + audio[i1] * frac;
    }
    return out;
  }

  // Deterministic PRNG (Mulberry32 + Box-Muller) for reproducible DiT noise
  function makeRng(seed) {
    let a = seed >>> 0;
    function next() {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    return function randn() {
      const u1 = Math.max(1e-12, next());
      const u2 = next();
      return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    };
  }

  function tensorStats(arr) {
    let nNan = 0;
    let mn = 1e9;
    let mx = -1e9;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (!Number.isFinite(v)) { nNan++; continue; }
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    return `nan=${nNan}/${arr.length} [${mn.toFixed(2)},${mx.toFixed(2)}]`;
  }

  // 2nd-order Biquad filter (Butterworth) for multiband envelope & consonant articulation preservation
  function makeBiquadFilter(type, fc, fs = SR_OUT, Q = 0.7071) {
    const w0 = (2.0 * Math.PI * fc) / fs;
    const cosW0 = Math.cos(w0);
    const sinW0 = Math.sin(w0);
    const alpha = sinW0 / (2.0 * Q);
    let b0, b1, b2, a0, a1, a2;
    if (type === "lowpass") {
      b0 = (1.0 - cosW0) / 2.0;
      b1 = 1.0 - cosW0;
      b2 = (1.0 - cosW0) / 2.0;
      a0 = 1.0 + alpha;
      a1 = -2.0 * cosW0;
      a2 = 1.0 - alpha;
    } else {
      b0 = (1.0 + cosW0) / 2.0;
      b1 = -(1.0 + cosW0);
      b2 = (1.0 + cosW0) / 2.0;
      a0 = 1.0 + alpha;
      a1 = -2.0 * cosW0;
      a2 = 1.0 - alpha;
    }
    return {
      b0: b0 / a0,
      b1: b1 / a0,
      b2: b2 / a0,
      a1: a1 / a0,
      a2: a2 / a0,
    };
  }

  function applyBiquad(wav, coeffs) {
    const out = new Float32Array(wav.length);
    const { b0, b1, b2, a1, a2 } = coeffs;
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < wav.length; i++) {
      const x0 = wav[i];
      const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      out[i] = y0;
      x2 = x1;
      x1 = x0;
      y2 = y1;
      y1 = y0;
    }
    return out;
  }

  class SeedVcEngine {
    static SAMPLE_RATE = SR_OUT;

    constructor() {
      this.ort = null;
      this.hasWebGpu = false;
      this.backendLabel = "WASM";
      this.banks = null;
      this.bankData = null;
      this.whisperCos = null;
      this.whisperSin = null;
      this.fft512 = makeFft(512);
      this.fft1024 = makeFft(1024);
      this.fft2048 = makeFft(2048);

      // Cached reference speaker data & ONNX sessions
      this.ref = null;
      this.sessions = new Map();
      this.ortLock = Promise.resolve();
      this.warmedShapes = new Set();
      this.resetStream();
    }

    async load(ort, onProgress = () => {}) {
      this.ort = ort;
      this.hasWebGpu = typeof navigator !== "undefined" && Boolean(navigator.gpu);
      this.backendLabel = this.hasWebGpu ? "WebGPU (FP16)" : "WASM (FP16)";
      onProgress("メルフィルタバンク読込中...");

      const modelBase = getModelBase();
      const [metaRes, binRes, fcpeMelRes] = await Promise.all([
        fetch(`${modelBase}/mel_banks.json`),
        fetch(`${modelBase}/mel_banks.bin`),
        fetch(`${modelBase}/fcpe_mel_128x513.bin`).catch(() => null),
      ]);
      if (!metaRes.ok || !binRes.ok) {
        throw new Error("mel_banks の読込に失敗しました");
      }
      this.banks = await metaRes.json();
      const buf = await binRes.arrayBuffer();
      this.bankData = new Float32Array(buf);
      this.fcpeMelBank = fcpeMelRes && fcpeMelRes.ok ? new Float32Array(await fcpeMelRes.arrayBuffer()) : null;

      // Precompute 400-point DFT tables for Whisper (201 x 400)
      const nFft = 400;
      const nFreq = 201;
      this.whisperCos = new Float32Array(nFreq * nFft);
      this.whisperSin = new Float32Array(nFreq * nFft);
      for (let k = 0; k < nFreq; k++) {
        const base = k * nFft;
        for (let n = 0; n < nFft; n++) {
          const ang = (-2.0 * Math.PI * k * n) / nFft;
          this.whisperCos[base + n] = Math.cos(ang);
          this.whisperSin[base + n] = Math.sin(ang);
        }
      }
      onProgress(`準備完了 (${this.backendLabel})`);
    }

    getBank(name) {
      const m = this.banks[name];
      return {
        rows: m.rows,
        cols: m.cols,
        data: this.bankData.subarray(m.offsetFloats, m.offsetFloats + m.lengthFloats),
      };
    }

    async getSession(filename, preferWebGpu = true) {
      const cpuParam = typeof location !== "undefined" ? new URLSearchParams(location.search).get("seedvc_cpu") : null;
      if (cpuParam && (cpuParam === "all" || filename.includes(cpuParam))) {
        preferWebGpu = false;
      }
      const key = `${filename}:${preferWebGpu ? "gpu" : "cpu"}`;
      if (this.sessions.has(key)) {
        return this.sessions.get(key);
      }
      const sess = await this.createSession(filename, preferWebGpu);
      this.sessions.set(key, sess);
      return sess;
    }

    async runSession(sess, feeds) {
      // ORT Web cannot execute two sessions concurrently on the same WASM/WebGPU runtime ("Session mismatch")
      const next = this.ortLock.then(
        () => sess.run(feeds),
        () => sess.run(feeds)
      );
      this.ortLock = next.catch(() => {});
      return await next;
    }

    async dispose() {
      for (const sess of this.sessions.values()) {
        try {
          await sess.release();
        } catch (_) {}
      }
      this.sessions.clear();
    }

    async createSession(filename, preferWebGpu = true) {
      const url = `${getModelBase()}/${filename}`;
      const eps = preferWebGpu && this.hasWebGpu ? ["webgpu", "wasm"] : ["wasm"];
      try {
        return await this.ort.InferenceSession.create(url, {
          executionProviders: eps,
          graphOptimizationLevel: "all",
        });
      } catch (e) {
        if (eps[0] === "webgpu") {
          console.warn(`[Seed-VC] WebGPU fallback to WASM for ${filename}:`, e);
          return await this.ort.InferenceSession.create(url, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all",
          });
        }
        throw e;
      }
    }

    // ── DSP Feature Extractors ───────────────────────────────────────────────

    computeWhisperFeatures(wav16, targetFrames = 3000) {
      const nFft = 400;
      const hop = 160;
      const nFrames = Math.max(150, Math.min(3000, targetFrames));
      const maxLen = nFrames * hop;
      const activeLen = Math.min(wav16.length, maxLen);
      const paddedSig = new Float32Array(maxLen);
      paddedSig.set(wav16.subarray(0, activeLen));

      const nFreq = 201;
      const win = periodicHann(nFft);
      const sig = reflectPad(paddedSig, 200, 200);
      const bank = this.getBank("whisper");
      const out = new Float32Array(80 * nFrames);

      const frame = new Float32Array(nFft);
      const pow = new Float32Array(nFreq);
      const activeFrames = Math.min(nFrames, Math.ceil(activeLen / hop));

      // Compute log10 mel spectrogram for active frames; fill silent remainder with min value
      let globalMax = -1e9;
      for (let t = 0; t < activeFrames; t++) {
        const off = t * hop;
        for (let n = 0; n < nFft; n++) {
          frame[n] = sig[off + n] * win[n];
        }
        for (let k = 0; k < nFreq; k++) {
          const base = k * nFft;
          let re = 0;
          let im = 0;
          for (let n = 0; n < nFft; n++) {
            const x = frame[n];
            re += x * this.whisperCos[base + n];
            im += x * this.whisperSin[base + n];
          }
          pow[k] = re * re + im * im;
        }
        for (let m = 0; m < 80; m++) {
          const bOff = m * nFreq;
          let acc = 0;
          for (let k = 0; k < nFreq; k++) {
            acc += bank.data[bOff + k] * pow[k];
          }
          const v = Math.log10(Math.max(acc, 1e-10));
          out[m * nFrames + t] = v;
          if (v > globalMax) globalMax = v;
        }
      }

      const floorVal = globalMax - 8.0;
      const padVal = (floorVal + 4.0) / 4.0;
      for (let m = 0; m < 80; m++) {
        const row = m * nFrames;
        for (let t = 0; t < activeFrames; t++) {
          const clamped = Math.max(out[row + t], floorVal);
          out[row + t] = (clamped + 4.0) / 4.0;
        }
        for (let t = activeFrames; t < nFrames; t++) {
          out[row + t] = padVal;
        }
      }
      return out;
    }

    computeRmvpeMel(wav16) {
      const nFft = 1024;
      const hop = 160;
      const nFreq = 513;
      const win = periodicHann(nFft);
      const sig = reflectPad(wav16, nFft / 2, nFft / 2);
      const nFrames = Math.floor((sig.length - nFft) / hop) + 1;
      const nPad = 32 * Math.ceil(nFrames / 32) - nFrames;
      const totalFrames = nFrames + nPad;
      const bank = this.getBank("rmvpe");
      const out = new Float32Array(128 * totalFrames);

      const re = new Float32Array(nFft);
      const im = new Float32Array(nFft);
      const mag = new Float32Array(nFreq);

      for (let t = 0; t < nFrames; t++) {
        const off = t * hop;
        for (let n = 0; n < nFft; n++) {
          re[n] = sig[off + n] * win[n];
          im[n] = 0;
        }
        this.fft1024(re, im);
        for (let k = 0; k < nFreq; k++) {
          mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
        }
        for (let m = 0; m < 128; m++) {
          const bOff = m * nFreq;
          let acc = 0;
          for (let k = 0; k < nFreq; k++) {
            acc += bank.data[bOff + k] * mag[k];
          }
          out[m * totalFrames + t] = Math.log(Math.max(acc, 1e-5));
        }
      }
      return { mel: out, nFrames, totalFrames };
    }

    decodeRmvpeF0(salience, nFrames, thred = 0.03) {
      const nBins = 360;
      const f0 = new Float32Array(nFrames);
      for (let t = 0; t < nFrames; t++) {
        const base = t * nBins;
        let maxVal = -1;
        let maxIdx = 0;
        for (let i = 0; i < nBins; i++) {
          const v = salience[base + i];
          if (v > maxVal) {
            maxVal = v;
            maxIdx = i;
          }
        }
        if (maxVal <= thred) {
          f0[t] = 0;
          continue;
        }
        let prod = 0;
        let wsum = 0;
        for (let d = -4; d <= 4; d++) {
          const idx = maxIdx + d;
          if (idx >= 0 && idx < nBins) {
            const w = salience[base + idx];
            const cent = 20.0 * idx + 1997.3794084376191;
            prod += w * cent;
            wsum += w;
          }
        }
        const cents = wsum > 0 ? prod / wsum : 0;
        const hz = 10.0 * Math.pow(2.0, cents / 1200.0);
        f0[t] = hz === 10.0 ? 0 : hz;
      }
      return f0;
    }

    computeKaldiFbank(wav16) {
      const frameLen = 400;
      const frameShift = 160;
      const nFft = 512;
      const nFreq = 257;
      const numFrames = Math.max(1, 1 + Math.floor((wav16.length - frameLen) / frameShift));
      const povey = new Float32Array(frameLen);
      for (let i = 0; i < frameLen; i++) {
        povey[i] = Math.pow(0.5 - 0.5 * Math.cos((2.0 * Math.PI * i) / (frameLen - 1)), 0.85);
      }
      const bank = this.getBank("kaldi");
      const out = new Float32Array(numFrames * 80);
      const colMeans = new Float32Array(80);

      const re = new Float32Array(nFft);
      const im = new Float32Array(nFft);
      const pow = new Float32Array(nFreq);
      const tmp = new Float32Array(frameLen);

      for (let t = 0; t < numFrames; t++) {
        const off = t * frameShift;
        let mean = 0;
        for (let i = 0; i < frameLen; i++) {
          const s = off + i < wav16.length ? wav16[off + i] : 0;
          tmp[i] = s;
          mean += s;
        }
        mean /= frameLen;
        for (let i = 0; i < frameLen; i++) {
          tmp[i] -= mean;
        }
        // Preemphasis 0.97 + Povey window
        re.fill(0);
        im.fill(0);
        re[0] = (tmp[0] - 0.97 * tmp[0]) * povey[0];
        for (let i = 1; i < frameLen; i++) {
          re[i] = (tmp[i] - 0.97 * tmp[i - 1]) * povey[i];
        }
        this.fft512(re, im);
        for (let k = 0; k < nFreq; k++) {
          pow[k] = re[k] * re[k] + im[k] * im[k];
        }
        for (let m = 0; m < 80; m++) {
          const bOff = m * nFreq;
          let acc = 0;
          for (let k = 0; k < nFreq; k++) {
            acc += bank.data[bOff + k] * pow[k];
          }
          const v = Math.log(Math.max(acc, 1.1920929e-7));
          out[t * 80 + m] = v;
          colMeans[m] += v;
        }
      }
      for (let m = 0; m < 80; m++) colMeans[m] /= numFrames;
      for (let t = 0; t < numFrames; t++) {
        const row = t * 80;
        for (let m = 0; m < 80; m++) {
          out[row + m] -= colMeans[m];
        }
      }
      return { fbank: out, numFrames };
    }

    computeBigvganMel(wav44) {
      const nFft = 2048;
      const hop = 512;
      const nFreq = 1025;
      const pad = (nFft - hop) / 2; // 768
      const win = periodicHann(nFft);
      const sig = reflectPad(wav44, pad, pad);
      const nFrames = Math.floor((sig.length - nFft) / hop) + 1;
      const bank = this.getBank("bigvgan");
      const out = new Float32Array(128 * nFrames);

      const re = new Float32Array(nFft);
      const im = new Float32Array(nFft);
      const mag = new Float32Array(nFreq);

      for (let t = 0; t < nFrames; t++) {
        const off = t * hop;
        for (let n = 0; n < nFft; n++) {
          re[n] = sig[off + n] * win[n];
          im[n] = 0;
        }
        this.fft2048(re, im);
        for (let k = 0; k < nFreq; k++) {
          mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k] + 1e-9);
        }
        for (let m = 0; m < 128; m++) {
          const bOff = m * nFreq;
          let acc = 0;
          for (let k = 0; k < nFreq; k++) {
            acc += bank.data[bOff + k] * mag[k];
          }
          out[m * nFrames + t] = Math.log(Math.max(acc, 1e-5));
        }
      }
      return { mel: out, nFrames };
    }

    // ── Model Helpers ────────────────────────────────────────────────────────

    async runWhisper(wav16, label, onProgress) {
      onProgress(`Whisper 特徴抽出中 (${label})...`);
      const neededMelFrames = Math.ceil(wav16.length / 160);
      // Bucket window size to [150 (1.5s), 300 (3s), 600 (6s), 1200 (12s), 3000 (30s)] so WebGPU caches only 5 shapes
      // and short streaming chunks avoid silence padding dilution in Self-Attention.
      let winFrames = 3000;
      if (neededMelFrames <= 140) winFrames = 150;
      else if (neededMelFrames <= 260) winFrames = 300;
      else if (neededMelFrames <= 540) winFrames = 600;
      else if (neededMelFrames <= 1100) winFrames = 1200;

      const feats = this.computeWhisperFeatures(wav16, winFrames);
      const nFrames = Math.floor((wav16.length / SR_16K) * 50);
      const sess = await this.getSession("whisper_small_enc.onnx", true);
      const inp = new this.ort.Tensor("float32", feats, [1, 80, winFrames]);
      const res = await this.runSession(sess, { input_features: inp });
      const full = res.hidden_states.data; // (1, winFrames/2, 768)
      const out = new Float32Array(nFrames * 768);
      out.set(full.subarray(0, Math.min(full.length, nFrames * 768)));
      return { data: out, frames: nFrames };
    }

    computeYinF0(wav16) {
      const sr = SR_16K;
      const win = 400;
      const hop = 160;
      const minTau = 21;   // ~762 Hz
      const maxTau = 213;  // ~75.1 Hz
      const nLags = maxTau - minTau + 1;
      const nFrames = Math.max(1, 1 + Math.floor((wav16.length - win) / hop));
      const padLeft = win >> 1;
      const padded = new Float32Array(wav16.length + padLeft + (win >> 1) + maxTau);
      padded.set(wav16, padLeft);

      const d = new Float32Array(nLags);
      const cmnd = new Float32Array(nLags);
      const rawF0 = new Float32Array(nFrames);

      for (let t = 0; t < nFrames; t++) {
        const refStart = t * hop + maxTau;
        let energy = 0;
        for (let j = 0; j < win; j++) {
          const v = padded[refStart + j];
          energy += v * v;
        }
        if (Math.sqrt(energy / win + 1e-12) < 0.005) continue;

        let cum = 0;
        let firstDip = -1;
        let minIdx = 0;
        let minVal = 1e9;
        for (let k = 0; k < nLags; k++) {
          const tau = minTau + k;
          const lagStart = refStart - tau;
          let sumSq = 0;
          for (let j = 0; j < win; j++) {
            const diff = padded[refStart + j] - padded[lagStart + j];
            sumSq += diff * diff;
          }
          d[k] = sumSq;
          cum += sumSq;
          const cVal = (sumSq * (k + 1)) / (cum + 1e-8);
          cmnd[k] = cVal;
          if (cVal < minVal) {
            minVal = cVal;
            minIdx = k;
          }
        }
        for (let k = 1; k < nLags - 1; k++) {
          if (cmnd[k] < 0.22 && cmnd[k] <= cmnd[k - 1] && cmnd[k] <= cmnd[k + 1]) {
            firstDip = k;
            break;
          }
        }
        const idx = firstDip >= 0 ? firstDip : minIdx;
        if (cmnd[idx] > 0.35) continue;

        let delta = 0;
        if (idx > 0 && idx < nLags - 1) {
          const denom = 2.0 * (cmnd[idx - 1] - 2.0 * cmnd[idx] + cmnd[idx + 1]);
          if (Math.abs(denom) > 1e-6) {
            delta = Math.max(-0.5, Math.min(0.5, (cmnd[idx - 1] - cmnd[idx + 1]) / denom));
          }
        }
        rawF0[t] = sr / (minTau + idx + delta);
      }

      // 3-frame median filter on voiced frames
      const out = new Float32Array(rawF0);
      for (let t = 1; t < nFrames - 1; t++) {
        const vals = [];
        if (rawF0[t - 1] > 0) vals.push(rawF0[t - 1]);
        if (rawF0[t] > 0) vals.push(rawF0[t]);
        if (rawF0[t + 1] > 0) vals.push(rawF0[t + 1]);
        if (vals.length >= 2) {
          vals.sort((a, b) => a - b);
          out[t] = vals[vals.length >> 1];
        }
      }
      return out;
    }

    computeFcpeMel(wav16) {
      const nFft = 1024;
      const hop = 160;
      const nFreq = 513;
      const pad = 512;
      const win = periodicHann(nFft);
      const sig = reflectPad(wav16, pad, pad);
      const nFrames = Math.floor((sig.length - nFft) / hop) + 1;
      const bank = this.fcpeMelBank;
      const out = new Float32Array(nFrames * 128);

      const re = new Float32Array(nFft);
      const im = new Float32Array(nFft);
      const mag = new Float32Array(nFreq);

      for (let t = 0; t < nFrames; t++) {
        const off = t * hop;
        for (let n = 0; n < nFft; n++) {
          re[n] = sig[off + n] * win[n];
          im[n] = 0;
        }
        this.fft1024(re, im);
        for (let k = 0; k < nFreq; k++) {
          mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k] + 1e-9);
        }
        const rowOut = t * 128;
        for (let m = 0; m < 128; m++) {
          const bOff = m * nFreq;
          let acc = 0;
          for (let k = 0; k < nFreq; k++) {
            acc += bank[bOff + k] * mag[k];
          }
          out[rowOut + m] = Math.log(Math.max(acc, 1e-5));
        }
      }
      return { mel: out, nFrames };
    }

    decodeFcpeF0(logits, nFrames, threshold = 0.006) {
      const nBins = 360;
      const cMin = 2051.148681640625;
      const cStep = (9151.2890625 - cMin) / 359.0;
      const f0 = new Float32Array(nFrames);
      for (let t = 0; t < nFrames; t++) {
        const base = t * nBins;
        let maxVal = -1e9;
        let maxIdx = 0;
        for (let i = 0; i < nBins; i++) {
          const v = logits[base + i];
          if (v > maxVal) {
            maxVal = v;
            maxIdx = i;
          }
        }
        if (maxVal <= threshold) continue;
        let prod = 0;
        let wsum = 0;
        const s0 = Math.max(0, maxIdx - 4);
        const s1 = Math.min(nBins, maxIdx + 5);
        for (let i = s0; i < s1; i++) {
          const w = Math.max(0, logits[base + i]);
          prod += w * (cMin + i * cStep);
          wsum += w;
        }
        const cents = wsum > 0 ? prod / wsum : cMin + maxIdx * cStep;
        f0[t] = 10.0 * Math.pow(2.0, cents / 1200.0);
      }
      return f0;
    }

    async runRmvpe(wav16, label, onProgress) {
      if (this.fcpeMelBank) {
        try {
          onProgress(`FCPE 歌唱F0抽出中 (${label})...`);
          const { mel, nFrames } = this.computeFcpeMel(wav16);
          const sess = await this.getSession("fcpe.onnx", true);
          // Process in <=500-frame (5s) windows so O(T^2) attention in WASM never freezes the UI thread
          const maxWin = 500;
          if (nFrames <= maxWin) {
            const res = await this.runSession(sess, {
              mel: new this.ort.Tensor("float32", mel, [1, nFrames, 128]),
            });
            const logits = res[sess.outputNames[0]].data;
            return this.decodeFcpeF0(logits, nFrames, 0.006);
          }
          const fullF0 = new Float32Array(nFrames);
          for (let off = 0; off < nFrames; off += maxWin) {
            const len = Math.min(maxWin, nFrames - off);
            const slice = mel.subarray(off * 128, (off + len) * 128);
            const res = await this.runSession(sess, {
              mel: new this.ort.Tensor("float32", slice, [1, len, 128]),
            });
            const logits = res[sess.outputNames[0]].data;
            const partF0 = this.decodeFcpeF0(logits, len, 0.006);
            fullF0.set(partF0, off);
            await new Promise((r) => setTimeout(r, 0));
          }
          return fullF0;
        } catch (err) {
          console.warn("[Seed-VC] FCPE fallback to YIN:", err);
        }
      }
      onProgress(`YIN F0抽出中 (${label})...`);
      return this.computeYinF0(wav16);
    }

    // ── Public API ───────────────────────────────────────────────────────────

    computeFeatureStats(data, frames, dim = 768, wav16 = null) {
      const mean = new Float32Array(dim);
      const std = new Float32Array(dim);
      if (frames <= 0) return { mean, std: std.fill(1) };

      let activeMask = null;
      let activeCount = frames;
      if (wav16 && wav16.length >= 320) {
        const hop = 320; // 20ms at 16kHz -> 50Hz Whisper frame rate
        const rms = new Float32Array(frames);
        let maxRms = 0;
        for (let t = 0; t < frames; t++) {
          const s = t * hop;
          const e = Math.min(wav16.length, s + hop);
          if (e <= s) continue;
          let sq = 0;
          for (let i = s; i < e; i++) sq += wav16[i] * wav16[i];
          const r = Math.sqrt(sq / (e - s));
          rms[t] = r;
          if (r > maxRms) maxRms = r;
        }
        const thr = Math.max(0.008, maxRms * 0.12);
        const mask = new Uint8Array(frames);
        let cnt = 0;
        for (let t = 0; t < frames; t++) {
          if (rms[t] > thr) {
            mask[t] = 1;
            cnt++;
          }
        }
        if (cnt >= 10) {
          activeMask = mask;
          activeCount = cnt;
        }
      }

      for (let t = 0; t < frames; t++) {
        if (activeMask && !activeMask[t]) continue;
        const off = t * dim;
        for (let d = 0; d < dim; d++) mean[d] += data[off + d];
      }
      for (let d = 0; d < dim; d++) mean[d] /= activeCount;
      for (let t = 0; t < frames; t++) {
        if (activeMask && !activeMask[t]) continue;
        const off = t * dim;
        for (let d = 0; d < dim; d++) {
          const diff = data[off + d] - mean[d];
          std[d] += diff * diff;
        }
      }
      for (let d = 0; d < dim; d++) {
        std[d] = Math.sqrt(std[d] / activeCount) + 1e-5;
      }
      return { mean, std };
    }

    applyWhisperAdaIN(data, frames, dim = 768, wav16 = null, useStreamEma = false, f0 = null) {
      if (!this.ref || !this.ref.sOriMean || !this.ref.sOriStd || frames <= 1) return data;
      let { mean: mSrc, std: sSrc } = this.computeFeatureStats(data, frames, dim, wav16);
      if (useStreamEma) {
        if (!this.streamAdaMean || !this.streamAdaStd || this.streamAdaMean.length !== dim) {
          this.streamAdaMean = new Float32Array(mSrc);
          this.streamAdaStd = new Float32Array(sSrc);
        } else {
          for (let d = 0; d < dim; d++) {
            this.streamAdaMean[d] = 0.75 * this.streamAdaMean[d] + 0.25 * mSrc[d];
            this.streamAdaStd[d] = 0.75 * this.streamAdaStd[d] + 0.25 * sSrc[d];
          }
        }
        mSrc = this.streamAdaMean;
        sSrc = this.streamAdaStd;
      }
      const mOri = this.ref.sOriMean;
      const sOri = this.ref.sOriStd;
      const out = new Float32Array(data.length);
      for (let t = 0; t < frames; t++) {
        const off = t * dim;
        // Voiced-Adaptive AdaIN: 有声(母音)フレームは100%話者適応(alpha=1.0)、無声子音フレームは元の子音ダイナミクスを75%保持(alpha=0.25)
        const isVoiced = f0 && t < f0.length ? f0[t] > 40.0 : true;
        const alpha = isVoiced ? 1.0 : 0.25;
        for (let d = 0; d < dim; d++) {
          const adainVal = ((data[off + d] - mSrc[d]) / sSrc[d]) * sOri[d] + mOri[d];
          out[off + d] = data[off + d] * (1.0 - alpha) + adainVal * alpha;
        }
      }
      return out;
    }

    computeVoicedF0Median(f0) {
      const voiced = [];
      for (let i = 0; i < f0.length; i++) {
        if (f0[i] > 40.0) voiced.push(f0[i]);
      }
      if (voiced.length === 0) return 0;
      voiced.sort((a, b) => a - b);
      return voiced[voiced.length >> 1];
    }

    async setReference(audioFloat32, sampleRate, onProgress = () => {}) {
      const t0 = performance.now();
      // DiT acoustic prompt uses up to 8s (~688 frames) for crisp 1-3kHz vocal timbre;
      // CAMPPlus speaker embedding is O(1) w.r.t. DiT attention so it can use up to 15s.
      const maxPromptSamples = Math.min(audioFloat32.length, Math.round(sampleRate * 8));
      const maxStyleSamples = Math.min(audioFloat32.length, Math.round(sampleRate * 15));
      const trimmedPrompt = audioFloat32.subarray(0, maxPromptSamples);
      const trimmedStyle = audioFloat32.subarray(0, maxStyleSamples);

      const ref44 = await resampleAudio(trimmedPrompt, sampleRate, SR_OUT);
      const ref16 = await resampleAudio(trimmedPrompt, sampleRate, SR_16K);
      const style16 =
        maxStyleSamples > maxPromptSamples
          ? await resampleAudio(trimmedStyle, sampleRate, SR_16K)
          : ref16;

      const { mel: mel2, nFrames: T_prompt } = this.computeBigvganMel(ref44);
      const S_ori = await this.runWhisper(ref16, "参照音声", onProgress);
      const F0_ori = await this.runRmvpe(ref16, "参照音声", onProgress);

      onProgress("CAMPPlus 話者特徴抽出中...");
      const { fbank, numFrames: fbFrames } = this.computeKaldiFbank(style16);
      const cpSess = await this.getSession("campplus_fp16.onnx", true);
      const fbTensor = new this.ort.Tensor("float32", fbank, [1, fbFrames, 80]);
      const cpRes = await this.runSession(cpSess, { fbank: fbTensor });
      const style2 = new Float32Array(cpRes[cpSess.outputNames[0]].data);

      onProgress("Length Regulator 処理中 (参照)...");
      const lrSess = await this.getSession("length_regulator_fp16.onnx", true);
      const lrRes = await this.runSession(lrSess, {
        semantic: new this.ort.Tensor("float32", S_ori.data, [1, S_ori.frames, 768]),
        ylens: new this.ort.Tensor("int64", BigInt64Array.from([BigInt(T_prompt)]), [1]),
        f0: new this.ort.Tensor("float32", F0_ori, [1, F0_ori.length]),
      });
      const promptCond = new Float32Array(lrRes[lrSess.outputNames[0]].data);

      const f0Med = this.computeVoicedF0Median(F0_ori);
      const { mean: sOriMean, std: sOriStd } = this.computeFeatureStats(S_ori.data, S_ori.frames, 768, ref16);
      this.ref = {
        mel2,
        T_prompt,
        style2,
        promptCond,
        f0Med,
        sOriMean,
        sOriStd,
        seconds: ref44.length / SR_OUT,
      };
      await this.warmupShaders(
        [
          { winFrames: 540, maxPromptFrames: 0, nSteps: 1, cfgRate: 0.0 },
          { winFrames: 96, maxPromptFrames: 258, nSteps: 1, cfgRate: 0.0 },
        ],
        onProgress
      );
      return {
        seconds: this.ref.seconds,
        frames: T_prompt,
        embedding: style2,
        buildMs: performance.now() - t0,
      };
    }

    async warmupShaders(winConfigs = [{ winFrames: 540, maxPromptFrames: 0, nSteps: 1, cfgRate: 0.4 }], onProgress = () => {}) {
      if (!this.ref || !this.hasWebGpu) return;
      for (const cfg of winConfigs) {
        const winFrames = typeof cfg === "number" ? cfg : cfg.winFrames;
        const maxPromptFrames = typeof cfg === "number" ? 0 : (cfg.maxPromptFrames || 0);
        const cfgRate = typeof cfg === "number" ? 0.0 : (cfg.cfgRate ?? 0.4);
        const effPrompt = maxPromptFrames > 0 ? Math.min(this.ref.T_prompt, maxPromptFrames) : this.ref.T_prompt;
        const key = `${effPrompt}:${winFrames}:${cfgRate > 0 ? 2 : 1}`;
        if (this.warmedShapes.has(key)) continue;
        onProgress(`WebGPU シェーダ事前コンパイル中 (${winFrames}fr)...`);
        const dummyCond = new Float32Array(winFrames * 768);
        await this.runDitAndBigvgan(dummyCond, winFrames, 1, cfgRate, () => {}, maxPromptFrames);
        this.warmedShapes.add(key);
      }
    }

    async convertAudio(audioFloat32, sampleRate, opts = {}, onProgress = () => {}) {
      if (!this.ref) {
        throw new Error("先にSTEP 1で参照音声を登録してください。");
      }
      const t0 = performance.now();
      const pitchShift = Number(opts.pitchShift || 0);
      const nSteps = Number(opts.steps || 3);
      const cfgRate = Number(opts.cfgRate ?? 0.85);
      const maxPromptFrames = Number(opts.maxPromptFrames || 0);

      const src44 = sampleRate === SR_OUT ? audioFloat32 : await resampleAudio(audioFloat32, sampleRate, SR_OUT);
      const src16 = sampleRate === SR_OUT ? resample44to16Fast(src44) : await resampleAudio(audioFloat32, sampleRate, SR_16K);

      const { nFrames: T_src } = this.computeBigvganMel(src44);
      const tY0 = performance.now();
      const F0_alt = await this.runRmvpe(src16, "変換元", onProgress);
      const tYinMs = performance.now() - tY0;

      const tW0 = performance.now();
      const S_alt = await this.runWhisper(src16, "変換元", onProgress);
      if (opts.useAdain !== false) {
        S_alt.data = this.applyWhisperAdaIN(S_alt.data, S_alt.frames, 768, src16, Boolean(opts.isStream), F0_alt);
      }
      const tWhisperMs = performance.now() - tW0;

      if (pitchShift !== 0) {
        const factor = Math.pow(2.0, pitchShift / 12.0);
        for (let i = 0; i < F0_alt.length; i++) {
          if (F0_alt[i] > 0) F0_alt[i] *= factor;
        }
      }

      onProgress("Length Regulator 処理中 (変換元)...");
      const lrSess = await this.getSession("length_regulator_fp16.onnx", true);
      const lrRes = await this.runSession(lrSess, {
        semantic: new this.ort.Tensor("float32", S_alt.data, [1, S_alt.frames, 768]),
        ylens: new this.ort.Tensor("int64", BigInt64Array.from([BigInt(T_src)]), [1]),
        f0: new this.ort.Tensor("float32", F0_alt, [1, F0_alt.length]),
      });
      const cond = new Float32Array(lrRes[lrSess.outputNames[0]].data);
      const { outWav, tDitMs, tBigvganMs, rms, peak } = await this.runDitAndBigvgan(
        cond,
        T_src,
        nSteps,
        cfgRate,
        onProgress,
        maxPromptFrames
      );

      // Multiband adaptive articulation envelope match
      const bLow = makeBiquadFilter("lowpass", 2500, SR_OUT);
      const bHigh = makeBiquadFilter("highpass", 2800, SR_OUT);
      const srcLow = applyBiquad(src44, bLow);
      const srcHigh = applyBiquad(src44, bHigh);
      const srcLowEnv = this.computeSmoothEnv(srcLow, T_src, 512, 6);
      const srcHighEnv = this.computeSmoothEnv(srcHigh, T_src, 512, 2);
      this.applyEnvelopeMatch(outWav, srcLowEnv, srcHighEnv, 0);

      return {
        audio: outWav,
        sampleRate: SR_OUT,
        elapsedMs: performance.now() - t0,
        timings: { tWhisperMs, tYinMs, tDitMs, tBigvganMs },
        rms,
        peak,
      };
    }

    async runDitAndBigvgan(cond, T_src, nSteps, cfgRate, onProgress = () => {}, maxPromptFrames = 0) {
      // Concatenate promptCond (1, T_prompt, 768) + cond (1, T_src, 768) -> (1, T_total, 768)
      const T_fullPrompt = this.ref.T_prompt;
      const T_prompt = maxPromptFrames > 0 ? Math.min(T_fullPrompt, maxPromptFrames) : T_fullPrompt;
      const T_total = T_prompt + T_src;
      const catCond = new Float32Array(T_total * 768);
      catCond.set(this.ref.promptCond.subarray(0, T_prompt * 768), 0);
      catCond.set(cond, T_prompt * 768);

      // Prepare DiT inputs (batch=2: [cond, uncond])
      const randn = makeRng(42);
      const x = new Float32Array(128 * T_total);
      for (let i = 0; i < x.length; i++) x[i] = randn();
      for (let c = 0; c < 128; c++) {
        const row = c * T_total;
        for (let t = 0; t < T_prompt; t++) x[row + t] = 0.0;
      }

      const promptX = new Float32Array(128 * T_total);
      for (let c = 0; c < 128; c++) {
        const dstRow = c * T_total;
        const srcRow = c * T_fullPrompt;
        promptX.set(this.ref.mel2.subarray(srcRow, srcRow + T_prompt), dstRow);
      }

      const tD0 = performance.now();
      const useStudentNoCfg = cfgRate <= 0;
      const useStudentMicroCfg = cfgRate > 0 && nSteps <= 2;
      if (useStudentNoCfg) {
        // Distilled 1-2 step No-CFG Student (batch=1)
        const ditSess = await this.getSession("dit_student_fp16.onnx", true);
        const xLensTensor = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(T_total)]), [1]);
        const pxTensor = new this.ort.Tensor("float32", promptX, [1, 128, T_total]);
        const cTensor = new this.ort.Tensor("float32", catCond, [1, T_total, 768]);
        const sTensor = new this.ort.Tensor("float32", this.ref.style2, [1, 192]);
        const dt = 1.0 / nSteps;
        const stride = 128 * T_total;
        for (let step = 0; step < nSteps; step++) {
          onProgress(`蒸留DiT 推論中 (${step + 1}/${nSteps})...`);
          const tCur = step / nSteps;
          const vRes = await this.runSession(ditSess, {
            x: new this.ort.Tensor("float32", x, [1, 128, T_total]),
            prompt_x: pxTensor,
            x_lens: xLensTensor,
            t: new this.ort.Tensor("float32", new Float32Array([tCur]), [1]),
            style: sTensor,
            cond: cTensor,
          });
          const v = vRes[ditSess.outputNames[0]].data;
          for (let i = 0; i < stride; i++) x[i] += dt * v[i];
          for (let c = 0; c < 128; c++) {
            const row = c * T_total;
            for (let t = 0; t < T_prompt; t++) x[row + t] = 0.0;
          }
        }
      } else {
        const pxIn = new Float32Array(2 * 128 * T_total);
        pxIn.set(promptX, 0);
        const cIn = new Float32Array(2 * T_total * 768);
        cIn.set(catCond, 0);
        const sIn = new Float32Array(2 * 192);
        sIn.set(this.ref.style2, 0);
        const xLensTensor = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(T_total), BigInt(T_total)]), [2]);
        const pxTensor = new this.ort.Tensor("float32", pxIn, [2, 128, T_total]);
        const cTensor = new this.ort.Tensor("float32", cIn, [2, T_total, 768]);
        const sTensor = new this.ort.Tensor("float32", sIn, [2, 192]);

        const ditSess = await this.getSession(useStudentMicroCfg ? "dit_student_fp16.onnx" : "dit_estimator_fp16.onnx", true);
        const xIn = new Float32Array(2 * 128 * T_total);
        const tSpan = new Float32Array(nSteps + 1);
        const sway = useStudentMicroCfg ? 0.0 : -0.8;
        for (let i = 0; i <= nSteps; i++) {
          const u = i / nSteps;
          tSpan[i] = u + sway * (Math.cos(Math.PI * 0.5 * u) - 1.0 + u);
        }
        for (let step = 0; step < nSteps; step++) {
          onProgress(`${useStudentMicroCfg ? "蒸留DiT micro-CFG" : "DiT Flow Matching"} 推論中 (${step + 1}/${nSteps})...`);
          const tCur = tSpan[step];
          const dt = tSpan[step + 1] - tSpan[step];
          xIn.set(x, 0);
          xIn.set(x, 128 * T_total);
          const vRes = await this.runSession(ditSess, {
            x: new this.ort.Tensor("float32", xIn, [2, 128, T_total]),
            prompt_x: pxTensor,
            x_lens: xLensTensor,
            t: new this.ort.Tensor("float32", new Float32Array([tCur, tCur]), [2]),
            style: sTensor,
            cond: cTensor,
          });
          const vBoth = vRes[ditSess.outputNames[0]].data;
          const stride = 128 * T_total;
          for (let i = 0; i < stride; i++) {
            const dphi = (1.0 + cfgRate) * vBoth[i] - cfgRate * vBoth[stride + i];
            x[i] += dt * dphi;
          }
          for (let c = 0; c < 128; c++) {
            const row = c * T_total;
            for (let t = 0; t < T_prompt; t++) x[row + t] = 0.0;
          }
        }
      }
      const tDitMs = performance.now() - tD0;

      const vcTarget = new Float32Array(128 * T_src);
      for (let c = 0; c < 128; c++) {
        const srcOff = c * T_total + T_prompt;
        vcTarget.set(x.subarray(srcOff, srcOff + T_src), c * T_src);
      }

      onProgress("BigVGAN 44.1kHz 波形生成中...");
      const tB0 = performance.now();
      const useBvFp16 = typeof location !== "undefined" && new URLSearchParams(location.search).has("bigvgan_fp16");
      const bvSess = await this.getSession(useBvFp16 ? "bigvgan_fp16.onnx" : "bigvgan.onnx", true);
      const bvRes = await this.runSession(bvSess, {
        mel: new this.ort.Tensor("float32", vcTarget, [1, 128, T_src]),
      });
      const outWav = new Float32Array(bvRes[bvSess.outputNames[0]].data);
      const tBigvganMs = performance.now() - tB0;

      let peak = 0;
      let sumSq = 0;
      for (let i = 0; i < outWav.length; i++) {
        const v = outWav[i];
        if (!Number.isFinite(v)) {
          outWav[i] = 0;
          continue;
        }
        const av = Math.abs(v);
        if (av > peak) peak = av;
        sumSq += v * v;
      }
      const rms = outWav.length > 0 ? Math.sqrt(sumSq / outWav.length) : 0;
      return { outWav, tDitMs, tBigvganMs, rms, peak };
    }

    async precomputeSourceCond(audioFloat32, sampleRate, opts = {}, onProgress = () => {}) {
      if (!this.ref) {
        throw new Error("先にSTEP 1で参照音声を登録してください。");
      }
      const pitchShift = Number(opts.pitchShift || 0);
      const src44 = await resampleAudio(audioFloat32, sampleRate, SR_OUT);
      const src16 = await resampleAudio(audioFloat32, sampleRate, SR_16K);
      const T_full = Math.max(1, Math.floor((src44.length + 2 * 768 - 2048) / 512) + 1);

      // Run Whisper in 30s (480,000 samples) windows so full-song semantic context is preserved
      const maxWin = 480000;
      let sData;
      let sFrames;
      if (src16.length <= maxWin) {
        const wRes = await this.runWhisper(src16, "変換元全体", onProgress);
        sData = wRes.data;
        sFrames = wRes.frames;
      } else {
        const parts = [];
        let totalF = 0;
        for (let off = 0; off < src16.length; off += maxWin) {
          const slice = src16.subarray(off, Math.min(src16.length, off + maxWin));
          const wRes = await this.runWhisper(slice, "変換元全体", onProgress);
          parts.push(wRes.data);
          totalF += wRes.frames;
        }
        sData = new Float32Array(totalF * 768);
        let pOff = 0;
        for (const p of parts) {
          sData.set(p, pOff);
          pOff += p.length;
        }
        sFrames = totalF;
      }
      const f0 = await this.runRmvpe(src16, "変換元全体", onProgress);
      if (opts.useAdain !== false) {
        sData = this.applyWhisperAdaIN(sData, sFrames, 768, src16, false, f0);
      }

      if (pitchShift !== 0) {
        const factor = Math.pow(2.0, pitchShift / 12.0);
        for (let i = 0; i < f0.length; i++) {
          if (f0[i] > 0) f0[i] *= factor;
        }
      }

      onProgress("Length Regulator 処理中 (変換元全体)...");
      const lrSess = await this.getSession("length_regulator_fp16.onnx", true);
      const lrRes = await this.runSession(lrSess, {
        semantic: new this.ort.Tensor("float32", sData, [1, sFrames, 768]),
        ylens: new this.ort.Tensor("int64", BigInt64Array.from([BigInt(T_full)]), [1]),
        f0: new this.ort.Tensor("float32", f0, [1, f0.length]),
      });
      const condFull = new Float32Array(lrRes[lrSess.outputNames[0]].data);

      // Multiband adaptive articulation envelope precomputation
      const bLow = makeBiquadFilter("lowpass", 2500, SR_OUT);
      const bHigh = makeBiquadFilter("highpass", 2800, SR_OUT);
      const srcLow = applyBiquad(src44, bLow);
      const srcHigh = applyBiquad(src44, bHigh);
      const srcLowEnv = this.computeSmoothEnv(srcLow, T_full, 512, 6);
      const srcHighEnv = this.computeSmoothEnv(srcHigh, T_full, 512, 2);
      const srcEnv = this.computeSmoothEnv(src44, T_full, 512, 10);
      return { condFull, T_full, src44, srcLowEnv, srcHighEnv, srcEnv };
    }

    computeSmoothEnv(wav, nFrames, hop = 512, radius = 10) {
      const raw = new Float32Array(nFrames);
      for (let f = 0; f < nFrames; f++) {
        const s = f * hop;
        const e = Math.min(wav.length, s + hop);
        if (e <= s) continue;
        let sq = 0;
        for (let i = s; i < e; i++) sq += wav[i] * wav[i];
        raw[f] = Math.sqrt(sq / (e - s));
      }
      const smooth = new Float32Array(nFrames);
      for (let f = 0; f < nFrames; f++) {
        const a = Math.max(0, f - radius);
        const b = Math.min(nFrames - 1, f + radius);
        let acc = 0;
        for (let k = a; k <= b; k++) acc += raw[k];
        smooth[f] = acc / (b - a + 1);
      }
      return smooth;
    }

    applyEnvelopeMatch(out, srcLowEnv, srcHighEnv = null, startFrame = 0) {
      const hop = 512;
      const nFrames = Math.ceil(out.length / hop);
      if (nFrames <= 0 || !srcLowEnv) return;

      // Legacy fallback if only single-band env is provided
      if (!srcHighEnv) {
        const outEnv = this.computeSmoothEnv(out, nFrames, hop, 10);
        const gains = new Float32Array(nFrames);
        for (let f = 0; f < nFrames; f++) {
          const sIdx = Math.min(srcLowEnv.length - 1, startFrame + f);
          const sRms = srcLowEnv[sIdx];
          const oRms = outEnv[f];
          gains[f] = sRms < 0.003 ? 0.0 : (oRms < 0.015 ? 1.0 : Math.min(1.0, sRms / oRms));
        }
        for (let f = 0; f < nFrames; f++) {
          const g0 = gains[f];
          const g1 = f + 1 < nFrames ? gains[f + 1] : g0;
          const s = f * hop;
          const e = Math.min(out.length, s + hop);
          const len = e - s;
          for (let i = 0; i < len; i++) {
            const t = i / len;
            out[s + i] *= g0 + (g1 - g0) * t;
          }
        }
        return;
      }

      // Multiband Adaptive Articulation Envelope Match
      // Low band (<2500Hz): Preserves target vocal timbre and prevents clipping (radius=6, gain <= 1.0)
      // High band (>2800Hz): Restores sharp consonant transients and fricative air (radius=2, adaptive gain up to 2.5x)
      const bLow = makeBiquadFilter("lowpass", 2500, SR_OUT);
      const bHigh = makeBiquadFilter("highpass", 2800, SR_OUT);
      const outLow = applyBiquad(out, bLow);
      const outHigh = applyBiquad(out, bHigh);

      const outLowEnv = this.computeSmoothEnv(outLow, nFrames, hop, 6);
      const outHighEnv = this.computeSmoothEnv(outHigh, nFrames, hop, 2);

      const gLow = new Float32Array(nFrames);
      const gHigh = new Float32Array(nFrames);

      for (let f = 0; f < nFrames; f++) {
        const sIdx = Math.min(srcLowEnv.length - 1, startFrame + f);
        const sl = srcLowEnv[sIdx];
        const ol = outLowEnv[f];
        gLow[f] = sl < 0.003 ? 0.0 : (ol < 0.015 ? 1.0 : Math.min(1.0, sl / ol));

        const sh = srcHighEnv[sIdx];
        const oh = outHighEnv[f];
        if (sh < 0.001) {
          gHigh[f] = 0.0;
        } else if (oh < 0.003) {
          gHigh[f] = 1.0;
        } else {
          const ratio = sh / oh;
          gHigh[f] = Math.max(0.5, Math.min(2.5, ratio));
        }
      }

      for (let f = 0; f < nFrames; f++) {
        const gl0 = gLow[f];
        const gl1 = f + 1 < nFrames ? gLow[f + 1] : gl0;
        const gh0 = gHigh[f];
        const gh1 = f + 1 < nFrames ? gHigh[f + 1] : gh0;
        const s = f * hop;
        const e = Math.min(out.length, s + hop);
        const len = e - s;
        for (let i = 0; i < len; i++) {
          const t = i / len;
          const gl = gl0 + (gl1 - gl0) * t;
          const gh = gh0 + (gh1 - gh0) * t;
          out[s + i] = outLow[s + i] * gl + outHigh[s + i] * gh;
        }
      }

      let pk = 0;
      for (let i = 0; i < out.length; i++) {
        const av = Math.abs(out[i]);
        if (av > pk) pk = av;
      }
      if (pk > 0.95) {
        const scale = 0.95 / pk;
        for (let i = 0; i < out.length; i++) out[i] *= scale;
      }
    }

    async convertCondWindow(condFull, T_full, startFrame, chunkFrames, leftFrames = 16, rightFrames = 8, opts = {}, onProgress = () => {}) {
      const t0 = performance.now();
      const winFrames = leftFrames + chunkFrames + rightFrames;
      const nOutFrames = Math.min(chunkFrames, Math.max(0, T_full - startFrame));
      const nOutSamples = nOutFrames * 512;
      const xfade = Math.min(this.streamXfadeSamples, nOutSamples);

      const condWin = new Float32Array(winFrames * 768);
      const wStart = startFrame - leftFrames;
      const srcStart = Math.max(0, wStart);
      const dstStart = srcStart - wStart;
      const srcEnd = Math.min(T_full, wStart + winFrames);
      if (srcEnd > srcStart) {
        condWin.set(condFull.subarray(srcStart * 768, srcEnd * 768), dstStart * 768);
      }

      const nSteps = Number(opts.steps ?? 4);
      const cfgRate = Number(opts.cfgRate ?? 0.85);
      const { outWav, tDitMs, tBigvganMs, rms, peak } = await this.runDitAndBigvgan(
        condWin,
        winFrames,
        nSteps,
        cfgRate,
        onProgress
      );

      const startOff = leftFrames * 512 - xfade;
      const out = new Float32Array(nOutSamples);
      const copyLen = Math.min(nOutSamples, Math.max(0, outWav.length - startOff));
      if (copyLen > 0) {
        out.set(outWav.subarray(startOff, startOff + copyLen), 0);
      }
      if (this.streamPrevTail && this.streamPrevTail.length === xfade) {
        for (let i = 0; i < xfade; i++) {
          const fadeOut = 0.5 * (1.0 + Math.cos((Math.PI * i) / xfade));
          out[i] = this.streamPrevTail[i] * fadeOut + out[i] * (1.0 - fadeOut);
        }
      }
      const tailStart = startOff + nOutSamples;
      if (tailStart + xfade <= outWav.length) {
        this.streamPrevTail = new Float32Array(outWav.subarray(tailStart, tailStart + xfade));
      } else {
        this.streamPrevTail = null;
      }

      if (opts.srcLowEnv && opts.srcHighEnv) {
        this.applyEnvelopeMatch(out, opts.srcLowEnv, opts.srcHighEnv, startFrame);
      } else if (opts.srcEnv) {
        this.applyEnvelopeMatch(out, opts.srcEnv, null, startFrame);
      }

      let outPeak = 0;
      let outSumSq = 0;
      for (let i = 0; i < out.length; i++) {
        const av = Math.abs(out[i]);
        if (av > outPeak) outPeak = av;
        outSumSq += out[i] * out[i];
      }
      const outRms = out.length > 0 ? Math.sqrt(outSumSq / out.length) : 0;

      return {
        audio: out,
        sampleRate: SR_OUT,
        elapsedMs: performance.now() - t0,
        timings: { tWhisperMs: 0, tYinMs: 0, tDitMs, tBigvganMs },
        rms: outRms,
        peak: outPeak,
        skippedSilence: false,
      };
    }

    resetStream() {
      this.streamCtxSamples = 48 * 512;     // 24,576 samples (~557ms left context, 0ms added latency)
      this.streamChunkSamples = 40 * 512;   // 20,480 samples (~464ms low-latency hop)
      this.streamRightSamples = 8 * 512;    // 4,096 samples (~93ms right context, discarded from output)
      this.streamXfadeSamples = 1024;       // ~23ms SOLA phase-aligned Hann crossfade
      this.streamLeftCtx = new Float32Array(this.streamCtxSamples);
      this.streamPrevTail = null;
      this.streamAutoSemi = null;
      this.streamAdaMean = null;
      this.streamAdaStd = null;
    }

    async convertChunkStream(newChunk44, opts = {}, onProgress = () => {}) {
      if (!this.ref) {
        throw new Error("先にSTEP 1で参照音声を登録してください。");
      }
      const t0 = performance.now();
      const nNew = newChunk44.length;
      const xfade = Math.min(this.streamXfadeSamples, nNew);
      const rightCtx = opts.rightCtx || null;

      // Silence gate to skip heavy GPU work during pauses
      let sumSq = 0;
      for (let i = 0; i < nNew; i++) sumSq += newChunk44[i] * newChunk44[i];
      const inRms = Math.sqrt(sumSq / Math.max(1, nNew));
      if (inRms < 0.004) {
        const silent = new Float32Array(nNew);
        if (this.streamPrevTail) {
          for (let i = 0; i < xfade; i++) {
            const w = 0.5 * (1.0 + Math.cos((Math.PI * i) / xfade));
            silent[i] = this.streamPrevTail[i] * w;
          }
          this.streamPrevTail = null;
        }
        if (nNew >= this.streamCtxSamples) {
          this.streamLeftCtx.set(newChunk44.subarray(nNew - this.streamCtxSamples));
        } else {
          this.streamLeftCtx.copyWithin(0, nNew);
          this.streamLeftCtx.set(newChunk44, this.streamCtxSamples - nNew);
        }
        return {
          audio: silent,
          sampleRate: SR_OUT,
          elapsedMs: performance.now() - t0,
          timings: { tWhisperMs: 0, tYinMs: 0, tDitMs: 0, tBigvganMs: 0 },
          rms: 0,
          peak: 0,
          skippedSilence: true,
        };
      }

      // Always pad to fixed 96 frames (48 left + 40 chunk + 8 right) so WebGPU never recompiles shaders
      // and output + crossfade tail stay >= 8 frames (93ms) away from right-edge BigVGAN/Whisper artifacts.
      const ctxLen = this.streamCtxSamples;
      const totalLen = ctxLen + this.streamChunkSamples + this.streamRightSamples;
      const comb = new Float32Array(totalLen);
      comb.set(this.streamLeftCtx, 0);
      comb.set(newChunk44, ctxLen);
      if (rightCtx && rightCtx.length > 0 && nNew === this.streamChunkSamples) {
        const rLen = Math.min(this.streamRightSamples, rightCtx.length);
        comb.set(rightCtx.subarray(0, rLen), ctxLen + this.streamChunkSamples);
      }

      const conv = await this.convertAudio(
        comb,
        SR_OUT,
        {
          pitchShift: opts.pitchShift ?? 0,
          steps: opts.steps ?? 1,
          cfgRate: opts.cfgRate ?? 0.0,
          maxPromptFrames: opts.maxPromptFrames ?? 258,
          isStream: true,
        },
        onProgress
      );

      // Extract aligned region starting xfade samples before newChunk44, with SOLA phase alignment
      const fullOut = conv.audio;
      let startOff = ctxLen - xfade;
      if (this.streamPrevTail && this.streamPrevTail.length === xfade) {
        const maxLag = 256; // +/- 5.8ms covers >1 full period down to 90Hz
        let bestLag = 0;
        let bestCorr = -1e18;
        const minOff = Math.max(0, startOff - maxLag);
        const maxOff = Math.min(fullOut.length - nNew - xfade, startOff + maxLag);
        for (let off = minOff; off <= maxOff; off += 2) {
          let dot = 0;
          let eNew = 1e-9;
          for (let i = 0; i < xfade; i += 2) {
            const a = this.streamPrevTail[i];
            const b = fullOut[off + i];
            dot += a * b;
            eNew += b * b;
          }
          const score = dot / Math.sqrt(eNew);
          if (score > bestCorr) {
            bestCorr = score;
            bestLag = off - startOff;
          }
        }
        startOff += bestLag;
      }

      const out = new Float32Array(nNew);
      const copyLen = Math.min(nNew, Math.max(0, fullOut.length - startOff));
      if (copyLen > 0) {
        out.set(fullOut.subarray(startOff, startOff + copyLen), 0);
      }

      if (this.streamPrevTail && this.streamPrevTail.length === xfade) {
        for (let i = 0; i < xfade; i++) {
          const fadeOut = 0.5 * (1.0 + Math.cos((Math.PI * i) / xfade));
          const fadeIn = 1.0 - fadeOut;
          out[i] = this.streamPrevTail[i] * fadeOut + out[i] * fadeIn;
        }
      }

      const tailStart = startOff + nNew;
      if (tailStart + xfade <= fullOut.length) {
        this.streamPrevTail = new Float32Array(fullOut.subarray(tailStart, tailStart + xfade));
      } else {
        this.streamPrevTail = null;
      }

      if (nNew >= this.streamCtxSamples) {
        this.streamLeftCtx.set(newChunk44.subarray(nNew - this.streamCtxSamples));
      } else {
        this.streamLeftCtx.copyWithin(0, nNew);
        this.streamLeftCtx.set(newChunk44, this.streamCtxSamples - nNew);
      }

      const bLow = makeBiquadFilter("lowpass", 2500, SR_OUT);
      const bHigh = makeBiquadFilter("highpass", 2800, SR_OUT);
      const cLow = applyBiquad(newChunk44, bLow);
      const cHigh = applyBiquad(newChunk44, bHigh);
      const nChunkFrames = Math.ceil(nNew / 512);
      const cLowEnv = this.computeSmoothEnv(cLow, nChunkFrames, 512, 6);
      const cHighEnv = this.computeSmoothEnv(cHigh, nChunkFrames, 512, 2);
      this.applyEnvelopeMatch(out, cLowEnv, cHighEnv, 0);

      let outPeak = 0;
      let outSumSq = 0;
      for (let i = 0; i < out.length; i++) {
        const av = Math.abs(out[i]);
        if (av > outPeak) outPeak = av;
        outSumSq += out[i] * out[i];
      }
      const outRms = out.length > 0 ? Math.sqrt(outSumSq / out.length) : 0;

      return {
        audio: out,
        sampleRate: SR_OUT,
        elapsedMs: performance.now() - t0,
        timings: conv.timings,
        rms: outRms,
        peak: outPeak,
        skippedSilence: false,
      };
    }

    reset() {
      this.resetStream();
    }
  }

  window.SeedVcEngine = SeedVcEngine;
})();
