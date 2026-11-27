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

  function buildSparseBank(data, rows, cols) {
    const starts = new Int32Array(rows);
    const ends = new Int32Array(rows);
    for (let m = 0; m < rows; m++) {
      const off = m * cols;
      let s = 0;
      while (s < cols && data[off + s] === 0) s++;
      let e = cols - 1;
      while (e >= s && data[off + e] === 0) e--;
      starts[m] = s;
      ends[m] = e;
    }
    return { data, rows, cols, starts, ends };
  }

  function disposeTensorMap(map, skipSet = null) {
    if (!map) return;
    for (const k of Object.keys(map)) {
      const t = map[k];
      if (t && (!skipSet || !skipSet.has(t)) && typeof t.dispose === "function") {
        try { t.dispose(); } catch (_) {}
      }
    }
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

  class SeedVcEngine {
    static SAMPLE_RATE = SR_OUT;

    constructor() {
      this.ort = null;
      this.hasWebGpu = false;
      this.backendLabel = "WASM";
      this.banks = null;
      this.bankData = null;
      this.sparseBanks = {};
      this.hann400 = periodicHann(400);
      this.hann1024 = periodicHann(1024);
      this.hann2048 = periodicHann(2048);
      this.wFrame = new Float32Array(400);
      this.wPow = new Float32Array(201);
      this.fcRe = new Float32Array(1024);
      this.fcIm = new Float32Array(1024);
      this.fcMag = new Float32Array(513);
      this.bvRe = new Float32Array(2048);
      this.bvIm = new Float32Array(2048);
      this.bvMag = new Float32Array(1025);
      this.wD25Cos = null;
      this.wD25Sin = null;
      this.wTwCos = null;
      this.wTwSin = null;
      this.wD16Cos = null;
      this.wD16Sin = null;
      this.wTmpRe = new Float32Array(400);
      this.wTmpIm = new Float32Array(400);
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

      // Precompute non-zero frequency ranges for sparse mel matmuls (~8-12x faster, bit-exact)
      this.sparseBanks = {};
      for (const name of Object.keys(this.banks)) {
        const b = this.getBank(name);
        this.sparseBanks[name] = buildSparseBank(b.data, b.rows, b.cols);
      }
      if (this.fcpeMelBank) {
        this.sparseBanks.fcpe = buildSparseBank(this.fcpeMelBank, 128, 513);
      }

      // Precompute exact 400-point (16 x 25) Cooley-Tukey FFT tables for Whisper (~7x faster than naive DFT)
      this.wD25Cos = new Float32Array(25 * 25);
      this.wD25Sin = new Float32Array(25 * 25);
      for (let k = 0; k < 25; k++) {
        for (let n = 0; n < 25; n++) {
          const ang = (-2.0 * Math.PI * k * n) / 25;
          this.wD25Cos[k * 25 + n] = Math.cos(ang);
          this.wD25Sin[k * 25 + n] = Math.sin(ang);
        }
      }
      this.wTwCos = new Float32Array(400);
      this.wTwSin = new Float32Array(400);
      for (let k2 = 0; k2 < 25; k2++) {
        for (let n1 = 0; n1 < 16; n1++) {
          const ang = (-2.0 * Math.PI * k2 * n1) / 400;
          this.wTwCos[k2 * 16 + n1] = Math.cos(ang);
          this.wTwSin[k2 * 16 + n1] = Math.sin(ang);
        }
      }
      this.wD16Cos = new Float32Array(16 * 16);
      this.wD16Sin = new Float32Array(16 * 16);
      for (let k = 0; k < 16; k++) {
        for (let n = 0; n < 16; n++) {
          const ang = (-2.0 * Math.PI * k * n) / 16;
          this.wD16Cos[k * 16 + n] = Math.cos(ang);
          this.wD16Sin[k * 16 + n] = Math.sin(ang);
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
      if (this.ref && this.ref.ditCache) {
        this.ref.ditCache.clear();
      }
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
      const win = this.hann400;
      const sig = reflectPad(paddedSig, 200, 200);
      const bank = this.sparseBanks.whisper;
      const bStarts = bank.starts;
      const bEnds = bank.ends;
      const bData = bank.data;
      const out = new Float32Array(80 * nFrames);

      const frame = this.wFrame;
      const pow = this.wPow;
      const activeFrames = Math.min(nFrames, Math.ceil(activeLen / hop));

      const d25Cos = this.wD25Cos;
      const d25Sin = this.wD25Sin;
      const twCos = this.wTwCos;
      const twSin = this.wTwSin;
      const d16Cos = this.wD16Cos;
      const d16Sin = this.wD16Sin;
      const tRe = this.wTmpRe;
      const tIm = this.wTmpIm;

      // Compute log10 mel spectrogram for active frames; fill silent remainder with min value
      let globalMax = -1e9;
      for (let t = 0; t < activeFrames; t++) {
        const off = t * hop;
        for (let n = 0; n < nFft; n++) {
          frame[n] = sig[off + n] * win[n];
        }
        // Exact 400-pt (16x25) real Cooley-Tukey FFT -> power[0..200]
        for (let n1 = 0; n1 < 16; n1++) {
          let s0 = 0;
          for (let n2 = 0; n2 < 25; n2++) s0 += frame[n2 * 16 + n1];
          tRe[n1] = s0;
          tIm[n1] = 0;
          for (let k2 = 1; k2 <= 12; k2++) {
            const b = k2 * 25;
            let re = 0;
            let im = 0;
            for (let n2 = 0; n2 < 25; n2++) {
              const v = frame[n2 * 16 + n1];
              re += v * d25Cos[b + n2];
              im += v * d25Sin[b + n2];
            }
            const twIdx = k2 * 16 + n1;
            const tc = twCos[twIdx];
            const ts = twSin[twIdx];
            tRe[twIdx] = re * tc - im * ts;
            tIm[twIdx] = re * ts + im * tc;
            const twcIdx = (25 - k2) * 16 + n1;
            const tcc = twCos[twcIdx];
            const tsc = twSin[twcIdx];
            tRe[twcIdx] = re * tcc + im * tsc;
            tIm[twcIdx] = re * tsc - im * tcc;
          }
        }
        for (let k2 = 0; k2 < 25; k2++) {
          const row = k2 * 16;
          const k1Max = k2 === 0 ? 8 : 7;
          for (let k1 = 0; k1 <= k1Max; k1++) {
            const dOff = k1 * 16;
            let re = 0;
            let im = 0;
            for (let n1 = 0; n1 < 16; n1++) {
              const xr = tRe[row + n1];
              const xi = tIm[row + n1];
              const c = d16Cos[dOff + n1];
              const s = d16Sin[dOff + n1];
              re += xr * c - xi * s;
              im += xr * s + xi * c;
            }
            pow[25 * k1 + k2] = re * re + im * im;
          }
        }
        for (let m = 0; m < 80; m++) {
          const bOff = m * nFreq;
          const kEnd = bEnds[m];
          let acc = 0;
          for (let k = bStarts[m]; k <= kEnd; k++) {
            acc += bData[bOff + k] * pow[k];
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
      const win = this.hann1024;
      const sig = reflectPad(wav16, nFft / 2, nFft / 2);
      const nFrames = Math.floor((sig.length - nFft) / hop) + 1;
      const nPad = 32 * Math.ceil(nFrames / 32) - nFrames;
      const totalFrames = nFrames + nPad;
      const bank = this.sparseBanks.rmvpe;
      const bStarts = bank.starts;
      const bEnds = bank.ends;
      const bData = bank.data;
      const out = new Float32Array(128 * totalFrames);

      const re = this.fcRe;
      const im = this.fcIm;
      const mag = this.fcMag;

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
          const kEnd = bEnds[m];
          let acc = 0;
          for (let k = bStarts[m]; k <= kEnd; k++) {
            acc += bData[bOff + k] * mag[k];
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
      const bank = this.sparseBanks.kaldi;
      const bStarts = bank.starts;
      const bEnds = bank.ends;
      const bData = bank.data;
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
          const kEnd = bEnds[m];
          let acc = 0;
          for (let k = bStarts[m]; k <= kEnd; k++) {
            acc += bData[bOff + k] * pow[k];
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
      const win = this.hann2048;
      const sig = reflectPad(wav44, pad, pad);
      const nFrames = Math.floor((sig.length - nFft) / hop) + 1;
      const bank = this.sparseBanks.bigvgan;
      const bStarts = bank.starts;
      const bEnds = bank.ends;
      const bData = bank.data;
      const out = new Float32Array(128 * nFrames);

      const re = this.bvRe;
      const im = this.bvIm;
      const mag = this.bvMag;

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
          const kEnd = bEnds[m];
          let acc = 0;
          for (let k = bStarts[m]; k <= kEnd; k++) {
            acc += bData[bOff + k] * mag[k];
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
      let res = null;
      try {
        res = await this.runSession(sess, { input_features: inp });
        const full = res.hidden_states.data; // (1, winFrames/2, 768)
        const out = new Float32Array(nFrames * 768);
        out.set(full.subarray(0, Math.min(full.length, nFrames * 768)));
        return { data: out, frames: nFrames };
      } finally {
        if (typeof inp.dispose === "function") try { inp.dispose(); } catch (_) {}
        disposeTensorMap(res);
      }
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
      const win = this.hann1024;
      const sig = reflectPad(wav16, pad, pad);
      const nFrames = Math.floor((sig.length - nFft) / hop) + 1;
      const bank = this.sparseBanks.fcpe;
      const bStarts = bank.starts;
      const bEnds = bank.ends;
      const bData = bank.data;
      const out = new Float32Array(nFrames * 128);

      const re = this.fcRe;
      const im = this.fcIm;
      const mag = this.fcMag;

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
          const kEnd = bEnds[m];
          let acc = 0;
          for (let k = bStarts[m]; k <= kEnd; k++) {
            acc += bData[bOff + k] * mag[k];
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
            const melTensor = new this.ort.Tensor("float32", mel, [1, nFrames, 128]);
            let res = null;
            try {
              res = await this.runSession(sess, { mel: melTensor });
              const logits = res[sess.outputNames[0]].data;
              return this.decodeFcpeF0(logits, nFrames, 0.10);
            } finally {
              if (typeof melTensor.dispose === "function") try { melTensor.dispose(); } catch (_) {}
              disposeTensorMap(res);
            }
          }
          const fullF0 = new Float32Array(nFrames);
          for (let off = 0; off < nFrames; off += maxWin) {
            const len = Math.min(maxWin, nFrames - off);
            const slice = mel.subarray(off * 128, (off + len) * 128);
            const melTensor = new this.ort.Tensor("float32", slice, [1, len, 128]);
            let res = null;
            try {
              res = await this.runSession(sess, { mel: melTensor });
              const logits = res[sess.outputNames[0]].data;
              const partF0 = this.decodeFcpeF0(logits, len, 0.10);
              fullF0.set(partF0, off);
            } finally {
              if (typeof melTensor.dispose === "function") try { melTensor.dispose(); } catch (_) {}
              disposeTensorMap(res);
            }
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

    applyWhisperAdaIN(data, frames, dim = 768, wav16 = null, useStreamEma = false, strength = 0.25) {
      if (!this.ref || !this.ref.sOriMean || !this.ref.sOriStd || frames <= 1 || strength <= 0) return data;
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

      // Silence-protected Soft Active-frame AdaIN:
      // Keeps silent frames (rms < 0.006) completely untouched (zero silence noise injection)
      // and applies soft strength (0.25) on active speech so Japanese phonemes never distort/lisp.
      const hop = 320; // 20ms at 16kHz -> exact 50Hz Whisper frame rate
      const out = new Float32Array(data.length);
      for (let t = 0; t < frames; t++) {
        let gate = 1.0;
        if (wav16 && wav16.length >= hop) {
          const s = t * hop;
          const e = Math.min(wav16.length, s + hop);
          let sq = 0;
          for (let i = s; i < e; i++) sq += wav16[i] * wav16[i];
          const rms = e > s ? Math.sqrt(sq / (e - s)) : 0;
          gate = Math.max(0.0, Math.min(1.0, (rms - 0.006) / 0.012));
        }
        const alpha = strength * gate;
        const off = t * dim;
        if (alpha <= 1e-6) {
          out.set(data.subarray(off, off + dim), off);
        } else {
          for (let d = 0; d < dim; d++) {
            const adainVal = ((data[off + d] - mSrc[d]) / sSrc[d]) * sOri[d] + mOri[d];
            out[off + d] = data[off + d] * (1.0 - alpha) + adainVal * alpha;
          }
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

      const rawRef44 = await resampleAudio(trimmedPrompt, sampleRate, SR_OUT);
      // Append 40ms smooth fade-out + 80ms clean silence tail so DiT never leaks
      // mid-utterance voice from the prompt boundary into frame 0 of converted audio.
      const fadeSamples = Math.min(rawRef44.length, Math.round(SR_OUT * 0.04));
      const padSamples = Math.round(SR_OUT * 0.08);
      const ref44 = new Float32Array(rawRef44.length + padSamples);
      ref44.set(rawRef44, 0);
      for (let i = 0; i < fadeSamples; i++) {
        ref44[rawRef44.length - fadeSamples + i] *= 1.0 - i / fadeSamples;
      }
      const ref16 = await resampleAudio(ref44, SR_OUT, SR_16K);
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
      let cpRes = null;
      let style2;
      try {
        cpRes = await this.runSession(cpSess, { fbank: fbTensor });
        style2 = new Float32Array(cpRes[cpSess.outputNames[0]].data);
      } finally {
        if (typeof fbTensor.dispose === "function") try { fbTensor.dispose(); } catch (_) {}
        disposeTensorMap(cpRes);
      }

      onProgress("Length Regulator 処理中 (参照)...");
      const lrSess = await this.getSession("length_regulator_fp16.onnx", true);
      const lrFeeds = {
        semantic: new this.ort.Tensor("float32", S_ori.data, [1, S_ori.frames, 768]),
        ylens: new this.ort.Tensor("int64", BigInt64Array.from([BigInt(T_prompt)]), [1]),
        f0: new this.ort.Tensor("float32", F0_ori, [1, F0_ori.length]),
      };
      let lrRes = null;
      let promptCond;
      try {
        lrRes = await this.runSession(lrSess, lrFeeds);
        promptCond = new Float32Array(lrRes[lrSess.outputNames[0]].data);
      } finally {
        disposeTensorMap(lrFeeds);
        disposeTensorMap(lrRes);
      }

      const f0Med = this.computeVoicedF0Median(F0_ori);
      const { mean: sOriMean, std: sOriStd } = this.computeFeatureStats(S_ori.data, S_ori.frames, 768, ref16);
      if (this.ref && this.ref.ditCache) {
        this.ref.ditCache.clear();
      }
      this.ref = {
        mel2,
        T_prompt,
        style2,
        promptCond,
        f0Med,
        sOriMean,
        sOriStd,
        seconds: ref44.length / SR_OUT,
        ditCache: new Map(),
      };
      // Warm only the live-streaming shape on reference registration (~3x faster setup);
      // offline long-window shapes compile on-demand if ever used.
      await this.warmupShaders(
        [{ winFrames: 96, maxPromptFrames: 258, nSteps: 1, cfgRate: 0.0 }],
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
        await this.runDitAndBigvgan(dummyCond, winFrames, 1, cfgRate, () => {}, maxPromptFrames, true);
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

      // Compute exact BigVGAN mel frame count without wasting ~40ms computing an unused 2048-pt STFT + mel matmul
      const T_src = Math.floor((src44.length + 1536 - 2048) / 512) + 1;
      const tW0 = performance.now();
      const S_alt = await this.runWhisper(src16, "変換元", onProgress);
      if (opts.useAdain !== false) {
        S_alt.data = this.applyWhisperAdaIN(S_alt.data, S_alt.frames, 768, src16, Boolean(opts.isStream), opts.adainStrength ?? 0.25);
      }
      const tWhisperMs = performance.now() - tW0;

      const tY0 = performance.now();
      const F0_alt = await this.runRmvpe(src16, "変換元", onProgress);
      const tYinMs = performance.now() - tY0;

      if (pitchShift !== 0) {
        const factor = Math.pow(2.0, pitchShift / 12.0);
        for (let i = 0; i < F0_alt.length; i++) {
          if (F0_alt[i] > 0) F0_alt[i] *= factor;
        }
      }

      onProgress("Length Regulator 処理中 (変換元)...");
      const lrSess = await this.getSession("length_regulator_fp16.onnx", true);
      const lrFeeds = {
        semantic: new this.ort.Tensor("float32", S_alt.data, [1, S_alt.frames, 768]),
        ylens: new this.ort.Tensor("int64", BigInt64Array.from([BigInt(T_src)]), [1]),
        f0: new this.ort.Tensor("float32", F0_alt, [1, F0_alt.length]),
      };
      let lrRes = null;
      let cond;
      try {
        lrRes = await this.runSession(lrSess, lrFeeds);
        cond = new Float32Array(lrRes[lrSess.outputNames[0]].data);
      } finally {
        disposeTensorMap(lrFeeds);
        disposeTensorMap(lrRes);
      }
      let ditFrames = T_src;
      const ditTailFrames = Number(opts.ditTailFrames || 0);
      if (ditTailFrames > 0 && ditTailFrames < T_src) {
        cond = cond.subarray((T_src - ditTailFrames) * 768);
        ditFrames = ditTailFrames;
      }
      const { outWav, tDitMs, tBigvganMs, rms, peak } = await this.runDitAndBigvgan(
        cond,
        ditFrames,
        nSteps,
        cfgRate,
        onProgress,
        maxPromptFrames,
        Boolean(opts.skipStats)
      );

      return {
        audio: outWav,
        sampleRate: SR_OUT,
        elapsedMs: performance.now() - t0,
        timings: { tWhisperMs, tYinMs, tDitMs, tBigvganMs },
        rms,
        peak,
      };
    }

    async runDitAndBigvgan(cond, T_src, nSteps, cfgRate, onProgress = () => {}, maxPromptFrames = 0, skipStats = false) {
      // Concatenate promptCond (1, T_prompt, 768) + cond (1, T_src, 768) -> (1, T_total, 768)
      const T_fullPrompt = this.ref.T_prompt;
      const T_prompt = maxPromptFrames > 0 ? Math.min(T_fullPrompt, maxPromptFrames) : T_fullPrompt;
      const T_total = T_prompt + T_src;

      // Reuse deterministic noise baseX, promptX, and scratch buffers per (T_prompt, T_src)
      if (!this.ref.ditCache) this.ref.ditCache = new Map();
      const cacheKey = `${T_prompt}:${T_src}`;
      let entry = this.ref.ditCache.get(cacheKey);
      if (!entry) {
        const catCond = new Float32Array(T_total * 768);
        catCond.set(this.ref.promptCond.subarray(0, T_prompt * 768), 0);

        const randn = makeRng(42);
        const baseX = new Float32Array(128 * T_total);
        for (let i = 0; i < baseX.length; i++) baseX[i] = randn();
        for (let c = 0; c < 128; c++) {
          const row = c * T_total;
          for (let t = 0; t < T_prompt; t++) baseX[row + t] = 0.0;
        }

        const promptX = new Float32Array(128 * T_total);
        for (let c = 0; c < 128; c++) {
          const dstRow = c * T_total;
          const srcRow = c * T_fullPrompt;
          promptX.set(this.ref.mel2.subarray(srcRow, srcRow + T_prompt), dstRow);
        }

        entry = {
          catCond,
          baseX,
          xBuf: new Float32Array(128 * T_total),
          promptX,
          vcTarget: new Float32Array(128 * T_src),
          xLens1: BigInt64Array.from([BigInt(T_total)]),
        };
        this.ref.ditCache.set(cacheKey, entry);
      }

      const catCond = entry.catCond;
      catCond.set(cond, T_prompt * 768);
      const x = entry.xBuf;
      x.set(entry.baseX);
      const promptX = entry.promptX;

      const tD0 = performance.now();
      const useStudentNoCfg = cfgRate <= 0;
      const useStudentMicroCfg = cfgRate > 0 && nSteps <= 2;
      if (useStudentNoCfg) {
        // Distilled 1-2 step No-CFG Student (batch=1)
        const ditSess = await this.getSession("dit_student_fp16.onnx", true);
        const xLensTensor = new this.ort.Tensor("int64", entry.xLens1, [1]);
        const pxTensor = new this.ort.Tensor("float32", promptX, [1, 128, T_total]);
        const cTensor = new this.ort.Tensor("float32", catCond, [1, T_total, 768]);
        const sTensor = new this.ort.Tensor("float32", this.ref.style2, [1, 192]);
        const dt = 1.0 / nSteps;
        const stride = 128 * T_total;
        try {
          for (let step = 0; step < nSteps; step++) {
            onProgress(`蒸留DiT 推論中 (${step + 1}/${nSteps})...`);
            const tCur = step / nSteps;
            const xTensor = new this.ort.Tensor("float32", x, [1, 128, T_total]);
            const tTensor = new this.ort.Tensor("float32", new Float32Array([tCur]), [1]);
            let vRes = null;
            try {
              vRes = await this.runSession(ditSess, {
                x: xTensor,
                prompt_x: pxTensor,
                x_lens: xLensTensor,
                t: tTensor,
                style: sTensor,
                cond: cTensor,
              });
              const v = vRes[ditSess.outputNames[0]].data;
              for (let i = 0; i < stride; i++) x[i] += dt * v[i];
            } finally {
              if (typeof xTensor.dispose === "function") try { xTensor.dispose(); } catch (_) {}
              if (typeof tTensor.dispose === "function") try { tTensor.dispose(); } catch (_) {}
              disposeTensorMap(vRes);
            }
            if (step + 1 < nSteps) {
              for (let c = 0; c < 128; c++) {
                const row = c * T_total;
                for (let t = 0; t < T_prompt; t++) x[row + t] = 0.0;
              }
            }
          }
        } finally {
          disposeTensorMap({ xLensTensor, pxTensor, cTensor, sTensor });
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
        try {
          for (let step = 0; step < nSteps; step++) {
            onProgress(`${useStudentMicroCfg ? "蒸留DiT micro-CFG" : "DiT Flow Matching"} 推論中 (${step + 1}/${nSteps})...`);
            const tCur = tSpan[step];
            const dt = tSpan[step + 1] - tSpan[step];
            xIn.set(x, 0);
            xIn.set(x, 128 * T_total);
            const xTensor = new this.ort.Tensor("float32", xIn, [2, 128, T_total]);
            const tTensor = new this.ort.Tensor("float32", new Float32Array([tCur, tCur]), [2]);
            let vRes = null;
            try {
              vRes = await this.runSession(ditSess, {
                x: xTensor,
                prompt_x: pxTensor,
                x_lens: xLensTensor,
                t: tTensor,
                style: sTensor,
                cond: cTensor,
              });
              const vBoth = vRes[ditSess.outputNames[0]].data;
              const stride = 128 * T_total;
              for (let i = 0; i < stride; i++) {
                const dphi = (1.0 + cfgRate) * vBoth[i] - cfgRate * vBoth[stride + i];
                x[i] += dt * dphi;
              }
            } finally {
              if (typeof xTensor.dispose === "function") try { xTensor.dispose(); } catch (_) {}
              if (typeof tTensor.dispose === "function") try { tTensor.dispose(); } catch (_) {}
              disposeTensorMap(vRes);
            }
            if (step + 1 < nSteps) {
              for (let c = 0; c < 128; c++) {
                const row = c * T_total;
                for (let t = 0; t < T_prompt; t++) x[row + t] = 0.0;
              }
            }
          }
        } finally {
          disposeTensorMap({ xLensTensor, pxTensor, cTensor, sTensor });
        }
      }
      const tDitMs = performance.now() - tD0;

      const vcTarget = entry.vcTarget;
      for (let c = 0; c < 128; c++) {
        const srcOff = c * T_total + T_prompt;
        vcTarget.set(x.subarray(srcOff, srcOff + T_src), c * T_src);
      }

      onProgress("BigVGAN 44.1kHz 波形生成中...");
      const tB0 = performance.now();
      const useBvFp16 = typeof location !== "undefined" && new URLSearchParams(location.search).has("bigvgan_fp16");
      const bvSess = await this.getSession(useBvFp16 ? "bigvgan_fp16.onnx" : "bigvgan.onnx", true);
      const melTensor = new this.ort.Tensor("float32", vcTarget, [1, 128, T_src]);
      let bvRes = null;
      let outWav;
      try {
        bvRes = await this.runSession(bvSess, { mel: melTensor });
        outWav = new Float32Array(bvRes[bvSess.outputNames[0]].data);
      } finally {
        if (typeof melTensor.dispose === "function") try { melTensor.dispose(); } catch (_) {}
        disposeTensorMap(bvRes);
      }
      const tBigvganMs = performance.now() - tB0;

      if (skipStats) {
        return { outWav, tDitMs, tBigvganMs, rms: 0, peak: 0 };
      }

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
        sData = this.applyWhisperAdaIN(sData, sFrames, 768, src16, false, opts.adainStrength ?? 0.25);
      }

      if (pitchShift !== 0) {
        const factor = Math.pow(2.0, pitchShift / 12.0);
        for (let i = 0; i < f0.length; i++) {
          if (f0[i] > 0) f0[i] *= factor;
        }
      }

      onProgress("Length Regulator 処理中 (変換元全体)...");
      const lrSess = await this.getSession("length_regulator_fp16.onnx", true);
      const lrFeeds = {
        semantic: new this.ort.Tensor("float32", sData, [1, sFrames, 768]),
        ylens: new this.ort.Tensor("int64", BigInt64Array.from([BigInt(T_full)]), [1]),
        f0: new this.ort.Tensor("float32", f0, [1, f0.length]),
      };
      let lrRes = null;
      let condFull;
      try {
        lrRes = await this.runSession(lrSess, lrFeeds);
        condFull = new Float32Array(lrRes[lrSess.outputNames[0]].data);
      } finally {
        disposeTensorMap(lrFeeds);
        disposeTensorMap(lrRes);
      }

      const srcEnv = this.computeSmoothEnv(src44, T_full, 512, 10);
      return { condFull, T_full, src44, srcEnv };
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

    applyEnvelopeMatch(out, srcEnv, startFrame = 0) {
      const hop = 512;
      const nFrames = Math.ceil(out.length / hop);
      if (nFrames <= 0 || !srcEnv) return;

      const outEnv = this.computeSmoothEnv(out, nFrames, hop, 10);
      const gains = new Float32Array(nFrames);
      for (let f = 0; f < nFrames; f++) {
        const sIdx = Math.min(srcEnv.length - 1, startFrame + f);
        const sRms = srcEnv[sIdx];
        const oRms = outEnv[f];
        if (sRms < 0.003) {
          gains[f] = 0.0;
        } else if (oRms < 0.015) {
          gains[f] = 1.0;
        } else {
          // Purely linear gain (capped at 1.0) to match source dynamics;
          // clean 1-step DiT outputs clip=0.00% natively without EQ/limiter hacks.
          gains[f] = Math.min(1.0, sRms / oRms);
        }
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
      const { outWav, tDitMs, tBigvganMs } = await this.runDitAndBigvgan(
        condWin,
        winFrames,
        nSteps,
        cfgRate,
        onProgress,
        0,
        true
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

      if (opts.srcEnv) {
        this.applyEnvelopeMatch(out, opts.srcEnv, startFrame);
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
      // 72 + 40 + 8 = 120 frames (~1.39s -> 139 Whisper mel frames <= 140),
      // keeping Whisper in the ultra-fast W150 bucket (4x faster attention than W300)
      // while still giving 1.5x longer past context than DiT's 48 frames with 0ms extra latency.
      this.streamWhisperCtxSamples = 72 * 512;  // 36,864 samples (~836ms past context for Whisper/FCPE)
      this.streamCtxSamples = 48 * 512;         // 24,576 samples (~557ms left context for DiT/BigVGAN)
      this.streamChunkSamples = 40 * 512;       // 20,480 samples (~464ms low-latency hop)
      this.streamRightSamples = 8 * 512;        // 4,096 samples (~93ms right context, discarded from output)
      this.streamXfadeSamples = 1024;           // ~23ms SOLA phase-aligned Hann crossfade
      this.streamLeftCtx = new Float32Array(this.streamWhisperCtxSamples);
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
        if (nNew >= this.streamWhisperCtxSamples) {
          this.streamLeftCtx.set(newChunk44.subarray(nNew - this.streamWhisperCtxSamples));
        } else {
          this.streamLeftCtx.copyWithin(0, nNew);
          this.streamLeftCtx.set(newChunk44, this.streamWhisperCtxSamples - nNew);
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

      // Feed 1.86s past context to Whisper/FCPE (0ms added latency) so Japanese phonemes stay crisp,
      // then slice only the tail 96 frames (L48/C40/R8) for fast fixed-shape DiT/BigVGAN inference.
      const whCtxLen = this.streamWhisperCtxSamples;
      const ctxLen = this.streamCtxSamples;
      const totalLen = whCtxLen + this.streamChunkSamples + this.streamRightSamples;
      const comb = new Float32Array(totalLen);
      comb.set(this.streamLeftCtx, 0);
      comb.set(newChunk44, whCtxLen);
      if (rightCtx && rightCtx.length > 0 && nNew === this.streamChunkSamples) {
        const rLen = Math.min(this.streamRightSamples, rightCtx.length);
        comb.set(rightCtx.subarray(0, rLen), whCtxLen + this.streamChunkSamples);
      }

      const conv = await this.convertAudio(
        comb,
        SR_OUT,
        {
          pitchShift: opts.pitchShift ?? 0,
          steps: opts.steps ?? 1,
          cfgRate: opts.cfgRate ?? 0.0,
          maxPromptFrames: opts.maxPromptFrames ?? 258,
          ditTailFrames: 96,
          isStream: true,
          skipStats: true,
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

      if (nNew >= this.streamWhisperCtxSamples) {
        this.streamLeftCtx.set(newChunk44.subarray(nNew - this.streamWhisperCtxSamples));
      } else {
        this.streamLeftCtx.copyWithin(0, nNew);
        this.streamLeftCtx.set(newChunk44, this.streamWhisperCtxSamples - nNew);
      }

      const chunkEnv = this.computeSmoothEnv(newChunk44, Math.ceil(nNew / 512), 512, 10);
      this.applyEnvelopeMatch(out, chunkEnv, 0);

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
