/**
 * kNN-VC browser engine (Baas et al., "Voice Conversion With Just Nearest Neighbors", MIT).
 *
 * Pipeline (16 kHz, 200 ms hop):
 *   audio window (24080 smp) -> WavLM-Large layer-6 (ONNX, int8 MatMul) -> 75x1024 features
 *   -> cosine kNN (k=4) against the reference-speaker feature set -> HiFi-GAN (ONNX) -> audio
 *
 * Latency model: 10 new frames per step (200 ms); WavLM right context RW=3 frames,
 * vocoder context LC=6 / RC=3 frames. The output stream lags the input by ~5160 samples.
 */
(function () {
  "use strict";

  const SR = 16000;
  const HOP = 320; // 20 ms per WavLM frame
  const WIN_FRAMES = 75;
  const WIN_SAMPLES = 400 + HOP * (WIN_FRAMES - 1); // 24080
  const HF = 10; // new frames per step
  const RW = 5; // WavLM right-context frames kept for emitted frames (100 ms)
  const LC = 6; // vocoder left context frames
  const RC = 3; // vocoder right context frames
  const OVERLAP = 320; // 20 ms Hann crossfade across 200 ms chunks
  const DIM = 1024;
  const TOPK = 4;
  const TEMP_WEIGHT = 15.0;
  const CONT_BONUS = 0.0;
  const CENTER_ALPHA = 0.0;
  const GATE_LOW = 0.005;
  const GATE_HIGH = 0.015;
  const CHUNK = HF * HOP; // 3200 samples = 200 ms
  const REF_EMIT = 40; // frames emitted per reference window
  const MAX_REF_SECONDS = 60;

  const WAVLM_URL = "/models/knn/wavlm6_w1500_int8mm.onnx";
  const HIFIGAN_URL = "/models/knn/hifigan_dyn.onnx";

  function trimSilence(x, thrDb) {
    const frame = HOP;
    const nFrames = Math.floor(x.length / frame);
    if (nFrames < 4) return x;
    const rms = new Float32Array(nFrames);
    let maxRms = 0;
    for (let f = 0; f < nFrames; f++) {
      let s = 0;
      const base = f * frame;
      for (let i = 0; i < frame; i++) {
        const v = x[base + i];
        s += v * v;
      }
      rms[f] = Math.sqrt(s / frame);
      if (rms[f] > maxRms) maxRms = rms[f];
    }
    if (maxRms < 1e-4) return x;
    const thr = maxRms * Math.pow(10, thrDb / 20);
    let first = 0;
    while (first < nFrames - 1 && rms[first] < thr) first++;
    let last = nFrames - 1;
    while (last > first && rms[last] < thr) last--;
    first = Math.max(0, first - 2);
    last = Math.min(nFrames - 1, last + 2);
    return x.subarray(first * frame, (last + 1) * frame);
  }

  async function resampleMono(float32, fromRate, toRate) {
    if (fromRate === toRate) return float32;
    const buf = new AudioBuffer({ length: float32.length, numberOfChannels: 1, sampleRate: fromRate });
    buf.copyToChannel(float32, 0);
    const outLen = Math.max(1, Math.ceil((float32.length * toRate) / fromRate));
    const offline = new OfflineAudioContext(1, outLen, toRate);
    const src = offline.createBufferSource();
    src.buffer = buf;
    src.connect(offline.destination);
    src.start(0);
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0);
  }

  async function tensorData(t) {
    const raw = typeof t.getData === "function" ? await t.getData() : t.data;
    return raw;
  }

  function fastNsdfF0(buf, start, len, sr, fmin, fmax, prevF0) {
    if (start < 0 || start + len > buf.length || len < 400) return { f0: 0, conf: 0 };
    let sq = 0;
    for (let i = 0; i < len; i += 2) {
      const v = buf[start + i];
      sq += v * v;
    }
    if (Math.sqrt((sq * 2) / len) < 0.008) return { f0: 0, conf: 0 };

    const minLag = Math.floor(sr / fmax);
    const maxLag = Math.min(Math.floor(len / 2), Math.floor(sr / fmin));
    const vals = new Float32Array(maxLag + 1);
    let bestLag = 0;
    let bestScore = -1;
    // Coarse search (step=2, subsampled i+=2)
    for (let lag = minLag; lag <= maxLag; lag += 2) {
      let ab = 0;
      let denom = 1e-9;
      const lim = len - lag;
      for (let i = 0; i < lim; i += 2) {
        const a = buf[start + i];
        const b = buf[start + i + lag];
        ab += a * b;
        denom += a * a + b * b;
      }
      const v = (2 * ab) / denom;
      vals[lag] = v;
      let score = v;
      if (prevF0 > 50) {
        const ratio = sr / lag / prevF0;
        if (ratio > 0.65 && ratio < 1.55) score += 0.03;
      }
      if (score > bestScore) {
        bestScore = score;
        bestLag = lag;
      }
    }
    if (bestLag === 0) return { f0: 0, conf: 0 };
    // Fine search around bestLag
    const r0 = Math.max(minLag, bestLag - 2);
    const r1 = Math.min(maxLag, bestLag + 2);
    bestScore = -1;
    for (let lag = r0; lag <= r1; lag++) {
      let ab = 0;
      let denom = 1e-9;
      const lim = len - lag;
      for (let i = 0; i < lim; i++) {
        const a = buf[start + i];
        const b = buf[start + i + lag];
        ab += a * b;
        denom += a * a + b * b;
      }
      const v = (2 * ab) / denom;
      vals[lag] = v;
      let score = v;
      if (prevF0 > 50) {
        const ratio = sr / lag / prevF0;
        if (ratio > 0.65 && ratio < 1.55) score += 0.03;
      }
      if (score > bestScore) {
        bestScore = score;
        bestLag = lag;
      }
    }
    const conf = vals[bestLag];
    if (conf < 0.45) return { f0: 0, conf };
    let lagRefined = bestLag;
    if (bestLag > r0 && bestLag < r1) {
      const y0 = vals[bestLag - 1];
      const y1 = vals[bestLag];
      const y2 = vals[bestLag + 1];
      const dp = 2 * (y0 - 2 * y1 + y2);
      if (Math.abs(dp) > 1e-9) {
        lagRefined += Math.max(-0.5, Math.min(0.5, (y0 - y2) / dp));
      }
    }
    return { f0: sr / lagRefined, conf };
  }

  function tdPsolaChunk(wav, s0, outLen, f0OutArr, confOutArr, f0TgtArr, confTgtArr, padLeft, padRight) {
    const aStart = Math.max(0, s0 - padLeft);
    const aEnd = Math.min(wav.length, s0 + outLen + padRight);
    const n = aEnd - aStart;
    const leftOff = s0 - aStart;
    const nFr = f0OutArr.length;

    const getFr = (sampleIdx) => Math.max(0, Math.min(nFr - 1, Math.floor((sampleIdx - leftOff) / HOP)));

    // 9-tap moving average low-pass (~900Hz) so pitch marks lock onto fundamental rather than harmonics
    const lp = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      let cnt = 0;
      const k0 = Math.max(0, i - 4);
      const k1 = Math.min(n - 1, i + 4);
      for (let k = k0; k <= k1; k++) {
        acc += wav[aStart + k];
        cnt++;
      }
      lp[i] = acc / cnt;
    }

    const marksPos = [];
    const marksTa = [];
    let pos = 0;
    while (pos < n) {
      const fo = f0OutArr[getFr(pos)];
      let Ta = fo > 50 ? Math.round(SR / fo) : 160;
      Ta = Math.max(22, Math.min(260, Ta));
      let mark = pos;
      if (fo > 50) {
        const p0 = Math.max(0, pos - Math.floor(Ta / 3));
        const p1 = Math.min(n, pos + Math.floor(Ta / 3) + 1);
        let maxV = -Infinity;
        for (let i = p0; i < p1; i++) {
          const v = lp[i];
          if (v > maxV) {
            maxV = v;
            mark = i;
          }
        }
      }
      marksPos.push(mark);
      marksTa.push(Ta);
      pos = mark + Ta > pos ? mark + Ta : pos + Ta;
    }

    const ySyn = new Float32Array(n);
    const wSum = new Float32Array(n);
    const numMarks = marksPos.length;
    let mIdx = 0;
    let tSyn = 0;

    while (tSyn < n) {
      const fr = getFr(tSyn);
      const fo = f0OutArr[fr];
      const ft = f0TgtArr[fr];
      const co = confOutArr[fr];
      const ct = confTgtArr[fr];
      let Ts = 160;
      if (fo > 50 && ft > 50) {
        const cMin = Math.min(co, ct);
        const wShift = Math.max(0, Math.min(1, (cMin - 0.45) / 0.2));
        const rawRatio = Math.max(0.55, Math.min(1.85, ft / fo));
        const ratio = 1 + wShift * (rawRatio - 1);
        Ts = SR / (fo * ratio);
      } else if (fo > 50) {
        Ts = SR / fo;
      }
      Ts = Math.max(22, Math.min(260, Ts));

      while (mIdx + 1 < numMarks && Math.abs(marksPos[mIdx + 1] - tSyn) <= Math.abs(marksPos[mIdx] - tSyn)) {
        mIdx++;
      }
      const ma = marksPos[mIdx];
      const Ta = marksTa[mIdx];
      // Adaptive half-width prevents comb-filtering/raspy overlap when shifting up to high notes
      const hw = Math.max(18, Math.min(Ta, Math.round(Math.max(Ts, Ta * 0.65))));
      const wl = 2 * hw + 1;
      const cs = Math.round(tSyn);
      const a0 = ma - hw;
      const sOff = cs - hw;
      const gl = Math.max(0, -a0, -sOff);
      const gr = Math.min(wl, n - a0, n - sOff);
      const invWl = wl > 1 ? (2 * Math.PI) / (wl - 1) : 0;
      for (let g = gl; g < gr; g++) {
        const winW = 0.5 * (1 - Math.cos(invWl * g));
        ySyn[sOff + g] += wav[aStart + a0 + g] * winW;
        wSum[sOff + g] += winW;
      }
      tSyn += Ts;
    }

    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const idx = leftOff + i;
      out[i] = wSum[idx] > 1e-4 ? ySyn[idx] / wSum[idx] : wav[aStart + idx];
    }
    return out;
  }

  class KnnVcEngine {
    constructor() {
      this.wavlm = null;
      this.hifigan = null;
      this.ready = false;
      this.refRaw = null; // Float32Array(N*DIM)
      this.refNorm = null; // Float32Array(N*DIM), row-normalised raw
      this.refCentNorm = null; // Float32Array(N*DIM), speaker-centered normalised
      this.nRef = 0;
      this.outGain = 1.8;
      this.pitchSemitones = 0;
      this.lastWavlmMs = 0;
      this.lastHifiganMs = 0;
      this.lastMatchMs = 0;
      this.chunkSamples = CHUNK;
      this.sampleRate = SR;
      this.reset();
    }

    setPitchShift(semitones) {
      this.pitchSemitones = Number.isFinite(semitones) ? semitones : 0;
    }

    async load(ort, onStatus) {
      const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
      if (onStatus) onStatus("WavLM (int8, 127MB) 読込中...");
      this.wavlm = await ort.InferenceSession.create(WAVLM_URL, opts);
      if (onStatus) onStatus("HiFi-GAN (66MB) 読込中...");
      this.hifigan = await ort.InferenceSession.create(HIFIGAN_URL, opts);
      this.ort = ort;
      this.ready = true;
      // Warm-up
      const zeros = new Float32Array(WIN_SAMPLES);
      await this._wavlmFeatures(zeros);
      await this._vocode(new Float32Array((LC + HF + RC) * DIM));
    }

    reset() {
      this.hist = new Float32Array(WIN_SAMPLES);
      this.matched = []; // array of Float32Array(DIM), keeps last LC+HF+RC frames
      this.rmsFifo = []; // aligned frame RMS for soft silence gating
      this.srcF0Fifo = []; // aligned source F0 for TD-PSOLA pitch matching
      this.srcConfFifo = []; // aligned source voicing confidence
      this.prevSrcF0 = 0;
      this.prevOutF0 = 0;
      this.prevTail = null; // OVERLAP samples for Hann crossfade
      this.prevTop1 = -10;
      this.srcMean = new Float32Array(DIM);
      this.srcMeanCount = 0;
      this.totalEmitted = 0;
    }

    async _wavlmFeatures(win) {
      const t = new this.ort.Tensor("float32", win, [1, WIN_SAMPLES]);
      const t0 = performance.now();
      const out = await this.wavlm.run({ audio: t });
      const data = new Float32Array(await tensorData(out.feat));
      this.lastWavlmMs = performance.now() - t0;
      t.dispose();
      if (typeof out.feat.dispose === "function") out.feat.dispose();
      return data; // (75*DIM)
    }

    async _vocode(featFlat) {
      const frames = featFlat.length / DIM;
      const t = new this.ort.Tensor("float32", featFlat, [1, frames, DIM]);
      const t0 = performance.now();
      const out = await this.hifigan.run({ feat: t });
      const data = new Float32Array(await tensorData(out.wav));
      this.lastHifiganMs = performance.now() - t0;
      t.dispose();
      if (typeof out.wav.dispose === "function") out.wav.dispose();
      return data;
    }

    /** Build the matching set from reference audio (Float32Array at `rate`). */
    async setReference(audio, rate, onProgress) {
      let a = audio.length > rate * MAX_REF_SECONDS ? audio.subarray(0, rate * MAX_REF_SECONDS) : audio;
      a = await resampleMono(a, rate, SR);
      a = trimSilence(a, -35);
      const L = a.length;
      if (L < WIN_SAMPLES / 2) throw new Error("参照音声が短すぎます（1秒以上を推奨、10秒以上で高品質）。");

      const lead = (WIN_FRAMES - RW - REF_EMIT) * HOP; // zero pad on the left so frame 0 is emitted
      const stepSamples = REF_EMIT * HOP;
      const validFrames = Math.floor((L - 400) / HOP) + 1;
      const rows = [];
      const t0 = performance.now();
      let emittedFrames = 0;
      let winIdx = 0;
      const totalWins = Math.ceil(validFrames / REF_EMIT);
      while (emittedFrames < validFrames) {
        const S = WIN_SAMPLES - lead + winIdx * stepSamples; // window end (global sample index)
        const start = S - WIN_SAMPLES;
        const win = new Float32Array(WIN_SAMPLES);
        for (let i = 0; i < WIN_SAMPLES; i++) {
          const g = start + i;
          win[i] = g >= 0 && g < L ? a[g] : 0;
        }
        const feat = await this._wavlmFeatures(win);
        const take = Math.min(REF_EMIT, validFrames - emittedFrames);
        const j0 = WIN_FRAMES - RW - REF_EMIT;
        for (let k = 0; k < take; k++) {
          rows.push(feat.subarray((j0 + k) * DIM, (j0 + k + 1) * DIM).slice());
        }
        emittedFrames += take;
        winIdx += 1;
        if (onProgress) onProgress(winIdx, totalWins);
      }
      const n = rows.length;
      const raw = new Float32Array(n * DIM);
      const norm = new Float32Array(n * DIM);
      const mean = new Float32Array(DIM);
      for (let r = 0; r < n; r++) {
        raw.set(rows[r], r * DIM);
        let s = 0;
        for (let d = 0; d < DIM; d++) {
          const v = rows[r][d];
          s += v * v;
          mean[d] += v;
        }
        const inv = 1 / Math.sqrt(s + 1e-9);
        for (let d = 0; d < DIM; d++) norm[r * DIM + d] = rows[r][d] * inv;
      }
      for (let d = 0; d < DIM; d++) mean[d] /= Math.max(1, n);

      const centNorm = new Float32Array(n * DIM);
      for (let r = 0; r < n; r++) {
        const base = r * DIM;
        let s = 0;
        for (let d = 0; d < DIM; d++) {
          const v = raw[base + d] - CENTER_ALPHA * mean[d];
          centNorm[base + d] = v;
          s += v * v;
        }
        const inv = 1 / Math.sqrt(s + 1e-9);
        for (let d = 0; d < DIM; d++) centNorm[base + d] *= inv;
      }
      this.refRaw = raw;
      this.refNorm = norm;
      this.refCentNorm = centNorm;
      this.nRef = n;
      this.reset();
      return { frames: n, seconds: L / SR, buildMs: performance.now() - t0 };
    }

    _matchFrame(q, frameRms) {
      const n = this.nRef;
      if (frameRms > 0.01) {
        this.srcMeanCount += 1;
        const am = 1 / Math.min(this.srcMeanCount, 100);
        const om = 1 - am;
        for (let d = 0; d < DIM; d++) {
          this.srcMean[d] = om * this.srcMean[d] + am * q[d];
        }
      }

      const useCent = CENTER_ALPHA > 0 && this.srcMeanCount >= 5;
      const qBuf = new Float32Array(DIM);
      let s = 0;
      if (useCent) {
        for (let d = 0; d < DIM; d++) {
          const v = q[d] - CENTER_ALPHA * this.srcMean[d];
          qBuf[d] = v;
          s += v * v;
        }
      } else {
        for (let d = 0; d < DIM; d++) {
          const v = q[d];
          qBuf[d] = v;
          s += v * v;
        }
      }
      const inv = 1 / Math.sqrt(s + 1e-9);
      for (let d = 0; d < DIM; d++) qBuf[d] *= inv;

      const ref = useCent ? this.refCentNorm : this.refNorm;
      const topS = new Float32Array(TOPK).fill(-Infinity);
      const topI = new Int32Array(TOPK).fill(0);
      const p1 = this.prevTop1;
      for (let r = 0; r < n; r++) {
        const base = r * DIM;
        let dot = 0;
        for (let d = 0; d < DIM; d += 4) {
          dot +=
            qBuf[d] * ref[base + d] +
            qBuf[d + 1] * ref[base + d + 1] +
            qBuf[d + 2] * ref[base + d + 2] +
            qBuf[d + 3] * ref[base + d + 3];
        }
        if (CONT_BONUS > 0 && p1 >= 0) {
          if (r === p1 + 1) dot += CONT_BONUS;
          else if (r === p1 + 2) dot += CONT_BONUS * 0.5;
          else if (r === p1) dot += CONT_BONUS * 0.25;
        }
        let mi = 0;
        for (let k = 1; k < TOPK; k++) if (topS[k] < topS[mi]) mi = k;
        if (dot > topS[mi]) {
          topS[mi] = dot;
          topI[mi] = r;
        }
      }

      let bestK = 0;
      for (let k = 1; k < TOPK; k++) if (topS[k] > topS[bestK]) bestK = k;
      this.prevTop1 = topI[bestK];
      const maxS = topS[bestK];

      const weights = new Float32Array(TOPK);
      let wSum = 0;
      for (let k = 0; k < TOPK; k++) {
        const w = Math.exp((topS[k] - maxS) * TEMP_WEIGHT);
        weights[k] = w;
        wSum += w;
      }

      const out = new Float32Array(DIM);
      for (let k = 0; k < TOPK; k++) {
        const w = weights[k] / wSum;
        const base = topI[k] * DIM;
        for (let d = 0; d < DIM; d++) out[d] += this.refRaw[base + d] * w;
      }
      return out;
    }

    /**
     * Feed exactly CHUNK (3200) input samples @16 kHz; returns CHUNK output samples @16 kHz.
     * Output lags input by ~5800 samples (aligned later by the caller if needed).
     */
    async processChunk(chunk) {
      if (!this.ready) throw new Error("kNN-VC エンジン未初期化");
      if (!this.nRef) throw new Error("参照音声が未設定です。");
      this.hist.copyWithin(0, CHUNK);
      this.hist.set(chunk, WIN_SAMPLES - CHUNK);
      const feat = await this._wavlmFeatures(this.hist);

      const t0 = performance.now();
      const j0 = WIN_FRAMES - RW - HF;
      const usePsola = Math.abs(this.pitchSemitones) >= 0.1;
      for (let k = 0; k < HF; k++) {
        const st = (j0 + k) * HOP;
        let sq = 0;
        for (let i = 0; i < 400; i++) {
          const v = this.hist[st + i];
          sq += v * v;
        }
        const frameRms = Math.sqrt(sq / 400);
        this.rmsFifo.push(frameRms);
        if (usePsola) {
          const f0St = Math.max(0, st - 100);
          const sf = fastNsdfF0(this.hist, f0St, 560, SR, 75, 700, this.prevSrcF0);
          if (sf.f0 > 0) this.prevSrcF0 = sf.f0;
          this.srcF0Fifo.push(sf.f0);
          this.srcConfFifo.push(sf.conf);
        } else {
          this.srcF0Fifo.push(0);
          this.srcConfFifo.push(0);
        }
        const q = feat.subarray((j0 + k) * DIM, (j0 + k + 1) * DIM);
        this.matched.push(this._matchFrame(q, frameRms));
      }
      this.totalEmitted += HF;
      this.lastMatchMs = performance.now() - t0;
      const need = LC + HF + RC;
      if (this.matched.length > need) this.matched.splice(0, this.matched.length - need);
      if (this.rmsFifo.length > need) this.rmsFifo.splice(0, this.rmsFifo.length - need);
      if (this.srcF0Fifo.length > need) this.srcF0Fifo.splice(0, this.srcF0Fifo.length - need);
      if (this.srcConfFifo.length > need) this.srcConfFifo.splice(0, this.srcConfFifo.length - need);

      const flat = new Float32Array(need * DIM);
      const pad = need - this.matched.length;
      const first = this.matched[0];
      for (let i = 0; i < need; i++) {
        const src = i < pad ? first : this.matched[i - pad];
        flat.set(src, i * DIM);
      }
      const wav = await this._vocode(flat);
      const s0 = LC * HOP;

      let seg;
      if (usePsola) {
        // Formant-preserving TD-PSOLA pitch shift when user adjusts #pitch-shift
        const nFrOut = HF + 1;
        const f0OutArr = new Float32Array(nFrOut);
        const confOutArr = new Float32Array(nFrOut);
        const f0TgtArr = new Float32Array(nFrOut);
        const confTgtArr = new Float32Array(nFrOut);
        const pitchMul = Math.pow(2, this.pitchSemitones / 12);
        const firstF0 = this.srcF0Fifo[0];
        const firstConf = this.srcConfFifo[0];
        for (let k = 0; k < nFrOut; k++) {
          const idx = LC + k;
          const sf = idx < pad ? firstF0 : idx - pad < this.srcF0Fifo.length ? this.srcF0Fifo[idx - pad] : this.srcF0Fifo[this.srcF0Fifo.length - 1];
          const sc = idx < pad ? firstConf : idx - pad < this.srcConfFifo.length ? this.srcConfFifo[idx - pad] : this.srcConfFifo[this.srcConfFifo.length - 1];
          f0TgtArr[k] = sf > 0 ? sf * pitchMul : 0;
          confTgtArr[k] = sc;
          const cst = Math.max(0, s0 + k * HOP - 100);
          const lenO = Math.min(560, wav.length - cst);
          const of = lenO >= 400 ? fastNsdfF0(wav, cst, lenO, SR, 75, 700, this.prevOutF0) : { f0: 0, conf: 0 };
          if (of.f0 > 0) this.prevOutF0 = of.f0;
          f0OutArr[k] = of.f0;
          confOutArr[k] = of.conf;
        }
        seg = tdPsolaChunk(wav, s0, CHUNK + OVERLAP, f0OutArr, confOutArr, f0TgtArr, confTgtArr, 320, 320);
      } else {
        seg = wav.subarray(s0, s0 + CHUNK + OVERLAP).slice();
      }
      if (this.prevTail) {
        for (let i = 0; i < OVERLAP; i++) {
          const a = 0.5 * (1 - Math.cos((Math.PI * i) / OVERLAP));
          seg[i] = this.prevTail[i] * (1 - a) + seg[i] * a;
        }
      }
      this.prevTail = seg.subarray(CHUNK, CHUNK + OVERLAP).slice();

      const out = new Float32Array(CHUNK);
      const g = this.outGain;
      const firstRms = this.rmsFifo[0];
      for (let k = 0; k < HF; k++) {
        const idx = LC + k;
        const rk = idx < pad ? firstRms : this.rmsFifo[idx - pad];
        const gate = Math.min(1.0, Math.max(0.0, (rk - GATE_LOW) / (GATE_HIGH - GATE_LOW)));
        const scale = g * gate;
        const base = k * HOP;
        for (let i = 0; i < HOP; i++) {
          out[base + i] = Math.tanh(seg[base + i] * scale);
        }
      }
      return out;
    }
  }

  KnnVcEngine.CHUNK = CHUNK;
  KnnVcEngine.SAMPLE_RATE = SR;
  KnnVcEngine.resampleMono = resampleMono;
  // With RW=5, output lags by an extra 2*320=640 samples compared to RW=3 (1960 + 640 = 2600).
  KnnVcEngine.ALIGN_TRIM_SAMPLES = 2600;
  window.KnnVcEngine = KnnVcEngine;
})();
