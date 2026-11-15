/**
 * AudioWorkletProcessor for 16kHz / 200ms (3200-sample) kNN-VC streaming.
 * Accumulates WebAudio quanta into 3200-sample chunks and plays converted chunks from a jitter buffer.
 */

const CHUNK_SAMPLES = 3200; // 200ms @ 16kHz
const RING_BUFFER_SIZE = 64000; // 4s ring
const PREROLL_SAMPLES = 9600; // wait for 3 converted chunks (600ms) before playback to absorb inference jitter
const MAX_QUEUED_SAMPLES = 25600; // cap latency drift (~1.6s)

class KnnStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.inBuffer = new Float32Array(CHUNK_SAMPLES);
    this.inWritePos = 0;

    this.outRing = new Float32Array(RING_BUFFER_SIZE);
    this.outReadPos = 0;
    this.outWritePos = 0;
    this.outAvailable = 0;
    this.primed = false;
    this.bypass = false;

    this.port.onmessage = (event) => {
      const data = event.data;
      if (!data) return;
      if (data.type === "output_chunk" && data.chunk) {
        this._pushOutputChunk(data.chunk);
      } else if (data.type === "reset") {
        this._resetBuffers();
      } else if (data.type === "set_bypass") {
        this.bypass = Boolean(data.value);
      }
    };
  }

  _resetBuffers() {
    this.inBuffer.fill(0);
    this.inWritePos = 0;
    this.outRing.fill(0);
    this.outReadPos = 0;
    this.outWritePos = 0;
    this.outAvailable = 0;
    this.primed = false;
  }

  _pushOutputChunk(chunk) {
    const len = chunk.length;
    if (this.outAvailable + len > MAX_QUEUED_SAMPLES) {
      const drop = this.outAvailable + len - CHUNK_SAMPLES * 3;
      if (drop > 0) {
        this.outReadPos = (this.outReadPos + drop) % RING_BUFFER_SIZE;
        this.outAvailable = Math.max(0, this.outAvailable - drop);
      }
    }
    for (let i = 0; i < len; i++) {
      this.outRing[this.outWritePos] = chunk[i];
      this.outWritePos = (this.outWritePos + 1) % RING_BUFFER_SIZE;
    }
    this.outAvailable = Math.min(RING_BUFFER_SIZE, this.outAvailable + len);
    if (!this.primed && this.outAvailable >= PREROLL_SAMPLES) {
      this.primed = true;
    }
  }

  process(inputs, outputs) {
    const inputChannels = inputs[0];
    const outputChannels = outputs[0];
    const hasInput = inputChannels && inputChannels.length > 0 && inputChannels[0].length > 0;
    const outPrimary = outputChannels && outputChannels.length > 0 ? outputChannels[0] : null;
    const frameCount = outPrimary ? outPrimary.length : hasInput ? inputChannels[0].length : 128;

    if (hasInput) {
      const inData = inputChannels[0];
      for (let i = 0; i < inData.length; i++) {
        this.inBuffer[this.inWritePos++] = inData[i];
        if (this.inWritePos >= CHUNK_SAMPLES) {
          let sumSq = 0.0;
          for (let j = 0; j < CHUNK_SAMPLES; j++) {
            const s = this.inBuffer[j];
            sumSq += s * s;
          }
          const inputRms = Math.sqrt(sumSq / CHUNK_SAMPLES);
          const chunkCopy = new Float32Array(this.inBuffer);
          this.port.postMessage({ type: "input_chunk", chunk: chunkCopy, inputRms }, [chunkCopy.buffer]);
          this.inWritePos = 0;
        }
      }

      if (this.bypass && outPrimary) {
        outPrimary.set(inData);
        for (let ch = 1; ch < outputChannels.length; ch++) {
          outputChannels[ch].set(inData);
        }
        return true;
      }
    }

    if (outPrimary) {
      if (this.primed && this.outAvailable >= frameCount) {
        for (let i = 0; i < frameCount; i++) {
          outPrimary[i] = this.outRing[this.outReadPos];
          this.outReadPos = (this.outReadPos + 1) % RING_BUFFER_SIZE;
        }
        this.outAvailable -= frameCount;
      } else {
        outPrimary.fill(0);
        if (this.primed && this.outAvailable === 0) {
          this.primed = false;
        }
      }
      for (let ch = 1; ch < outputChannels.length; ch++) {
        outputChannels[ch].set(outPrimary);
      }
    }
    return true;
  }
}

registerProcessor("knn-stream-processor", KnnStreamProcessor);
