import Foundation
import Accelerate

final class RingBuffer {
    private var buffer: [Float]
    private var readIdx = 0
    private var writeIdx = 0
    private var count = 0
    private let lock = NSLock()

    init(capacity: Int) {
        self.buffer = [Float](repeating: 0, count: max(1024, capacity))
    }

    var available: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func clear() {
        lock.lock()
        readIdx = 0
        writeIdx = 0
        count = 0
        lock.unlock()
    }

    func write(_ data: [Float]) {
        lock.lock()
        defer { lock.unlock() }
        let cap = buffer.count
        for sample in data {
            if count == cap {
                readIdx = (readIdx + 1) % cap
                count -= 1
            }
            buffer[writeIdx] = sample
            writeIdx = (writeIdx + 1) % cap
            count += 1
        }
    }

    func read(into out: UnsafeMutablePointer<Float>, count reqCount: Int) -> Int {
        lock.lock()
        defer { lock.unlock() }
        let toRead = min(reqCount, count)
        let cap = buffer.count
        for i in 0..<toRead {
            out[i] = buffer[readIdx]
            readIdx = (readIdx + 1) % cap
        }
        count -= toRead
        if toRead < reqCount {
            for i in toRead..<reqCount {
                out[i] = 0
            }
        }
        return toRead
    }

    func readArray(_ reqCount: Int) -> [Float]? {
        lock.lock()
        defer { lock.unlock() }
        guard count >= reqCount else { return nil }
        var out = [Float](repeating: 0, count: reqCount)
        let cap = buffer.count
        for i in 0..<reqCount {
            out[i] = buffer[readIdx]
            readIdx = (readIdx + 1) % cap
        }
        count -= reqCount
        return out
    }

    func peek(_ reqCount: Int) -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        let n = min(reqCount, count)
        var out = [Float](repeating: 0, count: reqCount)
        let cap = buffer.count
        var idx = readIdx
        for i in 0..<n {
            out[i] = buffer[idx]
            idx = (idx + 1) % cap
        }
        return out
    }
}

enum AudioDSP {
    static func rms(_ x: [Float]) -> Float {
        guard !x.isEmpty else { return 0 }
        var val: Float = 0
        vDSP_rmsqv(x, 1, &val, vDSP_Length(x.count))
        return val
    }

    static func resampleLinear(_ input: [Float], from srcRate: Double, to dstRate: Double) -> [Float] {
        if abs(srcRate - dstRate) < 1.0 || input.isEmpty { return input }
        let dstLen = max(1, Int(round(Double(input.count) * dstRate / srcRate)))
        var out = [Float](repeating: 0, count: dstLen)
        let ratio = Float(srcRate / dstRate)
        let maxIdx = input.count - 1
        for i in 0..<dstLen {
            let pos = Float(i) * ratio
            let i0 = min(maxIdx, Int(pos))
            let i1 = min(maxIdx, i0 + 1)
            let frac = pos - Float(i0)
            out[i] = input[i0] * (1.0 - frac) + input[i1] * frac
        }
        return out
    }

    static func stftMagnitude(signal: [Float], nFft: Int, hop: Int, winLen: Int, center: Bool = true) -> (mag: [Float], frames: Int) {
        let pad = center ? nFft / 2 : 0
        var padded = [Float](repeating: 0, count: signal.count + 2 * pad)
        for i in 0..<signal.count { padded[pad + i] = signal[i] }

        let nFrames = max(1, 1 + (padded.count - nFft) / hop)
        let nBins = nFft / 2 + 1
        var window = [Float](repeating: 0, count: nFft)
        let wOff = (nFft - winLen) / 2
        for i in 0..<winLen {
            window[wOff + i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(winLen)))
        }

        let log2n = vDSP_Length(log2(Float(nFft)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return ([], 0)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var mag = [Float](repeating: 0, count: nFrames * nBins)
        var frameBuf = [Float](repeating: 0, count: nFft)
        var realp = [Float](repeating: 0, count: nFft / 2)
        var imagp = [Float](repeating: 0, count: nFft / 2)

        for t in 0..<nFrames {
            let start = t * hop
            for i in 0..<nFft {
                frameBuf[i] = padded[start + i] * window[i]
            }
            realp.withUnsafeMutableBufferPointer { rPtr in
                imagp.withUnsafeMutableBufferPointer { iPtr in
                    var split = DSPSplitComplex(realp: rPtr.baseAddress!, imagp: iPtr.baseAddress!)
                    frameBuf.withUnsafeBufferPointer { fPtr in
                        fPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nFft / 2) { cPtr in
                            vDSP_ctoz(cPtr, 2, &split, 1, vDSP_Length(nFft / 2))
                        }
                    }
                    vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                    let scale: Float = 0.5
                    let outOff = t * nBins
                    mag[outOff] = abs(split.realp[0]) * scale
                    mag[outOff + nBins - 1] = abs(split.imagp[0]) * scale
                    for k in 1..<(nBins - 1) {
                        let re = split.realp[k] * scale
                        let im = split.imagp[k] * scale
                        mag[outOff + k] = sqrt(re * re + im * im)
                    }
                }
            }
        }
        return (mag, nFrames)
    }

    static func decodeFcpeF0(logits: [Float], nFrames: Int, nBins: Int = 360) -> [Float] {
        var f0 = [Float](repeating: 0, count: nFrames)
        let f0MinCent: Float = 1200.0 * log2(32.70319566 / 440.0) + 6900.0
        for t in 0..<nFrames {
            let off = t * nBins
            var maxVal: Float = -1e9
            var maxIdx = 0
            for b in 0..<nBins {
                let v = logits[off + b]
                if v > maxVal {
                    maxVal = v
                    maxIdx = b
                }
            }
            let prob = 1.0 / (1.0 + exp(-maxVal))
            if prob < 0.006 {
                f0[t] = 0
                continue
            }
            let lo = max(0, maxIdx - 4)
            let hi = min(nBins - 1, maxIdx + 4)
            var wSum: Float = 0
            var cSum: Float = 0
            for b in lo...hi {
                let p = 1.0 / (1.0 + exp(-logits[off + b]))
                wSum += p
                cSum += p * (f0MinCent + Float(b) * 20.0)
            }
            let cent = wSum > 1e-6 ? (cSum / wSum) : (f0MinCent + Float(maxIdx) * 20.0)
            f0[t] = 440.0 * pow(2.0, (cent - 6900.0) / 1200.0)
        }
        return f0
    }
}
