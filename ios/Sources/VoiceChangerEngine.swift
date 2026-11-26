import Foundation
import onnxruntime_objc

final class VoiceChangerEngine: ObservableObject {
    static let sampleRateOut: Double = 44100.0
    static let sampleRate16k: Double = 16000.0
    static let hopSize: Int = 512

    @Published var statusText: String = "モデル未読込"
    @Published var isReady: Bool = false
    @Published var hasReference: Bool = false
    @Published var refName: String = "未設定"

    private var env: ORTEnv?
    private var sessions: [String: ORTSession] = [:]
    private var fcpeMelBank: [Float] = []
    private var bigvganMelBank: [Float] = []

    // Reference state
    private var refStyle: [Float] = []
    private var refPromptMel: [Float] = []
    private var refPromptCond: [Float] = []
    private var refPromptFrames: Int = 0

    // Streaming state
    private var leftCtxSamples = [Float](repeating: 0, count: 16 * 512)
    private var prevTail: [Float]?

    var modelsDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("models", isDirectory: true)
    }

    init() {
        try? FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
    }

    private func resolveModelURL(_ filename: String) -> URL? {
        let docURL = modelsDirectory.appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: docURL.path) {
            return docURL
        }
        if let bundlePath = Bundle.main.path(forResource: filename, ofType: nil) {
            return URL(fileURLWithPath: bundlePath)
        }
        return nil
    }

    private func makeSession(_ filename: String, useCoreML: Bool = true) throws -> ORTSession {
        if let cached = sessions[filename] { return cached }
        guard let url = resolveModelURL(filename) else {
            throw NSError(domain: "VCiPhone", code: 404, userInfo: [NSLocalizedDescriptionKey: "モデルが見つかりません: \(filename)"])
        }
        if env == nil {
            env = try ORTEnv(loggingLevel: .warning)
        }
        let opts = try ORTSessionOptions()
        try opts.setIntraOpNumThreads(4)
        if useCoreML {
            let ep = ORTCoreMLExecutionProviderOptions()
            try? opts.appendCoreMLExecutionProvider(with: ep)
        }
        let sess = try ORTSession(env: env!, modelPath: url.path, sessionOptions: opts)
        sessions[filename] = sess
        return sess
    }

    func loadModels() async {
        await MainActor.run {
            self.isReady = false
            self.statusText = "モデル読込中 (CoreML / ANE)..."
        }
        do {
            // Load fcpe mel bank if available
            if let fcpeMelURL = resolveModelURL("fcpe_mel_128x513.bin"),
               let data = try? Data(contentsOf: fcpeMelURL) {
                fcpeMelBank = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            }
            if let bvMelURL = resolveModelURL("mel_banks.bin"),
               let data = try? Data(contentsOf: bvMelURL) {
                bigvganMelBank = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            }

            // Prefer lightweight causal whisper student if present, otherwise standard encoder
            let whisperName = resolveModelURL("whisper_causal_fp16.onnx") != nil
                ? "whisper_causal_fp16.onnx"
                : "whisper_small_enc.onnx"

            _ = try makeSession("campplus_fp16.onnx", useCoreML: true)
            _ = try makeSession("fcpe.onnx", useCoreML: false)
            _ = try makeSession(whisperName, useCoreML: true)
            _ = try makeSession("length_regulator_fp16.onnx", useCoreML: true)
            _ = try makeSession("dit_student_fp16.onnx", useCoreML: true)
            let vocoderName = resolveModelURL("vocos_44k_fp16.onnx") != nil
                ? "vocos_44k_fp16.onnx"
                : "bigvgan_fp16.onnx"
            _ = try makeSession(vocoderName, useCoreML: true)

            await MainActor.run {
                self.isReady = true
                self.statusText = "準備完了 (CoreML ANE + Metal)"
            }
        } catch {
            await MainActor.run {
                self.isReady = false
                self.statusText = "モデル未配置: ファイルアプリ > VCiPhone > models にONNXを配置してください (\(error.localizedDescription))"
            }
        }
    }

    func resetStream() {
        leftCtxSamples = [Float](repeating: 0, count: 16 * 512)
        prevTail = nil
    }

    func setReference(samples: [Float], sampleRate: Double, name: String) async throws {
        let wav16 = AudioDSP.resampleLinear(samples, from: sampleRate, to: Self.sampleRate16k)
        let cap16 = Array(wav16.prefix(Int(Self.sampleRate16k * 3.0)))
        let f0 = try runFcpe(wav16: cap16)
        let semantic = try runWhisper(wav16: cap16)
        let style = try runCampplus(wav16: cap16)
        let tPrompt = max(1, Int(Double(cap16.count) / Self.sampleRate16k * Self.sampleRateOut / Double(Self.hopSize)))
        let promptCond = try runLengthRegulator(semantic: semantic.data, sFrames: semantic.frames, yLen: tPrompt, f0: f0)

        await MainActor.run {
            self.refStyle = style
            self.refPromptCond = promptCond
            self.refPromptFrames = tPrompt
            self.refPromptMel = [Float](repeating: 0, count: 128 * tPrompt)
            self.hasReference = true
            self.refName = name
        }
    }

    func convertChunk(_ newChunk44: [Float], pitchShift: Float) throws -> [Float] {
        guard isReady, hasReference else { return newChunk44 }
        let nNew = newChunk44.count
        if AudioDSP.rms(newChunk44) < 0.004 {
            return [Float](repeating: 0, count: nNew)
        }

        // Combine left context (16 frames) + current chunk
        let comb = leftCtxSamples + newChunk44
        let wav16 = AudioDSP.resampleLinear(comb, from: Self.sampleRateOut, to: Self.sampleRate16k)

        var f0 = try runFcpe(wav16: wav16)
        if abs(pitchShift) > 0.05 {
            let factor = powf(2.0, pitchShift / 12.0)
            for i in 0..<f0.count where f0[i] > 0 {
                f0[i] *= factor
            }
        }

        let semantic = try runWhisper(wav16: wav16)
        let tSrc = max(1, comb.count / Self.hopSize)
        let cond = try runLengthRegulator(semantic: semantic.data, sFrames: semantic.frames, yLen: tSrc, f0: f0)

        let mel = try runDitStudent(cond: cond, tSrc: tSrc)
        let outFull = try runVocoder(mel: mel, tSrc: tSrc)

        // Update left context
        let ctxLen = leftCtxSamples.count
        if nNew >= ctxLen {
            leftCtxSamples = Array(newChunk44.suffix(ctxLen))
        } else {
            leftCtxSamples.removeFirst(nNew)
            leftCtxSamples.append(contentsOf: newChunk44)
        }

        // Extract aligned portion + 2048-sample Hann crossfade
        let xfade = min(2048, nNew)
        let startOff = max(0, ctxLen - xfade)
        let endOff = min(outFull.count, startOff + nNew)
        var out = [Float](repeating: 0, count: nNew)
        if endOff > startOff {
            for i in 0..<(endOff - startOff) {
                out[i] = outFull[startOff + i]
            }
        }

        if let prev = prevTail, prev.count == xfade {
            for i in 0..<xfade {
                let fadeOut = 0.5 * (1.0 + cos(Float.pi * Float(i) / Float(xfade)))
                out[i] = prev[i] * fadeOut + out[i] * (1.0 - fadeOut)
            }
        }
        let tailStart = startOff + nNew
        if tailStart + xfade <= outFull.count {
            prevTail = Array(outFull[tailStart..<(tailStart + xfade)])
        } else {
            prevTail = nil
        }

        return out
    }

    // MARK: - ORT Inference Helpers

    private func floatTensor(_ data: [Float], shape: [NSNumber]) throws -> ORTValue {
        let bytes = data.withUnsafeBytes { Data($0) }
        return try ORTValue(tensorData: NSMutableData(data: bytes), elementType: .float, shape: shape)
    }

    private func int64Tensor(_ data: [Int64], shape: [NSNumber]) throws -> ORTValue {
        let bytes = data.withUnsafeBytes { Data($0) }
        return try ORTValue(tensorData: NSMutableData(data: bytes), elementType: .int64, shape: shape)
    }

    private func extractFloats(_ val: ORTValue) throws -> [Float] {
        let data = try val.tensorData() as Data
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func runFcpe(wav16: [Float]) throws -> [Float] {
        let sess = try makeSession("fcpe.onnx", useCoreML: false)
        let (mag, nFrames) = AudioDSP.stftMagnitude(signal: wav16, nFft: 1024, hop: 160, winLen: 1024, center: true)
        var mel = [Float](repeating: 0, count: nFrames * 128)
        let nBins = 513
        if fcpeMelBank.count == 128 * nBins {
            for t in 0..<nFrames {
                let mOff = t * nBins
                let oOff = t * 128
                for m in 0..<128 {
                    var sum: Float = 0
                    let bOff = m * nBins
                    for k in 0..<nBins {
                        sum += fcpeMelBank[bOff + k] * mag[mOff + k]
                    }
                    mel[oOff + m] = logf(max(1e-5, sum))
                }
            }
        }
        let inp = try floatTensor(mel, shape: [1, NSNumber(value: nFrames), 128])
        let out = try sess.run(withInputs: ["mel": inp], outputNames: ["cent_logits"], runOptions: nil)
        let logits = try extractFloats(out["cent_logits"]!)
        return AudioDSP.decodeFcpeF0(logits: logits, nFrames: nFrames)
    }

    private func runWhisper(wav16: [Float]) throws -> (data: [Float], frames: Int) {
        if let sess = try? makeSession("whisper_causal_fp16.onnx", useCoreML: true) {
            let (mag, nFrames) = AudioDSP.stftMagnitude(signal: wav16, nFft: 400, hop: 160, winLen: 400, center: true)
            // Placeholder 80-bin log mel projection -> [1, 80, nFrames]
            var mel = [Float](repeating: 0, count: 80 * nFrames)
            for t in 0..<nFrames {
                for m in 0..<80 {
                    let k = min(200, (m * 200) / 80)
                    mel[m * nFrames + t] = log10f(max(1e-10, mag[t * 201 + k]))
                }
            }
            let inp = try floatTensor(mel, shape: [1, 80, NSNumber(value: nFrames)])
            let out = try sess.run(withInputs: ["mel": inp], outputNames: ["hidden_states"], runOptions: nil)
            let hidden = try extractFloats(out["hidden_states"]!)
            return (hidden, max(1, hidden.count / 768))
        }

        let sess = try makeSession("whisper_small_enc.onnx", useCoreML: true)
        let feats = [Float](repeating: -1.5, count: 80 * 3000)
        let inp = try floatTensor(feats, shape: [1, 80, 3000])
        let out = try sess.run(withInputs: ["input_features": inp], outputNames: ["hidden_states"], runOptions: nil)
        let full = try extractFloats(out["hidden_states"]!)
        let nFrames = max(1, Int(Double(wav16.count) / Self.sampleRate16k * 50.0))
        return (Array(full.prefix(nFrames * 768)), nFrames)
    }

    private func runCampplus(wav16: [Float]) throws -> [Float] {
        let sess = try makeSession("campplus_fp16.onnx", useCoreML: true)
        let nFrames = max(1, wav16.count / 160)
        let fbank = [Float](repeating: 0, count: nFrames * 80)
        let inp = try floatTensor(fbank, shape: [1, NSNumber(value: nFrames), 80])
        let outName = try sess.outputNames().first ?? "output"
        let out = try sess.run(withInputs: ["x": inp], outputNames: [outName], runOptions: nil)
        return try extractFloats(out[outName]!)
    }

    private func runLengthRegulator(semantic: [Float], sFrames: Int, yLen: Int, f0: [Float]) throws -> [Float] {
        let sess = try makeSession("length_regulator_fp16.onnx", useCoreML: true)
        let semT = try floatTensor(semantic, shape: [1, NSNumber(value: sFrames), 768])
        let lenT = try int64Tensor([Int64(yLen)], shape: [1])
        let f0T = try floatTensor(f0, shape: [1, NSNumber(value: f0.count)])
        let outName = try sess.outputNames().first ?? "cond"
        let out = try sess.run(withInputs: ["semantic": semT, "ylens": lenT, "f0": f0T], outputNames: [outName], runOptions: nil)
        return try extractFloats(out[outName]!)
    }

    private func runDitStudent(cond: [Float], tSrc: Int) throws -> [Float] {
        let sess = try makeSession("dit_student_fp16.onnx", useCoreML: true)
        let tTot = refPromptFrames + tSrc
        var catCond = [Float](repeating: 0, count: tTot * 768)
        for i in 0..<min(refPromptCond.count, refPromptFrames * 768) { catCond[i] = refPromptCond[i] }
        for i in 0..<min(cond.count, tSrc * 768) { catCond[refPromptFrames * 768 + i] = cond[i] }

        var x = [Float](repeating: 0, count: 128 * tTot)
        // Deterministic pseudo-random normal noise for 1-step CFM
        var seed: UInt64 = 42
        for c in 0..<128 {
            for t in refPromptFrames..<tTot {
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                let u1 = max(1e-6, Float(seed >> 32) / Float(UInt32.max))
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                let u2 = Float(seed >> 32) / Float(UInt32.max)
                x[c * tTot + t] = sqrtf(-2.0 * logf(u1)) * cosf(2.0 * Float.pi * u2)
            }
        }

        let promptX = [Float](repeating: 0, count: 128 * tTot)
        let inputs: [String: ORTValue] = [
            "x": try floatTensor(x, shape: [1, 128, NSNumber(value: tTot)]),
            "prompt_x": try floatTensor(promptX, shape: [1, 128, NSNumber(value: tTot)]),
            "x_lens": try int64Tensor([Int64(tTot)], shape: [1]),
            "t": try floatTensor([0.0], shape: [1]),
            "style": try floatTensor(refStyle, shape: [1, 192]),
            "cond": try floatTensor(catCond, shape: [1, NSNumber(value: tTot), 768]),
        ]
        let out = try sess.run(withInputs: inputs, outputNames: ["v"], runOptions: nil)
        let v = try extractFloats(out["v"]!)
        var melOut = [Float](repeating: 0, count: 128 * tSrc)
        for c in 0..<128 {
            let srcOff = c * tTot + refPromptFrames
            let dstOff = c * tSrc
            for t in 0..<tSrc {
                melOut[dstOff + t] = x[srcOff + t] + v[srcOff + t]
            }
        }
        return melOut
    }

    private func runVocoder(mel: [Float], tSrc: Int) throws -> [Float] {
        let name = resolveModelURL("vocos_44k_fp16.onnx") != nil ? "vocos_44k_fp16.onnx" : "bigvgan_fp16.onnx"
        let sess = try makeSession(name, useCoreML: true)
        let inName = try sess.inputNames().first ?? "mel"
        let outName = try sess.outputNames().first ?? "audio"
        let inp = try floatTensor(mel, shape: [1, 128, NSNumber(value: tSrc)])
        let out = try sess.run(withInputs: [inName: inp], outputNames: [outName], runOptions: nil)
        return try extractFloats(out[outName]!)
    }
}
