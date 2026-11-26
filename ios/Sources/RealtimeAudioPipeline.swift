import Foundation
import AVFoundation

final class RealtimeAudioPipeline: ObservableObject {
    @Published var isStreaming: Bool = false
    @Published var isFilePlaying: Bool = false
    @Published var isBypass: Bool = false
    @Published var pitchShift: Float = 0.0
    @Published var fileLoop: Bool = true
    @Published var fileProgress: Double = 0.0
    @Published var fileDuration: Double = 0.0
    @Published var fileName: String = ""
    @Published var inputRms: Float = 0.0
    @Published var outputRms: Float = 0.0
    @Published var lastInferMs: Double = 0.0
    @Published var lastRtf: Double = 0.0

    private let engine: VoiceChangerEngine
    private var avEngine: AVAudioEngine?
    private let inputRing = RingBuffer(capacity: 44100 * 6)
    private let outputRing = RingBuffer(capacity: 44100 * 6)
    private let chunkSamples = 44 * 512 // ~0.511s
    private var workerTask: Task<Void, Never>?

    // File loopback playback buffer
    private var fileSamples: [Float] = []
    private var fileReadPos: Int = 0
    private var isFileMode: Bool = false
    private var isFilePaused: Bool = false

    init(engine: VoiceChangerEngine) {
        self.engine = engine
    }

    func loadAudioFile(url: URL) throws -> (samples: [Float], sampleRate: Double) {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw NSError(domain: "VCiPhone", code: -1, userInfo: [NSLocalizedDescriptionKey: "PCMバッファ確保失敗"])
        }
        try file.read(into: buf)
        let nFrames = Int(buf.frameLength)
        let nCh = Int(format.channelCount)
        var mono = [Float](repeating: 0, count: nFrames)
        if let chData = buf.floatChannelData {
            for ch in 0..<nCh {
                let ptr = chData[ch]
                for i in 0..<nFrames {
                    mono[i] += ptr[i] / Float(nCh)
                }
            }
        }
        return (mono, format.sampleRate)
    }

    func startMicVC() throws {
        stop()
        isFileMode = false
        try startAudioGraph(useMic: true)
    }

    func startFileLoopback(url: URL) throws {
        stop()
        let (raw, sr) = try loadAudioFile(url: url)
        let resampled = AudioDSP.resampleLinear(raw, from: sr, to: VoiceChangerEngine.sampleRateOut)
        fileSamples = resampled
        fileReadPos = 0
        isFileMode = true
        isFilePaused = false
        isFilePlaying = true
        fileName = url.lastPathComponent
        fileDuration = Double(resampled.count) / VoiceChangerEngine.sampleRateOut
        fileProgress = 0.0
        try startAudioGraph(useMic: false)
    }

    func toggleFilePlayPause() {
        guard isFileMode else { return }
        isFilePaused.toggle()
        isFilePlaying = !isFilePaused
    }

    func seekFile(progress: Double) {
        guard isFileMode, !fileSamples.isEmpty else { return }
        let clamped = max(0.0, min(1.0, progress))
        fileReadPos = Int(clamped * Double(fileSamples.count - 1))
        fileProgress = clamped
        inputRing.clear()
        outputRing.clear()
    }

    func stop() {
        workerTask?.cancel()
        workerTask = nil
        avEngine?.stop()
        avEngine = nil
        inputRing.clear()
        outputRing.clear()
        isStreaming = false
        isFileMode = false
        isFilePlaying = false
        isFilePaused = false
        inputRms = 0
        outputRms = 0
    }

    private func startAudioGraph(useMic: Bool) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetoothA2DP])
        try session.setPreferredSampleRate(VoiceChangerEngine.sampleRateOut)
        try session.setPreferredIOBufferDuration(0.01)
        try session.setActive(true)

        engine.resetStream()
        inputRing.clear()
        outputRing.clear()

        let av = AVAudioEngine()
        let sr = VoiceChangerEngine.sampleRateOut
        guard let monoFormat = AVAudioFormat(standardFormatWithSampleRate: sr, channels: 1) else {
            throw NSError(domain: "VCiPhone", code: -2, userInfo: [NSLocalizedDescriptionKey: "オーディオフォーマット初期化失敗"])
        }

        if useMic {
            let inputNode = av.inputNode
            let hwFormat = inputNode.outputFormat(forBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 2048, format: hwFormat) { [weak self] buf, _ in
                guard let self = self, let ptr = buf.floatChannelData?[0] else { return }
                let arr = Array(UnsafeBufferPointer(start: ptr, count: Int(buf.frameLength)))
                let r44 = AudioDSP.resampleLinear(arr, from: hwFormat.sampleRate, to: sr)
                self.inputRing.write(r44)
                let r = AudioDSP.rms(r44)
                DispatchQueue.main.async { self.inputRms = r }
            }
        }

        let srcNode = AVAudioSourceNode(format: monoFormat) { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self = self else { return noErr }
            let abl = UnsafeMutableAudioBufferListPointer(audioBufferList)
            guard let outPtr = abl[0].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
            let count = Int(frameCount)
            if self.isFileMode && !self.isFilePaused && !self.fileSamples.isEmpty {
                var feed = [Float](repeating: 0, count: count)
                for i in 0..<count {
                    if self.fileReadPos >= self.fileSamples.count {
                        if self.fileLoop {
                            self.fileReadPos = 0
                        } else {
                            self.isFilePaused = true
                            DispatchQueue.main.async { self.isFilePlaying = false }
                            break
                        }
                    }
                    feed[i] = self.fileSamples[self.fileReadPos]
                    self.fileReadPos += 1
                }
                self.inputRing.write(feed)
                let r = AudioDSP.rms(feed)
                let prog = Double(self.fileReadPos) / Double(max(1, self.fileSamples.count))
                DispatchQueue.main.async {
                    self.inputRms = r
                    self.fileProgress = prog
                }
            }
            _ = self.outputRing.read(into: outPtr, count: count)
            return noErr
        }

        av.attach(srcNode)
        av.connect(srcNode, to: av.mainMixerNode, format: monoFormat)
        try av.start()
        avEngine = av
        isStreaming = true

        workerTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            while !Task.isCancelled && self.isStreaming {
                if let chunk = self.inputRing.readArray(self.chunkSamples) {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let out: [Float]
                    if self.isBypass {
                        out = chunk
                    } else {
                        out = (try? self.engine.convertChunk(chunk, pitchShift: self.pitchShift)) ?? chunk
                    }
                    let elapsedMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
                    let rtf = elapsedMs / (Double(chunk.count) / VoiceChangerEngine.sampleRateOut * 1000.0)
                    let oRms = AudioDSP.rms(out)
                    self.outputRing.write(out)
                    await MainActor.run {
                        self.lastInferMs = elapsedMs
                        self.lastRtf = rtf
                        self.outputRms = oRms
                    }
                } else {
                    try? await Task.sleep(nanoseconds: 10_000_000)
                }
            }
        }
    }
}
