import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var engine = VoiceChangerEngine()
    @StateObject private var pipeline: RealtimeAudioPipeline

    @State private var showRefPicker = false
    @State private var showSrcPicker = false
    @State private var alertMessage = ""
    @State private var showAlert = false

    init() {
        let eng = VoiceChangerEngine()
        _engine = StateObject(wrappedValue: eng)
        _pipeline = StateObject(wrappedValue: RealtimeAudioPipeline(engine: eng))
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Status Card
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Circle()
                                .fill(engine.isReady ? Color.green : Color.orange)
                                .frame(width: 10, height: 10)
                            Text(engine.statusText)
                                .font(.subheadline)
                                .fontWeight(.medium)
                        }
                        Button("モデル再読込 (CoreML ANE)") {
                            Task { await engine.loadModels() }
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(12)

                    // STEP 1: Reference Speaker
                    VStack(alignment: .leading, spacing: 10) {
                        Text("STEP 1: ターゲット参照音声 (ゼロショット)")
                            .font(.headline)
                        Text("現在の話者: \(engine.refName)")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        Button {
                            showRefPicker = true
                        } label: {
                            Label("参照音声ファイルを選択", systemImage: "folder")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!engine.isReady)
                    }
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(12)

                    // STEP 2: Realtime VC & File Loopback
                    VStack(alignment: .leading, spacing: 12) {
                        Text("STEP 2: リアルタイム音声変換")
                            .font(.headline)

                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("ピッチシフト")
                                Spacer()
                                Text(String(format: "%+.0f st", pipeline.pitchShift))
                                    .monospacedDigit()
                            }
                            Slider(value: $pipeline.pitchShift, in: -12...12, step: 1)
                        }

                        HStack(spacing: 10) {
                            Button {
                                if pipeline.isStreaming {
                                    pipeline.stop()
                                } else {
                                    do {
                                        try pipeline.startMicVC()
                                    } catch {
                                        alertMessage = error.localizedDescription
                                        showAlert = true
                                    }
                                }
                            } label: {
                                Label(pipeline.isStreaming ? "停止" : "マイク変換開始",
                                      systemImage: pipeline.isStreaming ? "stop.fill" : "mic.fill")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(pipeline.isStreaming ? .red : .accentColor)
                            .disabled(!engine.isReady)

                            Button {
                                pipeline.isBypass.toggle()
                            } label: {
                                Text(pipeline.isBypass ? "バイパス: ON" : "バイパス: OFF")
                            }
                            .buttonStyle(.bordered)
                        }

                        Divider()

                        Button {
                            showSrcPicker = true
                        } label: {
                            Label("ファイルからリアルタイム再生", systemImage: "music.note")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .disabled(!engine.isReady)

                        if !pipeline.fileName.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("再生中: \(pipeline.fileName)")
                                    .font(.caption)
                                Slider(
                                    value: Binding(
                                        get: { pipeline.fileProgress },
                                        set: { pipeline.seekFile(progress: $0) }
                                    ),
                                    in: 0...1
                                )
                                HStack {
                                    Button(pipeline.isFilePlaying ? "⏸️ 一時停止" : "▶️ 再生") {
                                        pipeline.toggleFilePlayPause()
                                    }
                                    .buttonStyle(.bordered)

                                    Toggle("ループ", isOn: $pipeline.fileLoop)
                                        .labelsHidden()
                                    Text("🔁 ループ")
                                        .font(.caption)
                                }
                            }
                        }

                        // Meters
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("Input").font(.caption).frame(width: 50, alignment: .leading)
                                ProgressView(value: Double(min(1.0, pipeline.inputRms * 3.5)))
                            }
                            HStack {
                                Text("Output").font(.caption).frame(width: 50, alignment: .leading)
                                ProgressView(value: Double(min(1.0, pipeline.outputRms * 3.5)))
                                    .tint(.green)
                            }
                        }

                        HStack {
                            Text(String(format: "推論: %.1f ms", pipeline.lastInferMs))
                            Spacer()
                            Text(String(format: "RTF: %.3f", pipeline.lastRtf))
                        }
                        .font(.caption)
                        .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(12)
                }
                .padding()
            }
            .navigationTitle("VC iPhone (CoreML)")
            .task {
                await engine.loadModels()
            }
            .fileImporter(
                isPresented: $showRefPicker,
                allowedContentTypes: [.audio],
                allowsMultipleSelection: false
            ) { result in
                if case .success(let urls) = result, let url = urls.first {
                    Task {
                        do {
                            let (samples, sr) = try pipeline.loadAudioFile(url: url)
                            try await engine.setReference(samples: samples, sampleRate: sr, name: url.lastPathComponent)
                        } catch {
                            alertMessage = error.localizedDescription
                            showAlert = true
                        }
                    }
                }
            }
            .fileImporter(
                isPresented: $showSrcPicker,
                allowedContentTypes: [.audio],
                allowsMultipleSelection: false
            ) { result in
                if case .success(let urls) = result, let url = urls.first {
                    do {
                        try pipeline.startFileLoopback(url: url)
                    } catch {
                        alertMessage = error.localizedDescription
                        showAlert = true
                    }
                }
            }
            .alert("エラー", isPresented: $showAlert) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(alertMessage)
            }
        }
    }
}
