# VC iPhone (iOS Native App - Swift + CoreML ANE)

Mac / Xcode が無い Windows 環境でも、GitHub Actions (`macos-latest`) を使って無署名 `.ipa` をビルドし、Sideloadly / AltStore 等で iPhone 実機へインストールできる iOS ネイティブボイスチェンジャーアプリです。

## 構成
- `project.yml`: XcodeGen プロジェクト定義 (`onnxruntime-swift-package-manager` / CoreML EP 有効)
- `Sources/App.swift`: アプリエントリポイント
- `Sources/AudioDSP.swift`: `Accelerate` (`vDSP`) による高速 STFT・リサンプル・FCPE F0 デコード・リングバッファ
- `Sources/VoiceChangerEngine.swift`: ONNX Runtime (CoreML ANE + Metal) ゼロショット推論エンジン
- `Sources/RealtimeAudioPipeline.swift`: `AVAudioEngine` 低遅延マイク＆ファイルループバック再生
- `Sources/ContentView.swift`: SwiftUI 操作画面（参照音声・リアルタイムマイク・ファイル再生/シーク/ループ・ピッチシフト・バイパス）

## ビルド方法 (Windows から Mac なしで `.ipa` を作る)
1. このリポジトリを GitHub へ push すると、`.github/workflows/ios-build.yml` が自動実行されます（または GitHub の **Actions** タブから手動実行）。
2. ビルド完了後、Artifact `VCiPhone-unsigned-ipa` (`VCiPhone-unsigned.ipa`) をダウンロードします。
3. **Sideloadly** または **AltStore**（Windows 版あり）で自分の Apple ID を使って署名し、iPhone 実機へインストールします。

## モデルファイルの配置方法 (iPhone 実機)
iPhone の **ファイル** アプリを開き、`このiPhone内 > VCiPhone > models` フォルダへ PC から以下をコピーしてください（`UIFileSharingEnabled=YES` 有効）：
- `whisper_causal_fp16.onnx` (約 16 MB、軽量因果版) ※無い場合は `whisper_small_enc.onnx`
- `campplus_fp16.onnx` (約 14 MB)
- `fcpe.onnx` (約 41 MB) + `fcpe_mel_128x513.bin`
- `length_regulator_fp16.onnx` (約 16 MB)
- `dit_student_fp16.onnx` (約 335 MB)
- `bigvgan_fp16.onnx` (約 236 MB) ※`vocos_44k_fp16.onnx` がある場合はそちらを優先
