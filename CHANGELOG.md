# CHANGELOG

## 2026-11-14
- プロジェクト初期化
  - 内容: リポジトリ初期化、発見メモ.mdの作成、WebGPUリアルタイムストリーミングVCのアーキテクチャ設計
  - 理由: iPhone（WebGPU / iOS Safari）でも動作する超低遅延・ゼロショットFlow Matchingボイスチェンジャーの開発プロジェクト立ち上げ
- Step 1 & Step 2 実装: Causal Flow Matching VCコアモデルとONNXエクスポート
  - 内容: `src/models/` 配下（因果STFT/COLA iSTFT, CausalConvNeXt, SpeakerEncoder, CausalContentEncoder/PitchConditioner, CausalFlowMatchingDecoder, CausalVocosVocoder, StreamingFlowMatchingVC）、`src/export_onnx.py`、`tests/test_models_and_onnx.py` を実装・検証
  - 理由: WebGPU / ONNX Runtime上で40msチャンク単位の超低遅延・因果的ゼロショットボイスチェンジを実現するため
