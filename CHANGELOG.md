# CHANGELOG

## 2026-11-14
- プロジェクト初期化
  - 内容: リポジトリ初期化、発見メモ.mdの作成、WebGPUリアルタイムストリーミングVCのアーキテクチャ設計
  - 理由: iPhone（WebGPU / iOS Safari）でも動作する超低遅延・ゼロショットFlow Matchingボイスチェンジャーの開発プロジェクト立ち上げ
- Step 1 & Step 2 実装: Causal Flow Matching VCコアモデルとONNXエクスポート
  - 内容: `src/models/` 配下（因果STFT/COLA iSTFT, CausalConvNeXt, SpeakerEncoder, CausalContentEncoder/PitchConditioner, CausalFlowMatchingDecoder, CausalVocosVocoder, StreamingFlowMatchingVC）、`src/export_onnx.py`、`tests/test_models_and_onnx.py` を実装・検証
  - 理由: WebGPU / ONNX Runtime上で40msチャンク単位の超低遅延・因果的ゼロショットボイスチェンジを実現するため
- Step 3 実装: WebGPU ＋ AudioWorklet ブラウザアプリ構築と実機検証
  - 内容: `src/server.py`（COOP/COEP対応ローカルサーバー）、`web/index.html`、`web/styles.css`、`web/app.js`、`web/vc-audio-worklet.js`、`tests/test_web_server.py` を実装し、ONNX Runtime Web (1.20.1) での話者抽出・20チャンク連続ベンチマーク（平均10.64ms / RTF 0.266）およびリロード時ウォームアップ順序バグ修正を実施
  - 理由: iPhone（iOS Safari）およびPCブラウザ上でサーバー処理なしのクライアント完結リアルタイムボイスチェンジャーと実機計測を実現するため
- Step 4 実装: 自己教師あり学習パイプラインとONNX自動再エクスポート連携
  - 内容: `src/training/`（`losses.py`, `dataset.py`, `trainer.py`）、`src/train.py`、`tests/test_training_pipeline.py` を実装・検証（全10テスト合格、3エポック学習で損失16.10→10.05低下・ONNX更新確認）
  - 理由: 非パラレルなマルチスピーカー音声データおよび合成ウォームスタートからモデル重みを学習し、即座にWebGPU用ONNXモデルを更新できるようにするため
- 方法B 実装: Hugging Face公開日本語音声コーパス自動取得・24kHz前処理とロールバック保護付き学習
  - 内容: `.gitignore` への `data/` 追加、`src/training/hf_dataset.py`（`TransactionalBackup` および `fetch_and_prepare_hf_dataset`）の実装、`src/train.py` の `--auto-hf` デフォルト化・CLI拡張、`tests/test_training_pipeline.py` への `test_transactional_backup_rollback` 追加、`litagin/jvnv_corpus_v1_no_nv` 20ファイル5エポック学習（総損失 45.3343→36.7219）・ONNX再エクスポート・全11テスト合格を確認
  - 理由: ユーザーの手動データ準備なしで高品質な日本語マルチスピーカー実音声を安全（失敗時ロールバック保証）かつ再現性高く学習データとして利用するため
- 追加学習: JVNV 400ファイルで20エポック再開学習
  - 内容: 総損失 23.95→5.00、checkpoints/vc_latest.pt と models/*.onnx を更新
  - 理由: 20ファイル学習では不足していた音声再構成品質（mel L1）を改善するため
- Common Voice擬似話者データ追加と追加学習
  - 内容: src/prepare_common_voice.py を追加（話者埋め込みで擬似話者分割、TransactionalBackup保護）、data/mixed（64話者969本）で20エポック学習（総損失 5.63→4.02）、ONNX更新
  - 理由: 4話者のみでは話者の多様性が不足し、ゼロショット汎化が見込めないため
- 追加学習（大規模化）: Common Voice 8シャード＋JVNV 232話者で20エポック再開学習
  - 内容: `data/mixed` を232話者（JVNV 4話者＋CV擬似228話者・2,588 WAV）へ拡張して Epoch 46→65 の20エポック学習を実行（`total_loss` 4.1324→3.2952、`mel_l1` 0.8431→0.6357、`spk_loss` 0.0614→0.0320、`vocoder_loss` 1.7974→1.4975）、`checkpoints/vc_latest.pt` と `models/*.onnx` を更新（一時ログ `prep.log` / `train_big.log` を整理）
  - 理由: さらに多様な話者分布でゼロショット話者類似度と再構成品質を向上させるため
