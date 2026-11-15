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
- Causal NSFボコーダ導入・Slaneyメル補正・話者F0/音色適応強化と再学習
  - 内容: `CausalVocosVocoder` の連続位相ハーモニック+ノイズNSF励起・`log_fb_sum`補正・33ビン局所周波数包絡正規化、`CausalFlowMatchingDecoder.out_proj.bias=-5.5` 初期化、`CausalPitchConditioner.spk_f0_head` の `LayerNorm` 付きMLP化、`SpeakerEncoder.encode_mel` によるサイクル話者損失追加、フォルマント摂動強化（`0.75〜1.25`）を実装し、`data/jvnv`（400発話）で15エポック新規学習（`total=9.5047→5.1588`, `voc=2.1188→1.8133`）および ONNX 再エクスポート・全11テスト合格を確認
  - 理由: 100Hzフレーム周期の低音ロボット声・Slaneyスケールずれによる高域/音量歪み・男女ターゲットF0未適応・声色転写不足を根本解消するため
- ハーモニックコームF0・無音ゲート厳格化・ゼロショット包絡/VTL転写・sub-F0抑制・低ノイズFlow刷新と再学習
  - 内容: `build_harmonic_comb_bank`・`extract_f0_autocorr`（`rms_thresh=0.015`, `voicing_thresh=0.55`）・`CausalPitchConditioner` の無音UVゲート強化と因果F0平滑化、`SpeakerEncoder` への80次元有声包絡・log-F0埋め込み及び低域重み付けプーリング、`CausalFlowMatchingDecoder.compute_target_base_mel`（15ビン包絡＋5タップVTLフォルマントワープ＋AGC＋`0.04*x_0` 低ノイズFlow）、`CausalVocosVocoder` の `sub_f0_cut`、及び `VCTrainer` クロス話者包絡損失を追加し、`data/jvnv` で15エポック学習（`total=5.3325→3.4105`）・ONNX再出力・定量検証・全11テスト合格を確認
  - 理由: ピッチの急激な高低飛び、低音の同時混入、無音ノイズ漏れ、フレーム間ノイズ揺れ、およびゼロショット話者類似度不足を解消するため

## 2026-11-15
- 動的フォルマント軌跡・自然抑揚・音量AGC・ボコーダコーム自然化と1000ms窓再学習
  - 内容: `CausalContentEncoder._normalize_frame_mel` と `CausalFlowMatchingDecoder.compute_target_base_mel` で毎フレーム15ビン包絡減算を撤廃して5ビン狭帯域平滑＋全フォルマントVTLワープ＋時間不変EQ差分へ改修、非飽和ラウドネスAGC（`-4.5±3.2`）・`SpeakerEncoder.encode_mel` の `canon_tilt` 残差クランプ（`±0.90`）・`CausalPitchConditioner` の自然抑揚（`0.85x`, `tanh/0.45`）・`CausalVocosVocoder` の柔軟コーム正規化（`clamp[0.20,3.5]`, `0.70` 重み）・`VCTrainer` の発話単位 `cross_env_loss` と `base_smooth_loss=12.0` を実装し、`data/jvnv`（100フレーム=1000ms, batch 4）で15エポック新規学習（`total=4.2143→2.8084`）および ONNX（7.11MB+31.36MB）再出力・定量検証を実施
  - 理由: 毎フレーム広帯域包絡の定数化による母音・子音フォルマント消失、抑揚圧縮による単調ブザー化、AGC飽和による低音量、および `web/app.js` 内蔵220Hzテスト音声使用時の極端な低域傾斜が引き起こした「完全な機械音」症状を解消するため
- 話者リーク除去・クロス話者ニューラル話者類似度最適化・低音量UV安定化・Schroeder位相/連続F0ボコーダ改善と再学習
  - 内容: `CausalPitchConditioner` の `comb_uv` 閾値(`max_sal - 0.42`, `low_band_energy - 0.025`)・時間F0平滑比重・UVバイアス見直し、`CausalVocosVocoder` のSchroeder位相分散・キャッシュ内 `prev_f0` を用いたC1連続F0位相補間・3kHz超倍音テーパー＋事前高域強調ノイズ `noise_hp_basis`・`_smooth_mel` 二重ぼかし撤廃、`CausalFlowMatchingDecoder.compute_target_base_mel` の3ビン平滑化と `w_broad` 元話者包絡除去、`VCTrainer` の `0.7*tgt_mel` リーク撤廃・マイク音量ランダム拡張(`0.12〜1.40x`)・クロス話者ニューラル話者類似度損失＋元話者反発項＋内容サイクル損失実装を実施し、`data/jvnv` で15エポック学習（`total=6.0475→3.7751`）・ONNX（`speaker_encoder.onnx` 7.11MB, `streaming_vc_core.onnx` 31.44MB）再出力・全11テスト合格を確認
  - 理由: 「機械的なロボットの声で自分の声に近い気がする」という症状（Flowへのターゲットメル70%混合・元話者包絡残存・無防備なゼロ位相パルス励起・低マイク音量でのUV崩壊）を根本解消するため

## 2026-11-15 (Round 3-4)
- **ContentEncoder 崩壊解消 & Flow 振幅制限緩和 (`cfa0d5d`)**: `content_inv_loss` を時間平均差し引き後の動的コサイン類似度＋VICReg分散維持ヒンジ (`std >= 0.65`) へ変更、`CausalFlowMatchingDecoder` の `forward_euler_streaming` 残差上限を `0.80 -> 2.80` へ緩和、`SpeakerEncoder` 入力のフレーム平均除去とチャネル拡張＋スタジオ/CVバランス型サンプラーを導入。理由: Round 2 診断で判明した `ContentEncoder` 定数化 (`time_std=0.12`)・Flow 平坦化・JVNV 話者クラスタ崩壊の解消。
- **Vocoder 高域倍音延長と谷深化 (`345ddec`) + 232話者再学習**: `hf_taper` を `3.2kHz -> 5.2kHz` へ引き上げ、`noise_hp` 低減 (`0.015 -> 0.005`) および `exc_norm` 下限 (`0.20 -> 0.10`)・指数 (`0.70 -> 0.85`) を調整し、`data/mixed` で計 23 エポック学習・ONNX 再出力。理由: ユーザー実音声変換での高域ノイズ感 (`HFflat`) 低減と倍音明瞭度 (`CPP`) 向上。
- **ファイル入力による音声変換UIの追加**: [web/index.html](file:///c:/Users/terus/Documents/antigravity/vc-iphone/web/index.html) および [web/app.js](file:///c:/Users/terus/Documents/antigravity/vc-iphone/web/app.js) の STEP 2 に `📂 ファイルから変換`（`src-file-input`）と変換結果の `<audio>` 再生・`💾 WAV保存` リンクを追加。理由: マイク入力だけでなくローカルの `.mp3` / `.wav` 等から同じ 40ms ストリーミング推論パイプラインで変換・試聴できるようにするため。
- **案A: 事前学習済み ContentVec 因果蒸留・F0 抑揚復元・Vocoder 自然化**: [src/training/extract_ssl_cache.py](file:///c:/Users/terus/Documents/antigravity/vc-iphone/src/training/extract_ssl_cache.py) で `fwerkor/content-vec-best` 768次元SSL特徴を事前抽出（2,588ファイル）、[src/models/content_f0_extractor.py](file:///c:/Users/terus/Documents/antigravity/vc-iphone/src/models/content_f0_extractor.py) で `CausalContentEncoder` `bottleneck_dim=48` 拡張・`CausalPitchConditioner` の F0 抑揚復元（`0.45*comb_dev + 0.55*pred`）、[src/models/vocos_vocoder.py](file:///c:/Users/terus/Documents/antigravity/vc-iphone/src/models/vocos_vocoder.py) で glottal shimmer (`0.06`) と HF breathiness (`0.009`) を追加し、`data/mixed` で計22エポック学習・ONNX再出力（`speaker_encoder.onnx` 7.11MB, `streaming_vc_core.onnx` 32.40MB）・全11テスト合格（`RTF=0.122`）を確認。理由: ゼロからの教師なし内容学習によるロボット声質とF0抑揚圧縮を解消するため。
- **SelfVCメル再構成・Vocoder低域ボトルネック解消**: [src/training/dataset.py](file:///c:/Users/terus/Documents/antigravity/vc-iphone/src/training/dataset.py) のフォルマント摂動を `(0.86, 1.16)` へ緩和、[src/training/trainer.py](file:///c:/Users/terus/Documents/antigravity/vc-iphone/src/training/trainer.py) で `ssl_proj` 2層化・`content_clean/feat` 併用・時間差分メル損失追加・`mel_l1` 重み `2.5`・Vocoder への `tgt_mel` 比率 `0.85`・`voc_mel_loss` 重み `1.5` へ強化、[src/models/vocos_vocoder.py](file:///c:/Users/terus/Documents/antigravity/vc-iphone/src/models/vocos_vocoder.py) の `delta_env` 上限を `±2.5` へ拡張し、追加12エポック学習（`rf=0.5154`, `ssl=0.5278`）・ONNX再出力・`SelfVC voiced mel L1: 0.9390→0.7152`・`GT_Voc: 0.7248→0.6651`・`CrossVC f0_std=44.8Hz` / `neu(t=+0.935)` への改善を確認。理由: 時間方向の過剰平滑化とVocoder低域再現誤差によるロボット的こもりを解消するため。
- **ファイルからのリアルタイム再生 UI & 目標話者シフト強化**: [web/index.html](file:///c:/Users/terus/Documents/antigravity/vc-iphone/web/index.html) / [web/app.js](file:///c:/Users/terus/Documents/antigravity/vc-iphone/web/app.js) に `📂 ファイルからリアルタイム再生`（40ms チャンク順次 `AudioContext` 再生・ライブピッチ/バイパス反映）と `⏹️ 再生停止` を追加。[src/models/speaker_encoder.py](file:///c:/Users/terus/Documents/antigravity/vc-iphone/src/models/speaker_encoder.py)・[src/models/content_f0_extractor.py](file:///c:/Users/terus/Documents/antigravity/vc-iphone/src/models/content_f0_extractor.py) のハーモニックコーム温度を `sal * 35.0` へ先鋭化、[src/models/causal_flow_matching.py](file:///c:/Users/terus/Documents/antigravity/vc-iphone/src/models/causal_flow_matching.py) の `w_detail` を `0.12` へ低減、[src/training/trainer.py](file:///c:/Users/terus/Documents/antigravity/vc-iphone/src/training/trainer.py) で `content_clean` Flow 注入廃止・`base_f0_loss` 重み `5.0`・`cross_env_loss` 重み `2.0`（`target_spk_sim 1.4`・`src_repel 1.2`）へ強化して 10 エポック追加学習・全11テスト合格（`RTF=0.101`）・`cent_mel=14.21`（参考全編 `14.34`）・`GT_Voc=0.6326` を確認。理由: 変換後音声が元話者寄りに留まる現象を解消し、ファイル入力を即座にストリーミング試聴できるようにするため。
