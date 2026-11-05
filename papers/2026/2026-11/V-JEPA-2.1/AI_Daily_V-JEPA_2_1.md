# AI Daily

## 2026-11-05 — V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning

**選中論文：** *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning*，作者為 Lorenzo Mur-Labadia、Matthew Muckley、Amir Bar、Mido Assran、Koustuv Sinha、Michael Rabbat、Yann LeCun、Nicolas Ballas、Adrien Bardes。研究團隊以 Meta FAIR 為主；Lorenzo Mur-Labadia 另列 Universidad de Zaragoza affiliation。論文收錄於 **ECCV 2026 poster**，並以 arXiv:2603.14482 發表；本文閱讀的是 arXiv v3。[1] [2] [3]

**Repo 去重：** 本次首輪排序第一的 Energy-based Transformer 論文 TEM 已有獨立文章，因此排除。V-JEPA 2.1 的 arXiv ID 在 AI Daily 既有文章中，只出現在 V-RAE 與 WA-JEPA 的相關研究／方法脈絡，並沒有同一篇論文的獨立深讀文章；因此新增本篇不會重複收錄。[6] [7]

### 為什麼今天選它

V-JEPA 2.1 命中近期偏好的 **JEPA**，也把一個具體的表示學習失衡問題講清楚：原始 V-JEPA 只要求 predictor 還原被遮住的 token，沒有直接要求可見的 context token 保留局部位置與物件結構。模型因此可能把計算集中在全局語義，留下不容易直接用於深度、分割、追蹤或操作規劃的局部特徵。V-JEPA 2.1 的核心改動，是將預測監督擴展到可見 token，再以多層自監督及影像—影片共同訓練，嘗試同時保住局部時空結構與全局理解。[1] [4]

它不是圖像生成器，也不是 energy-based Transformer；價值在於提供更適合下游生成、規劃與控制的視覺表徵。和同輪其他未重複候選相比，V-JEPA 2.1 的 JEPA 主題最直接，而且有 ECCV 2026 收錄、官方程式碼與多種尺度的已釋出 checkpoint。另一篇值得留意的候選 GT-SVJ 是 CVPR 2026 的影片評分模型，但主要用途是 reward/evaluation，而非 JEPA 表徵或直接生成影片。[2] [3]

### 核心貢獻

1. **Dense predictive loss。** 原始 V-JEPA 只對 masked tokens 做預測。新方法額外對 visible/context tokens 施加與 EMA target encoder 對齊的損失，讓局部 patch 也有明確學習訊號。[1] [4]
2. **距離加權的 context loss。** context token 越靠近遮罩區域，監督權重越大；這使模型更有動機在遮罩邊界附近維持局部連續性，而不是只靠全局線索完成補全。[1]
3. **Deep self-supervision。** 不只監督 encoder 最後一層，也串接三個中間層與輸出層的表示，讓 predictor 在四個層級產生預測並接受 loss。這補回單獨加入 context loss 後部分流失的全局分類能力。[1]
4. **跨影像與影片的共同訓練與擴展。** 使用影像和影片各自的 patch tokenizer，但共享 encoder；資料從原本約 1M 張 ImageNet 影像擴展為 VisionMix-163M，模型則擴至 2B 參數，並以高解析度 cooldown 完成訓練。[1] [3]

### 技術方法與公式

令 $x$ 為隨機遮罩後的影像或影片，$y$ 為同一輸入的未遮罩版本。online encoder $E_{\theta}$ 處理可見內容；target encoder $E_{\bar\theta}$ 以指數移動平均更新，並對 target 使用 stop-gradient。$P_{\phi}$ 是 predictor，$\Delta_y$ 是帶有時空位置的可學習 mask tokens。$M$ 表示遮罩位置，$C$ 表示可見的 context 位置。原始 V-JEPA 的 masked-token loss 為：[1]

$$
\mathcal{L}_{\mathrm{pred}}=\frac{1}{|M|}\sum_{i\in M}\left\|P_{\phi}(E_{\theta}(x),\Delta_y)_i-\operatorname{sg}\!\left(E_{\bar\theta}(y)_i\right)\right\|_1.
$$

V-JEPA 2.1 對可見 token 也做同類預測，並賦予每個 context token 個別權重：

$$
\mathcal{L}_{\mathrm{ctx}}=\frac{1}{|C|}\sum_{i\in C}\lambda_i\left\|P_{\phi}(E_{\theta}(x),\Delta_y)_i-\operatorname{sg}\!\left(E_{\bar\theta}(y)_i\right)\right\|_1,
\qquad
\mathcal{L}_{\mathrm{dense}}=\mathcal{L}_{\mathrm{pred}}+\mathcal{L}_{\mathrm{ctx}}.
$$

其中權重依 context token 到最近遮罩 token 的時空區塊距離 $d_{\min}(i,M)$ 設定：

$$
\lambda_i=\frac{\lambda}{\sqrt{d_{\min}(i,M)}}.
$$

因此近遮罩區域的 token 受到較強監督，較遠的可見 token 則不會被一律用相同強度拉向複製式預測。訓練也會將三個中間 encoder block 與最後一層的正規化表示串接，經 MLP 融合後送入 predictor；四個層級各自接受 masked-token 與 context-token 的預測監督。[1]

架構上的另一個實務重點是 modality-specific tokenizer：影片使用 3D tubelet patch embedding，影像使用 2D patch embedding，再加上模態標記；避免把靜態影像複製成多幀影片輸入。論文的標準階段訓練 135,000 iterations，影片為 16 frames、256×256，影像為 256×256；接著可再做 12,000 iterations cooldown，影片增至 64 frames、384×384，影像增至 512×512。VisionMix-163M 包含 142M 張 LVD-142M 影像，以及 YT-1B、Howto100M、Kinetics、SSv2 等影片來源；在資料混合中提高 YT-1B 與 SSv2 權重，以補足動態場景。[1]

### 實驗結果

在從 V-JEPA 2 開始的逐步消融中，ImageNet 線性分類、SSv2 動作分類、NYUv2 深度 RMSE、ADE20K 分割 mIoU 分別為 **82.2、72.8、0.682、22.2**。只加入 context loss 後，ADE20K 升至 **33.8**、NYUv2 RMSE 降至 **0.474**，但 ImageNet 降至 **72.6**，SSv2 降至 **62.5**。加入 multi-level predictor 後，SSv2 回升至 **72.1**，ADE20K 再升至 **38.6**。完整擴展資料、tokenizer、模型規模並完成 cooldown 後，四項指標為 **85.5、77.7、0.307、47.9**。這組消融是本篇最有價值的證據之一：局部監督能補足 dense feature，但若只加 loss，會損害全局辨識；最終結果來自一套組合 recipe，而非單一 loss 的獨立效果。[1]

在 2B ViT-G 的 frozen-feature 評估中，NYUv2 depth RMSE 為 **0.307**，略優於論文比較的 DINOv3 7B **0.309**；但這不是全面勝出：ADE20K mIoU 為 **47.9**，低於 DINOv3 7B 的 **55.9**，Cityscapes 為 **73.5**，也低於 **81.1**。其他結果包括 Pascal VOC12 **85.0 mIoU**、YouTube-VOS **72.7 J&F-Mean**。模型在 Ego4D short-term object-interaction anticipation 達 **7.71 mAP**，在 EPIC-KITCHENS-100 action anticipation 達 **40.8 mean-class Recall@5**。[1] [5]

機器人部分需精確界定「zero-shot」：作者先以 DROID robot data 訓練 action-conditioned predictor，再將系統部署至新環境的 Franka Panda；零樣本指的是新部署環境，而不是整套模型完全不經下游訓練。於論文主表相同設定下，V-JEPA 2 baseline 抓取成功率 **60%**，V-JEPA 2.1 為 **70%**。值得特別註明：摘要寫「20-point improvement」，但主表的 60%→70% 是 **10 個百分點**；正文另有「10% improvement」的表述，三者不完全一致。本文以可核對的主表數字為準，不把摘要的 20 點當成與表格一致的結果。[1]

導航實驗把 V-JEPA 2.1 表徵接到另外訓練的 latent world model 與 CEM 規劃器。TartanDrive 的 ATE 為 **5.687**，NWM baseline 為 **5.831**；作者報告平均 planning time 為 **10.6 秒**，對照 NWM 的 **103.2 秒**。因此應將優勢說成該論文特定 open-loop planning setup 的時間與軌跡誤差結果，不應簡化成「JEPA encoder 單獨帶來 10 倍即時控制速度」。[1]

### 相關研究與後續脈絡

**V-JEPA 2** 以 masked latent prediction 在大量影片上學習動態表徵，並把 loss 主要放在被遮罩位置；V-JEPA 2.1 的關鍵差異，是把監督擴展到 context tokens，目標由偏全局的理解與預測延伸到保留局部空間結構。[4] **DINOv3** 則代表另一條以大規模影像自監督學習高品質 dense features 的路線。兩者的比較顯示，影片 JEPA 可以在 temporal features 與 depth 上具競爭力，但不能據此推論它在所有 segmentation 或 dense-vision 任務都優於大型影像 encoder。[1] [5]

兩篇 repo 已有的後續研究把 V-JEPA 2.1 表徵接到不同目標。**V-RAE** 將凍結視覺 foundation model 表徵轉成影片生成 latent，並報告 V-JEPA 2.1 變體在 Kinetics-600 的 rFVD 為 **2.13**；這讓 V-JEPA 特徵與 generative modeling 直接接軌。[6] **WA-JEPA** 則把 V-JEPA 式表示學習改造成因果 future prediction，再以 conditional flow matching 聯合預測 scene latent 與 action trajectory，走向自動駕駛 world-action model。[7] 這兩條路徑提示：V-JEPA 2.1 的實際影響不只在單一 benchmark，而是它能否成為 generation、planning 與 action prediction 共用的高密度狀態空間。

### 個人評價與研究發想

本篇最有啟發的地方，是將「可見 token 沒有直接監督」視為表示學習中的結構性缺口，而不是只用更大的模型或更多資料掩蓋問題。距離加權 context loss 把 masked prediction 的局部鄰域變成明確訓練訊號，deep self-supervision 則處理 dense 與 global 任務互相拉扯的問題。它的證據也有清楚邊界：最終分數同時受 2B 規模、大量資料、模態 tokenizer 與高解析度 cooldown 影響；而 segmentation 上仍落後 DINOv3 7B。它應被視為一套有說服力的 JEPA 訓練 recipe，不是「所有任務都更強」的證明。[1] [5]

我認為值得追的三個問題是：

1. **JEPA × Energy-based Transformer：** 將 token 預測殘差與跨層一致性定義成局部相容能量，測試能量是否比固定的距離權重更能找出不可靠區域。可先在 NYUv2/ADE20K 測 frozen feature，再檢查能量 gate 能否改善遮擋或 domain shift 下的 dense prediction。這是延伸提案，不是原論文內容。
2. **JEPA × VAR／Flow Matching：** 以 V-JEPA 2.1 的 dense token 作為生成軌跡的結構教師，在每個 VAR scale 或 flow timestep 加入局部一致性與時間一致性評估；與 V-RAE 的影片 latent 路線區隔，先測影像 next-scale 生成中物件邊界、數量及空間關係是否改善。
3. **Training-free attention modulation：** 由 frozen JEPA 特徵在推論時產生局部結構信心圖，調節 diffusion/VAR 的 cross-attention 或 preservation strength，並在未見過的資料域驗證是否能在無額外訓練下保住背景、邊界與物件身份。應以同等 NFE、同一生成 backbone 比較，避免把額外算力誤認成表示本身的增益。

### References

[1]: https://arxiv.org/html/2603.14482v3 "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning"
[2]: https://eccv.ecva.net/virtual/2026/poster/4392 "ECCV 2026 official poster page"
[3]: https://github.com/facebookresearch/vjepa2 "Official V-JEPA 2, V-JEPA 2-AC and V-JEPA 2.1 code and models"
[4]: https://arxiv.org/abs/2506.09985 "V-JEPA 2: Self-Supervised Video Models Enable Understanding Prediction and Planning"
[5]: https://arxiv.org/abs/2508.10104 "DINOv3"
[6]: https://arxiv.org/abs/2608.13556 "V-RAE: Rethinking Video Latent Spaces for Generation"
[7]: https://arxiv.org/abs/2608.20974 "WA-JEPA: Rethinking the Video JEPA Paradigm for World-Action Modeling in Autonomous Driving"
