# AI Daily

每日精選 AI 前沿論文閱讀與深度解析。聚焦深度學習、圖像生成、表徵學習、擴散模型等前沿方向。

**Last Updated: 2026-08-22**

📚 **[完整論文索引(141 篇,按月份)](INDEX.md)**

---

## 今日閱讀

- **[SASMA — 2026-08-22：MLLM-Guided Semantic Correction for Text-to-Video Generation；以 Semantic Assessment Supervisor 產生 clean intermediate preview，再由 MLLM 輸出診斷、正向修正與負向約束，透過 Semantic Modification Assistant 的 semantic dilution → injection → trajectory resumption 在 diffusion 中途修正語義漂移；CogVideoX1.5 的 VBench Total 0.7457→0.7723、Semantic 0.6419→0.6689，且 78.79% samples 在三次 MLLM 評估內早停，但 runtime 約 254.43s→590.04s；浙江大學、Huawei Cloud（arXiv:2608.16513）](papers/2026/2026-08/SASMA/AI_Daily_SASMA.md)**

本文精選 **SASMA**，把 MLLM 從生成前的 prompt planner 或生成後的 evaluator 改造成 sampling loop 內的語義監督器。方法不更新 T2V backbone 參數，卻以 clean preview、condition residual 與雙向 latent intervention 進行 inference-time correction；報告也特別指出 training-free 不等於低成本，constraint prompt 的可靠度、MLLM bias、理論改善條件與 VRAM 表格的單位疑點仍需審慎看待。這篇工作對 **Energy-based trajectory reranking、JEPA latent critic、VAR scale-wise token correction、training-free attention modulation 與 zero-shot generation controller** 提供了具體研究接口。

---

- **[Hydra-0 — 2026-08-21：Action Flow for Generalist World Modeling and Control；將 robot command 轉成具備 visibility 的 camera-plane pixel trajectories，統一 human hands、UMI grippers、bimanual/unimanual robots 的視覺動力條件；在 Cosmos 2.5、Wan2.2 上相對 action-conditioned baseline 將 robot-motion error 降低 90.4%、object-motion error 降低 60.2%，RoboLab replayed/reference success rates 達 Pearson $r=0.96$，並由 human-demo object flow 反推 executable robot action；NVIDIA、Brown、Columbia、Harvard（arXiv:2608.18077）](papers/2026/2026-08/Hydra-0/AI_Daily_Hydra-0.md)**

本文精選 **Hydra-0**，把跨 embodiment world modeling 的條件空間從 joint-space command 改寫成 action flow：訓練時由影片追蹤取得 video-derived flow，部署時由 Isaac Lab physics rollout 與 camera projection 取得 kinematically grounded flow，再以 Gaussian feature propagation 建立 motion field 與 presence gate。它同時支援 causal autoregressive rollout、open-loop policy evaluation，以及只給 desired object flow 的 inverse world-action mode；Wan2.2 A14B 的 4-step student 在排除 guidance/VAE decode 的 generation-only 計時達 61.98 FPS、相對 bidirectional teacher 為 16.0×。這篇工作對 **JEPA physical-state grounding、Energy-based action compatibility、VAR scale-wise flow routing、training-free attention modulation 與 zero-shot visual control** 提供了清楚的實驗接口，但報告也特別區分 open-loop 與 closed-loop、generation-only 與 end-to-end latency。

---

- **[SparsePR — 2026-08-20：Partition the Support, Reconstruct the Residual，將 training-free block-sparse attention 拆成 Response-Coupled Partitioning 與 Probe-Fitted Residual Reconstruction；在 HunyuanVideo、Wan2.2、Cosmos-Predict2.5、Cosmos3-Nano 以 21.92%–25.96% executed-pair density 保持生成品質，端到端加速 1.48×–2.61×，Texas A&M University（arXiv:2608.18484）](papers/2026/2026-08/SparsePR/AI_Daily_SparsePR.md)**

本文精選 **SparsePR**，指出 per-query attention concentration 不等於可執行的 shared-route sparsity：Wan2.2 在 90% attention mass 下單列 median support 為 6.2%，八個 query 共用 route 後擴張至 22.9%。論文以 sampled-query response geometry 建立 paired K/V 與 query groups，再用少量 exact probe rows 擬合 sparse output 到 post-softmax residual 的 affine correction；Wan2.2 latency 由 1650 秒降至 917 秒，probe repair 僅佔 1.1%。這個把 residual risk、attention route 與 online calibration 接在一起的視角，對 Energy-based Transformer、JEPA predictive latent、VAR scale-wise routing 與 zero-shot attention modulation 都提供了直接研究接口。

---

- **[GATO-Vid — 2026-08-19：以 closed-form cross-attention score 與 RMSNorm-aware query injection，實現 training-free、gradient-free 的 spatially-grounded text-to-video；在 Wan2.2 的 Set 1/Set 2 將 IoU 提升至 0.363/0.324、CD 降至 0.059/0.121，額外 runtime 僅 0.4%，但 Dynamic Degree 與 Aesthetic Quality 下降，官方專案頁標示 ECCV 2026（Sorbonne Université、CNRS/ISIR、Obvious Research、Valeo.ai；arXiv:2608.13037）](papers/2026/2026-08/GATO-Vid/AI_Daily_GATO-Vid.md)**

本文精選 **GATO-Vid**，把影片生成中的空間 grounding 從 inference-time backpropagation 重寫成 pre-softmax logit 空間的三項 surrogate score：框內提高目標詞、框內抑制非目標詞、框外抑制目標詞。作者由 prompt key 的幾何差異直接求得 $b^+$/$b^-$，再以 RMSNorm ellipsoid projection、Gaussian spatial modulation 與前期 block injection 保持控制穩定；Wan2.2 上定位大幅改善，但 DD/AQ 下降，清楚呈現「更準確的位置控制」與「影片自然度」的 trade-off。這篇工作對 **Energy-based steering、JEPA predictive latent、VAR scale-wise attention modulation 與 zero-shot visual control** 提供了可直接實驗的閉式控制原語。

---

- **[Scalable Energy-Based Models — 2026-08-18：以 Dual Adversarial Training（DAT）將 JEM 的 SGLD negative phase 改為 PGD contrastive samples + BCE energy learning，並以 discriminative adversarial training 與 two-stage training 統一分類、robustness、生成與 counterfactual；ImageNet 256×256 的 DAT ConvNeXt-L 達 FID 3.29、robust accuracy 56.40%、198M 參數、36 steps，接近 VAR-d16 的 FID 3.30（MIT／ICLR 2026；arXiv:2510.13872）](papers/2026/2026-08/Scalable-EBM/AI_Daily_Scalable_EBM.md)**

本文精選 **Scalable Energy-Based Models via Adversarial Training: Unifying Discrimination and Generation**。論文提出 DAT，以 PGD 將 OOD／noise 影像推向低能量區，再用 BCE 學習 real-versus-contrastive energy；分類端同時使用 adversarial cross-entropy，兩者合成 JEM 的 joint objective。它在 ImageNet 256×256 上以 **FID 3.29** 接近 VAR-d16 的 3.30，且以約 36 個 PGD steps 取得比 ADM-G／LDM-4-G 少得多的採樣步數；但不應誤述為勝過所有 diffusion，因同表 DiT-XL/2-G 的 FID 為 2.27。這篇工作對 Energy-based Transformer 的能量地形、VAR 的 scale-wise energy guidance、JEPA 的 latent predictive energy，以及 training-free inference-time steering 都提供了清楚的研究接口。

---

- **[EditMod — 2026-08-17：Model the Edit, Not the Image: Visual Autoregressive Editing from a Source-Centric Perspective；以 source token hierarchy 為編輯狀態，在 shared autoregressive context 下用 \(Q_t-Q_s\) 建立 scale-wise residual，training-free、inversion-free、mask-free，Infinity-2B 1K 編輯 1.57 秒，source-preservation CLIP-I 0.9120 / DINO 0.8641 / LPIPS 0.2212（最新 arXiv v2: 2608.09057）](papers/2026/2026-08/EditMod/AI_Daily_EditMod.md)**

本文精選 **EditMod**，將 VAR 圖像編輯從「target generation 加 source constraint」改寫成「保留 source state，只建模 condition-induced change」。它在 coarse scales 直接 prefill source tokens，中間 scales 比較 source／target-conditioned predictions 並在 bitwise probability space 更新，最後於 fine scales 進行 target-conditioned refinement；不需訓練、反演、mask、attention control 或 per-image optimization，並對 Energy-based scale-wise editing、JEPA predictive critic 與 adaptive attention modulation 提供可直接延伸的研究接口。

---

- **[Semantic Steering — 2026-08-16：在 MM-DiT 內部以單一語義向量實現 Training-Free Concept Erasure；從中間 block 的 text-branch paired prompts 建構 steering vector，注入連續 early+middle blocks，SDv3.5 celebrity GIPHY/LLaVA 0.020/0.002、FLUX.1 nudity NudeNet 0.023，並標示 Accepted to ACM MM 2026（中國科學院信息工程研究所；arXiv:2608.12829）](papers/2026/2026-08/Semantic-Steering/AI_Daily_Semantic_Steering.md)**

本文精選 **Semantic Steering**，將 MM-DiT 的 concept erasure 從 prompt/latent guidance 推進到模型內部 representation geometry：在中間 block、intermediate timestep 以「目標概念—安全替代概念」paired prompts 的 text-token 差值建立單一向量，再於所有 denoising steps 注入 early+middle blocks。它不更新 frozen backbone，卻能同時處理 celebrity、art style、nudity，並在 FLUX.1 的四類 adversarial prompts 上將 NudeNet 降至 **0.057 / 0.014 / 0.016 / 0.013**。這篇工作對 **Energy-based semantic steering、JEPA predictive-latent safety critic、VAR scale-wise early commitment，以及 selective attention modulation** 都提供了可直接延伸的介面。

---

- **[xLARD — 2026-08-15：Self-Corrected Image Generation with Explainable Latent Rewards；以可解釋 counting/color/position reward 訓練輕量 latent corrector，推理時單次 residual 修正，GenEval 0.81、DPG-Bench 86.45（CMU、SMU、William & Mary；CVPR 2026；arXiv:2603.24965）](papers/2026/2026-08/xLARD/AI_Daily_xLARD.md)**

本文精選 **xLARD**，把多模態理解、可解釋 reward 與 latent residual 接成生成—理解閉環。它以 frozen generator 保留原有生成先驗，另訓練 Understanding-Guided Reinforcement Corrector 與 Latent Reward Projection，將數量、顏色、空間關係的 image-level 評估映射成可微的三維 latent reward；OmniGen2 上 GenEval 由 0.77 升至 **0.81**、DPG-Bench 由 83.48 升至 **86.45**，推理不增加 sampling 或 reward evaluation。這篇論文也提醒我們嚴格區分 frozen-backbone 與 training-free：xLARD 不是完全免訓練，而是將大型 backbone 的重訓改成小型、可解釋、可蒸餾的 controller，對 VAR scale-wise guidance、JEPA representation critic 與 Energy-based latent modulation 都提供直接接口。

---

- **[V-RAE — 2026-08-14：以凍結視覺基礎模型表徵直接定義影片生成 latent；時間注意力池化保留語義、3D RoPE 解碼連續動態，V-JEPA 2.1 變體 K600 rFVD 2.13、gFVD 19.16，並提出以時間中點插值檢驗生成友善 latent 幾何的 tFVD（NUS & University of Oxford；arXiv:2608.13556）](papers/2026/2026-08/V-RAE/AI_Daily_V-RAE.md)**

本文精選 **V-RAE**（National University of Singapore、University of Oxford），把 frozen DINOv3、SigLIP2、EUPE 與 V-JEPA 2.1 的視覺表徵，透過可學時間 attention pooling 壓縮為可生成的影片狀態，再以 3D RoPE Transformer 解碼。其核心洞見是：重建最優的 latent 不必最適合生成或世界模型 rollout。作者以 **tFVD** 對相鄰 latent 做中點插值，發現其與下游 gFVD 的相關性在 K600 為 **0.919**，顯著高於 rFVD 的 0.473；在固定 conditional DiT 與訓練預算的 Cityscapes 預測中，V-RAE 雖有較高 rFVD，卻將 gFID / gFVD 降至 **11.52 / 111.36**。這為 JEPA latent、Energy-based 平滑度、attention modulation 與影片 VAR 的長期 rollout 提出可實驗化的新問題。

---

- **[JoyAI-Video-Edit — 2026-08-13：以自迴歸擴散實現即時、開放式串流影片編輯；SA-DMD 以來源錨定蒸餾抵抗長期漂移，16B 模型單張 B200 達 720p 約 30 FPS (Joy Future Academy, JD; arXiv:2608.03974)](papers/2026/2026-08/JoyAI-Video-Edit/AI_Daily_JoyAI_Video_Edit.md)**

本文精選 **JoyAI-Video-Edit**（Joy Future Academy, JD）的 16B 自迴歸擴散影片編輯系統。它以 chunk 內雙向、chunk 間因果的注意力與 bounded-history KV cache，在不讀取未來畫格、也不預設影片時長的條件下持續輸出編輯結果。核心的 **Source-Anchored Distribution Matching Distillation（SA-DMD）** 把文字遵從與當前來源影片保真度拆為兩條 CFG 軸，將 source-aware 指引蒸餾進兩步 student；再以 **Long-Horizon Autoregressive Distillation** 對長 rollout 做分段反傳，抑制自生成歷史帶來的漂移。LongV2VBench 一分鐘測試中，模型以 **3.30** overall 與 **30.19 FPS** 領先串流基線，對 attention modulation、JEPA latent critic、energy-based source anchoring 與低延遲 video editing 都具有直接啟發。

---

- **[AdaLN-Zero — 2026-08-12：解開 DiT 條件化的初始化祕密，從零初始化到 Gaussian 初始化與 SE 式殘差門控 (IEEE TPAMI 2026)](papers/2026/2026-08/AdaLN-Zero/AI_Daily_AdaLN_Zero.md)**

本文精選 **Unveiling the Secret of AdaLN-Zero in Diffusion Transformer**（北京大學、UC Berkeley、百度；IEEE TPAMI 2026 已接收）。論文以析因實驗拆解 DiT 的 adaLN-Zero：SE-like 通道調制、零初始化與漸進式更新。核心結論是「良好的初始化位置」比短暫更新順序更關鍵，並據此提出 **adaLN-Gaussian**：將條件調制權重由全零改為零均值高斯。ImageNet-1K 256×256 的 DiT-XL/2 在 400K steps 由 FID **20.02** 降至 **17.86**，800K 由 **14.73** 降至 **13.14**；結合 SE-like 結構後，參數由 **676M** 降至 **582M**，FID 仍降至 **18.76**。這項工作直接啟發 attention modulation、Energy-based Transformer、JEPA predictor 與 VAR 的條件殘差設計：先理解目標分佈，再選擇 gate 的初始化與調制方式。

---

- **[HRDiT — 2026-08-10：免訓練將現成DiT模型擴展至高解析度圖像生成 (ECCV 2026)](papers/2026/2026-08/HRDiT/AI_Daily_HRDiT.md)**
本文提出 **HRDiT**，一個專為 Diffusion Transformer (DiT) 設計的免訓練（Training-Free）高解析度圖像生成框架。針對現成 DiT 模型（如 FLUX, Stable Diffusion 3）在高解析度下容易出現的「空間混亂」與「生成時間過長」兩大痛點，作者提出了兩個即插即用的模組：(1) **空間位置對齊 (SPA)**：透過 Bundle 和 Slide 操作重新調整位置編碼輸入，解決高解析度下位置信號表達能力不足的問題；(2) **自適應頭部注意力剪枝 (HAP)**：利用泰勒展開式在單次前向傳播中估算各注意力頭的最佳局部窗口大小，大幅削減冗餘計算。實驗證明，HRDiT 在 4K 甚至 8K 解析度下，不僅生成品質顯著超越現有方法，推理速度也提升了近一倍，為 DiT 的高解析度應用提供了極具價值的新思路。

---
**[EG-FM — 2026-08-09：Energy-Guided Flow Matching: 以熱核濾波動態頻譜端點取代固定端點，讓 Flow Matching 生成軌跡顯式遵循 coarse-to-fine 頻率演化，sample-adaptive heat-time scheduling 根據每張圖的頻譜能量自適應釋放高頻訊號，ImageNet 256×256 FID 1.45（4× 訓練加速），512×512 FID 1.58（僅 40 epoch 微調），GenEval 0.85 / DPG-Bench 83.9，即插即用無需修改 backbone，對 Energy-based、Training-Free 軌跡設計與 Zero-shot 生成均有深遠啟示 (CASIA & JD.com, arXiv:2608.05811)](papers/2026/2026-08/EG-FM/AI_Daily_EG-FM.md)**

本文提出 **EG-FM（Energy-Guided Flow Matching）**（中科院自動化研究所 CASIA & 京東），一個從頻譜能量視角重新設計 Flow Matching 生成軌跡的創新框架。標準 Flow Matching 將噪聲插值到一個固定的乾淨圖像端點，頻譜演化完全交由模型隱式學習，增加了優化難度。EG-FM 的核心創新在於：引入一個由 heat-kernel 濾波器生成的**移動頻譜端點（Moving Spectral Endpoint）** $y_t(x) = \mathcal{F}^{-1}(R(h(x,t),\rho)*\hat{x})$，使生成過程顯式地從低頻流形（Low-Pass Manifold）逐步過渡到全頻流形（Full-Image Manifold）。高頻訊號的釋放速度由每張圖像的頻譜能量 $E(\rho)=\|\hat{x}(\rho)\|_2^2$ 決定，透過全域釋放時鐘 $q(t)$ 對齊不同樣本的恢復比例：$G_x(h(x,t))/\tilde{G}_x=q(t)$，確保紋理豐富的圖像更早釋放高頻訊號。由於端點移動，訓練目標速度包含端點運動項：$v_t = y_t(x)-\epsilon + t\partial_t y_t(x)$，整個框架無需修改 backbone 架構，以即插即用方式應用於 PixelDiT、DeCo、HyperDiT 等現有模型。實驗顯示，PixelDiT-XL + EG-FM 在 **200 epochs** 即達 FID **1.55**（原版需 800 epochs），600 epochs 進一步降至 **1.45**，實現近 **4× 訓練加速**；512×512 僅需 40 epoch 微調達 FID **1.58**；Text-to-Image 在 GenEval 達 **0.85**、DPG-Bench 達 **83.9**。這種「把 coarse-to-fine 從直覺變成可解析、可微、可對齊的 spectral schedule」的思路，對 Energy-based Transformer 的能量函數設計、Training-Free 推理期軌跡控制，以及 Zero-shot 複雜語義生成均有深刻啟發。

---
**[UDT — 2026-08-08：UDT: Reconciling U-Nets and Diffusion Transformers with Data-Adaptive Token Reduction，提出以 Training-Free 的 Token Merging（ToMe）取代傳統空間下採樣，在保持 Token 隱藏維度 $D$ 不變的前提下構建 U-Net 式 encoder-decoder 結構，解決 DiT 的 encoder-decoder 不平衡問題，XL 模型 40 epoch 達 FID 7.6（SiT 需 1400 epoch），CFG 下 FID 1.35（VA-VAE），完美相容 REPA 且可作為 T2I/MMDiT 的 Drop-in Backbone，University of Minnesota (arXiv:2608.01298)](papers/2026/2026-08/UDT/AI_Daily_UDT.md)**

本文提出 **UDT（U-Net Diffusion Transformer）**（明尼蘇達大學），一個以 **Data-Adaptive Token Merging** 為核心的 Diffusion Transformer 架構創新。DiT 由等向性 Transformer 區塊組成，其去噪目標迫使深層網路聚焦於高頻細節重建，導致語義表徵品質在中後層達峰後急劇下降，形成「編碼器過長、解碼器過短」的不平衡。現有 U-Net DiTs（U-DiT、SiT↓/UREPA）雖引入多尺度結構，但依賴固定 $2\times2$ 空間網格下採樣，破壞了 Transformer 的全域 token 互動，且因 token 維度不匹配而難以與 REPA 直接結合。UDT 的核心設計原則是：**保持 token 隱藏維度 $D$ 不變，僅在 token 序列長度上做 data-adaptive 縮減**。具體而言，Encoder 階段利用 ToMe 的二分軟匹配（Bipartite Soft Matching）根據 key 相似度逐層合併語義相近 token：$\mathbf{x}_{1,2}^{\text{m}} = (s_1\mathbf{x}_1 + s_2\mathbf{x}_2)/(s_1+s_2)$；自注意力加入比例修正 $\text{softmax}(\mathbf{QK}^T/\sqrt{d}+\log\mathbf{s})$ 以修正合併偏差；Decoder 依據記錄的 merge indices 精確 unmerge，Skip Connection 補充高頻資訊。這種設計使 UDT 能優先壓縮背景等冗餘區域，保留細節豐富的 token，同時完美相容 Cross-Attention（T2I）與 REPA（無需維度轉換模組）。實驗顯示，UDT-XL/2 在 ImageNet 256×256 上無 CFG 僅需 **80 epochs** 達到 FID 7.7（SiT 需 1400 epochs，加速 ~20×），結合 REPA 後 **40 epochs** 達 FID 7.6（加速 ~40×）；使用 CFG 時以 SD-VAE 達 FID **1.38**（320 epochs），VA-VAE 達 FID **1.35**（500 epochs）。UDT 可作為 DiT、SiT、JiT、MMDiT 的 Drop-in Replacement，對 Training-Free 加速、Attention Modulation 及未來 JEPA/Energy-Based 世界模型的階層式 Token 設計均有深遠啟示。

---
**[RTD — 2026-08-07：Rectify Then Diffuse: Disentangling Concepts Before Denoising Trajectory Unfolds，將多概念生成失敗重新定義為「邊界條件問題」而非「軌跡控制問題」，提出 training-free 的 SOD（Soft-Overlap Disentanglement）以可微 soft IoU 量化初始 attention 的概念空間重疊，搭配 IGR（Isotropic Gradient Rectification）對初始 latent 做一次各向同性的 one-shot 修正，去噪後完全回到原始 sampler，AE-Bench O-O 子集 BLIP-VQA 0.7503（+45.8% vs CO3），推理開銷僅 +6.3%，比 CO3 快 2.3×，UESTC (arXiv:2608.03135)](papers/2026/2026-08/RTD/AI_Daily_RTD.md)**

本文提出 **RTD（Rectify-then-Diffuse）**（電子科技大學），一個針對多概念文本到圖像生成的 training-free 框架。現有方法（Attend-and-Excite、CO3 等）均在去噪過程中反覆干預 attention 或修正 score，屬於「軌跡控制」策略。RTD 的核心洞見是：**生成失敗的根本原因在於去噪開始前，初始 latent 對各概念的空間分配（spatial allocation）就已重疊**——即「早期分配瓶頸（Early Allocation Bottleneck）」。RTD 在高噪聲 pilot timestep（$t_\text{pilot}=980$）做一次 diagnostic forward pass，提取每個概念的 cross-attention map $A^{(k)}$，以 max-min normalization 得 soft occupancy map $M^{(k)}$，再以 soft IoU 定義分離目標 $\mathcal{S}(x_T)=1-\frac{2}{K(K-1)}\sum_{i<j}\mathrm{O}_{ij}$（其中 $\mathrm{O}_{ij}=\langle M^{(i)},M^{(j)}\rangle/(\|M^{(i)}\|_1+\|M^{(j)}\|_1-\langle M^{(i)},M^{(j)}\rangle+\epsilon)$）。最後以 IGR 對初始 latent 做一次各向同性修正：$x_T'=x_T+\rho\|x_T\|_2\hat{g}$（$\hat{g}=g/\max(\|g\|_2,\epsilon)$），之後完全回到原始 sampler 不再干預。在 AE-Bench O-O 子集，RTD 達到 BLIP-VQA **0.7503**（+45.8% vs CO3）、ImageReward **1.2144**（+19.6% vs CO3），early overlap 最低（S-IoU₅ 0.2113），推理開銷僅增 6.3%，比 CO3 快 2.3×。這種「先診斷初始條件、一次性修正邊界、完全信任原始模型」的思路，對 attention modulation、training-free zero-shot 生成，以及 Energy-Based / JEPA 視角下的 latent space shaping 均有深刻啟發。

---
**[Perceptual Anchoring (PTC) — 2026-08-06：Perceptual Anchoring: Prototype-Guided Text Calibration for Training-free Open-Vocabulary Semantic Segmentation，首次從文本端出發解決 Training-free OVSS 中的語義鴻溝問題，提出 PTC 模組透過「感知錨定」概念，以 Margin-based 可靠性評估從輸入圖像構建 Category-specific Visual Prototype，再以證據驅動的自適應校準將通用文本嵌入錨定至實例特定視覺外觀，無需任何訓練或外部模型，即插即用地在 8 個 benchmark、6 個 baseline 上全面提升 mIoU（NACLIP +2.2%、ProxyCLIP +1.9%、CorrCLIP +0.9%），對 Cross-Attention Modulation 與 Zero-Shot 圖像生成具有重要啟發意義，HUST & Hangzhou Dianzi University (arXiv:2608.03991)](papers/2026/2026-08/PTC/AI_Daily_PTC.md)**

本文提出 **PTC（Prototype-Guided Text Calibration）**（華中科技大學、杭州電子科技大學），一個針對 Training-free 開放詞彙語義分割（OVSS）的即插即用文本校準模組。現有 training-free OVSS 方法大多聚焦於修補視覺特徵（如改進 CLIP 的自注意力機制或引入 SAM/DINO 等外部模型），卻普遍忽略了通用類別文本嵌入（Generic Text Embeddings）與輸入圖像中特定實例視覺外觀（Instance-specific Visual Representations）之間的**語義鴻溝**。PTC 從認知機器人學的「感知錨定（Perceptual Anchoring）」概念汲取靈感，在推理時動態地從輸入圖像中提取可靠的視覺證據，構建類別特定的視覺原型，再以證據驅動的自適應強度校準對應的文本嵌入。具體而言，PTC 以「得分邊距（Score Margin）」$\Delta_i = S_{i,\hat{c}_i} - \max_{c' \neq \hat{c}_i} S_{i,c'}$ 篩選可靠 token，以混合策略 $K_c = \min(N_c, \max(K_{\min}, \lfloor \rho N_c \rfloor))$ 決定證據數量，並以對數自適應校準強度 $\mu_c = \mu \cdot \min(1, \log(1+n_c^{\text{ev}})/\log(1+\lambda K_{\min}))$ 防止語義偏移。最終校準公式 $t_c^{\text{cal}} = (1-\mu_c)t_c + \mu_c V_c^{\text{proto}}$ 在保留通用語義的同時引入實例特定的視覺線索。實驗在 8 個 benchmark（VOC、Context、COCO、Cityscapes、ADE20K）、6 個代表性 baseline 上全面驗證，NACLIP 平均 mIoU 從 39.0% 提升至 41.2%（+2.2%），ProxyCLIP 從 42.3% 提升至 44.2%（+1.9%），即使已整合 SAM 的 CorrCLIP 也從 51.0% 提升至 51.9%（+0.9%）。這種「視覺證據反向調製文本條件」的思路，對 Diffusion Model 的 Cross-Attention Modulation 與 Zero-Shot 圖像生成/編輯方向具有重要啟發意義。

---
**[SPARE — 2026-08-05：SPARE: Structural Parameter-Free Affinity Regularization for Flow Matching，提出針對 Flow Matching 與 DiT 的免參數結構性親和力正則化方法。透過計算中間層 Token 間的餘弦親和力，並將其與乾淨 VAE 潛在特徵的親和力分佈進行 KL 散度對齊（包含圖像內與跨圖像同位置特徵），在零額外參數與僅增 0.08 GB 內存下，ImageNet 256×256 SiT-XL/2 FID 達 13.86，全面超越 Dispersive Loss，並可與 REPA 疊加使用將 FID 進一步降至 1.90，證明了 VAE 潛在特徵本身即具備強大的幾何結構先驗 (arXiv:2608.01990)](papers/2026/2026-08/SPARE/AI_Daily_SPARE.md)**

本文提出 **SPARE (Structural Parameter-Free Affinity Regularization)**，一種針對 Flow Matching 與 Diffusion Transformers (DiT) 的免參數（Parameter-Free）結構性親和力正則化方法。現有加速 DiT 訓練的表示正則化方法中，Target-based（如 REPA）需引入外部編碼器與投影頭，增加訓練成本；而 Target-free（如 Dispersive Loss）雖免參數但僅透過排斥批次內樣本來增加多樣性，忽略了資料的空間結構。SPARE 的核心洞見在於：**乾淨的資料潛在特徵（Clean Data Latent）本身就蘊含了豐富的空間與語義結構，且可透過 Token 間的「親和力（Affinity）」來表示**。SPARE 計算中間層 Token 的成對親和力，並與乾淨潛在特徵的親和力分佈進行對齊。更重要的是，SPARE 不僅對齊單張圖像內的親和力，還對齊跨圖像同位置（Cross-Image Same-Position）的親和力，打破了過去認為跨圖像特徵應盲目排斥的思維。實驗顯示，在 ImageNet 256×256 上，SPARE 在零額外參數下達到了所有免參數方法中的最佳 FID，且能與 REPA 完美結合，進一步提升生成品質，對未來設計更高效的 Training-Free 訓練框架極具啟發性。

---
**[Signed Rectified Flow — 2026-08-04：Signed Rectified Flow: Negativity-Controlled Generation，將 Rectified Flow 推廣至帶符號測度 $\pi^{\mathtt{sign}}=(1+\alpha)\pi^+-\alpha\pi^-$，提供數學保證的負向排斥生成框架，State-Aware CFG 在 ImageNet 256×256 以 16 NFE 將 FID 從 2.38 降至 1.82，Data Repulsive Flow 反記憶化 FID 2.03 同時增大訓練集最近鄰距離，SD 3.5 裸露 ASR 從 15.19% 降至 6.33%，UT Austin (arXiv:2607.18516)](papers/2026/2026-08/SignedRF/AI_Daily_SignedRF.md)**

本文提出 **Signed Rectified Flow (Signed RF)**（UT Austin，Qiang Liu 課題組），一個將 Rectified Flow 推廣至**帶符號測度（Signed Measure）**的生成框架。核心思想是定義帶符號目標 $\pi^{\mathtt{sign}}=(1+\alpha)\pi^+-\alpha\pi^-$，雖然帶符號測度不能直接採樣，但 Signed RF 構建了一個有效的 ODE 生成過程，能在數學上**保證**生成樣本集中在 $\pi^+$ 的正區域，同時完全排斥被 $\pi^-$ 主導的區域。理論分析揭示了三個關鍵概念：**可達區域（Reachable Region）**、**負區域（Negative Region）**和**幽靈區域（Ghost Region）**，帶電粒子類比提供了直觀的物理圖像。實踐上，Signed RF 作為 State-Aware CFG 在 ImageNet 256×256 以 16 NFE 將 FID 從 2.38 降至 **1.82**；作為 Data Repulsive Flow 在反記憶化測試中 FID 保持 **2.03**（基礎模型 2.07）同時顯著增大與訓練集的最近鄰距離；在 SD 3.5 安全生成中將對抗性提示詞的裸露 ASR 從 15.19% 降至 **6.33%**。這項工作完美融合了 Flow Matching 理論、Energy-based Model 的負向排斥思想，以及 Training-Free 推理期引導，對 Alignment、版權保護、Safe Generation 等方向均有深遠啟示。

---

**[Explorative Modeling — 2026-08-03：Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation，UIUC & Harvard 提出生成式建模的「第三個預訓練軸」——探索（Exploration），與參數和資料並列，分解訓練迴圈而非生成過程以实現端到端生成，Forward XM 最小化 $\min_{i\in\{1,\ldots,K\}}J(\hat{y}_{i},x)$，Reverse XM 最小化 $\min_{i\in\{1,\dots,K\}}J(\hat{y},x_{i})$，FLOP 效率提升 4.1× / 樣本效率提升 6.2× / 參數效率提升 47%，ImageNet 256×256 無引導 gFID **1.43**，探索增益隨規模成長（資料效益 7%→36%，模型效益 13%→23%），與 Energy-Based Transformers 及 JEPA 世界模型結合有深遠啟示 (arXiv:2607.27372)](papers/2026/2026-08/ExploratveModeling/AI_Daily_Explorative_Modeling.md)**

本文提出 **Explorative Modeling (XM)**（UIUC & Harvard），一個顏覆性的新生成式建模範式。現有可擴展的生成模型（Diffusion、Autoregressive、Flow Matching）均透過分解「生成過程（Generation Procedure）」來處理多眾數分佈，這根本上阻礙了端到端訓練。XM 改為分解「訓練迴圈（Training Loop）」：在每個訓練步驟，模型探索 $K$ 個候選生成與資料的配對，只對最佳匹配進行反向傳播。這一設計使「生成式表達力（Generative Expressivity）」成為第三個可獨立擴展的訓練軸，與參數和資料並列。實驗表明，在最強 RAE 方案上加入探索，可將 FLOP 效率提升 **4.1×**、樣本效率提升 **6.2×**、參數效率提升 **47%**，在 ImageNet 256×256 無引導生成下達到近 SOTA 的 **gFID 1.43**。探索帶來的增益隨規模成長：資料效益從 7% 升至 36%，模型效益從 13% 升至 23%。作為獨立的端到端生成方案，Explorative Policy 匹配 Diffusion Policy 在 Robomimic 上的表現，但推理僅需 **1 次前向傳遞**（NFE: 1）而非 100 次。論文明確指出 XM 與 Energy-Based Transformers 結合、以及與 JEPA 世界模型配對是兩大未來方向，為近期關注這兩個方向的研究者提供了極具啟發性的新規訓練范式。

---

**[Chimera — 2026-08-02：Chimera: Designing and Chinchilla-Scaling Hybrid Visual Diffusion Transformers，Adobe Research 提出混合視覺擴散主幹（KDA 線性注意力 + MLA 全域注意力 + 模態感知短卷積 + 稀疏 MoE），完全捨棄位置編碼（NoPE）實現 Zero-Shot 長度外推（5s→30s，FID 僅退化 6.5%），HeteroP 超參轉移首次為異質視覺擴散模型建立 Chinchilla Scaling Laws（圖像 $N_{opt}\propto C^{0.48-0.52}$，影片 $N_{opt}\propto C^{0.53-0.56}$），完整版 Chimera 計算效率達 Wan 2.1 的 7.3×，KDA/MLA 主幹支援 1.68× 更長序列且推理速度快 2.14×，GenEval 0.82 / DPG-Bench 85.12，Adobe Research (arXiv:2607.28611)](papers/2026/2026-08/Chimera/AI_Daily_Chimera.md)**

Adobe Research 提出 **Chimera**，一個將混合視覺擴散主幹架構與系統化 Scaling Recipe 共同設計的新框架，專為「Token 密集型（token-extensive）」的高解析度圖像與長影片生成而生。核心架構由三個互補機制構成：**Kimi Delta Attention (KDA)** 以 $\mathcal{O}(N)$ 複雜度進行長上下文狀態追蹤，**Multi-head Latent Attention (MLA)** 週期性插入以恢復全域互動，以及**模態感知短卷積（mShortConv）**在 KDA 更新前捕捉局部時空上下文。這三者的組合使模型能以單一時間優先光柵掃描處理多模態 token，完全捨棄了傳統位置編碼（NoPE），換來了驚人的 Zero-Shot 長度外推能力——僅在 5 秒影片上訓練，直接生成 30 秒影片時 FID 僅退化 6.5%，遠優於 Wan 2.1（50.5%）和 HunyuanVideo-1.5（53.6%）。在 Scaling 方面，論文提出 **HeteroP** 超參轉移方法，根據每個模組的功能性 fan-in 獨立計算縮放比例，首次為視覺擴散模型建立了 Chinchilla-style 計算最佳化定律，揭示圖像預訓練的最佳資源分配近似均衡（$N_{opt}\propto C^{0.48-0.52}$），而影片預訓練在高算力下略偏向增大模型（$N_{opt}\propto C^{0.53-0.56}$）。在僅約 600 H100 Days 的訓練預算下，完整版 Chimera 達到 Wan 2.1 基線的 **7.3 倍計算效率**，GenEval **0.82**，DPG-Bench **85.12**，並能 Zero-Shot 直接生成 2K/4K 高解析度圖像，為資源受限下的高效視覺生成提供了一條系統性的設計與擴展路徑。

---

**[FreqForcing — 2026-08-01：FreqForcing: Autoregressive Long Video Generation via Spectral Self-Anchoring，首個從頻域視角系統性解決自迴歸長影片誤差累積的 Training-Free 框架，Spectral Self-Anchoring (SSA) 以雙分支注意力（Local + Anchor）在 3D FFT 頻域融合低頻穩定性與高頻動態，Gaussian Low-pass Filter 頻率選擇性注入 $\lambda H_{\mathrm{lp}}(\hat{A}_{\mathrm{anc}}-\hat{A}_{\mathrm{loc}})$，將 Self-Forcing 從 5s 預訓練 Zero-Shot 外推至 2 分鐘（24× 外推），VBench-Long Dynamic Degree 59.58 / Overall Consistency 20.94 超越所有 Training-Free 方法，SJTU & Tencent HY Team (arXiv:2607.27110)](papers/2026/2026-08/FreqForcing/AI_Daily_FreqForcing.md)**

本文提出 **FreqForcing**（上海交通大學、騰訊 HY 團隊），一個**完全免訓練（Training-Free）**的自迴歸長影片生成框架。論文的核心洞見在於：自迴歸影片生成的誤差累積，在頻域上表現為 DC 與低頻頻段的**頻譜能量漂移（Spectral Energy Drift）**，而非單純的時序退化。Attention Sink 雖能緩解此問題，但無法根本解決。為此，FreqForcing 提出 **Spectral Self-Anchoring (SSA)**：在推理時維護一個固定容量的 Anchor Cache（保存預訓練範圍內的高品質幀），並與標準滑動窗口注意力的輸出在頻域進行融合。融合公式為 $\hat{A}_{\mathrm{fused}} = \hat{A}_{\mathrm{loc}} + \lambda H_{\mathrm{lp}}(\hat{A}_{\mathrm{anc}} - \hat{A}_{\mathrm{loc}})$，其中時空高斯低通濾波器 $H_{\mathrm{lp}}$ 選擇性地從錨點注意力中提取低頻穩定成分注入局部注意力，同時保留高頻動態細節。在 VBench-Long 基準上，FreqForcing 在 60s 生成中 Dynamic Degree 達 **59.58**、Overall Consistency 達 **20.94**，在 120s 生成中同樣以 **58.97 / 20.98** 超越所有 Training-Free 方法（Infinity-RoPE、Deep Forcing），並能與需大量計算的 Training-based 方法（LongLive、Rolling Forcing）競爭。這種推理時頻域調製的思想對 JEPA 世界模型長期 Rollout 穩定化、Energy-based Transformer 的 Zero-Shot 控制等前沿方向均有深遠啟示。

---

**[SANA-Video 2.0 — 2026-07-31：SANA-Video 2.0: Hybrid Linear Attention with Attention Residuals for Efficient Video Generation，NVIDIA 提出混合線性-Softmax 注意力（75% 線性 + 25% Softmax 錨點，3:1 比例）搭配 Block Attention Residuals (AttnRes) 的影片擴散模型，5B 模型在單張 H100 上生成 720p/5s 僅需 13.06 秒，速度是 Wan 2.2-A14B 的 120 倍，VBench Total 84.30 / Quality 85.61，DiT 前向速度比全 Softmax 快 3.2×，AttnRes 提升深層有效秩 ~12%，Sol-Engine 全端最佳化再加速 3.58×，NVIDIA (arXiv:2607.21553)](papers/2026/2026-07/2026-07-31-SANA-Video-2.0.md)**

本文提出 **SANA-Video 2.0**（NVIDIA），一個在 5B 和 14B 參數規模下的混合注意力影片擴散模型。核心創新在於 **Hybrid Linear-Softmax Attention**：以 3:1 比例混合門控線性注意力（$O(N)$ 複雜度）與週期性 Softmax 錨點，在保持長序列擴展效率的同時恢復全秩 token 互動。**Block Attention Residuals (AttnRes)** 將每 8 層的完成特徵摘要跨深度路由至後續線性層，路由公式為 $h_l(x) = \sum_{v_i \in \mathcal{V}_l} \alpha^{(\tau)}_{i \to l}(x) v_i(x)$，有效提升深層有效秩約 12%。透過代理實驗確認 25% Softmax 為最佳品質-效率 Pareto 點，並從頭訓練（無需後線性化），搭配 Sol-Engine 全端最佳化（算子融合、快取、稀疏注意力），5B 模型在單張 H100 上生成 720p/5s 影片僅需 **13.06 秒**，是 Wan 2.2-A14B 的 **120 倍**，VBench Total **84.30**（品質分 **85.61**）與 14B 全 Softmax 模型相當，為消費級硬體上的高品質長影片生成提供了一條務實的技術路徑。

---

**[UniGen-AR — 2026-07-30：UniGen-AR: Unifying Visual Generation with Auto-Regressive Modeling，首個將 VAR（Visual Auto-Regressive）下一尺度預測擴展至完整統一視覺生成（UVG）的框架，MLLM（Qwen2.5-VL）+ VAR Decoder（Infinity）混合架構，Block-wise 因果遮罩統一參考 Token 與目標 Token 序列，支援超過 15 種任務（T2I、修復、感知、編輯），推理延遲最高降低 19×，NYUv2 深度 RMSE 0.245 / Rain100L PSNR 33.71，揭示 VQ-VAE Codebook 設計為 VAR 可擴展性關鍵，CMU & UIUC & Toyota Research Institute (arXiv:2607.24157)](papers/2026/2026-07/2026-07-30-UniGen-AR.md)**

本文提出 **UniGen-AR**（CMU、UIUC、Toyota Research Institute），是首個將視覺自迴歸（VAR）下一尺度預測範式與多模態語言模型（MLLM）條件化相結合，並擴展到完整統一視覺生成（UVG）設定的框架。現有擴散模型雖在 UVG 中佔主導地位，但迭代採樣帶來的巨大推理延遲限制了實際部署。UniGen-AR 以 Qwen2.5-VL 作為靈活的多模態編碼器，搭配 Infinity 風格的 VAR 解碼器，透過 Block-wise 因果遮罩將參考 Token 序列 $r^{\text{ref}}_K$ 作為非預測性上下文前綴，統一了超過 15 種任務的訓練。MLLM 的文字嵌入透過交叉注意力調製 VAR 解碼過程，訓練目標為 $\mathcal{L}_{\text{VAR}} = \sum_{k=1}^{K} \sum_{i=1}^{n_k} \text{CE}(p_{\theta}(\cdot|r_{<k}), r_k^{(i)})$。實驗顯示，UniGen-AR 在感知與修復任務上顯著超越先前 AR-based UVG 系統（Rain100L PSNR **33.71**，NYUv2 RMSE **0.245**），推理延遲相比擴散模型降低最高 **19 倍**，消融研究揭示 VQ-VAE Codebook 大小與層次結構是 VAR 可擴展性的關鍵瓶頸。

---

**[Twins — 2026-07-29：Twins: Learn to Predict Unified Representations with Focal Loss，打破視覺 Tokenization「不可能的三角」，Channel-wise 拼接 ViT 語義特徵與 VAE 細節特徵構建統一連續表示空間，揭示 DiT 聯合建模的三重優化不平衡（頻率偏置/內在維度/條件依賴），Focal Regression 自適應加權 VAE 困難維度提升 gFID 最高 10.57，重建 PSNR 31.46 / rFID 0.11 SOTA，理解 GQA 64.93 / MME-S 1971.0，Tencent Hunyuan Team (ICML 2026, arXiv:2607.22531)](papers/2026/2026-07/Twins/AI_Daily_Twins.md)**

本文提出 **Twins**（Tencent Hunyuan 團隊），發表於 **ICML 2026**。這是一個打破視覺 Tokenization「不可能的三角」的統一連續視覺表示框架。傳統連續表示方法在語義理解（ViT 特徵）與高保真生成（VAE 特徵）之間長期存在鴻溝，Twins 透過最簡潔的 **Channel-wise 拼接** $\mathbf{z} = [f_{vit}(I), f_{vae}(I)]$ 將兩者統一，且不增加 token 長度，不增加注意力的平方級計算成本。然而，作者發現在 Diffusion Transformer 聯合建模時出現嚴重的**優化不平衡**：模型偏好低頻、低內在維度、與條件高度對齊的 ViT 特徵，而難以學習高頻、高複雜度的 VAE 特徵。為此，論文將 Focal Loss 引入 Flow Matching，對誤差較大的 VAE 維度施加自適應加權 $w_i = |v_i - v_{\theta,i}|^{2\gamma}$，顯著改善優化平衡。實驗顯示，Twins 在重建（**PSNR 31.46 / rFID 0.11**）、理解（**GQA 64.93 / MME-S 1971.0**）與生成（**gFID 提升最高 10.57**）三個維度上全面超越前代統一表示方法，為下一代多模態基礎模型提供了一條優雅且高效的統一表示之路。

---

**[Text Template Tokens — 2026-07-28：Text Template Tokens Are Implicit Semantic Registers in Diffusion Transformers，首個針對 Chat-Templated T2I DiT 的因果可解釋性框架，揭示聊天模板結構化標記（如 `<|im_end|>`）作為主導 Attention Sinks 並隱式承載語義的雙重角色，透過跨軌跡頭移植（Head Transplantation）與層級因果遮罩（Causal Masking）證明「讀取 Prompt 的頭」在因果上是惰性的，語義流向為 $\mathcal{S}\rightarrow\mathcal{I}\rightarrow\mathcal{R}$，Training-Free Head Pruning 剪除 20% FLOPs 僅損失 1.4 分 GenEval，揭示 DiT 的 Early Commit / Middle Carry / Late Refine 三階段深度分工，Nanjing University & Alibaba Group (arXiv:2607.19139)](papers/2026/2026-07/TextTemplateTokens/AI_Daily_TextTemplateTokens.md)**

本文提出一個針對現代 Chat-Templated Text-to-Image Diffusion Transformers 的**因果可解釋性框架**（南京大學、阿里巴巴 Qwen 團隊、浙江大學），揭示了一個違反直覺的核心發現：聊天模板中的結構化標記（Structural Tokens，如 `<|im_end|>`）雖然在 VLM 編碼器輸出中幾乎不攜帶 Prompt 語義，卻在 DiT 內部成為**主導的 Attention Sinks** 並作為**隱式語義暫存器（Implicit Semantic Registers）**。在 GenEval 基準上，這 5 個模板標記吸收了 76%-78% 的影像到文字注意力質量（$\bar{m}_{\mathcal{R}}=0.187$），每個結構標記吸收的注意力是語義標記的 6.4-6.9 倍。透過**漸進式頭部移植（Progressive Head Swap）**實驗，作者證明「最讀語義標記的頭」在因果上是惰性的——僅替換約 18% 的「幾乎不讀語義標記的頭」就能翻轉物體身份，揭示語義的隱式傳遞路徑為 $\mathcal{S}\rightarrow\mathcal{I}\rightarrow\mathcal{R}$。基於此，作者提出 **Training-Free Head Pruning** 規則：剪除 360/1440 個頭（20% attention FLOPs），GenEval 僅從 76.1 降至 74.7（損失 1.4 points），而隨機剪枝則災難性地降至 51.3%。此外，論文揭示了 DiT 的深度分工：Early Layers（L1-10）負責 Commit 物體身份，Middle Layers（L11-50）負責 Carry，Late Layers（L51-60）負責 Refine，與 LLM 的 Stages of Inference 高度吻合。本文對 Attention Modulation、Training-Free 加速、Zero-Shot 影像編輯以及 Register Token 設計均有深遠啟示。

---

**[dRAE — 2026-07-24：dRAE: Representation Autoencoder with Hyper-Spherical Codes，首個從幾何視角解決高維語義特徵量化崩潰的離散 Tokenizer，HSQ（Hyper-Spherical Quantization）以 Angular Routing 解耦語義方向與特徵大小，詞表擴展至 131,072 仍保持 100% Codebook 利用率，rFID 0.42 / PSNR 24.52 大幅超越 VQRAE，T2I 生成 GenEval 0.63 僅用 12M 數據，UCAS & Zhejiang University & Peking University & Ant Group (arXiv:2607.22148)](papers/2026/2026-07/2026-07-27-dRAE.md)**

本文提出 **dRAE（discrete Representation Autoencoder）**（中科院大學、浙江大學、北京大學、螞蟻集團），一個解決高維視覺語義特徵離散化瓶頸的新型 Tokenizer。論文的核心洞見在於：現有 VQ 方法在高維 representation 空間出現 **Codebook Collapse** 的根本原因是 **Metric Mismatch**——視覺基礎模型（如 CLIP、DINO、JEPA）的語義主要編碼在特徵向量的**方向（Angle）**上，而傳統 VQ 依賴歐式距離，容易被 **Magnitude** 主導，導致 code 分配與語義無關。為此，論文提出 **HSQ（Hyper-Spherical Quantization）**，在 code 分配階段改用餘弦相似度 $I_i = argmax_j rac{Z_i cdot C_j}{|Z_i|_2|C_j|_2}$，Codebook Loss 改為球面目標 $mathcal{L}_{	ext{codebook}} = 1 - mathrm{sg}[Z/|Z|_2] cdot Z_q/|Z_q|_2$，但刻意保留 Euclidean Commitment Loss 以維持 Magnitude 資訊（對重建不可或缺）。這種「分配看角度，約束看距離」的混合設計，使 dRAE 在詞表擴展至 $131,072$ 時依然穩定增長，達到 **rFID 0.42**、**PSNR 24.52**，並在僅 12M 訓練數據下於 GenEval 達到 **0.63**。論文同時揭示了 JEPA/DINO 等模型特徵的 Thin-Shell 幾何結構，對後續 VAR、AR 生成、JEPA 世界模型設計均有深遠啟示。

---

**[Awakening Diffusion Transformers — 2026-07-26：EMA (Eliciting Massive Activation)，首個系統性分析 DiT 內部 Massive Activations 的 Training-Free 框架，DG 以 MA 抑制建構反事實分支提升細節生成（SD3 FID 9.52→5.77），MREP 以 AdaLN 調製提取更具辨識度的視覺表徵（ADE20K mIoU 54.8→56.1），Partial-Forward 設計使推理延遲低於 CFG（SD3.5 latency 15.7s→10.6s），SJTU & HKU & CityU (arXiv:2607.02968)](papers/2026/2026-07/2026-07-26-Awakening-Diffusion-Transformers.md)**

本文提出 **EMA（Eliciting Massive Activation）**（上海交通大學、香港大學、香港城市大學等），一個**完全免訓練（Training-Free）**的統一框架，透過對 DiT 內部「海量激活（Massive Activations, MAs）」的系統性機理分析，同時提升生成品質與視覺理解能力。研究發現 MAs 廣泛分佈於所有空間 Token，但高度集中在少數固定 Channel 維度，且與 AdaLN 的殘差縮放因子高度對齊，主要受去噪時間步調控。在生成端，**DG（Detail Guidance）**透過自適應抑制 MA 維度（$\hat{z}_{t,i}^k = \rho_t z_{t,i}^k$ for $i \in \mathcal{I}_{\mathrm{MA}}^k$）建構缺乏細節的反事實分支，並以其殘差作為細節引導方向，在 DiT-XL/2 上將 FID 從 9.52 大幅降至 **5.77**，IS 從 122.79 提升至 **179.26**；在理解端，**MREP** 利用預訓練 AdaLN 調製壓制 MA 方向主導性並保留空間響應圖，在 ADE20K 上將 mIoU 從 54.8 提升至 **56.1**，NYUv2 深度估計 RMSE 降至 **0.220**。此外，DG 的 Partial-Forward 設計使 SD3.5 推理延遲從 15.7s 降至 **10.6s**，效率優於 CFG。本文完美契合 training-free、activation modulation、zero-shot enhancement 與生成-理解統一等前沿研究方向。

---

**[SpheRoPE — 2026-07-24：SpheRoPE: Zero-Shot Optimization-Free 360° Panorama Generation with Spherical RoPE，首個完全免訓練且免優化的零樣本 360° 全景生成框架，Spherical RoPE 頻譜分離策略（低頻 3D 笛卡爾坐標 + 高頻諧波量化）直接修改推理時 RoPE 編碼，Semantic Distortion CFG 三向引導機制，FAED 25.40 超越所有微調基線，支援 FLUX.1/FLUX.2/LTX-Video 多主幹，Amazon Prime Video & Tel-Aviv University (arXiv:2606.32033)](papers/2026/2026-07/SpheRoPE/SpheRoPE.md)**

本文提出 **SpheRoPE**（Amazon Prime Video、Tel-Aviv University、Hebrew University of Jerusalem），一個**完全免訓練 (Training-Free)** 且**免優化 (Optimization-Free)** 的零樣本 360° 全景圖像與視頻生成框架。核心創新在於 **Spherical RoPE**：通過頻譜分離策略，將 RoPE 通道分為低頻（使用 3D 笛卡爾坐標 $X(r,c) = (\cos\theta\cos\phi+1)R$，$Y(r,c) = (\cos\theta\sin\phi+1)R$ 編碼球面流形，滿足水平週期性 C1 和極點收斂性 C2）和高頻（諧波量化 $\hat{\omega}_i = \text{round}(k_i)\cdot\omega_{\text{fund}}$ 保留局部紋理一致性）兩部分，在推理時直接修改預訓練 DiT 的位置編碼。此外，**Semantic Distortion CFG** 引入三向引導機制 $\hat{\epsilon} = \epsilon_{\text{uncond}} + w_{\text{sem}}(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}}) + \gamma(\epsilon_{\text{geo}} - \epsilon_{\text{cond}})$，利用幾何錨定提示詞放大 ERP 畸變先驗。在 ODI-SR 基準上，SpheRoPE (FLUX.2) 以 **FAED 25.40** 超越所有微調基線（PAR 34.79, SMGD 33.55），視頻生成在 LTX 2.3 上以 1.11 秒/幀的速度全面領先基於優化的方法（DynamicScaler 51.56 秒/幀），用戶偏好研究中 88.5%-95.2% 的評估者更偏好 SpheRoPE。這是首個能在不修改任何模型權重的情況下，通過純推理時 Attention/RoPE Modulation 實現球面拓撲約束的框架，完美契合 training-free、zero-shot、attention modulation 研究方向。

---

**[Appearance Pointers — 2026-07-23：Appearance Pointers: Multimodal Region Control of Diffusion Transformers，首個模態無關的 DiT 區域控制介面，無需重新訓練基礎模型，單次去噪過程同步處理多個文本/圖像區域條件，Region Correspondence + Aggregation Transformer 將注意力複雜度從 O(T·(RN)²) 降至 O(R(N/k)²)，AppearancePointers-37K 資料集，Brown University & Adobe Research (arXiv:2607.19344)](papers/2026/2026-07/AppearancePointers/AI_Daily_AppearancePointers.md)**

本文提出 **Appearance Pointers**（Brown University & Adobe Research），一種輕量級、**模態無關（Modality-Agnostic）**的 DiT 區域控制機制，**無需重新訓練基礎模型**即可實現精確的多模態區域感知圖像生成。核心創新在於引入「外觀指標（Appearance Pointers）」——一種緊湊的 Token 表示法，不直接攜帶外觀資訊，而是作為路由指令，告訴 DiT 在正確的空間位置尋找對應的文本或圖像特徵。框架由兩個輕量模組組成：**Region Correspondence Transformer（$\Phi_{RC}$）**負責將 Mask、圖像 Token 和文本 Token 三者對齊，產生針對 DiT 雙流（image stream / text stream）的語義特徵圖；**Region Aggregation Transformer（$\Phi_A$）**則透過 region-wise self-attention 將多區域特徵壓縮為單一畫布大小的指標 Token，將注意力計算複雜度從 $\mathcal{O}(T\cdot(RN_{\text{reg}})^2)$ 降至 $\mathcal{O}(R(N_{\text{reg}}/k)^2)$，且指標只需在整個推理流程前計算一次。在 AppearancePointers-37K 基準上，圖像條件區域生成達到 CLIP-I **93.29**、DINO-I **69.31**、CLIP-IQA **95.57**，全面超越 MS-Diffusion 和 DreamRenderer*；文本條件區域生成在 CLIP-I（90.40）、DINO-I（56.09）、CLIP-IQA（95.02）均為最佳。這是首個能在**單次去噪過程**中同時處理多個文本和圖像區域條件的統一框架，完美契合 attention modulation、training-free 控制與 DiT 可控生成等研究方向。

---

**[Mage-Flow — 2026-07-22：Mage-Flow: An Efficient Native-Resolution Foundation Model for Image Generation and Editing，緊湊 4B 規模生成堆棧，系統級共設計實現高效原生解析度圖像生成與編輯，輕量級 Tokenizer 減少編碼解碼成本 12-22 倍，堆棧級 CUDA 核心融合實現 2.5 倍訓練加速，1024² 解析度 0.59 秒生成，Microsoft (arXiv:2607.19064)](papers/2026/2026-07/Mage-Flow/AI_Daily_Mage_Flow.md)**

本文提出 **Mage-Flow**（Microsoft Mage Team），一個精心設計的 4B 規模生成堆棧，針對高效文本到圖像生成和指令式圖像編輯進行了系統級優化。核心創新包括：（1）**Mage-VAE** 輕量級 Tokenizer，通過一步擴散式編碼解碼與錨點潛在正則化，在保持重建質量的同時將編碼和解碼成本分別降低 **12 倍**和 **22 倍**；（2）**原生解析度多模態擴散 Transformer（NR-MMDiT）**，採用原生解析度打包方案支持靈活的解析度和寬高比（512-2048 像素），通過變長序列打包和堆棧級 CUDA 核心融合實現 **2.5 倍**訓練加速；（3）**統一生成-編輯框架**，基於共享 Mage-VAE 潛在空間和 NR-MMDiT 骨幹，開發了完整的模型族系（Base、RL 對齊、Turbo 變體）。在 1024² 解析度下，Mage-Flow-Turbo 可在單個 A100 GPU 上 **0.59 秒**生成圖像，Mage-Flow-Edit-Turbo 可在 **1.02 秒**編輯圖像，峰值 GPU 內存僅 18-20 GB。該工作展示了系統級共設計如何在緊湊規模下實現與 6B-80B 大型系統相競爭的性能，對資源受限的研究和部署場景具有重要啟示。

---

**[MrFlow — 2026-07-21：Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling，訓練無關多解析度流匹配加速，10 倍端到端加速保持 1% 質量損失，FLUX.1-dev 和 Qwen-Image 上 SOTA，可與時間步蒸餾組合達 25 倍加速，北京航空航天大學 & 瑞士聯邦理工學院 (arXiv:2607.01642)](papers/2026/2026-07/MrFlow/AI_Daily_MrFlow.md)**

本文提出 **MrFlow** (Multi-Resolution Flow Matching)，一種**訓練無關（Training-Free）**的多解析度加速策略，專門針對現代流匹配（Flow Matching）擴散模型的推理加速。MrFlow 通過精妙的四階段管道設計——低解析度快速生成全局結構、像素空間超解析度、低強度噪聲注入、高解析度單步精化——實現了 **10 倍端到端加速**，同時保持生成質量在 **1% 以內的損失**。核心創新包括：（1）在像素空間進行超解析度而非潛在空間，充分利用自然圖像先驗；（2）基於信噪比條件推導的低強度噪聲注入理論基礎；（3）利用流匹配框架特性實現單步高解析度精化。在 FLUX.1-dev 上 8.25 倍加速下 GenEval 保持 0.63（原生 0.66），在 Qwen-Image 上 10.3 倍加速下 GenEval 保持 0.86（原生 0.88）。MrFlow 可直接與時間步蒸餾方法組合，在 Qwen-Image 上達到 **25.1 倍加速**。這項工作完美符合用戶的研究興趣：訓練無關方法、Flow Matching 方向、Zero-Shot 應用。

---

**[VPG — 2026-07-20：VPG: Visual Prefix Guidance for Autoregressive Image and Video Generation，訓練無關推理時引導方法，前綴後驗支撐改進 VAR FID 0.36、InfinityStar VBench 0.49，NUS & HUST (arXiv:2605.30317)](papers/2026/2026-07/VPG/AI_Daily_VPG.md)**

本文提出 **VPG (Visual Prefix Guidance)**（新加坡國立大學 & 華中科技大學），一種**訓練無關（Training-Free）**的推理時引導方法，用於改進視覺自迴歸模型的生成質量。VPG 從**前綴後驗支撐**的新視角解決自迴歸模型的暴露偏差問題。在訓練時，自迴歸模型採用教師強制策略，每個預測步驟基於真實前綴；但推理時必須基於自生成前綴，導致訓練-推理不匹配。VPG 的核心創新在於提出了一個新的引導軸：不是強化外部條件（如 CFG），而是確保下一步預測對已生成前綴提供強後驗支撐。通過配對預測對比（真實前綴 vs. 腐蝕前綴）和同尺度全嵌入替換，VPG 實現了簡潔而有效的前綴後驗目標。在 VAR 上 FID 平均降低 **0.36**（最高 0.63），在 Infinity 上改進文本對齐指標，在 InfinityStar 上 VBench 分數提升 **0.49**。VPG 完全免訓練、即插即用，與 CFG 沿不同軸工作，可組合使用。這項工作完美符合用戶的多個研究興趣方向：訓練無關方法、注意力調製、VAR 模型改進、零樣本應用。

---

**[SparVAR — 2026-07-20：SparVAR: Exploring Sparsity in Visual AutoRegressive Modeling for Training-Free Acceleration，利用注意力稀疏性實現訓練無關加速，1.57× 加速保持高頻細節，8B 模型 1024×1024 圖像生成降至 1 秒，CVPR 2026 (Chinese Academy of Sciences & City University of Hong Kong)](papers/2026/2026-07/SparVAR/AI_Daily_SparVAR.md)**

本文提出 **SparVAR**（中科院自動化所、北京人工智能研究院、南京理工大學、香港城市大學），發表於 **CVPR 2026**。這是一項針對視覺自迴歸（VAR）模型推理加速的突破性工作，通過系統分析 VAR 注意力激活模式，揭示了三個關鍵稀疏性性質：**強注意力 Sink**（早期尺度 Token 持續吸引高注意力權重，充當全局錨點）、**跨尺度激活相似性**（相鄰尺度的注意力模式高度相似，可跨尺度轉移）、**明顯的空間局部性**（高解析度尺度的注意力集中在局部空間帶狀區域）。基於這些發現，SparVAR 設計了兩個協同模組：**跨尺度自相似稀疏注意力（$CS^4A$）**通過高效索引映射動態預測後續尺度的稀疏注意力模式，**跨尺度局部稀疏注意力（CSLA）**實現了塊級稀疏核，前向計算速度比 FlashAttention 快 5 倍以上。SparVAR 無需任何重新訓練，即插即用，在 8B 模型上實現 **1.57× 加速**（相比 FlashAttention 基線），生成 1024×1024 圖像時間降至 **1 秒**，同時幾乎完全保留高頻細節（PSNR、SSIM、LPIPS 與基線接近），相比尺度跳過方法（FastVAR、SkipVAR）有本質優勢。

---
**[HAM — 2026-06-13：HAM: A Training-Free Style Transfer via Heterogeneous Attention Modulation for Diffusion Models，異質注意力調變免訓練風格轉換，CLIP-T 0.223 SOTA，LPIPS 0.479 全面領先，CVPR 2026 Findings (Hangzhou Dianzi University & ICT CAS)](papers/2026/2026-06/HAM/AI_Daily_HAM.md)**

本文提出 **HAM (Heterogeneous Attention Modulation)**（杭州電子科技大學 & 中科院計算所），發表於 **CVPR 2026 Findings**。這是一種針對擴散模型的**免訓練（Training-Free）**風格轉換方法，透過三個協同模組解決「風格-內容平衡」的核心難題：**全局注意力調節（GAR）**在自注意力層以 AdaIN 統計對齊融合內容與風格特徵；**局部注意力移植（LAT）**在交叉注意力層移植風格的 Key/Value，並以加權融合保護內容 Query；**注入風格的噪聲初始化（SINI）**在初始時間步 $T$ 以 AdaIN 融合內容與風格初始噪聲。HAM 的關鍵洞見在於：自注意力偏向空間結構，交叉注意力偏向語義注入，兩者需要「異質」的差異化操作，而非統一的替換策略。在 MS-COCO + WikiArt 基準上，HAM 在 LPIPS（0.479）和 LPIPS-Gray（0.362）指標上大幅領先所有對手，CLIP-T（0.223）達到最佳，ArtFID（15.151）、DC（2.113）、CC（2.057）全面超越 StyleID、DiffArtist 等 SOTA，實現了「強風格遷移」與「精確內容保留」的雙贏。

---
**[IDEAL — 2026-06-12：In-DEpth ALignment Makes A Discrete Representation AutoEncoder，深度對齊打造新一代離散表示自編碼器，AR 圖像生成新 SOTA gFID 1.89，Zero-Shot 語義保留 80.89% Top-1，同時刷新重建 rFID 0.61 (Fudan University & UMD)](papers/2026/2026-06/IDEAL/AI_Daily_IDEAL.md)**

本文提出 **IDEAL (In-DEpth ALignment)**（復旦大學、上海創新研究院 & 馬里蘭大學），一個創新的離散表示自編碼器框架。現有的 VFM-based tokenizer（如 VFMTok）僅使用深層 VFM 特徵進行量化，雖然語義豐富，但深層特徵缺乏低階空間細節，導致離散化後重建品質不佳。IDEAL 的核心洞見在於：**VFM 的淺層特徵保留了豐富的外觀細節，深層特徵攜帶高階語義，兩者在深度上具有天然的互補性。** 通過一個輕量級交叉注意力模組 $z = \text{AttnFuse}(f^{(d)}, f^{(s)})$，IDEAL 將淺層特徵（Block 8）和深層特徵（Block 24）融合為統一表示再進行向量量化。在解碼端，引入雙重對齊損失 $\mathcal{L}_{\text{deep}}$ 和 $\mathcal{L}_{\text{shallow}}$，強制重建特徵同時恢復語義結構和空間細節。在 ImageNet 上，IDEAL 達到 **rFID 0.61**（比前最佳好 0.28），Zero-Shot 分類 **Top-1 80.89%**（接近原始 SigLIP2 的 83.23%），且 AR 生成在 3B 參數下達到 **gFID 1.89**，創下自迴歸圖像生成的新 SOTA。

---
**[OmniGen-AR — 2026-06-11：OmniGen-AR: AutoRegressive Any-to-Image Generation，統一自迴歸 Any-to-Image 框架，Disentangled Causal Attention (DCA) 解決多模態條件生成的資訊洩漏問題，1.5B 模型 GenEval 0.63、VBench 80.02（首個離散 Token AR 模型突破 80 分），Training-Free Inference (Fudan University & ByteDance Seed, NeurIPS 2026)](papers/2026/2026-06/OmniGen-AR/OmniGen-AR.md)**

本文提出 **OmniGen-AR**（復旦大學 & 字節跳動 Seed），發表於 **NeurIPS 2026**。這是一個統一的自迴歸 Any-to-Image 生成框架，透過共享 Visual Tokenizer 將深度圖、語意分割圖、參考影像等多種視覺條件統一離散化為 Token，使單一模型同時支援 Text-to-Image、Text-to-Video、Image Editing、Depth-to-Image 與 Seg-to-Image 等五種以上任務。核心創新 **Disentangled Causal Attention (DCA)** 將全序列因果遮罩拆分為條件因果注意力與內容因果注意力：在訓練時以 10% 機率套用，阻斷內容 Token 對條件 Token 的直接依賴，防止 Shortcut Learning；推理時完全退化為標準 Next-Token Prediction，實現 Training-Free Inference。DCA 遮罩的數學設計為：當 Query 屬於內容區間 $C=[M+N_1, M+N_1+N_2)$、Key 屬於條件區間 $B=[M, M+N_1)$ 時，強制注意力權重為 $-\infty$。1.5B 模型在 GenEval 達 **0.63**，在 VBench 達 **80.02**，作者指出這是首次基於離散 Token 的純自迴歸模型突破 VBench 80 分大關。

---
**[HACK++ — 2026-06-10：HACK++: Towards More Effective Head-Aware Key-Value Compression for Efficient Visual Autoregressive Modeling，揭示 VAR 注意力頭的語義/結構二元性，Training-Free 解耦壓縮框架，Infinity-8B 內存節省 2.04× 吞吐提升 1.52×，KV Cache 可壓縮至 1% 仍保持近無損生成 (SJTU & Tsinghua)](papers/2026/2026-06/2026-06-10-HACK-plus-plus.md)**

本文提出 **HACK++ (Head-Aware Key-Value Compression)**（上海交通大學 & 清華大學），是一項針對 Visual Autoregressive (VAR) 模型的 **Training-Free KV Cache 壓縮**框架。作者首先深入分析了 VAR 模型中注意力頭的行為，發現其可穩定地分為兩類：**Contextual Heads**（語境頭，呈垂直條紋注意力模式，負責語義一致性）與 **Structural Heads**（結構頭，呈多對角線注意力模式，負責空間連貫性）。這種二元性使得「一刀切」的壓縮策略在 VAR 上必然失效。HACK++ 的核心創新在於：**解耦注意力計算與 KV Cache 壓縮**，分別使用獨立的預算 $B_a$（注意力）和 $B_c$（Cache），允許 Cache 被更激進地壓縮；並為兩類頭設計了特定模式的重要性估計策略（語境頭用 query-subset attention，結構頭用離線 scale-prior × value norm）；同時引入依賴感知的自適應預算分配，動態調整不同頭、層、生成步驟的 cache 預算。在 Infinity-8B 上，HACK++ 在 30% 注意力預算、10% Cache 預算下實現 **2.04× 內存節省**和 **1.52× 吞吐提升**，且在極限 1% Cache 預算下仍保持近無損的生成質量。

---
**[DAVE — 2026-06-09：Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation，透過表示調製打破早期 DC 鎖定，實現 Training-Free 的圖像生成多樣性增強 (ICML 2026)](papers/2026/2026-06/2026-06-09-DAVE.md)**

本文提出 **DAVE (DC Attenuation for diVersity Enhancement)**，發表於 **ICML 2026**。這是一項針對文本到圖像（Text-to-Image）生成模型多樣性不足問題的突破性免訓練（Training-Free）解決方案。作者深入分析了 Transformer 內部特徵，發現在生成早期，不同隨機種子下的零頻率空間平均值（即 DC 分量）會迅速收斂，導致「早期 DC 鎖定（Early DC Lock-in）」現象，這限制了後續生成過程中的結構變化。

DAVE 的核心創新在於：**在早期生成階段選擇性地衰減 DC 分量**。通過一個極其簡單的空間平均和縮放操作，DAVE 成功打破了這種過早的結構承諾，放大了特定種子的空間殘差的相對影響力。在 Stable Diffusion 3.5、FLUX.1-dev 等模型上的實驗表明，DAVE 在幾乎不增加計算開銷的情況下，顯著提升了生成圖像的多樣性（如 Recall、Coverage 和 Vendi Score），同時完美保持了與提示詞的一致性和極具競爭力的圖像質量。這項研究證明了理解模型內部動態並進行表示調製（Representation Modulation）往往比設計複雜的外部約束更為有效。

---
**[SeaCache — 2026-06-08：SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models 頻譜演化感知快取，透過頻域濾波器精準捕捉擴散模型內容冗餘，實現無需訓練的即插即用推理加速 (Sungkyunkwan University & NAVER Cloud, CVPR 2026 Oral)](papers/2026/2026-06/2026-06-08-SeaCache.md)**

本文提出 **SeaCache (Spectral-Evolution-Aware Cache)**（成均館大學 & NAVER Cloud），發表於 **CVPR 2026 (Oral)**。這是一項針對擴散模型 (Diffusion Models) 推理加速的突破性 Training-Free 動態快取策略。傳統的快取加速方法（如 TeaCache、DiCache）通常直接在原始特徵空間中測量相鄰時間步的距離來決定是否重用特徵。然而，這種設計忽略了擴散模型的「頻譜演化 (Spectral Evolution)」先驗：早期時間步主要生成低頻結構，後期則專注於高頻細節，直接測量原始特徵距離會被高頻隨機噪聲嚴重干擾。

SeaCache 的核心創新在於：**引入了頻譜演化感知 (SEA) 濾波器，將特徵轉換到更適合評估內容變化的空間。** 基於最優線性去噪器的理論推導，作者設計了一個隨時間步變化的頻域濾波器 $G_t(f)$，該濾波器能放大與內容相關的低頻信號，同時抑制由隨機變化主導的高頻噪聲。在經過密度歸一化後，SeaCache 透過 FFT/iFFT 將這個濾波器應用於輸入特徵，並在濾波後的特徵空間中測量 $\ell_1$ 距離。這種設計使得快取決策更聚焦於生成內容的實質變化，而非隨機噪聲的擾動。實驗表明，在 FLUX.1-dev、HunyuanVideo 和 Wan2.1 等頂級視覺生成模型上，SeaCache 展現了最先進的 Latency-Quality Trade-off。在 FLUX.1-dev 的激進加速設定（約 30% 刷新率）下，SeaCache 不僅延遲最低，且在 PSNR、LPIPS、SSIM 以及 CycleReward 人類偏好指標上均全面超越 TeaCache 和 TaylorSeer，完美保留了原始模型生成的語義內容和感知質量。

---
**[DDT — 2026-06-07：DDT: Decoupled Diffusion Transformer 解耦語義編碼器與速度解碼器，打破 Diffusion Transformer 優化困境，ImageNet 256×256 FID=1.31（4× 訓練加速），ImageNet 512×512 FID=1.28 SOTA，統計動態規劃推理加速 3× (Nanjing University & ByteDance Seed Vision, CVPR 2026)](papers/2026/2026-06/2026-06-07-DDT-Decoupled-Diffusion-Transformer.md)**

本文提出 **DDT (Decoupled Diffusion Transformer)**（南京大學 & ByteDance Seed Vision），發表於 **CVPR 2026**。DDT 針對傳統 Diffusion Transformer（如 DiT、SiT）中「語義編碼」與「細節解碼」在同一模塊中相互競爭的**優化困境**，提出了一個優雅的解耦架構：專用的 **Condition Encoder** 負責從帶噪聲輸入中提取低頻語義自條件特徵 $\boldsymbol{z}_t$，而 **Velocity Decoder** 則接收 $\boldsymbol{z}_t$ 作為引導，專注於高頻細節的速度場預測。Encoder 採用 REPA 表示對齊損失 $\mathcal{L}_{enc} = 1 - \cos(r_*, h_\phi(\mathbf{h}_i))$ 與 DINOv2 特徵對齊，不僅加速收斂，還賦予了相鄰時間步自條件特徵的高度局部一致性（餘弦相似度 > 0.8）。基於此，作者提出**統計動態規劃（Statistic DP）**，將尋找最優 Encoder 共享策略轉化為最小和路徑問題，實現 3× 推理加速且幾乎無質量損失。消融實驗揭示了「重 Encoder、輕 Decoder」的非對稱設計（如 22En6De）隨模型規模增大效果越顯著。DDT-XL/2 在 ImageNet 256×256 上僅需 **256 Epoch** 即達 **FID=1.31**（REPA 需 800 Epoch），在 ImageNet 512×512 上達 **FID=1.28** 的全新 SOTA。

---

**[SSG — 2026-06-06：Guiding a Diffusion Model by Swapping Its Tokens Training-Free Token 交換引導，無需條件即可實現 CFG 級別的圖像品質提升，SDXL FID 從 119.04 降至 70.91，CVPR 2026 Oral (SJTU & vivo)](papers/2026/2026-06/SSG_Diffusion/AI_Daily_SSG.md)**

本文提出 **SSG (Self-Swap Guidance)**（上海交通大學 MoE AI 重點實驗室 & vivo），發表於 **CVPR 2026 (Oral)**。這是一種極簡卻高效的 **Training-Free、Condition-Free** 擴散模型引導方法，通過在推理階段選擇性地交換語義最不相似的 Token Latents，構建弱化預測分支作為負面參考，從而引導採樣走向更高保真度的分佈。不同於 SAG、PAG、SEG 等在全局層面進行粗粒度擾動的方法，SSG 在 Token 粒度上精準操作，同時支援空間維度（破壞結構一致性）與通道維度（擾動紋理細節）的對抗性交換。引導公式為 $\tilde{\epsilon}(x_t) = \epsilon_{\text{ori}}(x_t) + \omega(\epsilon_{\text{ori}}(x_t) - \epsilon_{\text{pert}}(x_t))$，可作為即插即用模組插入任何現有擴散模型，並與 CFG 疊加使用。在 SDXL 無條件生成（MS-COCO 2014）中，SSG 將 FID 從 119.04 大幅降至 **70.91**，IS 從 9.08 提升至 **16.44**；條件生成 FID 達 **21.73**，ImageReward 達 **+0.276**，全面超越 SAG/PAG/SEG。

---

**[Internal Guidance — 2026-06-05：Guiding a Diffusion Transformer with the Internal Dynamics of Itself 零額外採樣成本的內部引導，中間層多尺度輔助監督，刷新 ImageNet 256x256 生成 SOTA FID = 1.19 (UESTC & NUS & SYSU)](papers/2026/2026-06/2026-06-05-Internal-Guidance.md)**

本文提出 **Internal Guidance (IG，內部引導)**（電子科技大學、新加坡國立大學、中山大學、華北計算技術研究所），發表於 **CVPR 2026 (Highlight)**。這是一項針對 Diffusion Transformer (DiT) 的突破性引導採樣與訓練加速技術。

傳統的 Classifier-Free Guidance (CFG) 雖然能有效引導生成路徑，但過高的引導係數容易導致圖像過度飽和、多樣性崩塌或邊緣失真。而替代方案 Autoguidance (bad version guidance) 則需要精心設計退化策略，甚至需要額外訓練和運行一個較弱的模型，導致雙倍的模型前向傳播開銷。

Internal Guidance 的核心創新在於：**「利用模型自己尚未發育完全的中間層輸出，來引導最終深層輸出的方向」**。在訓練階段，團隊在 DiT 的中間層（例如第 8 層）後方添加一個額外的輸出投影層，並對中間與最終預測值同時施加去噪監督損失（$\mathcal{L} = \mathcal{L}_{\text{final}} + \lambda \mathcal{L}_{\text{inter}}$）。這僅需極其簡單的輔助監督，即可達到甚至超越複雜自監督表徵學習（如 REPA）的加速收斂效果，顯著緩解深層網絡梯度消失。

在採樣階段，IG 僅需一次前向傳播，即可同時獲得中間層的「較弱預測」 $D_i$ 與最終層的「較強預測」 $D_f$。通過外推公式 $D_w = D_i + w(D_f - D_i)$，採樣方向會沿着「從較弱的中間表示指向更成熟的深層表示」的方向進行外推，從而以**零額外採樣成本**消除了低概率分佈中的異常噪聲。在 ImageNet 256x256 任務上，結合 IG 的 `SiT-XL/2` 僅訓練 80 個 epoch 即可達到 **FID = 5.31**；在 `LightningDiT-XL/1` 上，IG 實現了 **FID = 1.34** 的驚人表現。當進一步與 CFG 和引導區間（Guidance Interval）結合時，更是達到了 **SOTA FID = 1.19**，在不損失多樣性的前提下，完美保留圖像細節。

---

**[Markov-VAR — 2026-06-04：Markovian Scale Prediction 視覺自迴歸生成的馬可夫新紀元，滑動窗口歷史補償機制，1024×1024解析度峰值記憶體暴降 83.8% (Tongji University & University of Bristol)](papers/2026/2026-06/2026-06-04-Markov-VAR.md)**
本文提出 **Markov-VAR (Markovian Scale Prediction)**（同濟大學、布里斯托大學、麥考瑞大學），發表於 **CVPR 2026**。這是一項徹底解決視覺自迴歸模型（VAR）「全上下文依賴（Full-Context Dependency）」瓶頸的突破性工作。傳統 VAR 在預測當前尺度時，注意力必須覆蓋所有歷史尺度，導致 Token 序列超線性膨脹，在生成高解析度（如 1024×1024）圖像時，GPU 記憶體開銷呈災難性增長。
Markov-VAR 的核心創新在於：**將視覺自迴歸重新表述為一個非全上下文的馬可夫過程。** 為了補償因丟棄早期尺度原始資訊而造成的損失，團隊提出了一個極其優雅且輕量的**滑動窗口歷史補償機制**。該機制將最近 $N$ 個尺度的特徵放入滑動窗口，利用交叉注意力與可學習的全局查詢向量（Learnable Query）將其壓縮為一個固定維度的歷史補償向量。該向量作為歷史資訊的「充分統計量（Sufficient Statistic）」，與當前尺度特徵在通道維度進行拼接，構建出動態馬可夫狀態。在推理生成時，Markov-VAR **完全不需要維護龐大的 KV Cache**。實驗表明，在 ImageNet 1024×1024 解析度下，Markov-VAR-d24 將峰值 GPU 記憶體消耗從 117.9GB 驚人地降低至 **19.1GB**（降幅高達 **83.8%**），同時在 256×256 解析度下將 FID 降低了 **10.5%**（從 3.61 降至 **3.23**），真正實現了「高生成品質」與「極致計算效率」的雙贏。

---

**[CVQ — 2026-06-03：Channel-wise Vector Quantization 通道維度向量量化，100% 代碼本利用率，解鎖漸進式「下一通道預測」自迴歸圖像生成新範式 (Westlake University & ZJU)](papers/2026/2026-06/2026-06-03-CVQ.md)**
本文提出 **CVQ (Channel-wise Vector Quantization)**（西湖大學與浙江大學），這是一個顛覆性的圖像代碼化與自迴歸生成範式。傳統的 VQGAN 方案自提出以來一直採用空間補丁（Patch-wise）量化，但這種二維網格表徵面臨著嚴重的**代碼本崩潰（Codebook Collapse）**問題，且與一維自迴歸模型的單向預測模式存在結構失配。
CVQ 的核心創新在於：**將量化軸從空間補丁轉向特徵通道。** 通過將圖像表徵為一維特徵通道的序列，CVQ 巧妙利用了通道特徵天然的「語義-細節解耦」與高可分性，無需任何額外的複雜技巧（如低維投影或輔助損失），即可在 16K 甚至 65K 的大規模代碼本下實現 **100% 的代碼本利用率**。在此基礎上，團隊提出 **CAR (Channel-wise Auto-Regressive)** 模型，通過**嵌套通道丟棄（Nested Channel Dropout）**訓練，CAR 實現了「先勾勒全局輪廓，再逐步填充細節」的漸進式生成。在 ImageNet-1K 重建中，CVQ-1024 取得了 **0.88** 的極低 rFID；在 Text-to-Image 生成中，CAR-8B 達到了 **0.79** 的 GenEval 分數與 **86.72** 的 DPG 分數，與最強的混合多尺度自迴歸模型並駕齊驅，顯著超越了採用傳統補丁預測的 Emu3-8B，證明了一維通道序列比強行展平的二維空間序列更適合自迴歸預測學習。
---

**[AttnRouter — 2026-06-02：MMDiT 時代的 Training-Free 圖像編輯新範式，KVInject 單次前向 α-混合 + 基於類別的注意力路由，在 Qwen-Image-Edit-2511 上定位 L30–45/S0–7 編輯子電路 (iFLYTEK)](papers/2026/2026-05/AttnRouter/README.md)**

本文提出 **AttnRouter**（iFLYTEK），一個針對多模態擴散 Transformer（MMDiT）架構的**訓練免除（Training-Free）**圖像編輯框架，包含兩大核心貢獻：**KVInject** 與 **AttnRouter** 路由機制。核心洞見在於：UNet 時代的 MasaCtrl 在 MMDiT 上直接移植會導致 Composite Score 崩塌 31%，根本原因是 MMDiT 中來源圖像與噪聲流共用同一個聯合注意力（Joint Attention）通道，傳統的兩次前向傳播（Two-Forward）策略所記錄的 K/V 缺乏編輯語義。

KVInject 通過**同一次前向傳播**中對來源半部（Source-Half）的 K/V 與噪聲半部（Noise-Half）的 K/V 進行 $\alpha$-混合（$K_{\text{noise}}^{\prime} = \alpha \cdot K_{\text{src}} + (1-\alpha) \cdot K_{\text{noise}}$），在零參數、$<2\%$ 計算開銷的條件下實現結構保留。通過系統性的消融分析，作者成功定位了 Qwen-Image-Edit-2511（60 層 MMDiT）中的**編輯有效子電路**：層區間 **L30–45**、去噪步區間 **S0–7**（僅前 7 步即可恢復 99% 的增益），以及穩定的混合強度甜點 **$\alpha \in [0.3, 0.5]$**。AttnRouter 通過 CLIP Zero-Shot 分類器自動識別編輯類別（Replace/Attribute/Background → $\alpha=0.3$；Remove/Style → $\alpha=0.5$；Add → Baseline），並路由到對應的 KVInject 配置。儘管分類器準確率僅 55%，Auto Router（Composite 0.4113）仍能閉合 98% 的 Oracle 差距（0.4127），因為容易混淆的類別天然共享相同路由。在 ImgEdit-Bench-100 上，AttnRouter 相比 Baseline 提升 Composite Score **+6.3%**（0.3879 → 0.4127），DINO-I 提升 **+8.5%**，同時完全規避了 MasaCtrl 的 Prompt-Mismatch 失效模式。

---

**[VPG — 2026-05-28：Visual Prefix Guidance 視覺前綴引導，免訓練對抗自回歸曝光偏差，解鎖超強 compositional 生成 (NUS & HUST)](papers/2026/2026-05/VPG/AI_Daily_VPG.md)**

本文提出 **VPG (Visual Prefix Guidance)**（新加坡國立大學 NUS 與 華中科技大學 HUST），一個專為視覺自回歸模型（VAR、Infinity、InfinityStar）設計的**免訓練（Training-Free）**、**即插即用（Drop-In）**的推理端引導採樣規則。核心洞見在於：自回歸模型在訓練時使用 Teacher Forcing（以 ground-truth 歷史為條件），但在推理時必須以自己生成的歷史（即 Prefix）為條件，這種訓練與推理的不匹配導致嚴重的 **Exposure Bias（曝光偏差）** 與 **Prefix Drift（前綴漂移）**。先前的 CFG 只針對外部語意條件，而忽視了歷史前綴本身的累積誤差。

VPG 通過引入在推理時構建的**損毀前綴（Corrupted Prefix）**作為對照分支，並在 Logit 空間中進行外推：$\ell_k^{\text{VPG}} = (1+\lambda)\ell_k^{\text{gen}} - \lambda \ell_k^{\text{corr}}$，強制模型在每一步預測時優先選擇能增強已生成前綴後驗支持的候選 token。損毀前綴採用**同尺度全嵌入替換（Same-Scale Full-Embedding Replacement）**構造，在不引入 OOD 噪聲的前提下打破語意-位置綁定。在 ImageNet $256\times256$ 上，VPG 平均降低 **0.36** FID（VAR-d24+VPG 以 FID 1.83 超越 VAR-d30 基準 1.94）；在 InfinityStar 視頻模型上，多物體生成（+2.13）與語意對齊（+0.77）取得 VBench **SOTA**，整體得分提升 **+0.49**。

---

**[LeJEPA World Model — 2026-05-31：首個 JEPA 線性可識別性數學保證，Yann LeCun & Balestriero 嚴格證明高斯分佈是唯一能讓 LeJEPA 學習世界模型的潛在分佈，潛空間規劃等價性定理 (NYU & Brown University & Cold Spring Harbor Lab)](papers/2026/2026-05/LeJEPA/AI_Daily_LeJEPA.md)**

本文由 **David Klindt（Cold Spring Harbor Laboratory）、Yann LeCun（NYU & Meta FAIR）、Randall Balestriero（Brown University）** 共同提出，是自監督學習與世界模型領域的一項**里程碑式理論工作**。本文首次為 JEPA 提供了**線性可識別性（Linear Identifiability）**的嚴格數學保證：在具有高斯潛變量與平穩加性噪聲轉移（Ornstein-Uhlenbeck 過程）的世界中，LeJEPA（對齊損失 + SIGReg 高斯正則化）能夠**線性恢復**世界的潛在變量（僅存在正交旋轉二義性）。

核心理論突破在於利用 **Hermite 多項式的譜分解**：任何 $d$ 階 Hermite 多項式的轉移算子特徵值為 $\rho^d$，因此任何非線性畸變都會嚴格降低正樣本對的互相關（$\mathbb{E}[h_i(z')h_i(z)] \leq \rho$），等號成立若且唯若 $h_i$ 為純線性函數。這意味著 LeJEPA 的對齊損失會無情地懲罰所有非線性分量，逼迫表徵收斂到最優的線性解 $h(z) = Qz$（$Q$ 為正交矩陣）。**逆向定理**進一步利用 Sturm-Liouville 理論證明，在平穩加性噪聲轉移的物理世界中，**高斯分佈是唯一**能讓 LeJEPA 達成線性可識別性的潛在分佈——這顛覆了線性 ICA 中「高斯是詛咒」的傳統認知。**近似可識別性定理**則給出了優雅降級的誤差界限：$\mathbb{E}[\|h(z) - Qz\|^2] \leq D + (\varepsilon + D)^2$，其中 $D = \sqrt{n\delta / 2\rho(1-\rho)}$。

實驗在 2D 複雜非線性混合（螺旋/拋物線/RealNVP 等）、高達 **1024 維**潛空間（SIGReg 保持 $R^2 > 0.999$），以及 DMC Reacher 機械臂像素級控制任務中全面驗證了理論。**潛空間規劃等價性定理**表明：高斯編碼器（$R^2=0.95$）的直線插值規劃與 Oracle 真實規劃統計上無顯著差異，而 RL Trajectory 編碼器（$R^2=0.80$）的規劃路徑出現嚴重的非物理彎曲。所有定理均在 **Lean 4 定理證明器**中完成形式化驗證。

---

**[JEPA-Guided Diffusion — 2026-05-30：超越生成器先驗，JEPA 世界模型引導擴散採樣生成真實世界罕見樣本，Training-Free 跨任務通用，ICML 2026 (Seoul National University)](papers/2026/2026-05/JEPA-Guided-Diffusion/AI_Daily_JEPA_Guided_Diffusion.md)**

本文提出 **JEPA Guidance**（首爾大學），一個發表於 **ICML 2026** 的訓練免除 (Training-Free) 擴散採樣引導框架，從根本上重新定義了「少數樣本 (Minority Sample)」的概念。傳統少數採樣方法將罕見性定義在生成器自身的隱式先驗（Generator-Centric）中，導致生成的「少數樣本」只是在特定訓練集分佈下罕見，而非真實世界中語意罕見的樣本。本論文提出**世界中心 (World-Centric)** 少數採樣：利用在海量真實世界數據上預訓練的 JEPA 編碼器（如 DINOv2），其表徵空間雅可比矩陣的奇異值之和（JEPA-SCORE）作為真實世界密度的代理，引導擴散模型的逆向採樣走向真實世界先驗的低密度區，生成如「隱形戰機」、「老年女性軍人」等在現實中真正罕見的語意樣本。

技術上，本論文結合**隨機 SVD (Randomized SVD)** 和**包絡定理 (Envelope Theorem)** 解決了高維雅可比矩陣 SVD 的計算瓶頸，並提供嚴格的誤差上界。透過**延遲引導 (Deferred Guidance)** 策略，JEPA Guidance 自然延伸至類別條件和文字條件生成任務。在 CelebA 64×64 上，本論文方法的 cFID 達到 **8.50**（遠優於 SGMS 的 61.76 和 BnS 的 67.10），JEPA-SCORE 達到 **-300.79**（最低）。在 SDXL-Lightning 文字條件生成中，JEPA-SCORE 達到 **-337.88**，同時 CLIP/PickScore 幾乎不受影響。下游分類器增強實驗中，僅 30K 的 JEPA Guidance 增強樣本即超越 50K 其他方法的增強效果（F1: 0.775 vs 0.757）。

---

**[VIAR — 2026-05-29：視覺隱式自迴歸模型 (VIAR)：將顯式深層堆疊塌縮為單一隱式均衡層，解鎖常數訓練記憶體與每尺度彈性計算控制 (ICML 2026)](papers/2026/2026-05/VIAR/AI_Daily_VIAR.md)**

本文提出 **VIAR (Visual Implicit Autoregressive Modeling)**（TeleAI），這是一個發表於 **ICML 2026** 的突破性視覺自迴歸生成框架。傳統的視覺自迴歸模型（VAR）雖然將自迴歸重新定義為「下一尺度預測（next-scale prediction）」，並實現了尺度內的並行化，但其在每個尺度轉換中仍依賴於深度堆疊的顯式 Transformer 網路。這導致隨著影像解析度的提高與模型寬度增加，記憶體開銷（特別是 KV 快取）急劇膨脹，且每個尺度的計算量被固定，無法實現靈活的「按需計算」。

VIAR 的核心創新在於：**利用深層均衡模型（DEQs）的隱式固定點（fixed-point）層，來替代 VAR 中深層的中間顯式堆疊。** 透過將顯式中間層塌縮為單一隱式均衡層，中間區塊參數減少了 **93.3%**，整體模型參數減少了 **61.6%**（從 2.0B 壓縮至 770.9M）。此外，VIAR 採用**隨機雅可比無梯度反向傳播（S-JFB）**訓練隱式層，實現了常數級的訓練記憶體，反向傳播記憶體與網路「深度」解耦，訓練參數/梯度記憶體減少 **61.6%**。在推理端，VIAR 暴露了每尺度迭代次數旋鈕（per-scale iteration knob），可在細尺度上減少迭代次數，在幾乎不損失影像品質的前提下，將峰值記憶體降低 **42.0%**，吞吐量提升 **2.1 倍**，徹底解鎖了彈性、可控的邊端影像生成。

---

**[SRC-Flow — 2026-05-28：緊湊語義表示空間解鎖正規化流 SOTA，ImageNet gFID 1.65，快手 Kling 團隊 (USTC & Kuaishou)](papers/2026/2026-05/SRC-Flow/AI_Daily_SRC-Flow.md)**

本文提出 **SRC-Flow**（中國科學技術大學 & 快手 Kling 團隊），首次指出正規化流 (NF) 長期落後於擴散模型的根本原因：**語義容量不匹配 (Semantic-Capacity Mismatch)**。擴散模型可通過時間步相關的噪聲調度動態分配高維通道的學習壓力，而正規化流必須學習一個**單一固定雙射映射**，迫使其對完整高維表示空間的每一個維度都進行精確的可逆建模。RAE (Representation Autoencoder) 雖然提供了語義豐富的特徵，但其特徵通道高度過完整（前 32 個主成分即可解釋 99.06% 的方差），直接在完整 RAE 空間訓練 NF 效率極低（Naive Baseline gFID 僅 3.54，擴大模型寬度也無改善）。

SRC-Flow 的核心是引入**語義表示壓縮器 (SRC)**：在凍結的 RAE 編碼器與解碼器之間插入一個由 $L=4$ 層 Transformer 組成之輕量壓縮器，將 RAE 特徵從 $n$ 維壓縮至 $d=32$ 維的緊湊語義空間，再在此空間上訓練 Transformer 自回歸流 (TAF)。此外，針對 NF 學習單一固定雙射的特性，本文提出**常數噪聲正則化**（固定 $\sigma_{\text{flow}}=0.4$），替代 RAE 訓練中的每樣本隨機噪聲，顯著降低了流模型的擬合難度。在 ImageNet $256\times256$ 上，SRC-Flow 以 **gFID 1.65**（有 CFG）刷新了所有正規化流方法的歷史紀錄，在 $512\times512$ 上達到 **gFID 2.07**，同時保留了精確似然計算和確定性可逆採樣的優良數學性質。

---

**[AlignVid — 2026-05-27：Training-Free 注意力縮放調製，解決 TI2V 語義忽視問題，ICML 2026 (HKUST & UCF)](papers/2026/2026-05/AlignVid/AI_Daily_AlignVid.md)**

本文提出 **AlignVid**（HKUST, UCF, BAAI, CUHK），一種**免訓練（Training-Free）**的即插即用干預機制，專門解決文本引導圖像到視頻（TI2V）生成中普遍存在的**語義忽視（Semantic Negligence）**問題。核心洞見在於：當文本提示要求對參考圖像進行大幅修改（新增/刪除/修改物體）時，現有模型往往因**視覺主導（Visual Dominance）**而忽略文本指令——參考圖像的強大視覺先驗導致交叉注意力過度分散，抑制了新語義信息的整合。

作者通過 Pilot Study 發現，對輸入圖像施加高斯模糊能改善語義遵從性，且從能量視角分析，這對應於更低熵的交叉注意力分佈。基於此，AlignVid 提出兩大模組：(1) **注意力縮放調製（ASM）**：通過對 Q/K 矩陣乘以縮放係數 $\gamma > 1$，等效於提高注意力 Softmax 的逆溫度，從而單調降低條件塊的注意力熵，實現語義特徵的聚焦；(2) **參考引導交叉注意力（RGCA）**：利用參考圖像在時間步上的自適應交叉注意力特徵，動態引導生成過程中的關鍵結構特徵對齊。在 WebVid-10M 和多項下游任務的實驗中，AlignVid 在零額外參數和訓練成本的前提下，顯著提升了文本視頻語義對齊分數，同時在 VBench 上取得了優異的視覺逼真度，為未來的即插即用圖像到視頻生成提供了一條高效實用的全新解決方案。

---
**[UniJEPA — 2026-08-11：UniJEPA: A Unified Joint-Embedding Predictive Architecture for Task-Agnostic Visual World Modeling，首個將圖像級光度預測與視頻級時間預測統一在單一潛在空間的JEPA框架。透過單一下一嵌入預測損失與可證明防止崩潰的高斯正則化器，無需EMA或停止梯度即可端到端訓練。在保持極高計算效率的同時，ImageNet線性探測達74.9%，SSv2動作識別達78.1%，並支持零樣本動作條件規劃（規劃速度較生成式世界模型快44倍），為視覺世界建模與具身智能提供了優雅且高效的新範式 (ICML 2026, arXiv:2608.07409)](papers/2026/2026-08/UniJEPA/AI_Daily_UniJEPA.md)**

本文提出 **UniJEPA**，這是一個統一的 JEPA 框架，能夠在同一個共享的潛在空間中聯合學習光度預測（圖像級別變換）和時間預測（視頻級別下一狀態動態）。現有的 JEPA 方法高度碎片化，圖像、視頻和動作條件預測通常依賴各自獨立的編碼器與防崩潰技巧。UniJEPA 的核心創新在於：提出統一的目標函數，將光度預測和時間預測視為同一潛在預測任務的兩個實例；引入一個高斯正則化器，並在數學上證明其能有效防止表示崩潰（Representational Collapse），無需依賴 EMA 或預訓練編碼器；證明了同一個潛在空間支持可控的抽象化，光度預測學習不變結構，而時間預測學習等變動態。在預訓練後，透過在離線軌跡上對預測器進行動作條件後訓練，UniJEPA 能夠將目標特徵視為預測目標，實現零樣本的模型預測控制（MPC）規劃。實驗結果顯示，UniJEPA 在 ImageNet 線性探測達 74.9%，在 Something-Something-v2 達 78.1% Top-1 準確率，規劃成功率達 75.8%（超越 DINO-WM），且規劃速度比基於像素的生成式世界模型快上 44 倍。這項工作完美契合了 JEPA 與 Zero-shot Planning 的前沿研究，證明了「預測未來」與「理解當前變換」可以在同一個潛在空間中高效完成。
