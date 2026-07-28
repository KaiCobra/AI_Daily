# AI Daily

每日精選 AI 前沿論文閱讀與深度解析。聚焦深度學習、圖像生成、表徵學習、擴散模型等前沿方向。

**Last Updated: 2026-07-28**

📚 **[完整論文索引(111 篇,按月份)](INDEX.md)**

---

## 今日閱讀
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
