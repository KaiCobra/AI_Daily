# AI Daily

每日精選 AI 前沿論文閱讀與深度解析。聚焦深度學習、圖像生成、表徵學習、擴散模型等前沿方向。

**Last Updated: 2026-06-05**

---

## 今日閱讀
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
