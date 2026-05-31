# AI Daily

每日精選 AI 前沿論文閱讀與深度解析。聚焦深度學習、圖像生成、表徵學習、擴散模型等前沿方向。

**Last Updated: 2026-05-31**

---

## 今日閱讀

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

本文提出 **VIAR (Visual Implicit Autoregressive Modeling)**（TeleAI），這是一個發表於 **ICML 2026** 的突破性視覺自迴歸生成框架。傳統的視覺自迴歸模型（VAR）雖然將自迴歸重新定義為「下一尺度預測（next-scale prediction）」，並實現了尺度內的並行化，但其在每個尺度轉換中仍依賴於深度堆疊的顯式 Transformer 網路。這導致隨著影像解析度的提高與模型寬度的增加，記憶體開銷（特別是 KV 快取）急劇膨脹，且每個尺度的計算量被固定，無法實現靈活的「按需計算」。

VIAR 的核心創新在於：**利用深層均衡模型（DEQs）的隱式固定點（fixed-point）層，來替代 VAR 中深層的中間顯式堆疊。** 透過將顯式中間層塌縮為單一隱式均衡層，中間區塊參數減少了 **93.3%**，整體模型參數減少了 **61.6%**（從 2.0B 壓縮至 770.9M）。此外，VIAR 採用**隨機雅可比無梯度反向傳播（S-JFB）**訓練隱式層，實現了常數級的訓練記憶體，反向傳播記憶體與網路「深度」解耦，訓練參數/梯度記憶體減少 **61.6%**。在推理端，VIAR 暴露了每尺度迭代次數旋鈕（per-scale iteration knob），可在細尺度上減少迭代次數，在幾乎不損失影像品質的前提下，將峰值記憶體降低 **42.0%**，吞吐量提升 **2.1 倍**，徹底解鎖了彈性、可控的邊端影像生成。

---

**[SRC-Flow — 2026-05-28：緊湊語義表示空間解鎖正規化流 SOTA，ImageNet gFID 1.65，快手 Kling 團隊 (USTC & Kuaishou)](papers/2026/2026-05/SRC-Flow/AI_Daily_SRC-Flow.md)**

本文提出 **SRC-Flow**（中國科學技術大學 & 快手 Kling 團隊），首次指出正規化流 (NF) 長期落後於擴散模型的根本原因：**語義容量不匹配 (Semantic-Capacity Mismatch)**。擴散模型可通過時間步相關的噪聲調度動態分配高維通道的學習壓力，而正規化流必須學習一個**單一固定雙射映射**，迫使其對完整高維表示空間的每一個維度都進行精確的可逆建模。RAE (Representation Autoencoder) 雖然提供了語義豐富的特徵，但其特徵通道高度過完整（前 32 個主成分即可解釋 99.06% 的方差），直接在完整 RAE 空間訓練 NF 效率極低（Naive Baseline gFID 僅 3.54，擴大模型寬度也無改善）。

SRC-Flow 的核心是引入**語義表示壓縮器 (SRC)**：在凍結的 RAE 編碼器與解碼器之間插入一個由 $L=4$ 層 Transformer 組成的輕量壓縮器，將 RAE 特徵從 $n$ 維壓縮至 $d=32$ 維的緊湊語義空間，再在此空間上訓練 Transformer 自回歸流 (TAF)。此外，針對 NF 學習單一固定雙射的特性，本文提出**常數噪聲正則化**（固定 $\sigma_{\text{flow}}=0.4$），替代 RAE 訓練中的每樣本隨機噪聲，顯著降低了流模型的擬合難度。在 ImageNet $256\times256$ 上，SRC-Flow 以 **gFID 1.65**（有 CFG）刷新了所有正規化流方法的歷史紀錄，在 $512\times512$ 上達到 **gFID 2.07**，同時保留了精確似然計算和確定性可逆採樣的優良數學性質。

---

**[AlignVid — 2026-05-27：Training-Free 注意力縮放調製，解決 TI2V 語義忽視問題，ICML 2026 (HKUST & UCF)](papers/2026/2026-05/AlignVid/AI_Daily_AlignVid.md)**

本文提出 **AlignVid**（HKUST, UCF, BAAI, CUHK），一種**免訓練（Training-Free）**的即插即用干預機制，專門解決文本引導圖像到視頻（TI2V）生成中普遍存在的**語義忽視（Semantic Negligence）**問題。核心洞見在於：當文本提示要求對參考圖像進行大幅修改（新增/刪除/修改物體）時，現有模型往往因**視覺主導（Visual Dominance）**而忽略文本指令——參考圖像的強大視覺先驗導致交叉注意力過度分散，抑制了新語義信息的整合。

作者通過 Pilot Study 發現，對輸入圖像施加高斯模糊能改善語義遵從性，且從能量視角分析，這對應於更低熵的交叉注意力分佈。基於此，AlignVid 提出兩大模組：(1) **注意力縮放調製（ASM）**：通過對 Q/K 矩陣乘以縮放係數 $\gamma > 1$，等效於提高注意力 Softmax 的逆溫度，從而單調降低條件塊的注意力熵，实现「語義銳化」；(2) **引導調度（GS）**：通過模組級（Block-level）與步驟級（Step-level）的雙重調度，將 ASM 限制在前景敏感模組和語義決定性的去噪步驟中，避免美學質量下降。此外，本文推出了首個專門評估語義忽視的基準 **OmitI2V**（367 個人工標注樣本，VQA 評估協議）。在 FramePack 和 Wan2.1 上，AlignVid 在語義對齊指標上分別提升最高 **+6.82%（Modification）/ +7.79%（Addition）/ +6.34%（Deletion）**，同時美學質量幾乎不受影響。

---

**[RiT — 2026-05-26：自監督表徵空間的流匹配幾何學優勢，Vanilla DiT + x-Prediction 達成超高效生成 (Mila & Utrecht University)](papers/2026/2026-05/RiT/AI_Daily_RiT.md)**

本文提出 **RiT (Representation Image Transformer)**（Mila – Québec AI Institute & Utrecht University），一個直接在凍結的自監督特徵表示空間（DINOv2）中進行流匹配（Flow Matching）的極簡 Vanilla Diffusion Transformer 框架。

核心洞見在於：**自監督表徵空間（DINOv2）在幾何與統計學性質上天然契合流匹配的傳輸路徑**。作者通過對比 Pixel、SD-VAE 與 DINOv2 空間，量化證明了 DINOv2 具備四大幾何優勢：(1) **7.3 倍的有效秩**（方差分佈極其均勻），(2) **35 倍的協方差條件數改善**（避免去噪後期過擬合），(3) **11.5 倍的超額峰度降低**（邊際分佈高度接近高斯），(4) **1.7 倍的流形內線性插值誤差降低**（線性路徑不偏離數據流形）。這些優良性質消除了先前表徵擴散模型所需的複雜專用預測頭（如 RAE 的 DDT 頭）或黎曼流形傳輸（如 RJF）。

針對 DINOv2 模長集中所導致的徑向干涉（Radial Interference）病理特性，RiT 通過將預測目標改為 **$x$-prediction**（直接預測乾淨特徵 $z_0$），強制將預測目標約束在數據流形上，從而在輸出端完美消除徑向歧義。此外，RiT 引入了**聯合 [CLS]-Patch 建模**（利用 `[CLS]` 作為雙向注意力引導）與**維度感知雜訊排程**（透過 Time-Shift 補償高維空間的 SNR 衰減）。

在 ImageNet $256 \times 256$ 基準上，僅 676M 參數的 RiT-XL 僅需 100 個 Epoch 即可達到其他方法 800 Epoch 的效果（**4~7 倍訓練加速**），並在 800 Epoch 時達到無引導 FID **1.45**，有引導 FID **1.14**，全面超越參數量更大的 DiT-DH-XL（839M）。得益於極佳的幾何條件，RiT 的 ODE 去噪軌跡截斷誤差衰減速率是像素空間的 **12.9 倍**，在無任何蒸餾的情況下，**僅需 5 步 Heun 採樣即可達到 FID 1.99，10 步達到 FID 1.25**，解鎖了超高效的 Few-Step 生成能力。

---

**[GRAM — 2026-05-25：Generative Recursive Reasoning 概率多軌跡推理，無條件生成與並行 Test-Time Scaling (KAIST & Mila)](papers/2026/2026-05/GRAM/AI_Daily_GRAM.md)**

本文提出 **GRAM (Generative Recursive reAsoning Models)**（KAIST, Mila, NYU, UdeM），一個將遞歸潛在推理轉化為**概率多軌跡計算**的框架。核心洞見在於：現有遞歸推理模型（RRMs）大多是確定性的，容易在多解任務中陷入模式崩潰。GRAM 將推理建模為隨機的潛在軌跡，在每次遞歸時加入依賴於狀態的隨機引導 $\epsilon_t$。這種設計不僅使模型能夠探索多種假設 and 解決策略，更解鎖了全新的**並行推理時間擴展（Test-Time Scaling）**維度：除了增加遞歸深度，還能透過並行採樣多條軌跡（Width）來提升性能。在 Sudoku-Extreme 任務中，10M 參數的 GRAM 使用 16 次迭代並行採樣 20 條軌跡，準確率達 97.0%，顯著超越 320 次迭代的確定性基線 TRM（90.5%）。此外，在無輸入條件下，相同的遞歸過程可作為無條件生成模型，在二值化 MNIST 和 Sudoku 生成中表現優異。

---

**[SEGA — 2026-05-24：頻譜能量引導注意力動態縮放 RoPE，Training-Free DiT 超解析度外推達 6144×6144，全面超越 YaRN/DyPE/UltraImage (University of Toronto)](papers/2026/2026-05/SEGA/AI_Daily_SEGA.md)**

本文提出 **SEGA (Spectral-Energy Guided Attention)**（University of Toronto & Vector Institute），一種**免訓練（Training-Free）**的 Diffusion Transformer 高解析度外推方法。核心洞見在於：現有 RoPE 外推方法（如 YaRN）對所有頻率維度施加均勻縮放，導致全局結構與精細細節之間存在固有的 trade-off。SEGA 透過對每個去噪步驟的潛在特徵執行 2D FFT 頻譜分析，提取軸向功率譜（Axis-wise profiles） and 徑向頻譜（Radial profile），並計算三個組件：(1) **參考尺度** $m_{\text{ref}} = (R_{\text{target}}/R_{\text{train}})^\kappa$，(2) **維度級校正** $s_d^{(a)} = \phi(z_d^{(a)}) - \mathbb{E}[\phi(z^{(a)})]$（零和重分配，tanh 非線性），(3) **全局振幅因子** $\sigma = 1 - \text{SF}(\mathcal{E}_{\text{iso}})^\gamma$（頻譜平坦度 Wiener entropy），最終縮放因子 $m_d^{(a)} = m_{\text{ref}} \cdot (1 - \sigma \cdot s_d^{(a)})$。SEGA 對低能量頻段施加較強縮放以保留位置區分度，對高能量頻段施加較弱縮放以避免過度放大，並在去噪初期（頻譜平坦）自動抑制動態調整。在 Flux 模型 4096×4096 解析度上，SEGA 的 ImageReward（**1.26**）、CLIP Score（**29.22**）和 FID（**150.05**）全面超越 YaRN（0.88 / 28.30 / 160.48），並在 Qwen 模型上取得相同趨勢的 SOTA 結果。方法無需微調或架構修改，可直接整合至任何 RoPE-based DiT 管線。
