# AI Daily

每日精選 AI 前沿論文閱讀與深度解析。聚焦深度學習、圖像生成、表徵學習、擴散模型等前沿方向。

---

## 今日閱讀

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

本文提出 **SEGA (Spectral-Energy Guided Attention)**（University of Toronto & Vector Institute），一種**免訓練（Training-Free）**的 Diffusion Transformer 高解析度外推方法。核心洞見在於：現有 RoPE 外推方法（如 YaRN）對所有頻率維度施加均勻縮放，導致全局結構與精細細節之間存在固有的 trade-off。SEGA 透過對每個去噪步驟的潛在特徵執行 2D FFT 頻譜分析，提取軸向功率譜（Axis-wise profiles）和徑向頻譜（Radial profile），並計算三個組件：(1) **參考尺度** $m_{\text{ref}} = (R_{\text{target}}/R_{\text{train}})^\kappa$，(2) **維度級校正** $s_d^{(a)} = \phi(z_d^{(a)}) - \mathbb{E}[\phi(z^{(a)})]$（零和重分配，tanh 非線性），(3) **全局振幅因子** $\sigma = 1 - \text{SF}(\mathcal{E}_{\text{iso}})^\gamma$（頻譜平坦度 Wiener entropy），最終縮放因子 $m_d^{(a)} = m_{\text{ref}} \cdot (1 - \sigma \cdot s_d^{(a)})$。SEGA 對低能量頻段施加較強縮放以保留位置區分度，對高能量頻段施加較弱縮放以避免過度放大，並在去噪初期（頻譜平坦）自動抑制動態調整。在 Flux 模型 4096×4096 解析度上，SEGA 的 ImageReward（**1.26**）、CLIP Score（**29.22**）和 FID（**150.05**）全面超越 YaRN（0.88 / 28.30 / 160.48），並在 Qwen 模型上取得相同趨勢的 SOTA 結果。方法無需微調或架構修改，可直接整合至任何 RoPE-based DiT 管線。
