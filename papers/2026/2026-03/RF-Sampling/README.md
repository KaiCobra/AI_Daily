# AI Daily

## Reflective Flow Sampling Enhancement (RF-Sampling)

**發布日期**: 2026-03-06
**論文連結**: [arXiv:2603.06165](https://arxiv.org/abs/2603.06165)
**作者**: Zikai Zhou, Muyao Wang, Shitong Shao, Lichen Bai, Haoyi Xiong, Bo Han, Zeke Xie (HKUST-GZ, UTokyo, Microsoft, HKBU)

### 1. 論文背景與動機

隨著文本到圖像（Text-to-Image, T2I）生成技術的快速發展，基於 Flow Matching 演算法訓練的擴散模型（如 FLUX）因其卓越的生成品質和高效的採樣速度，已成為傳統擴散模型的強大替代方案。

為了進一步提升生成品質和文本對齊度，研究人員提出了多種推論增強（Inference Enhancement）策略（如 Z-Sampling）。然而，這些現有技術主要針對傳統擴散模型設計，依賴於無分類器引導（Classifier-Free Guidance, CFG）機制的特性。對於 FLUX 這類採用 **CFG-distilled** 架構的 Flow Matching 模型，由於傳統的引導信號已被蒸餾到模型權重中，缺乏顯式的無條件分支（unconditional branch），導致現有的推論增強方法往往無法直接應用或效果不佳。

為了解決這一痛點，本文提出了 **Reflective Flow Sampling (RF-Sampling)**，這是一個專為 Flow Models 設計的 **Training-free** 推論增強框架，特別適用於 FLUX 等 CFG-distilled 變體。

### 2. 核心方法：RF-Sampling

RF-Sampling 的核心思想是透過插值文本表示並結合 Flow Inversion，讓模型能夠探索與輸入提示（Prompt）更一致的噪聲空間。作者將這種 Flow Inversion 稱為 **Reflective Flow**。

與以往基於啟發式（heuristic）的噪聲操作不同，RF-Sampling 具有嚴格的數學理論基礎。作者證明了透過「高權重去噪（High-Weight Denoising） $\rightarrow$ 低權重反演（Low-Weight Inversion）」機制合成的潛在變數（latent），本質上是文本-圖像對齊分數（Alignment Score）梯度的近似值。

#### 2.1 理論基礎：作為梯度上升的 RF-Sampling

在推論階段，目標是找到一個潛在變數 $x_t$，使其最大化對齊分數 $J(x_t) = \log p(c|x_t)$。根據基於分數的模型理論，該分數的梯度與語義向量場的差異成正比：
$$ \nabla_x J(x_t) \propto v_\theta(x_t, c) - v_\theta(x_t, \emptyset) $$

由於 CFG-distilled 模型缺乏顯式的無條件分支 $v_\theta(x_t, \emptyset)$，RF-Sampling 引入了反射位移向量（Reflective Displacement Vector） $\Delta_{RF}$ 來估計該梯度：
$$ \Delta_{RF} = \delta t \cdot [v_\theta(x_t, t, c_{high}) - v_\theta(x_t - \delta t, t - \delta t, c_{low})] $$

作者透過 **Theorem 1 (First-Order Validity)** 證明了 $\Delta_{RF}$ 與 $\nabla_x J(x_t)$ 的內積大於零，確保了這是一個嚴格遞增的方向。這意味著 RF-Sampling 實際上是在潛在狀態上執行**梯度上升（Gradient Ascent）**，迭代地將軌跡更新向具有更高文本-圖像對齊機率的區域，而無需顯式的 CFG 計算或反向傳播。

#### 2.2 演算法流程

RF-Sampling 在 ODE 求解器的每個積分步驟中實作為一個三階段過程：

1.  **Stage 1: High-Weight Denoising（高權重去噪）**
    使用較高的插值權重和放大權重獲得混合文本嵌入 $c_{high}$，並執行標準的去噪步驟，確保與給定文本提示的強烈對齊。
2.  **Stage 2: Low-Weight Inversion（低權重反演）**
    不直接使用第一階段的結果，而是使用較弱的文本嵌入 $c_{low}$ 執行反向 ODE 求解（Inversion），退回到先前的時間步。這一步產生了 Reflective Flow。
3.  **Stage 3: Normal-Weight Denoising（正常權重去噪）**
    結合前兩階段的位移，使用正常權重的文本嵌入完成最終的去噪步驟。

![RF-Sampling 效果對比](./asset/figure_1.png)
*圖 1：RF-Sampling 與標準採樣在不同 Flow Models 上的視覺對比。*

### 3. 實驗結果與貢獻

作者在多個基準測試（如 HPDv2, Pick-a-Pic, DrawBench）上進行了廣泛的實驗，結果表明 RF-Sampling 在美學品質（Aesthetic Quality）和語義忠實度（Semantic Faithfulness）上均有顯著提升。

#### 3.1 核心貢獻

1.  **專為 Flow Models 設計的新框架**：有效解決了 CFG-distilled 變體（如 FLUX）無法使用傳統引導方法的限制。
2.  **堅實的理論基礎**：提供了嚴格的數學推導，證明 RF-Sampling 隱式地執行了文本-圖像對齊分數的梯度上升。
3.  **卓越的性能與 Test-time Scaling 能力**：
    *   在標準基準測試中達到 State-of-the-Art (SOTA) 表現。
    *   **首次在 FLUX 上展現出 Test-time Scaling 特性**：隨著推論計算量（Inference Time）的增加，生成品質能夠持續提升（見圖 2）。
4.  **強大的泛化能力**：可無縫整合到各種生成任務中，包括 LoRA 組合、圖像編輯和影片生成。

![Test-time Scaling 表現](./asset/figure_2.png)
*圖 2：RF-Sampling 在 FLUX 模型上展現出的 Test-time Scaling 能力。隨著推論時間增加，性能持續提升。*

### 4. 總結

RF-Sampling 為 Flow Matching 模型的推論增強提供了一個優雅且理論完備的解決方案。它不僅克服了 CFG-distilled 架構帶來的挑戰，還解鎖了 FLUX 等先進模型的 Test-time Scaling 潛力，為未來的 Training-free 圖像生成控制研究指明了新的方向。
