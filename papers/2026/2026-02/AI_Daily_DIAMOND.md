# AI Daily: DIAMOND - 無需訓練，用直接推論修正 Flow Matching 模型中的生成瑕疵

> 論文名稱：DIAMOND: Directed Inference for Artifact Mitigation in Flow Matching Models
> 
> 論文連結：[https://arxiv.org/abs/2602.00883](https://arxiv.org/abs/2602.00883)
> 
> 發表單位：Jagiellonian University, Silesian University of Technology, IDEAS Research Institute (波蘭)
> 
> 發表時間：2026年1月31日
> 
> 核心領域：圖像生成、Flow Matching、無需訓練、偽影修正
> 
> 可能投稿：ICML 2026

---

## 論文核心貢獻

近年來，以 FLUX.2 為代表的 Rectified Flow 模型在圖像生成品質上設立了新標竿，但其生成結果中頻繁出現的**視覺和解剖學偽影**（如多餘的手指、扭曲的物體）成為了專業應用的主要障礙。現有方法多半採用**事後修復**（post-hoc）或需要**侵入式修改模型權重**，不僅計算成本高昂，也難以在生成過程中有效干預。

為了解決這些挑戰，來自波蘭多所頂尖大學及研究機構的團隊提出了 **DIAMOND (Directed Inference for Artifact Mitigation in Flow Matching Models)**，一個**無需訓練 (training-free)** 且**零樣本 (zero-shot)** 的框架。它透過在生成過程的每一步進行**軌跡校正 (trajectory correction)**，主動引導模型遠離會產生偽影的潛在狀態，從而實現高保真、無瑕疵的圖像合成。DIAMOND 不僅適用於 Flow Matching 模型，更能擴展至標準的擴散模型 (Diffusion Models)，展現了其廣泛的適用性。

![DIAMOND 方法概覽](asset/2026-02-DIAMOND/figure3_method_overview.webp)
*圖 1：DIAMOND 方法概覽。該技術在推論過程中，透過一個可微分的偽影偵測器（Artifact Detector）來計算偽影損失，並利用其梯度來修正生成軌跡，從而主動避免瑕疵的產生。*

## 技術方法詳解

DIAMOND 的核心思想是在生成過程的每一步，估計出當前的「乾淨圖像」，並利用一個預訓練的、可微分的**偽影偵測器 (Artifact Detector)** 來識別潛在的瑕疵區域，然後計算一個修正梯度來引導下一步的生成方向。

### Flow Matching 與乾淨樣本估計

Rectified Flow 模型旨在學習一個向量場 $v_\theta$，將一個簡單的高斯噪聲分佈 $\pi_1 = \mathcal{N}(0, \mathbf{I})$ 盡可能「拉直」地轉換為真實的數據分佈 $\pi_0$。這個過程由一個常微分方程 (ODE) 描述：

$$
\frac{dx_t}{dt} = v_\theta(x_t, t)
$$

其中 $x_t$ 是時間 $t$（從 1 到 0）的潛在表示。理想情況下，軌跡是線性的，即 $x_t = (1-t)x_0 + tx_1$。透過這個線性假設，我們可以從任何中間時間步 $t$ 的潛在表示 $x_t$ 和當時預測的速度場 $v_\theta(x_t, t)$ 來估計最終的乾淨圖像 $x_0$：

$$
\hat{x}_{0,t} = x_t - t \cdot v_\theta(x_t, t)
$$

這個公式是 DIAMOND 的基石，它讓模型能在生成未完成時「預見」最終結果的樣貌，從而提前發現並修正問題。

### 基於梯度的軌跡校正

在得到每一步的乾淨圖像估計 $\hat{x}_{0,t}$ 後，DIAMOND 將其輸入一個預訓練的偽影偵測器 $\mathcal{AD}$（例如，一個專門用來檢測手部變形或文字錯誤的分割模型），並計算一個像素級的偽影損失 $\mathcal{L}_a$。接著，計算這個損失相對於當前潛在表示 $x_t$ 的梯度：

$$
\nabla_{x_t} \mathcal{L}_a = \nabla_{x_t} \mathcal{L}_a(\mathcal{AD}(\hat{x}_{0,t}))
$$

這個梯度指明了如何調整 $x_t$ 以減少偽影。為了確保校正的穩定性，論文強調了對梯度進行**歸一化 (Normalization)** 的重要性，只保留其方向，而強度則由一個動態調度的超參數 $\lambda_t$ 控制。最終，校正後的更新步驟如下：

$$
x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t) - \lambda_t \cdot \frac{\nabla_{x_t} \mathcal{L}_a}{\|\nabla_{x_t} \mathcal{L}_a\|_2 + \epsilon}
$$

這個過程在整個生成軌跡中反覆進行，從而實現了對偽影的漸進式修正。

![DIAMOND 修正效果](asset/2026-02-DIAMOND/figure2_artifact_comparison.webp)
*圖 2：DIAMOND 在多種主流模型（包括 FLUX.2, FLUX.1, SDXL）上的修正效果。無論是咖啡杯的形狀、船槳的結構，還是人物的手指，DIAMOND 都能在不影響整體畫面的情況下，顯著改善局部偽影。*

## 實驗結果與性能

論文在多個數據集（動物、文字、人物）和多種模型（FLUX.2, FLUX.1, SDXL）上進行了廣泛的實驗，結果令人信服。

### 定量分析

下表展示了 DIAMOND 與其他方法的比較。可以看到，在所有測試案例中，DIAMOND 都能顯著降低**平均偽影頻率 (Mean Artifact Freq)** 和**偽影像素比例 (Artifact Pixel Ratio)**，同時保持了與原始圖像的高度相似性 (低的 MAE) 和優秀的圖像品質 (高的 ImageReward) 及文圖對齊度 (高的 CLIP-T)。

| 模型與數據集 | 方法 | 平均偽影頻率 (%) ↓ | 偽影像素比例 (%) ↓ | ImageReward ↑ | MAE ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FLUX.2 [dev]** (animals) | Baseline | 100.0 | 0.325 | 1.354 | - |
| | **+ DIAMOND** | **30.8** | **0.074** | 1.343 | 9.402 |
| **FLUX.1 [dev]** (words) | Baseline | 100.0 | 0.152 | 0.790 | - |
| | + DiffDoctor | 47.0 | 0.065 | 0.744 | 11.469 |
| | **+ DIAMOND** | **9.8** | **0.009** | **0.819** | **8.744** |
| **FLUX.1 [schnell]** (people) | Baseline | 100.0 | 0.812 | 1.175 | - |
| | **+ DIAMOND** | **32.3** | **0.103** | 1.162 | **5.135** |
| **SDXL** (people) | Baseline | 100.0 | 0.651 | 1.289 | - |
| | **+ DIAMOND** | **36.5** | **0.128** | 1.281 | **4.981** |

*表 1：DIAMOND 在不同模型和數據集上的定量評估結果。數據表明，DIAMOND 在大幅減少偽影的同時，對圖像品質和內容的影響最小。*

### 定性比較

定性比較結果（如下圖）進一步證明了 DIAMOND 的優越性。與其他方法（如 HandsXL）相比，DIAMOND 在修正手部偽影時，能夠更好地保持圖像的原始風格和細節，避免了引入新的不自然紋理。

![定性比較](asset/2026-02-DIAMOND/figure4_qualitative_comparison.webp)
*圖 3：與其他方法的定性比較。DIAMOND 在修正人物手部等複雜結構時，效果自然，且不會對圖像其他部分造成負面影響。*

## 個人評價與意義

DIAMOND 論文提出了一個非常聰明且實用的偽影修正框架，其最大的亮點在於**「無需訓練」**和**「通用性」**。

1.  **激發新思路**：它將偽影修正從一個「事後處理」的難題，轉變為一個可以在生成過程中動態引導的「控制問題」。這種「預測-評估-修正」的閉環思路，為可控生成領域提供了極具價值的參考，特別是在需要高保真度的專業應用場景。

2.  **實用價值高**：無需修改模型權重或進行昂貴的微調，意味著它可以作為一個即插即用的模塊，輕鬆集成到現有的各種生成模型（Flow Matching 或 Diffusion）工作流中，極大地降低了部署門檻。

3.  **解決核心痛點**：偽影問題，特別是手部和臉部的解剖學錯誤，一直是限制 AI 生成圖像走向商業應用的主要瓶頸之一。DIAMOND 提供了一個優雅且高效的解決方案，有望顯著提升生成圖像的可用性和可靠性。

總體而言，DIAMOND 不僅在技術上具有創新性，更在應用層面展現了巨大的潛力。它完美契合了當前對**訓練無關、零樣本、可控生成**的研究趨勢，為我們打開了一扇通往更高質量、更可靠的 AI 內容創作的大門。這項研究無疑是近期圖像生成領域最值得關注的進展之一。
