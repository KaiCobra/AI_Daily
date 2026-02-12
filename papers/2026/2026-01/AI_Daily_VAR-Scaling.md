# AI Daily: VAR-Scaling - 探索視覺自回歸模型的推理時縮放新策略

**作者**: Manus AI
**日期**: 2026-01-14

---

## 論文基本資訊

- **標題**: Inference-Time Scaling for Visual AutoRegressive modeling by Searching Representative Samples
- **作者**: Weidong Tang, Xinyan Wan, Siyu Li, Xiumei Wang
- **機構**: School of Electronic Engineering, Xidian University, Xi'an, China
- **會議**: Accepted to PRCV 2025 (Pattern Recognition and Computer Vision)
- **arXiv**: [2601.07293](https://arxiv.org/abs/2601.07293)
- **代碼**: [https://github.com/007mh/VAR-Scaling](https://github.com/007mh/VAR-Scaling)

![論文首頁](../../../assets/var_scaling_page1.webp)

---

## 核心貢獻與創新點

這篇論文首次將**推理時縮放 (Inference-Time Scaling)** 的概念應用於**視覺自回歸模型 (Visual Autoregressive modeling, VAR)**，提出了一個名為 **VAR-Scaling** 的通用框架。傳統的VAR模型在向量量化 (Vector-Quantized, VQ) 的離散潛在空間中操作，這使得如擴散模型中常見的連續路徑搜索變得不可能。VAR-Scaling通過以下創新點解決了這個核心挑戰：

1.  **離散空間的連續化**：引入**核密度估計 (Kernel Density Estimation, KDE)**，將離散的採樣空間映射到一個準連續的特徵空間。在這個空間中，高密度區域的樣本被視為穩定且高質量的解，從而實現了對採樣分佈的有效導航。

2.  **密度自適應混合採樣策略**：為了平衡生成質量與多樣性，論文提出了一種混合採樣策略：
    - **Top-k 採樣**：專注於高密度區域，以保留接近分佈眾數的樣本質量。
    - **Random-k 採樣**：探索低密度區域，以維持生成結果的多樣性，防止過早收斂到單一模式。

3.  **關鍵尺度的保真度優化**：通過在推理過程的關鍵尺度上優化樣本的保真度，VAR-Scaling 能夠顯著提升最終生成圖像的質量。

實驗結果表明，該框架在類別條件生成和文本到圖像生成任務上均取得了顯著的性能提升，例如在VAR模型上將IS指標提升了8.7%，同時保持了穩定的FID。

---

## 技術方法簡述

### 1. VAR的生成先驗與挑戰

VAR模型將圖像生成定義為一個從粗到細 (coarse-to-fine) 的任務，其自回歸單元是整個Token Map，而非單個Token。自回歸似然公式如下：

$$ p(r_1, r_2, ..., r_K) = \prod_{k=1}^{K} p(r_k | r_1, r_2, ..., r_{k-1}) $$

其中，$r_k$ 是在尺度 $k$ 的Token Map。然而，VQ導致的離散潛在空間阻礙了平滑的優化路徑搜索。

### 2. VAR-Scaling 框架

為了解決上述問題，VAR-Scaling框架的核心思想是將離散的採樣空間轉化為可以進行有效搜索的連續空間。

**核密度估計 (KDE)**

論文使用帶有高斯核的KDE來估計樣本的概率密度。對於一組從離散碼本中採樣的候選Tokens \({x_j}\)_{j=1}^N \subset \mathbb{R}^d$，其多變量密度估計函數為：

$$ f(x) = \frac{1}{n(2\pi)^{d/2}h^d} \sum_{j=1}^{N} \exp(-\frac{||x - x_j||^2}{2h^2}) $$

其中，$h$ 是控制平滑度的帶寬參數，根據Silverman's rule of thumb進行選擇：

$$ h = \sigma[\frac{n(d + 2)}{4}]^{-\frac{1}{d+4}} $$

通過這種方式，每個離散的Token都可以被映射到一個連續的密度分數上，從而構建出一個準連續的特徵空間。

![方法示意圖](../../../assets/var_scaling_method.webp)

**密度引導的代表性樣本選擇**

論文發現，VAR的生成過程在不同尺度上表現出不同的模式：
- **通用模式 (General patterns)**：在早期尺度（如0-1），模型主要定義整體的空間結構和輪廓。
- **特定模式 (Specific patterns)**：在後期尺度（如2-9），模型則專注於細化紋理、邊緣等局部細節。

基於此，VAR-Scaling採用了密度自適應的採樣策略。首先，計算當前樣本池的平均密度 `Density Mean`，並定義一個動態閾值 `D_current`。

$$ \text{Density Mean} = \frac{1}{n} \sum_{i=1}^{n} f(x_i) \quad , \quad D_{\text{current}} = \text{Density Mean} \times \alpha $$

然後根據樣本密度進行分類：

$$ \text{Density Classification} = \begin{cases} \text{low-density} & \text{if } D_{\text{current}} < D_{\text{mean}} \\ \text{high-density} & \text{otherwise} \end{cases} $$

- 在**高密度區域**，採用 **Top-k 採樣**，優先選擇密度最高的樣本以保證生成質量。
- 在**低密度區域**，則切換到 **Random-k 採樣**，隨機探索以增加多樣性。

---

## 實驗結果與性能指標

VAR-Scaling在多個基準測試上都取得了優異的表現。論文在ImageNet 256x256數據集上對VAR和FlexVAR模型進行了評估。

**主要性能提升**：

- **VAR**: Inception Score (IS) 提升 **+8.7%**。
- **FlexVAR**: Inception Score (IS) 提升 **+6.3%**。
- **Infinity (文本到圖像)**: General score 提升 **+1.1%**。

下圖展示了在ImageNet-50k上，不同採樣策略對FID和IS指標的影響。可以看出，Top-k採樣（高密度）能夠持續降低FID並提升IS，而另外兩種策略則效果較差，驗證了高密度樣本對應高質量輸出的假設。

![實驗結果圖](../../../assets/var_scaling_results.webp)

---

## 相關研究背景

本研究建立在**視覺自回歸模型 (VAR)** 的基礎之上。VAR由Keyu Tian等人在NeurIPS 2024的獲獎論文 *"Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"* 中首次提出。其核心思想是將圖像的自回歸學習從傳統的“next-token prediction”（光柵掃描式）重新定義為“next-scale prediction”（從粗到細的多尺度預測）。

這一範式使得GPT風格的自回歸模型在圖像生成質量、推理速度和擴展性上首次超越了擴散模型（Diffusion Transformers），並展現出類似大型語言模型（LLMs）的**縮放定律 (Scaling Laws)** 和 **零樣本泛化能力**。

VAR-Scaling 正是為了解決VAR在離散潛在空間中難以進行推理時優化的問題而提出的，是VAR生態系統中的一次重要探索。

---

## 個人評價與意義

VAR-Scaling為視覺自回歸模型帶來了一個新穎且有效的推理時優化框架。它巧妙地繞過了VQ模型離散潛在空間的限制，通過KDE將其轉化為一個可供搜索的連續空間，這種思路極具啟發性。

**核心價值**：

- **通用性**：該框架是通用的，可以應用於不同類型的VAR模型（如VAR, FlexVAR, Infinity），展示了其廣泛的適用性。
- **性能與效率的平衡**：密度自適應採樣策略在追求更高生成質量的同時，兼顧了結果的多樣性，避免了模式崩塌。
- **激發新思路**：這項工作為“Training-Free”或“Inference-Time”優化方向提供了新的思路。它表明，即使不重新訓練模型，僅通過在推理階段進行更智能的採樣和搜索，也能顯著提升生成模型的性能。這對於降低模型迭代成本、激發社區對現有模型潛力的挖掘具有重要意義。

總體而言，VAR-Scaling不僅是對VAR模型的一次重要補強，也為整個生成模型領域，特別是那些依賴離散表示的模型，提供了一個值得借鑒的優化範式。它讓我們看到，在模型推理階段，依然有廣闊的創新空間等待探索。

---

### 參考文獻

1.  [Tang, W., Wan, X., Li, S., & Wang, X. (2026). *Inference-Time Scaling for Visual AutoRegressive modeling by Searching Representative Samples*. arXiv preprint arXiv:2601.07293.](https://arxiv.org/abs/2601.07293)
2.  [Tian, K., Jiang, Y., Yuan, Z., Peng, B., & Wang, L. (2024). *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*. arXiv preprint arXiv:2404.02905.](https://arxiv.org/abs/2404.02905)
