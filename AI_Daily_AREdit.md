# AI Daily: AREdit - 無需訓練的VAR圖像編輯新範式

**日期**: 2026年01月09日

## 論文基本信息

| 項目 | 內容 |
| --- | --- |
| **論文標題** | Training-Free Text-Guided Image Editing with Visual Autoregressive Model |
| **arXiv編號** | [2503.23897v1](https://arxiv.org/abs/2503.23897) |
| **提交日期** | 2025年3月31日 |
| **作者** | Yufei Wang, Lanqing Guo, Zhihao Li, Jiaxing Huang, Pichao Wang, Bihan Wen, Jian Wang |
| **研究機構** | Snap Research, 南洋理工大學, UT Austin |
| **代碼庫** | [https://github.com/wyf0912/AREdit](https://github.com/wyf0912/AREdit) |
| **關鍵詞** | `Visual Autoregressive (VAR)`, `Training-Free`, `Image Editing`, `Zero-Shot` |

---

## 核心貢獻與創新點

在文本引導的圖像編輯領域，現有的主流方法（如基於Diffusion或Rectified Flow的模型）普遍依賴**inversion**技術。然而，inversion過程的不準確性以及文本與圖像特徵的糾纏，常常導致編輯結果保真度下降或產生非預期的全局變化。

為了解決這些挑戰，來自Snap Research和南洋理工大學等機構的研究者提出了**AREdit**，一個基於**視覺自回歸模型 (Visual Autoregressive Model, VAR)** 的全新圖像編輯框架。這項工作標誌著圖像編輯領域的一次重要範式轉移，其核心創新點可歸納如下：

1.  **首個基於VAR的圖像編輯框架**：AREdit是第一個將VAR模型應用於文本引導圖像編輯的工作。它完全拋棄了傳統的inversion和迭代去噪過程，採用VAR的**next-scale prediction**策略，以coarse-to-fine的方式高效、結構化地生成圖像，從根本上避免了inversion帶來的誤差傳播問題。

2.  **無需訓練 (Training-Free)**：該框架無需任何額外訓練或微調，直接利用預訓練好的VAR模型（如Infinity-2B）即可實現高質量的圖像編輯，極大地降低了應用門檻和計算成本。

3.  **三大核心技術組件**：
    *   **隨機性緩存 (Randomness Caching)**：通過一次前向傳播，緩存原始圖像在生成過程中的token索引和概率分佈。這些緩存信息精準地捕捉了源提示詞與圖像內容的對應關係，功能上類似於diffusion模型中的結構化噪聲，但更加靈活和精確。
    *   **自適應細粒度遮罩 (Adaptive Fine-Grained Masking)**：通過比較源提示詞和目標提示詞在生成過程中的概率分佈差異，AREdit能夠動態、精準地計算出需要修改的圖像區域，生成細粒度的遮罩。這確保了編輯只作用於目標區域，完美保護了背景和無關內容。
    *   **Token重組 (Token Re-assembling)**：在編輯過程中，該策略巧妙地重用緩存中的低頻特徵以維持圖像的全局結構和一致性，同時在高頻細節上根據新提示詞進行重新採樣，從而兼顧了編輯的保真度、多樣性和可控性。

4.  **卓越的性能**：AREdit在性能上取得了巨大突破。在A100 GPU上處理一張1K解析度的圖像，首次運行僅需2.5秒，後續編輯更是快至**1.2秒**，比現有的SOTA方法快約**9倍**，在定量和定性評估中均達到甚至超越了基於diffusion和rectified flow的複雜方法。

![AREdit編輯效果展示](assets/aredit_figure1_examples.webp)
*圖1：AREdit能夠高效處理多種編輯任務，包括物體移除、添加、屬性修改和風格變換，同時保持非編輯區域的高度保真。*

---

## 技術方法詳解

AREdit的整體框架建立在預訓練的VAR模型**Infinity-2B**之上。VAR模型的核心思想是將圖像生成過程建模為一個多尺度的自回歸預測任務。

### 1. VAR模型基礎

VAR模型包含一個視覺tokenizer（編碼器`ℰ`和解碼器`D`）和一個Transformer。圖像`I`首先被編碼為多尺度的離散殘差序列 `(R₁, R₂, ..., Rₖ)`。Transformer則以自回歸的方式，根據之前的殘差序列和文本嵌入`Ψ(t)`來預測下一個尺度的殘差：

$$ p(R_1, ..., R_K) = \prod_{k=1}^{K} p(R_k | R_1, ..., R_{k-1}, \Psi(t)) $$

其中，`Rₖ`代表第`k`個尺度的殘差，`Ψ(t)`是來自文本編碼器（如Flan-T5）的文本嵌入。這種coarse-to-fine的生成方式為精確控制提供了可能。

### 2. 隨機性緩存 (Randomness Caching)

這是AREdit實現training-free編輯的基石。在編輯前，模型對原始圖像和源提示詞`t_S`進行一次前向傳播，並將每一尺度`k`的預測概率分佈`P_k`和最終採樣的二進制位元標籤`R_k`存儲在一個緩存隊列`P_queue`和`R_queue`中。這個過程相當於“記錄”了原始圖像的生成路徑。

### 3. 自適應細粒度遮罩與Token重組

在進行編輯時，AREdit根據目標提示詞`t_T`重新進行生成，但此時會利用緩存信息進行精確控制。

- **低頻信息保留**：對於較早的生成步驟（`k ≤ γ`，`γ`為超參數），直接重用緩存中的`R_k`。這一步確保了圖像的整體佈局、顏色和結構等低頻信息得以保留。

- **高頻信息編輯**：對於後續步驟（`k > γ`），模型會計算一個**自適應遮罩 `M_k`**。該遮罩的計算基於當前步驟下，由源提示詞`t_S`（從緩存中讀取）和目標提示詞`t_T`（重新計算）分別得到的概率分佈`P_k`和`P_k_tgt`之間的差異。這個差異（例如，通過KL散度衡量）揭示了哪些像素區域需要被修改。

$$ M_k = \text{CalculateMask}(P_k, P_{k}^{\text{tgt}}) $$

- **Token重組**：最後，利用遮罩`M_k`對新的採樣結果和緩存的舊結果進行融合：

$$ R_{k}^{\text{final}} = M_k \odot R_{k}^{\text{new}} + (1 - M_k) \odot R_{k}^{\text{cached}} $$

其中`R_k_new`是根據`P_k_tgt`新採樣的token。這個過程確保了只有被遮罩標記的區域會被更新，從而實現了精確的局部編輯。

![AREdit框架示意圖](assets/aredit_figure2_framework.webp)
*圖2：AREdit的整體框架。通過緩存機制和自適應遮罩，模型能夠在保留原始圖像結構的基礎上，精準地融入新的文本指令。*

---

## 實驗結果與性能

AREdit在多個公開基準測試中展現了其卓越的性能，能夠靈活應對各種編輯任務，包括：

- **物體添加/移除**：在圖像中無縫添加或刪除物體，且背景自然。
- **屬性修改**：改變物體的顏色、材質、表情等屬性。
- **風格變換**：將圖像轉換為不同的藝術風格，如“3D迪士尼卡通風格”。

與現有的SOTA方法（如MasaCtrl, PhotoGuard, RF-Solver）相比，AREdit在保持更高圖像保真度（LPIPS、PSNR等指標更優）的同時，推理速度實現了數量級的提升。

## 個人評價與意義

AREdit不僅僅是一項技術創新，更可能引領圖像編輯領域進入一個新的發展階段。它巧妙地將VAR模型的生成能力與無需訓練的編輯思想相結合，完美地解決了當前主流方法的核心痛點。

- **對研究的啟示**：這項工作證明了自回歸模型在圖像編輯任務上的巨大潛力。其“緩存-比較-融合”的思路非常優雅，為未來基於生成模型的編輯任務提供了全新的視角。特別是對於追求`training-free`和`zero-shot`能力的研究者來說，AREdit提供了一個極具價值的參考框架。

- **對應用的價值**：其驚人的推理速度和高保真度的編輯效果，使其具備了大規模商業應用的潛力。無論是專業設計師的創作工具，還是普通用戶的日常照片美化，AREdit都能提供前所未有的高效和便捷體驗。

總體而言，AREdit是一篇構思巧妙、實驗紮實、效果驚艷且具有開創性意義的工作。它不僅激發了我們對VAR模型應用邊界的想像，也為實現更加智能、高效、可控的AIGC內容創作鋪平了道路。

---

### 參考文獻

[1] Wang, Y., Guo, L., Li, Z., Huang, J., Wang, P., Wen, B., & Wang, J. (2025). *Training-Free Text-Guided Image Editing with Visual Autoregressive Model*. arXiv preprint arXiv:2503.23897.
