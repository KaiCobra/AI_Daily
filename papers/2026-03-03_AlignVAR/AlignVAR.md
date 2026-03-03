# AI Daily: AlignVAR - 實現全域一致性的視覺自回歸圖像超解析度

**Date:** March 03, 2026

**Paper:** [AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution](https://arxiv.org/abs/2603.00589)

**Authors:** Cencen Liu, Dongyang Zhang, Wen Yin, Jielei Wang, Tianyu Li, Ji Guo, Wenbo Jiang, Guoqing Wang, Guoming Lu

---

---

## 總結

這篇來自電子科技大學的論文 **AlignVAR**，被 CVPR 2026 Findings 接受，旨在解決視覺自回歸（Visual Autoregressive, VAR）模型在圖像超解析度（Image Super-Resolution, ISR）任務中的兩個核心挑戰：**空間不一致性（Spatial Inconsistency）**和**層級不一致性（Hierarchical Inconsistency）**。現有的 VAR 方法如 VARSR [1]，雖然展示了潛力，但其自注意力機制的**局部性偏誤（locality-biased attention）**會導致生成圖像的空間結構破碎，而僅依賴**殘差監督（residual-only supervision）**則會讓誤差在不同尺度間累積放大，最終損害重建圖像的全域一致性。

為此，AlignVAR 提出了一個專為 ISR 設計的全域一致性視覺自回歸框架，其包含兩個關鍵組件：

1.  **空間一致性自回歸（Spatial Consistency Autoregression, SCA）**：透過一個自適應遮罩（adaptive mask）來重新加權注意力，使其關注結構上相關的區域，從而減輕過度的局部性並增強長距離依賴。
2.  **層級一致性約束（Hierarchical Consistency Constraint, HCC）**：在每個尺度上增加對完整重建的監督，而不僅僅是殘差，從而及早暴露累積的偏差，穩定從粗到細的優化過程。

大量實驗證明，AlignVAR 在提升結構連貫性和感知真實度方面，顯著優於現有的生成方法。與領先的擴散模型相比，其推論速度快了10倍以上，參數數量減少了近50%，為高效的ISR建立了一個新的典範。

---

## 核心概念與方法

AlignVAR 的核心在於透過 SCA 和 HCC 兩個模組，分別從**尺度內（intra-scale）**和**尺度間（inter-scale）**兩個維度來解決一致性問題。

### 空間一致性自回歸 (SCA)

為了解決 VAR 模型中自注意力機制的局部性偏誤，SCA 引入了一個**結構感知（structure-aware）**的調節機制。它首先利用拉普拉斯算子從低解析度輸入中提取結構指導圖，該圖突顯了邊緣和紋理等高頻細節。接著，一個輕量級的 MLP 遮罩生成器會根據自回歸的 token 和結構指導圖，預測一個空間調製場（spatial modulation field）。

這個調製場會對 token 進行重新加權，增強結構清晰區域的權重，同時抑制平滑或不確定區域的權重。如此一來，模型在進行注意力計算時，能夠優先關注結構上相關的遠距離區域，而不是僅僅局限於相鄰的 token，從而捕捉更廣泛的上下文依賴，提升空間一致性。

### 層級一致性約束 (HCC)

傳統的 VAR 模型僅在每個尺度上監督預測的**殘差 token**，這使得來自較粗尺度的預測誤差會被傳播並放大到較細的尺度。為了解決這個問題，HCC 引入了**全尺度（full-scale）**的監督。

具體來說，除了監督殘差外，HCC 還會將每個尺度預測的**累積（cumulative）**潛在表示，與對應的真實潛在表示進行對齊。這個真實潛在表示是透過將高解析度圖像的 VAE 潛在特徵下採樣到各個尺度得到的。透過這種方式，HCC 在訓練過程中為每個尺度提供了更強的監督信號，迫使模型在早期就修正偏差，從而抑制了誤差的累積，確保了從粗到細的重建過程中的層級一致性。

### 訓練目標

AlignVAR 的最終訓練目標結合了傳統的交叉熵損失（用於監督殘差 token）和新提出的 HCC 損失（用於監督全尺度潛在表示），並透過一個超參數 λ 來平衡兩者。這種聯合優化策略確保了模型在學習局部細節的同時，也能保持全域結構的連貫性。

---

## 實驗結果

實驗在合成資料集（DIV2K-Val）和真實世界資料集（DRealSR, RealSR）上進行，並與多種基於 GAN、擴散模型和 VAR 的 SOTA 方法進行了比較。

### 量化比較

在 DIV2K-Val 資料集上，AlignVAR 在所有感知指標上均優於 GAN 和擴散模型，取得了最低的 FID（25.71）和 LPIPS（0.2955）。在 RealSR 資料集上，與其前身 VARSR 相比，AlignVAR 在 MUSIQ 和 CLIPIQA 等無參考指標上取得了顯著提升，證明了其在真實世界場景中的有效性。

### 定性比較

視覺比較結果顯示，GAN 模型容易產生局部失真和鋸齒狀邊緣，而擴散模型則可能產生幻覺紋理並削弱結構對齊。相比之下，AlignVAR 能夠重建出更清晰的邊緣、更連貫的紋理和更自然的色彩過渡，更符合人類的視覺感知。

### 消融實驗

- **SCA 的有效性**：移除 SCA 會輕微提升保真度指標（PSNR/SSIM），但會顯著降低感知分數。這證明了 SCA 在增強空間一致性和平衡保真度與感知品質方面的重要性。
- **HCC 的有效性**：引入 HCC 能持續改善保真度和感知指標，表明對齊全尺度潛在表示能有效抑制誤差累積，增強尺度間的一致性。

### 複雜度比較

AlignVAR 的推論速度極快，重建一張 512x512 的圖像僅需 0.43 秒，比主流擴散模型快 10 倍以上，同時參數數量也減少了近 50%。

---

## 討論與結論

AlignVAR 成功地識別並解決了現有 VAR 模型在 ISR 任務中的兩個核心痛點：空間局部性偏誤和層級誤差累積。透過創新的 SCA 和 HCC 模組，AlignVAR 在保持高效率的同時，顯著提升了重建圖像的全域一致性和感知品質，為基於 VAR 的圖像超解析度研究提供了一個新的視角和強有力的基準。

這項工作證明，透過精巧的設計，自回歸模型完全有潛力在生成品質和效率上超越主流的擴散模型，成為下一代高效生成模型的有力競爭者。

---

## 參考文獻

[1] Qu, Y., Yuan, K., Hao, J., Zhao, K., Xie, Q., Sun, M., & Zhou, C. (2025). Visual Autoregressive Modeling for Image Super-Resolution. *arXiv preprint arXiv:2501.18993*.
