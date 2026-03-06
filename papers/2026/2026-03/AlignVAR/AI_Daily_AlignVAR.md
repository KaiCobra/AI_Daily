# AI Daily：AlignVAR - 實現全域一致性的視覺自回歸圖像超解析度

**日期：** 2026年03月06日

**論文標題：** AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution

**作者：** Cencen Liu, Dongyang Zhang, Wen Yin, Jielei Wang, Tianyu Li, Ji Guo, Wenbo Jiang, Guoqing Wang, Guoming Lu

**來源：** CVPR 2026 Findings

**連結：** [https://arxiv.org/abs/2603.00589](https://arxiv.org/abs/2603.00589)

---

### 論文摘要

視覺自回歸（Visual Autoregressive, VAR）模型作為一種新興的圖像生成方法，透過「次級尺度預測」（next-scale prediction）實現了穩定的訓練過程、非迭代式的快速推理以及高保真度的圖像合成，在圖像生成領域展現了巨大潛力。這也促使研究者們開始探索將VAR模型應用於圖像超解析度（Image Super-Resolution, ISR）任務。然而，將VAR直接應用於ISR面臨兩大挑戰：首先是**局部偏向的注意力機制（locality-biased attention）**，這會導致生成圖像的空間結構破碎；其次是**僅依賴殘差的監督訊號（residual-only supervision）**，這會讓誤差在不同尺度間累積，嚴重損害重建影像的全域一致性。

為了解決這些問題，本文提出了**AlignVAR**，一個專為ISR設計、旨在實現全域一致性的視覺自回歸框架。AlignVAR包含兩個關鍵元件：

1.  **空間一致性自回歸（Spatial Consistency Autoregression, SCA）**：此模組透過一個自適應遮罩（adaptive mask）來重新加權注意力，使其更關注與結構相關的區域，從而減輕過度的局部性偏好，並增強長距離依賴關係。
2.  **層級一致性約束（Hierarchical Consistency Constraint, HCC）**：此模組在每個尺度上都增加對完整重建結果的監督，而不僅僅是殘差。這有助於及早發現並修正跨尺度累積的偏差，從而穩定由粗到精的細化過程。

大量的實驗證明，AlignVAR能夠持續提升重建影像的結構連貫性與感知真實度。與主流的基於擴散模型的方法相比，AlignVAR在推理速度上提升了超過10倍，參數數量也減少了近50%，為高效的圖像超解析度任務建立了一個新的典範。

---

### 核心創新

AlignVAR的核心在於解決了現有VAR模型應用於ISR時的兩大根本性矛盾：**空間不一致性（Spatial Inconsistency）**和**層級不一致性（Hierarchical Inconsistency）**。

#### 1. 空間一致性自回歸 (SCA)

傳統VAR模型中的自注意力機制有強烈的局部偏好，注意力權重幾乎完全集中在相鄰區域（如下圖Figure 2所示），限制了模型整合全域上下文的能力，導致空間上不連續的偽影，例如破碎的紋理和結構扭曲。

![AlignVAR注意力分佈圖](asset/alignvar_fig2_attention_map.webp)
*Figure 2: VARSR（左）與AlignVAR（右）的注意力分佈比較。VARSR的注意力高度局部化，而AlignVAR透過SCA捕捉了更廣泛的上下文依賴。*

SCA透過引入一個**結構感知（structure-aware）**的調節機制來解決這個問題。它利用從低解析度輸入中提取的結構線索（如邊緣），生成一個自適應遮罩，引導模型關注結構上相關的遠距離區域，而不僅僅是局部鄰域。這使得模型能夠聚合長距離上下文，保持空間連續性。

![空間不一致性導致的紋理與結構扭曲](asset/alignvar_fig3_spatial_inconsistency.webp)
*Figure 3: 空間不一致性導致的紋理不連續與結構扭曲問題。*

#### 2. 層級一致性約束 (HCC)

VAR模型的「次級尺度預測」範式中，每個尺度的預測都基於前一個（較粗糙）尺度的不完美輸出。這種僅依賴殘差的監督方式，會讓微小的預測誤差在尺度間傳播並被放大，導致最終重建影像出現顏色偏移和結構錯位等問題（如下圖Figure 4所示）。

![層級不一致性導致的顏色偏移與結構錯位](asset/alignvar_fig4_hierarchical_inconsistency.webp)
*Figure 4: 層級不一致性導致的顏色偏移與結構錯位問題。*

HCC透過在每個尺度上增加對**完整潛在表示（full-scale latent representation）**的監督來解決這個問題，而不僅僅是監督殘差。這種方式讓模型在每個尺度都能校準與真實影像的偏差，從而抑制誤差的累積，穩定整個由粗到精的生成過程。

---

### 方法論

AlignVAR的整體架構如下圖所示。它由SCA和HCC兩個互補的元件組成，協同工作以提升生成影像的品質。

![AlignVAR整體架構圖](asset/alignvar_fig5_overall_architecture.webp)
*Figure 5: AlignVAR整體架構圖。*

1.  **SCA模組**：首先，使用拉普拉斯濾波器從低解析度影像中提取結構引導（Structural Guidance）。然後，一個輕量級的遮罩生成器（Mask Generator）會結合自回歸的徵與結構引導，預測出一個空間調製場（spatial modulation field）。這個調製場會被用來重新加權特徵，使得模型能夠關注結構上重要的區域。

2.  **HCC模組**：在訓練過程中，除了計算預測殘差與真實殘差之間的交叉熵損失（CE Loss）外，HCC還會計算每個尺度下累積預測的潛在表示與真實潛在表示之間的L2損失（HCC Loss）。這迫使模型在每個層級都與真實影像保持一致。

總的訓練目標是這兩個損失的加權和：

`L_total = L_CE + λ * L_HCC`

透過這種聯合優化，AlignVAR能夠在每個尺度上生成空間上連貫的預測，並在整個重建過程中保持層級上的一致性。

---

### 實驗結果

AlignVAR在多個合成與真實世界的數據集上進行了評估，並與多種基於GAN和擴散模型的SOTA方法進行了比較。結果顯示，AlignVAR在多個感知指標（如LPIPS, DISTS, FID, MANIQA, CLIPIQA, MUSIQ）上都取得了領先或具有競爭力的表現，尤其是在FID（Fréchet Inception Distance）指標上，顯著優於其他方法，證明其生成的影像在分佈上與真實影像更為接近。

| **方法** | **PSNR ↑** | **SSIM ↑** | **LPIPS ↓** | **FID ↓** | **MANIQA ↑** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BSRGAN | 24.42 | 0.6164 | 0.3511 | 50.99 | 0.3547 |
| Real-ESRGAN | 24.30 | 0.6324 | 0.3267 | 44.34 | 0.3756 |
| SwinIR | 23.77 | 0.6186 | 0.3910 | 44.45 | 0.3411 |
| StableSR | 23.26 | 0.5670 | 0.3228 | 28.32 | 0.4173 |
| DiffBIR | 23.49 | 0.5568 | 0.3638 | 34.55 | 0.4598 |
| VARSR | 24.41 | 0.6189 | 0.2985 | 28.64 | 0.4137 |
| **AlignVAR** | **24.35** | **0.6021** | **0.2955** | **25.71** | **0.4665** |

*Table 1: 在DIV2K-Val數據集上的量化比較。AlignVAR在多個感知指標上表現出色。*

更重要的是，AlignVAR在保持高品質生成的同時，其推理速度遠超基於擴散模型的方法，參數效率也更高，這使其在實際應用中極具潛力。

---

### 結論

AlignVAR透過引入**空間一致性自回歸（SCA）**和**層級一致性約束（HCC）**，成功地解決了現有視覺自回歸模型在應用於圖像超解析度任務時所面臨的空間與層級不一致性問題。該框架不僅顯著提升了重建影像的結構連貫性和感知品質，還保持了VAR模型高效推理的優點。AlignVAR的提出，為開發高效且高品質的生成式圖像超解析度模型提供了一個全新的、有前景的研究方向。

---

### 參考文獻

[1] Tian, K., Jiang, Y., Yuan, Z., Peng, B., Luan, F., & Liu, Z. (2024). Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction. In *Advances in Neural Information Processing Systems*.

[2] Qu, Y., Chen, J., Wang, Z., Liu, J., & Lu, T. (2025). Visual Autoregressive Modeling for Image Super-Resolution. *arXiv preprint arXiv:2501.18993*.

[3] Chen, C., et al. (2025). Adversarial Diffusion Compression for Real-World Image Super-Resolution. In *CVPR*.

[4] Wang, Z., et al. (2023). SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution. *arXiv preprint arXiv:2311.16518*.


---

> *以下內容整合自另一版本的報告*

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
