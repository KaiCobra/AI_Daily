# AI Daily: 用遮罩位元建模突破自回歸圖像生成的極限

> 論文標題：Autoregressive Image Generation with Masked Bit Modeling
> 
> 論文連結：[https://arxiv.org/abs/2602.09024](https://arxiv.org/abs/2602.09024)
> 
> 程式碼連結：[https://github.com/amazon-far/BAR](https://github.com/amazon-far/BAR)
> 
> 發表單位：Google
> 
> 發表時間：2026年2月9日
> 
> 關鍵字：Autoregressive, Masked Bit Modeling, Discrete Tokenizer, Bit Budget, Scalable Generation

## 核心貢獻：離散方法的逆襲，gFID 0.99 刷新 SOTA

長期以來，視覺生成領域一直由連續管道（如擴散模型、GAN）主導，而離散方法（如自回歸模型）則因其在重建品質和擴展性上的限制而處於次要地位。然而，來自 Google 的最新研究 **BAR (masked Bit AutoRegressive modeling)** 徹底顛覆了這一局面。該論文不僅系統性地揭示了離散與連續方法性能差距的根源，更提出了一種創新的 **遮罩位元建模 (Masked Bit Modeling, MBM)** 框架，**首次在 ImageNet-256 上實現了 0.99 的 gFID 分數**，超越了所有已知的連續和離散生成模型，為自回歸圖像生成開闢了新的篇章。

研究的核心洞察在於，離散 tokenizer 的性能瓶頸並非其內在缺陷，而是**潛在空間中分配的總位元數（Bit Budget）不足**所致。透過擴大 codebook 的大小（即增加位元預算），離散方法完全有能力匹敵甚至超越連續方法。然而，傳統的自回歸模型在處理大詞彙時會面臨「詞彙擴展問題」，導致訓練成本過高或性能下降。BAR 提出的 MBM head 巧妙地繞過了這個問題，透過**漸進式生成 token 的組成位元**，實現了對任意大小 codebook 的可擴展支援，同時顯著降低了採樣成本並加快了收斂速度。

## 技術方法：遮罩位元建模 (Masked Bit Modeling)

BAR 的核心是其創新的預測頭設計，它將傳統的單步 token 預測轉化為一個多步的位元生成過程。這使得模型能夠在不犧牲性能的情況下，處理極大規模的離散詞彙表。

### 1. 問題根源：位元預算 (Bit Budget) 決定性能上限

論文首先建立了一個統一的比較框架，用「位元預算」來衡量 tokenizer 的資訊容量。對於一個將圖像壓縮到 $\frac{H}{f} \times \frac{W}{f}$ 潛在空間的 tokenizer，其位元預算計算方式如下：

- **離散 Tokenizer (codebook 大小為 C)**：
  $$B_{discrete} = \frac{H}{f} \times \frac{W}{f} \times \log_2 C$$

- **連續 Tokenizer (潛在維度為 D)**：
  $$B_{continuous} = \frac{H}{f} \times \frac{W}{f} \times 16D$$

實驗證明，一旦給予足夠的位元預算，離散 tokenizer 的重建品質就能超越連續方法（如 SD-VAE），這也揭示了擴大 codebook 的重要性。

![BAR 詞彙擴展比較](assets/bar_vocabulary_scaling_comparison.webp)
*圖1：隨著位元預算的增加，離散 tokenizer (BAR-FSQ) 的重建誤差持續下降，並最終超越了連續方法。*

### 2. 解決方案：Masked Bit Modeling (MBM) Head

傳統自回歸模型直接預測下一個 token 的索引，當 codebook 很大時（例如 $2^{32}$），這相當於一個有數十億類別的分類問題，計算上不可行。BAR 則將這個過程分解，不再預測整個 token，而是預測組成該 token 的位元。

MBM Head 的工作流程如下圖所示：

![BAR 框架架構](assets/bar_framework_architecture.webp)
*圖2：BAR 框架概覽。(a) 自回歸 Transformer 負責上下文建模；(b) 傳統線性頭在小詞彙下有效，但無法擴展；(c) 直接預測位元的 bit head 雖然可擴展，但性能較差；(d) 提出的 MBM head 透過漸進式 unmasking 生成位元，兼顧了可擴展性和生成品質。*

其核心公式如下：

1.  **上下文建模**：自回歸 Transformer $\mathcal{F}$ 根據之前的 tokens $\{x_1, ..., x_{i-1}\}$ 生成上下文表示 $z_{i-1}$。
    $$z_{i-1} = \mathcal{F}(x_1, x_2, ..., x_{i-1})$$

2.  **位元生成**：MBM Head $G_\theta$ 以 $z_{i-1}$ 為條件，透過一個漸進式的 unmasking 過程來預測下一個 token $x_i$ 的所有位元。
    $$\hat{x}_i = G_\theta(\text{Mask}_{M}(x_i) \mid z_{i-1}, M)$$

    其中 $\text{Mask}_{M}$ 是一個遮罩函數，它會隨機隱藏一部分位元，讓模型來預測它們。這個過程類似於 BERT 中的遮罩語言模型，但在位元層級上操作。

3.  **訓練目標**：優化預測位元與真實位元之間的交叉熵損失。
    $$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} \text{CrossEntropy}_{bit}(x_i, \hat{x}_i)$$

這種設計將一個巨大的分類問題分解為一系列小的二元分類問題，極大地提高了訓練的穩定性和效率，使得模型能夠輕鬆擴展到極大的 codebook 尺寸。

## 實驗結果：全面超越 SOTA

BAR 在 ImageNet 256x256 的生成任務上進行了廣泛評估，結果令人驚艷。

### 1. 生成品質：gFID 0.99

如下表所示，BAR-L 模型在僅使用 1.4B 參數的情況下，達到了 **0.99 的 gFID**，不僅超越了所有離散自回歸模型（如 VAR, RAR），也擊敗了所有頂級的擴散模型和流匹配模型（如 DiT, RAE, MDT）。

![ImageNet 比較表格](assets/bar_imagenet_comparison_table.webp)
*表1：在 ImageNet 256x256 上的生成結果比較。BAR-L 在 gFID 指標上刷新了紀錄，達到了 0.99。*

### 2. 採樣效率

除了生成品質，BAR 在採樣速度上也表現出色。高效的 BAR-B/2 變體在保持高品質（gFID 1.35）的同時，採樣速度達到了 **150.52 images/sec**，遠超其他同類模型。即使是最高品質的 BAR-L 模型，其速度也比 RAE 等頂級擴散模型快了近一倍。

### 3. 消融研究

論文進行了詳細的消融實驗，驗證了 MBM head 設計的優越性。實驗表明，相較於傳統的線性頭或直接預測位元的 bit head，MBM head 在各種 codebook 尺寸下都能實現最佳的性能與可擴展性權衡。

![消融研究表格](assets/bar_ablation_studies.webp)
*圖3：消融研究結果。MBM head 在 masking 策略、head 尺寸和採樣策略等多方面都展現出其設計的魯棒性和優越性。*

## 相關研究背景

這項工作建立在離散視覺表示學習的基礎之上，特別是 VQ-VAE、VQGAN 以及最近的 lookup-free 量化方法（如 FSQ）。同時，它也與一系列試圖擴展自回歸模型詞彙表的研究（如 MaskBit, Infinity）形成對話。BAR 的獨特之處在於它沒有直接預測 token 索引，而是透過預測位元來間接生成 token，從而優雅地解決了詞彙擴展的難題。

## 個人評價與意義

BAR 無疑是近年來圖像生成領域最具突破性的研究之一。它不僅在性能指標上取得了 SOTA，更重要的是，它從根本上挑戰了「離散方法不如連續方法」的普遍看法，為自回歸模型這一經典路線注入了新的活力。

**其核心意義在於**：

1.  **理論層面**：清晰地指出了「位元預算」是連接離散與連續方法性能的橋樑，為未來的 tokenizer 設計提供了明確的指導方向。
2.  **技術層面**：提出的 MBM head 是一個極具擴展性的優雅解決方案，巧妙地將大詞彙預測問題分解，有望被廣泛應用於多模態、影片生成等需要巨大離散詞彙表的領域。
3.  **實踐層面**：在性能和效率上雙雙取得突破，證明了自回歸模型不僅可以在品質上媲美甚至超越擴散模型，還能保持更快的採樣速度，這對於實際應用部署具有重大價值。

總而言之，BAR 的出現可能會引發圖像生成領域的一次範式轉移，促使研究界重新審視和投入離散生成方法的潛力。我們可以期待，未來將有更多基於位元建模的強大生成模型問世。
