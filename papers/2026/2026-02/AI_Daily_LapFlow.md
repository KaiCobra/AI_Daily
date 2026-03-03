# AI Daily: LapFlow - 當拉普拉斯金字塔遇上流匹配，實現高效多尺度圖像生成

**日期:** 2026年02月25日

**作者:** Manus AI

**論文標題:** Laplacian Multi-scale Flow Matching for Generative Modeling [1]

**來源:** arXiv, 2026年02月23日 (已被 ICLR 2026 接受)

---

### 摘要

隨著生成模型對更高解析度和更複雜內容的需求不斷增長，模型的可擴展性成為一個巨大的挑戰。傳統的單尺度模型（如 DiT）在處理高解析度圖像時，計算成本高昂。為了解決這個問題，來自喬治亞理工學院的研究者們提出了一個名為 **Laplacian Multi-scale Flow Matching (LapFlow)** 的新型生成模型框架。該論文已被 ICLR 2026 會議接受。

LapFlow 的核心思想是將圖像分解為**拉普拉斯金字塔（Laplacian Pyramid）**的多尺度殘差表示，並通過一個創新的**混合變壓器（Mixture-of-Transformers, MoT）**架構，以並行的方式處理不同尺度的信息。與需要為每個尺度訓練獨立模型或在尺度間進行複雜橋接的級聯方法不同，LapFlow 使用單一的統一模型，並通過**因果注意力（causal attention）**機制確保從低解析度到高解析度的自然信息流。實驗證明，LapFlow 在 CelebA-HQ 和 ImageNet 數據集上，不僅生成質量超越了現有的單尺度和多尺度流匹配基線，同時還實現了更少的計算量（GFLOPs）和更快的推理速度。

![論文摘要](assets/lapflow_abstract.webp)

### 背景與動機

流匹配（Flow Matching）作為一種新興的生成模型，因其簡單有效的訓練方式和確定性的生成路徑而備受關注。然而，現有的流匹配方法，如 **LFM (Latent Flow Matching)** [2]，大多繼承了擴散模型的單尺度生成範式，即一次性生成完整解析度的圖像。這種方式在生成高解析度圖像（如 1024x1024）時，會導致巨大的計算和內存開銷。

多尺度生成（Multi-scale generation）提供了一個解決方案，它將生成過程分解為從粗到細的多個階段。經典方法如 **LapGAN** [3] 和級聯擴散模型（Cascaded Diffusion Models）證明了這種層級生成的有效性。然而，這些方法通常需要為每個解析度訓練和維護一個獨立的網絡，增加了實現的複雜性。近期的工作如 **Pyramidal Flow** [4] 雖然統一了模型，但在從零開始訓練圖像生成任務上的探索尚不充分。

LapFlow 的動機正是為了解決這些挑戰：設計一個既能利用多尺度優勢，又保持模型和訓練流程簡潔高效的統一框架。

### 核心方法：LapFlow

LapFlow 框架包含三個核心組件：拉普拉斯金字塔分解、帶有因果注意力的混合變壓器（MoT）架構，以及漸進式多階段訓練策略。

#### 1. 拉普拉斯金字塔分解與多尺度流場

LapFlow 首先將圖像分解為一個拉普拉斯金字塔。這意味著圖像被表示為一組不同尺度的殘差圖像。例如，一個三層的金字塔包含一個低解析度的基礎圖像和兩個高頻殘差圖像。生成時，模型只需要學會生成這些殘差，最後通過上採樣和累加即可重建出完整圖像。

![LapFlow 多尺度生成過程](assets/lapflow_figure1_multiscale_process.webp)

如上圖所示，生成過程從一個隨機噪聲開始，遵循一個從粗到細的策略。模型首先在時間區間 [0, T2] 內生成最粗糙的尺度，然後在 [T2, T1] 內生成中等尺度，最後在 [T1, 1] 內生成最精細的尺度。這種設計確保了全局結構首先被確定，然後逐步添加細節。

#### 2. 混合變壓器 (MoT) 與因果注意力

為了用單一模型處理多尺度信息，LapFlow 採用了 **Mixture-of-Transformers (MoT)** [5] 架構。MoT 為每個尺度（專家）分配了獨立的處理模塊（如 QKV 投射、前饋網絡等），但所有尺度共享一個全局的自註意力機制。這種設計兼顧了尺度特異性建模和計算效率。

![LapFlow MoT 架構圖](assets/lapflow_figure2_mot_architecture.webp)

上圖展示了 LapFlow 的核心架構。其關鍵創新在於全局自註意力層中引入了**因果掩碼（Causal Mask）**。這個掩碼限制了信息流動的方向，確保任何一個尺度 `k` 的 token 只能關注到比它更粗糙或與其相同尺度的 token。這強制模型學習從全局結構到局部細節的層級依賴關係，避免了信息的洩漏，從而生成更連貫的圖像。

#### 3. 漸進式多階段訓練

LapFlow 還採用了一種漸進式的訓練策略。在訓練的不同階段，模型會專注於不同尺度的子集。例如，在初始階段，模型只訓練最粗糙的尺度；在後續階段，逐漸加入更高解析度的尺度進行聯合訓練。這種策略將計算資源優先分配給對最終圖像質量貢獻最大的尺度，從而提高了訓練效率。

### 實驗結果與分析

LapFlow 在 CelebA-HQ 和 ImageNet 數據集上進行了廣泛的實驗，並與 LFM、Pyramidal Flow 等 SOTA 方法進行了比較。

**CelebA-HQ 上的主要結果：**

| 解析度 | 方法 | FID ↓ | GFLOPs ↓ | 推理時間 (s) ↓ |
| :--- | :--- | :---: | :---: | :---: |
| 256x256 | LFM [2] | 5.26 | 22.1 | 1.70 |
| 256x256 | Pyramidal Flow [4] | 11.20 | 14.2 | 1.85 |
| **256x256** | **LapFlow (Ours)** | **3.53** | **16.5** | **1.51** |
| 512x512 | LFM [2] | 6.35 | 43.5 | 2.90 |
| **512x512** | **LapFlow (Ours)** | **4.04** | **41.7** | **2.60** |
| 1024x1024 | LFM [2] | 8.12 | 154.8 | 4.20 |
| **1024x1024** | **LapFlow (Ours)** | **5.51** | **148.2** | **3.30** |

從上表可以看出，在所有解析度下，LapFlow 的 FID 分數（越低越好）都顯著優於基線方法，同時計算成本（GFLOPs）和推理時間也更低。

**ImageNet 上的主要結果：**

在類別條件生成的 ImageNet 256x256 任務上，LapFlow 同樣表現出色。使用 DiT-XL/2 作為骨幹網絡時，LapFlow 取得了 **14.38** 的 FID，優於 Pyramidal Flow 的 17.10 和 DiT 的 19.50，並且 GFLOPs 僅為 DiT 的約70%。

**消融研究亮點：**

- **VAE 的選擇**：使用 **EQVAE** [6]（一種保持等變性的 VAE）比標準的 SDVAE 能為 LapFlow 帶來顯著的性能提升，證明了多尺度方法對潛在空間的結構更加敏感。
- **因果掩碼的必要性**：消融實驗證明，因果掩碼是實現最佳性能的關鍵，若無掩碼或僅進行尺度內自註意力，FID 分數會明顯下降。
- **尺度數量的影響**：在 256x256 解析度下，兩層金字塔的效果最好。這表明尺度的選擇需要與圖像的潛在空間大小相匹配。

### 結論與個人反思

LapFlow 巧妙地將拉普拉斯金字塔的經典思想與現代的流匹配和 Transformer 架構相結合，為高解析度圖像生成提供了一個優雅且高效的解決方案。它通過單一的 MoT 模型和因果注意力機制，成功地解決了傳統多尺度方法中模型冗餘和訓練複雜的問題。

這項工作最重要的貢獻在於展示了**如何在統一模型中高效地建模多尺度依賴關係**。因果注意力的引入，確保了從粗到細的生成流程的連貫性，而 MoT 架構則在不犧牲性能的前提下，極大地提高了計算效率。實驗結果令人信服地證明了 LapFlow 在生成質量和效率上的雙重優勢。

對於未來的研究，LapFlow 的思想可以被擴展到其他生成任務，如影片生成和3D建模。其在效率和性能之間的出色平衡，使其成為未來大規模生成模型發展的一個極具潛力的方向。

---

### 參考文獻

[1] Zelin Zhao, Petr Molodyk, Haotian Xue, Yongxin Chen. (2026). Laplacian Multi-scale Flow Matching for Generative Modeling. *arXiv:2602.19461*.

[2] Quoc-An Dao, Binh-Son Hua, et al. (2023). Flow Matching in Latent Space. *arXiv:2307.08698*.

[3] Emily Denton, Soumith Chintala, et al. (2015). Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks. *Advances in Neural Information Processing Systems 28 (NIPS 2015)*.

[4] Jin, Y., et al. (2025). Pyramidal Flow Matching for Efficient Video Generative Modeling. *arXiv:2410.05954*.

[5] Wenbo Liang, Yuzhang Shang, et al. (2024). A Sparse and Scalable Architecture for Multi-Modal Foundation Models. *arXiv:2411.04996*.

[6] Leonidas Kouzelis, et al. (2025). EQ-VAE: Equivariance Regularized Latent Space for Improved Generative Modeling. *International Conference on Machine Learning (ICML 2025)*.
