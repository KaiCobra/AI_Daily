# AI Daily: PixelRush - 20秒產出4K高解析度圖像的單步擴散模型

**論文標題：** PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion
**論文連結：** [https://arxiv.org/abs/2602.12769](https://arxiv.org/abs/2602.12769)
**發表會議：** CVPR 2026
**作者：** Hong-Phuc Lai, Phong Nguyen, Anh Tran (Qualcomm AI Research)

---

### 總結

PixelRush 是一篇發表於 CVPR 2026 的論文，提出了一個無需訓練（training-free）的框架，旨在解決現有擴散模型在生成高解析度圖像時速度過慢的問題。現有的方法通常需要數分鐘才能生成一張4K圖像，而 PixelRush 透過創新的 **部分反轉（Partial Inversion）**、**無縫融合（Seamless Blending）** 和 **噪聲注入（Noise Injection）** 技術，成功地將這個過程縮短到約20秒，實現了10到35倍的速度提升，同時在圖像品質上超越了當前的 SOTA 方法。

### 核心問題

預訓練的擴散模型（如 SDXL）雖然能生成高品質圖像，但其生成解析度受限於訓練時的數據集解析度（如 1024x1024）。直接在推理時生成更高解析度的圖像會導致嚴重的品質下降與結構失真。為了解決這個問題，現有的 training-free 方法主要分為兩類：

1.  **直接推理法（Direct Inference）**：如 ScaleCrafter 和 FreeScale，透過修改模型內部結構（如卷積層的 dilation rate）來擴大感受野。這類方法雖然能緩解物體重複的問題，但記憶體消耗巨大，且容易引入不自然的紋理。
2.  **基於補丁的方法（Patch-based）**：如 MultiDiffusion 和 DemoFusion，將高解析度潛在空間分割成小塊（patches）進行處理，雖然解決了記憶體瓶頸，但為了保證生成品質，它們依賴於一個完整且漫長的多步逆向擴散過程（通常需要50步或更多），導致生成速度極其緩慢。

這兩種方法共同的痛點是 **推理速度慢**，使得高解析度圖像的實際應用變得不切實際。

### PixelRush 的創新方法

PixelRush 的核心洞見在於：對於一個已經具備基本結構的粗糙高解析度圖像而言，完整的逆向擴散過程是多餘且浪費計算資源的。模型的早期去噪步驟主要用於生成低頻的全局結構，而這在粗糙圖像中已經存在。因此，PixelRush 專注於高頻細節的合成，並提出了三個關鍵技術：

#### 1. 部分反轉策略（Partial Inversion Strategy）

傳統方法會將粗糙的高解析度圖像（透過插值放大）完全變成高斯噪聲（t=T），然後再從頭開始去噪。PixelRush 則不同，它使用 DDIM Inversion 將粗糙圖像僅反轉到一個 **中間噪聲水平（t=K, K < T）**。這一步保留了圖像的全局結構，並使得後續可以利用高效的 **少步擴散模型（few-step models）**，如 SDXL-Turbo，僅需一步或幾步就能完成高頻細節的精煉。這也是其速度大幅提升的關鍵。

![Training-free high-resolution pipeline synthesize images hierarchically.](asset/PixelRush_partial_inversion_insight.webp)
*圖：論文指出，逆向擴散過程是分層的，早期（t值大）生成全局結構，晚期（t值小）添加細節。PixelRush 的部分反轉策略正是基於此洞見，跳過了冗餘的全局結構生成階段。*

#### 2. 無縫融合演算法（Seamless Blending Algorithm）

在少步或單步生成中，傳統的補丁平均融合方法會失效，在補丁邊界產生明顯的棋盤格偽影。PixelRush 受到圖像羽化（image feathering）技術的啟發，使用高斯濾波器對補丁的重疊區域生成一個平滑的權重蒙版，使得像素權重從一個補丁的中心向邊緣平滑過渡到另一個補丁，從而完美地消除了邊界偽影。

#### 3. 噪聲注入機制（Noise Injection Mechanism）

少步生成模型為了快速收斂，其更新步長較大，容易導致生成結果過於平滑，缺乏高頻細節。為了解決這個問題，PixelRush 在去噪的最後階段，透過球面插值（slerp）將模型預測的噪聲與一個隨機噪聲進行微小的混合。這種方法能夠有效地「擾動」數據分佈，促使模型生成更多高頻細節，從而解決過平滑問題。

### 方法總覽

PixelRush 的整體流程是一個兩階段系統：

1.  **基礎生成階段（Base Generation）**：使用一個標準的擴散模型（如 SDXL）生成一張低解析度（如 1024x1024）的基礎圖像。
2.  **級聯上採樣階段（Cascade Upsampling）**：這是一個逐步放大的過程。在每一個放大步驟中（例如從 2K 到 4K），系統會執行以下操作：
    *   **上採樣**：將當前解析度的圖像在像素空間中透過雙三次插值（bicubic interpolation）放大。
    *   **編碼**：將放大後的粗糙圖像透過 VAE Encoder 轉換為粗糙的潛在表示（Coarse Latent）。
    *   **精煉**：應用 PixelRush 的核心精煉流程（部分反轉 -> 少步去噪 -> 融合與噪聲注入），得到高品質的潛在表示（High Quality Latent）。
    *   **解碼**：將精煉後的潛在表示透過 VAE Decoder 轉換回像素空間，得到最終的高解析度圖像。

![An overview of two-stage system with for high-resolution generation with Cascade Upsampling.](asset/PixelRush_pipeline_overview.webp)
*圖：PixelRush 的兩階段高解析度生成流程。*

![The PixelRush Refinement Stage.](asset/PixelRush_refinement_stage.webp)
*圖：PixelRush 核心的精煉階段流程圖。*

### 實驗結果

PixelRush 在定量和定性比較中均表現出色。在定量指標上，無論是 2K 還是 4K 解析度，PixelRush 的 FID 分數（越低越好）和 IS 分數（越高越好）都優於現有的 SOTA 方法，如 FreeScale 和 DemoFusion。最引人注目的是其生成速度，生成一張 4K 圖像僅需約20秒，而其他方法則需要323秒到680秒不等。

| 方法 | FID (↓) - 4K | IS (↑) - 4K | 時間 (秒) - 4K |
| :--- | :---: | :---: | :---: |
| SDXL-DI | 153.53 | 7.32 | 247 |
| FouriScale | 98.97 | 8.54 | 680 |
| DemoFusion | 74.75 | 12.57 | 507 |
| FreeScale | 58.28 | 13.35 | 323 |
| **PixelRush*** | **54.67** | **13.75** | **20** |

*表：4K 解析度下的定量比較。PixelRush* 使用 SDXL-Turbo 作為精煉模型。*

在定性比較中，PixelRush 能夠有效避免其他方法的常見問題，例如 DemoFusion 的物體重複（如多個龍頭）和 FreeScale 的不自然紋理。其生成的圖像在細節上更銳利、更自然，且結構更一致。

### 結論

PixelRush 成功地打破了高解析度圖像生成中「品質」與「速度」不可兼得的困境。它首次證明了在 training-free 的 patch-based 框架下，使用少步甚至單步擴散模型是完全可行的。透過創新的部分反轉、無縫融合和噪聲注入技術，PixelRush 在大幅提升生成速度的同時，也樹立了新的品質標竿，為高解析度生成式 AI 的實際應用鋪平了道路。

---

### 相關研究

*   **SDXL-Turbo & Adversarial Diffusion Distillation (ADD):** PixelRush 的速度優勢很大程度上得益於像 SDXL-Turbo 這樣的少步擴散模型。這類模型是透過一種名為「對抗性擴散蒸餾」（ADD）的技術訓練而來，能夠將一個龐大的多步擴散模型（老師）蒸餾成一個僅需1-4步就能生成高質量圖像的輕量模型（學生）。
*   **MultiDiffusion & DemoFusion:** 這是 patch-based 方法的代表。MultiDiffusion 首次提出了將高解析度潛在空間分割成小塊並行處理的想法，而 DemoFusion 在此基礎上進行了改進，但兩者都受限於多步去噪帶來的緩慢速度。
*   **FreeScale & ScaleCrafter:** 這是 direct inference 方法的代表，它們在推理時直接修改模型以適應更大的解析度，但受記憶體和偽影問題的困擾。

