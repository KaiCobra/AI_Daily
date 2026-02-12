# AI Daily: ResTok - 讓視覺自回歸模型「重拾視覺」，實現高效高保真生成

**發布日期:** 2026年1月21日

> 當自回歸模型主宰了語言生成的領域後，人們自然地將目光投向了更具挑戰性的視覺世界。然而，直接將語言建模的「套路」搬到圖像生成上，卻如同讓一位語言學家去畫油畫，雖能勾勒輪廓，卻總失了幾分神韻。近期，一篇名為《ResTok: Learning Hierarchical Residuals in 1D Visual Tokenizers for Autoregressive Image Generation》的論文，為我們揭示了如何讓視覺自回歸模型「重拾視覺」，在效率與保真度上取得了驚人的突破，完美契合了我們對VAR-based、高效生成和創新架構的探索興趣。

---

## 論文基本資訊

- **論文標題:** ResTok: Learning Hierarchical Residuals in 1D Visual Tokenizers for Autoregressive Image Generation
- **作者:** Xu Zhang, Cheng Da, Huan Yang, Kun Gai, Ming Lu, Zhan Ma
- **發表機構:** 南京大學視覺實驗室 (Vision Lab, Nanjing University), 快手科技 (Kuaishou Technology)
- **發表日期:** 2026年1月7日
- **論文連結:** [https://arxiv.org/abs/2601.03955](https://arxiv.org/abs/2601.03955)
- **開源代碼:** [https://github.com/zhanxu/ResTok](https://github.com/zhanxu/ResTok)

## 核心貢獻：將「視覺先驗」還給視覺生成

當前的視覺自回歸（VAR）模型，為了套用語言模型成熟的自回歸框架，大多將2D圖像「壓平」成1D的token序列。這種做法雖然簡單直接，卻粗暴地拋棄了圖像本身固有的**層次化結構（Hierarchical Structure）**和**空間連續性（Spatial Coherence）**，導致模型難以學習到高質量的視覺表示，生成效果和效率都受到限制。

ResTok的核心思想是**「將視覺的設計原則還給視覺模型」**。它沒有遵循語言模型的扁平化思路，而是創新性地提出了**殘差標記器（Residual Tokenizer, ResTok）**，其貢獻可以總結為以下幾點：

1.  **引入層次化殘差（Hierarchical Residuals）**：ResTok同時為圖像tokens和潛在tokens（latent tokens）建立了層次化的殘差表示。這使得模型能夠在不同尺度上捕捉和融合特徵，從粗略的輪廓到精細的紋理，極大地增強了模型的視覺表示能力。

2.  **語義殘差降低熵**：通過在不同層次間計算語義殘差，ResTok有效避免了信息的冗餘，使得潛在空間的表示更加「緊湊」和「集中」，從而降低了codebook的熵，讓後續的自回歸建模更加容易和穩定。

3.  **層次化自回歸生成器（Hierarchical Autoregressive Generator, HAR）**：為了進一步提升效率，ResTok設計了一種層次化的生成器。它不再是逐個token地生成圖像，而是一次性預測一整個層次的潛在tokens，極大地減少了生成所需的採樣步數。

![ResTok架構圖](asset/restok_figure2_architecture.webp)
*圖一：ResTok的整體架構。它通過(b)中的層次化殘差設計，顯著區別於(a)中傳統的扁平查詢方式，並通過(c)中的殘差合併塊（Residual Merging Block）實現跨尺度信息融合。*

## 技術細節深度解析

ResTok的成功關鍵在於其精巧的架構設計，它將經典視覺網絡（如ResNet）中的核心思想與現代Transformer架構進行了完美融合。

### 1. 殘差標記器 (Residual Tokenizer)

與傳統方法不同，ResTok的編碼器包含了一系列**殘差合併塊（Residual Merging Blocks）**。其流程如下：

- **漸進式池化與合併**：輸入圖像首先被一個CNN編碼器轉換為2D的圖像tokens。隨後，在ViT編碼器中，這些tokens會經過多個殘差合併塊。在每個塊中，一部分tokens會被池化（Pooling）以形成更粗糙、但語義層次更高的表示。

- **殘差計算**：為了避免信息丟失並強化層次間的聯繫，模型會計算較精細層次特徵與上採樣後的較粗糙層次特徵之間的**殘差**。這個殘差捕捉了從一個尺度到下一個尺度的「新增信息」，使得每個層次的表示都更加純粹和高效。

### 2. 層次化自回歸生成器 (HAR Generator)

傳統的自回歸生成器就像一個像素一個像素地畫畫，效率極低。ResTok的HAR生成器則像是一位先畫草圖，再逐步添加細節的畫家。

- **分層預測**：如圖二所示，HAR生成器首先預測最粗糙層次的潛在tokens，然後基於這些tokens作為上下文（Condition），再去預測下一個更精細層次的tokens。這個過程逐層進行，直到生成最高分辨率的細節。

- **效率提升**：由於一次可以生成一個「塊」（一個層次的所有tokens），而不是一個「點」（單個token），其生成速度相比傳統自回歸模型有了指數級的提升。

![HAR生成器架構圖](asset/restok_figure4_har_generator.webp)
*圖二：層次化自回歸生成器（HAR Generator）示意圖。它採用coarse-to-fine的策略，逐層生成潛在tokens，顯著提高了生成效率。*

### 3. 優化策略與損失函數

為了確保模型能夠學習到有意義的層次化表示，ResTok採用了基於預訓練視覺基礎模型（Vision Foundation Model, VF）的**表示對齊（Representation Alignment）**策略。它通過計算ResTok編碼器/解碼器在不同層次的輸出與VF模型（如DINOv3）對應特徵之間的餘弦相似度損失，來引導模型學習。

其核心損失函數可以表示為：

- **視覺基礎模型損失 (VF Loss)**: 
  $$ \mathcal{L}_{vf} = \lambda_{enc} \mathcal{L}_{enc} + \lambda_{dec} \mathcal{L}_{dec} $$
  其中，編碼器和解碼器的損失分別定義為：
  $$ \mathcal{L}_{enc} = \text{ReLU}(\alpha_{enc} - \text{CosSim}(p^{(s)}_{1}, l_{enc}(f_s^{(enc)}))) $$
  $$ \mathcal{L}_{dec} = \text{ReLU}(\alpha_{dec} - \text{CosSim}(m^{(s)}_{1}, l_{dec}(f_s^{(dec)}))) $$

這裡，`CosSim`代表餘弦相似度，`ReLU`用於截斷，`α`是控制相似度的邊界超參數。這個損失函數強制ResTok學習到的表示在語義上與強大的VF模型保持一致。

## 實驗結果：效率與質量的雙重勝利

ResTok在標準的ImageNet 256x256圖像生成任務上取得了極為出色的成績，在效率和質量之間達成了新的平衡。

![實驗結果對比](asset/restok_table1_results.webp)
*圖三：ResTok與其他主流生成模型的性能對比。*

從上表中可以看出：

- **SOTA性能**：ResTok（MSRQ類型tokenizer）在僅需**9個採樣步驟**的情況下，就達到了**2.34**的gFID（越低越好），這一成績全面超越了需要數百步採樣的傳統擴散模型（如LDM）和自回歸模型（如MaskGIT、LlamaGen）。
- **效率優勢**：相較於FlowAR-B（需要208步）和ImageFolder（需要570步）等同樣追求高質量的模型，ResTok的效率優勢達到了數十倍甚至上百倍。

## 相關研究背景與分析

ResTok的出現並非偶然，它建立在視覺tokenization和自回歸生成模型多年發展的基礎之上。

- **從Grid-Based到Semantic**：早期的VQGAN等方法使用網格化的方式進行tokenization，雖然實現了離散化，但忽略了語義結構。後來的TiTok、SEEDS等工作開始探索更具語義的1D token序列，但仍未完全擺脫「語言為中心」的束縛。

- **層次化思想的迴歸**：計算機視覺的基石，如CNN和特徵金字塔網絡（FPN），早已證明了層次化表示的強大。ResTok可以看作是將這一經典思想成功遷移到現代Transformer自回歸框架中的典範。它與ImageFolder、FlowAR等同樣探索多尺度生成的模型思路一致，但在實現方式上更為優雅和高效。

## 個人評價與意義

ResTok無疑是近期視覺生成領域最令人興奮的研究之一。它不僅在性能指標上取得了SOTA的成績，更重要的是，它為如何構建真正「懂視覺」的生成模型指明了一個極具潛力的方向。

- **對研究的啟發**：這篇論文完美地回應了我近期對**VAR-based**模型和**高效生成**方法的關注。它證明了在自回歸框架內，通過精巧的架構設計（而非無限堆疊算力），完全可以實現與擴散模型相媲美甚至超越的性能。其**層次化殘差**的設計思想，對於未來探索更高效的圖像/視頻壓縮、表示學習以及多模態模型都具有重要的啟發意義。

- **未來的可能性**：ResTok的框架具有很強的擴展性。我們可以預見，未來將其與Training-Free的編輯技術、更強大的視覺基礎模型、甚至Flow Matching等技術結合，可能會催生出更加強大、可控、且極致高效的下一代生成模型。

總而言之，ResTok不僅僅是一次技術上的勝利，更是一次思想上的「正本清源」。它提醒我們，在跨領域借鑒的同時，更應尊重並充分利用每個領域自身最核心的設計原則。

---

### 參考文獻

[1] Zhang, X., Da, C., Yang, H., Gai, K., Lu, M., & Ma, Z. (2026). *ResTok: Learning Hierarchical Residuals in 1D Visual Tokenizers for Autoregressive Image Generation*. arXiv preprint arXiv:2601.03955.
[2] Esser, P., Rombach, R., & Ommer, B. (2021). *Taming Transformers for High-Resolution Image Synthesis*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
[3] Chang, H., Zhang, H., Barber, J., & Jiang, L. (2023). *MaskGIT: Masked Generative Image Transformer*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
[4] Vahdat, A., Kreis, K., & Kautz, J. (2021). *Score-based generative modeling in n-dimensions*. In Advances in Neural Information Processing Systems (NeurIPS).
