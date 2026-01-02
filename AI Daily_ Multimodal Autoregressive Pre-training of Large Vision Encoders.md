# AI Daily: Multimodal Autoregressive Pre-training of Large Vision Encoders

**日期**: 2026年1月1日
**作者**: Manus AI

## 論文基本信息

- **論文標題**: Multimodal Autoregressive Pre-training of Large Vision Encoders [1]
- **作者**: Enrico Fini, Mustafa Shukor, Alaaeldin El-Nouby, et al.
- **研究單位**: Apple
- **發表會議**: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025

---

## 核心貢獻與創新點

在大型多模態模型（LMMs）的發展中，一個主流範式是將一個大型語言解碼器與一個視覺編碼器配對。然而，如何有效預訓練視覺編碼器以適應生成式的下游任務，一直是個挑戰。傳統方法如CLIP [2] 採用對比學習（contrastive learning），這在預訓練階段（判別式目標）和下游應用（生成式目標）之間造成了目標不一致的問題。

為解決此問題，Apple的研究團隊提出了**AIMV2**，一個為大規模視覺編碼器設計的多模態自回歸預訓練框架。其核心創新在於**統一了預訓練和下游任務的目標**，通過一個多模態解碼器，自回歸地生成原始圖像補丁（raw image patches）和文本標記（text tokens）。

這種方法不僅解決了目標不一致的問題，還展現了卓越的**可擴展性（scalability）**和**訓練效率**。AIMV2在多項視覺和多模態基準測試中均取得了頂尖性能，尤其在多模態圖像理解方面，持續優於如CLIP和SigLIP [3] 等先進的對比學習模型。

> 我們介紹了一種新穎的大規模視覺編碼器預訓練方法。基於視覺模型自回歸預訓練的最新進展，我們將此框架擴展到多模態設置，即圖像和文本。在本文中，我們提出了AIMV2，一個通用視覺編碼器家族，其特點是預訓練過程直接、可擴展性強，並在一系列下游任務中表現出色。 [1]

## 技術方法簡述

AIMV2的架構基於一個視覺編碼器（Vision Transformer, ViT）和一個因果多模態解碼器（Causal Multimodal Decoder）。其預訓練過程如下：

1.  **輸入處理**：將圖像分割成一系列非重疊的patches，並將對應的文本轉換為tokens。
2.  **聯合編碼**：將圖像patches序列和文本tokens序列連接起來，輸入到模型中。
3.  **自回歸生成**：模型被訓練來預測序列中的下一個元素，無論是圖像patch還是文本token。這通過一個統一的自回歸目標實現，結合了圖像的**像素級均方誤差損失（Pixel MSE Loss）**和文本的**交叉熵損失（Cross-entropy Loss）**。
4.  **Prefix Vision Encoder**：在架構上，AIMV2採用了Prefix Vision Encoder，允許文本tokens關注圖像特徵，從而實現更強的跨模態融合。

這種設計使得AIMV2能夠在一個統一的框架內同時學習視覺表示和多模態對齊，其訓練目標與大型語言模型（LLMs）的生成式預訓練範式高度一致。

| 組件 | 作用 |
| :--- | :--- |
| **Vision Encoder (ViT)** | 提取圖像的視覺特徵。 |
| **Causal Multimodal Decoder** | 接收視覺和文本特徵，自回歸地生成圖像和文本。 |
| **Prefix Attention** | 增強文本對視覺特徵的關注，促進跨模態理解。 |
| **統一損失函數** | 結合圖像重建損失和文本生成損失，進行端到端訓練。 |

## 實驗結果與性能指標

AIMV2在多項基準測試中展現了其優越性。最引人注目的成果是，**AIMV2-3B**模型在ImageNet-1k分類任務上，僅使用凍結的骨幹網絡（frozen trunk）就達到了**89.5%的準確率**。

研究表明，AIMV2具有強大的**縮放法則（Scaling Laws）**。如下圖所示，無論是增加模型參數數量還是擴展訓練數據量，模型的性能都能穩定提升，這與大型語言模型中觀察到的現象一致。

此外，與僅使用文本描述（captioning-only）的基線模型相比，AIMV2的圖像級目標（image-level objective）在所有數據集和模型規模上都表現出更強的泛化能力和更少的性能飽和現象。

## 相關研究背景

AIMV2的出現並非偶然，而是建立在一系列視覺自回歸模型研究的基礎之上。

- **對比學習的興起與局限**：以**CLIP** [2] 和**SigLIP** [3] 為代表的對比學習模型在視覺-語言預訓練領域取得了巨大成功。它們通過對比圖像和文本的相似性來學習多模態表示，具有強大的零樣本分類能力。然而，其判別式的訓練目標與生成式的下游任務存在根本性的不匹配。

- **視覺自回歸模型的探索**：受LLMs成功的啟發，研究人員開始探索將自回歸範式應用於視覺領域。Apple的前作**AIM** [4] 證明了單純的自回歸目標可以訓練出強大的視覺模型，並展現了良好的縮放特性。與此同時，**Visual Autoregressive Modeling (VAR)** [5] 提出「下一尺度預測」，在圖像生成任務上首次超越了擴散模型（Diffusion Models），並榮獲NeurIPS 2024最佳論文獎，證明了自回歸模型在視覺生成領域的巨大潛力。

- **AIMV2的承上啟下**：AIMV2可以視為AIM的自然演進，將單模態的自回歸預訓練擴展到了多模態領域。它借鑒了VAR所驗證的自回歸建模在視覺領域的強大能力，並將其應用於解決大型多模態模型中視覺編碼器的預訓練難題，成功地將表示學習和生成建模統一起來。

## 個人評價與意義

AIMV2的提出具有重要的學術和應用價值。它為大型多模態模型的發展提供了一條更為統一和高效的技術路徑。

從**學術角度**看，AIMV2成功地將語言模型中的自回歸範式無縫遷移到多模態視覺預訓練中，解決了長期存在的預訓練與下游任務目標不一致的問題。這不僅為視覺編碼器的訓練提供了新的SOTA方法，也進一步驗證了自回歸作為一種通用學習框架的潛力。

從**應用角度**看，AIMV2所展現出的高效率和強大的縮放法則，意味著未來可以通過擴展模型和數據規模，持續提升模型的性能，從而構建能力更強的視覺-語言模型。這對於圖像理解、視覺問答、圖像生成和多模態交互等應用領域將產生深遠影響。

總體而言，AIMV2不僅是一項技術上的突破，更可能引領下一代大型多模態模型預訓練範式的變革，推動AI向更通用、更統一的目標邁進。

---

## 參考文獻

[1] Fini, E., Shukor, M., Li, X., et al. (2025). Multimodal Autoregressive Pre-training of Large Vision Encoders. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. (https://openaccess.thecvf.com/content/CVPR2025/html/Fini_Multimodal_Autoregressive_Pre-training_of_Large_Vision_Encoders_CVPR_2025_paper.html)

[2] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *OpenAI*. (https://openai.com/index/clip/)

[3] Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). Sigmoid Loss for Language Image Pre-Training. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. (https://arxiv.org/abs/2303.15343)

[4] Apple Machine Learning Research. (2024). Scalable Pre-training of Large Autoregressive Image Models. (https://machinelearning.apple.com/research/autoregressive-image-models)

[5] Tian, K., Jiang, Y., Yuan, Z., et al. (2024). Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction. In *Advances in Neural Information Processing Systems (NeurIPS)*. (https://arxiv.org/abs/2404.02905)
