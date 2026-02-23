# AI Daily: ARPG - 解鎖隨機並行解碼的自回歸圖像生成新境界

**作者: Manus AI**

**日期: 2026年02月23日**

## 導讀

自回歸（Autoregressive, AR）模型在大型語言模型（LLM）領域取得了巨大成功，其核心的「next-token prediction」範式也被應用於視覺生成任務。然而，將此範式從一維的文本直接套用到二維的圖像上，面臨著根本性的挑戰。傳統AR模型必須將圖像展平為一維序列，並遵循光柵掃描（raster-scan）等固定的生成順序，這不僅效率低下，更限制了模型在需要非因果依賴的零樣本（zero-shot）任務（如圖像修復）上的泛化能力。

為了解決這些問題，研究者們提出了多種替代方案，例如基於遮罩（masking）的並行生成（如MaskGIT [1]）和塊狀自回歸（block-wise AR）模型（如VAR [2]）。但這些方法要麼因為無法使用KV快取而計算成本高昂，要麼受限於固定的生成順序和排程。近期備受關注的RandAR [3] 雖然透過引入位置指令通證（positional instruction tokens）實現了任意順序的生成，卻也付出了序列長度加倍、記憶體和計算成本劇增的代價。

在此背景下，一篇被ICLR 2026接受為Poster的論文 **「Autoregressive Image Generation with Randomized Parallel Decoding」（ARPG）** [4] 提出了一種創新的視覺自回歸模型，旨在實現高效、靈活且高畫質的圖像生成。ARPG透過其獨特的 **解耦解碼（decoupled decoding）** 機制，成功地在完全隨機的通證順序下進行訓練和推理，突破了傳統AR模型的瓶頸。

本文將深入探討ARPG的核心思想、技術細節及其在多項圖像生成任務上的卓越表現。

## ARPG 核心創新：解耦位置與內容的雙通道解碼

ARPG的核心洞見在於：**有效的隨機順序建模需要明確的位置引導（positional guidance）**。為了實現這一點，ARPG拋棄了傳統單一解碼器的設計，提出了一個新穎的 **雙通道解碼器（Two-Pass Decoder）** 架構，將位置預測與內容表示學習徹底解耦。

這個架構包含兩個關鍵部分：

1.  **通道一：內容精煉（Content Refinement Pass）**：此通道是一個標準的因果自註意力解碼器，但它的唯一目的不是預測下一個通證，而是處理隨機順序的圖像通證序列，並生成一組豐富的、包含上下文信息的內容表示（Key-Value pairs）。這個過程確保了內容表示的學習是無標籤洩漏的。

2.  **通道二：位置引導預測（Position-Guided Prediction Pass）**：此通道由因果交叉注意力（causal cross-attention）構成。在這裡，與數據無關的 `[MASK]` 通證被賦予目標位置信息，作為「目標感知查詢（target-aware queries）」。這些查詢會對第一通道生成的內容表示（Key-Value pairs）進行注意力計算，從而預測出目標位置的通證。

這種設計的精妙之處在於，它將預測過程分解為「看見了什麼」（內容）和「在哪裡看見」（位置），並分別處理。這帶來了三大優勢：

*   **訓練效率高**：內容表示可以獨立學習，並且所有通證都在單一訓練步驟中得到充分學習，避免了遮罩模型中非遮罩通證梯度稀疏的問題。
*   **解碼質量好**：位置查詢可以直接利用由深層網絡生成的、語義豐富的內容表示，而非淺層的、信息量較少的表示。
*   **並行推理快**：在推理時，可以一次性輸入多個位置查詢，並行預測多個通證，極大地提升了生成速度。

| 特性 | 標準AR (如LlamaGen) | 遮罩AR (如MaskGIT) | RandAR | **ARPG (本文)** |
| :--- | :--- | :--- | :--- | :--- |
| **生成順序** | 固定 (光柵掃描) | 隨機 | 隨機 | **隨機** |
| **注意力機制** | 因果自註意力 | 雙向注意力 | 因果自註意力 | **因果自註意力 + 交叉注意力** |
| **KV快取** | 支持 | 不支持 | 支持 (成本加倍) | **支持 (高效)** |
| **並行解碼** | 不支持 | 支持 | 支持 | **支持 (靈活且高效)** |
| **零樣本泛化** | 差 | 好 | 好 | **極佳** |

## 卓越的實驗結果：速度、效率與質量的三豐收

ARPG在多個基準測試中展現了其卓越的性能，不僅在生成質量上超越了現有的AR模型，更在推理效率上實現了數量級的提升。

在ImageNet-1K 256x256的基準測試中，ARPG-XXL模型僅用32個解碼步驟就達到了 **1.83的FID分數**，這一成績不僅優於LlamaGen [5] 等傳統AR模型，甚至超越了許多擴散模型。更令人矚目的是其驚人的效率：

*   **速度**：相較於傳統的光柵掃描AR模型，ARPG實現了近 **30倍** 的推理加速；相較於近期的並行AR模型（如PAR [6]），也取得了 **3倍** 的加速。
*   **記憶體**：記憶體消耗降低了 **75%**，這得益於其高效的KV快取機制和無需儲存額外位置通證的設計。

| 模型 | 參數 | 步驟 | 吞吐量 (img/s) | 記憶體 (GB) | FID (↓) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DiT-XL/2 (擴散) | 675M | 250 | 0.91 | 2.14 | 2.27 |
| LlamaGen-XXL (AR) | 1.4B | 576 | 1.58 | 26.22 | 2.62 |
| RandAR-XXL (並行AR) | 1.4B | 88 | 10.46 | 21.77 | 2.15 |
| **ARPG-XXL** | **1.3B** | **32** | **55.28** | **7.22** | **1.83** |

## 解鎖零樣本泛化與可控生成

ARPG靈活的隨機順序生成能力使其天然具備強大的零樣本泛化能力，無需任何針對性微調即可完成多種圖像編輯任務。

*   **圖像修復（Inpainting）與編輯**：只需將已知區域的通證預先填入第一通道，然後在第二通道中生成被遮罩的目標區域。
*   **圖像外擴（Outpainting）與解析度擴展**：同樣可以通過預設上下文並生成未知區域來實現。

實驗結果表明，ARPG生成的內容不僅畫質高，而且與周圍上下文語義一致性極強。此外，通過引入條件輸入（如Canny邊緣圖、深度圖）作為第二通道的查詢，ARPG還能實現高質量的可控生成，其性能顯著優於ControlVAR和ControlAR等近期工作。

## 結論

ARPG通過其創新的雙通道解耦解碼架構，成功地解決了視覺自回歸模型中長期存在的效率與靈活性矛盾。它不僅在圖像生成質量上達到了新的高度，更在推理速度和記憶體效率上取得了突破性進展，同時展現了強大的零樣本泛化能力。這項工作為視覺生成領域提供了一個全新的、高效的、可擴展的範式，並為未來構建統一的視覺理解與生成模型鋪平了道路。

## 參考文獻

[1] Chang, H., Zhang, H., Barber, J., Maschinot, A., Lezama, J., Jiang, L., ... & Freeman, W. T. (2022). Maskgit: Masked generative image transformer. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 11335-11345).

[2] Tian, Y., Li, T., Li, H., & He, K. (2024). Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction. *arXiv preprint arXiv:2404.02905*.

[3] Pang, Z., Zhang, T., Luan, F., Man, Y., Tan, H., Zhang, K., ... & Wang, Y. (2025). RandAR: Decoder-only Autoregressive Visual Generation in Random Orders. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.

[4] Li, H., Yang, J., Li, G., & Wang, H. (2026). Autoregressive Image Generation with Randomized Parallel Decoding. In *International Conference on Learning Representations*.

[5] Sun, Q., Li, T., Li, H., & He, K. (2024). Autoregressive model beats diffusion: Llama for scalable image generation. *arXiv preprint arXiv:2403.10835*.

[6] Wang, Y., Ren, J., Dai, J., & Zhou, P. (2025). Parallelized Autoregressive Visual Generation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.
