# AI Daily: Mirai - 讓自回歸視覺生成擁有「預見未來」的能力

> **論文標題**: Mirai: Autoregressive Visual Generation Needs Foresight
> **發表單位**: 東京大學、日本國立情報學研究所、北京大學
> **發表時間**: 2026年1月21日
> **論文連結**: [https://arxiv.org/abs/2601.14671](https://arxiv.org/abs/2601.14671)
> **項目頁面**: [https://y0uroy.github.io/Mirai](https://y0uroy.github.io/Mirai)

---

## 核心貢獻：為AR模型注入「Foresight」，解決全局一致性難題

傳統的自回歸（Autoregressive, AR）視覺生成模型，如同在沒有參考圖的情況下拼湊拼圖，僅依賴於「下一個token」的預測進行訓練。這種嚴格的因果監督機制雖然在語言模型中取得了巨大成功，但在處理具有高度空間依賴性的視覺數據時，卻常常導致生成圖像的**全局一致性欠佳**和**收斂速度緩慢**。生成的圖像可能在局部細節上無可挑剔，但整體結構卻顯得支離破碎，例如生成一隻頭身分離的鸚鵡。

為了解決這一根本性難題，來自東京大學等機構的研究者們提出了一個名為 **Mirai**（日語「未來」）的通用訓練框架。其核心思想是在訓練階段為AR模型注入「**Foresight**」（預見）能力，即引入來自未來token的訓練信號。這使得模型在學習生成當前token時，能夠「預見」到未來的圖像結構，從而學習到更具全局觀的內部表徵，有效提升了生成圖像的結構連貫性。

![Mirai生成結果與基線模型LlamaGen-B的比較](assets/mirai_visual_comparison.webp)
*圖1：Mirai（下）與基線模型（上）在生成結果上的比較。Mirai能夠生成結構更完整、全局一致性更強的圖像，如完整的火箭發射煙霧。*

最重要的是，Mirai框架的設計極具巧思：它**無需對模型架構進行任何修改**，且**不增加任何額外的推理開銷**。Foresight機制僅在訓練階段發揮作用，推理時則完全移除，確保了模型的生成效率不受影響。

## 技術方法簡述：Mirai框架的兩種實現

Mirai框架通過一個輔助的對齊損失（alignment loss）來實現Foresight的注入。研究者們通過一系列嚴謹的診斷實驗發現，將Foresight信號對齊到AR模型**內部表徵**的**2D網格**上，而非輸出層或1D序列，是提升模型性能的關鍵。

標準的AR模型通過最大化下一個token的對數似然來進行訓練，其損失函數為：

$$\mathcal{L}_{\text{NTP}}(\theta) = -\mathbb{E}_{\bm{x}\sim p_{\text{data}}}\left[\frac{1}{N}\sum_{n=1}^{N}\log p_\theta(x_n \mid \bm{x}_{<n})\right]$$

Mirai在此基礎上，引入了一個Foresight損失項 $\mathcal{L}_{\text{foresight}}$，它衡量AR模型在位置 $n$ 的隱藏狀態 $\bm{h}_n$ 與來自未來tokens的特徵 $f_n$ 之間的一致性：

$$\mathcal{L}_{\text{foresight}} = \mathbb{E}\left[\frac{1}{NK}\sum_{n=1}^{N}\sum_{k=1}^{K}\ell(f_n^{(k)}, \rho_n(\bm{h}_n))\right]$$

其中，$R(\cdot)$ 是一個Foresight Encoder，負責從未來tokens中提取特徵。根據Foresight Encoder的不同，Mirai框架分為兩種實現：

1.  **Mirai-E (Explicit Foresight)**：使用模型自身的指數移動平均（EMA）作為Foresight Encoder。這種方式從單向的AR模型中提取顯式的、帶有位置索引的未來特徵，實現了內部狀態與鄰近未來幾個位置的對齊。

2.  **Mirai-I (Implicit Foresight)**：使用一個預訓練好的、**凍結的雙向Encoder**（如DINOv2）作為Foresight Encoder。這種方式提供了更豐富、更具上下文的隱式未來信息，通過將AR模型的內部狀態與雙向Encoder在對應空間位置的特徵進行對齊，讓AR模型學習到更強的全局表徵。

![Mirai框架概覽](assets/mirai_figure1_overview.webp)
*圖2：Mirai框架探索的三個維度：(a) 注入層級，(b) 定位方式，(c) Foresight來源。實驗證明，在內部表徵層（Internal）進行2D網格對齊效果最佳。*

## 實驗結果與性能指標

實驗結果有力地證明了Mirai框架的有效性。在ImageNet 256x256的類別條件圖像生成任務上，與基線模型LlamaGen-B相比，Mirai取得了顯著的性能提升。

### 訓練加速與性能提升

-   **驚人的訓練加速**：Mirai-I能夠將LlamaGen-B的收斂速度**提升高達10倍**，而Mirai-E也能達到5倍的加速效果。這意味著達到相同的生成質量（以FID指標衡量），Mirai所需的訓練時間大幅縮短。
-   **生成質量顯著改善**：在經過充分訓練後，Mirai-I將基線模型的FID從5.34降低到了**4.34**，這是一個非常顯著的提升，表明生成圖像的保真度和多樣性都得到了改善。

![Mirai訓練加速曲線](assets/mirai_training_acceleration.webp)
*圖3：Mirai-I（藍線）和Mirai-E（紅線）相比基線模型LlamaGen-B（黃線）在訓練過程中的FID曲線，展現了顯著的加速效果。*

### 關鍵設計的消融實驗

| Model | Inj. Lvl. | Layout | K | FID↓ | IS↑ |
|---|---|---|---|---|---|
| LlamaGen-B | - | - | - | 6.36 | 185.54 |
| + Foresight | Output | 2D | 3 | 6.48 | 185.57 |
| **+ Foresight** | **Internal** | **2D** | **3** | **5.22** | **197.14** |

*表1：Foresight注入方式的消融實驗結果。將Foresight注入到內部表徵層（Internal）並採用2D網格對齊（2D）時，取得了最佳效果（FID 5.22）。*

## 相關研究背景

自回歸模型在視覺生成領域的應用由來已久，從早期的PixelCNN到近期的VQVAE/VQGAN，再到大規模的Transformer模型如Parti、CogView和LlamaGen，研究者們一直在探索如何將AR模型的序列建模能力應用於圖像生成。然而，如何解決AR模型固有的單向性與圖像數據的2D空間依賴性之間的矛盾，始終是一個核心挑戰。Mirai的研究正是建立在這一系列工作之上，並從一個全新的角度——**訓練時的未來信息引導**——為解決這一挑戰提供了創新且高效的方案。

## 個人評價與意義

Mirai論文為視覺自回歸生成領域帶來了令人耳目一新的思路。它巧妙地借鑒了「未來」的信息來指導「現在」的生成，卻又不在推理時增加任何負擔，這種「訓練時增強，推理時無損」的設計哲學極具實用價值。

這項工作最大的啟示在於，它挑戰了傳統AR模型嚴格的因果建模範式，證明了在訓練中引入非因果信號（Foresight）不僅是可行的，而且是極其有效的。特別是Mirai-I的設計，通過對齊一個強大的、預訓練的雙向視覺模型，為AR模型提供了一個「視覺常識」的老師，使其能夠在訓練早期就建立起對全局結構的感知能力。這不僅僅是技術上的改進，更是對如何將不同模型的優勢（AR的生成能力和雙向模型的理解能力）進行融合的一次成功探索。

對於追求高效訓練和高質量生成的研究者和開發者而言，Mirai提供了一個即插即用、成本效益極高的解決方案。我們可以期待，這種「Foresight」思想未來會被更廣泛地應用於多模態生成、影片生成等更複雜的自回歸任務中，激發出更多創新的火花。
