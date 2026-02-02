# AI Daily: SoftCFG - 無需訓練，用不確定性引導穩定視覺自回歸模型

**發布日期**: 2026年02月02日

## 論文基本資訊

- **論文標題**: SoftCFG: Uncertainty-guided Stable Guidance for Visual Autoregressive Model
- **作者**: Dongli Xu, Aleksei Tiulpin, Matthew B. Blaschko
- **所屬單位**: KU Leuven, University of Oulu
- **發表會議**: ICLR 2026 (Submission)
- **論文連結**: [https://openreview.net/forum?id=G7tqQ5Upcs](https://openreview.net/forum?id=G7tqQ5Upcs)
- **arXiv連結**: [https://arxiv.org/abs/2510.00996](https://arxiv.org/abs/2510.00996)

---

## 核心貢獻與創新點

自回歸（Autoregressive, AR）模型已成為圖像生成領域的強大工具，但將廣受歡迎的**Classifier-Free Guidance (CFG)** 技術直接應用於AR模型時，會面臨兩大挑戰：**引導消退（Guidance Diminishing）**和**過度引導（Over-Guidance）**。前者指隨著解碼步驟增加，引導信號迅速減弱；後者則指過強的引導會扭曲圖像的視覺連貫性，產生不自然的偽影。

為解決這些問題，研究人員提出了**SoftCFG**，一種無需訓練、模型無關的推理時引導方法。其核心創新在於引入了**「歷史到未來」的引導機制**，利用已生成token的不確定性來動態調整引導強度，從而穩定生成過程並提升圖像質量。

SoftCFG的主要貢獻可以總結為以下三點：

1.  **不確定性引導的擾動（Uncertainty-guided Perturbation）**: 讓每個已生成的token根據其預測置信度（certainty）貢獻加權引導，確保引導信號在整個生成過程中持續存在，並解決文本引導與視覺上下文之間的衝突。
2.  **步驟歸一化（Step Normalization）**: 引入一種歸一化機制，限制每一步累積的擾動總量，從而穩定長序列的生成過程，防止模型偏離軌道。
3.  **即插即用且高效**: 作為一種無需訓練的方法，SoftCFG可以無縫集成到現有的AR模型（如AliTok、LuminaGPT）中，且幾乎不增加額外的計算開銷，同時在ImageNet 256x256數據集上達到了AR模型中最先進的FID分數。

![CFG vs SoftCFG 對比圖](assets/softcfg_figure1_comparison.png)
*圖1：標準CFG（a）僅在第一步應用引導，而SoftCFG（b）則在整個生成過程中根據token置信度自適應地調整引導強度。實驗結果（c）顯示SoftCFG在多個AR模型上均顯著降低了FID。*

---

## 技術方法簡述

SoftCFG的實現巧妙地修改了標準CFG在AR模型中的應用方式。標準CFG通過在條件和無條件預測之間進行插值來增強控制力，其公式如下：

$$ z_t^{CFG} = z_t^{cond} + \gamma \cdot (z_t^{cond} - z_t^{uncond}) $$

其中，$z_t^{cond}$ 和 $z_t^{uncond}$ 分別是模型在第 $t$ 步的條件和無條件預測的logits，$\gamma$ 是引導尺度（guidance scale）。

在AR模型中，由於每一步的預測都依賴於之前所有步驟生成的token所構成的KV快取（KV Cache），導致條件和無條件分支的KV快取迅速趨同，這就是「引導消退」的根源。

![Guidance Diminishing 問題](assets/softcfg_figure3_guidance_diminishing.png)
*圖2：在AR模型中，標準CFG的引導信號（橙線與藍線的差異）隨著生成步驟的增加而迅速消失。*

SoftCFG的核心思想是**在推理過程中持續地、自適應地對無條件分支的KV快取進行擾動**，從而維持一個健康的引導差距。

### 1. 不確定性引導的Value Cache擾動

SoftCFG並非粗暴地注入噪聲，而是根據每個已生成token的**預測置信度**來決定擾動的強度。置信度高的token（通常對應圖像的結構化部分）獲得較小的擾動，而置信度低的token（通常對應模糊或不確定的區域）則接受較大的引導。這種機制使得引導更加「智能」，專注於模型最需要幫助的地方。

具體來說，在每一步，SoftCFG會對無條件分支的Value Cache（$V_{uncond}$）進行修改：

$$ V_{uncond, i}' = V_{uncond, i} + (1 - P_{max, i})^k \cdot (V_{cond, i} - V_{uncond, i}) $$

其中：
- $V_{uncond, i}'$ 是擾動後的Value向量。
- $P_{max, i}$ 是第 $i$ 個token的最大預測概率，作為其置信度的代理。
- $(1 - P_{max, i})$ 代表不確定性，不確定性越高的token獲得的引導越強。
- $k$ 是一個超參數，用於調整不確定性權重的影響力。

![SoftCFG 方法架構圖](assets/softcfg_figure6_method_diagram.png)
*圖3：SoftCFG（b）與傳統擾動策略（a）的對比。SoftCFG根據每個token的預測概率（P）來施加柔和、加權的擾動，而不是對所有token一視同仁。*

### 2. 步驟歸一化（Step Normalization）

為了防止在長序列生成中，持續的擾動累積可能導致模型最終偏離穩定軌道，SoftCFG引入了步驟歸一化。該機制確保在每一步中，施加於所有歷史token的總擾動量保持在一個恆定的範圍內，從而保證了整個生成過程的穩定性。

---

## 實驗結果與性能指標

SoftCFG在標準的ImageNet 256x256圖像生成任務上進行了廣泛評估，並與基線模型及標準CFG進行了比較。

### 主要性能指標

實驗結果表明，SoftCFG顯著優於標準CFG，在不增加訓練成本的情況下，將SOTA AR模型（AliTok-XL）的FID從1.37降低到**1.27**。

| 模型 | 方法 | FID (↓) | IS (↑) | sFID (↓) | Precision (↑) | Recall (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| AliTok-XL | Baseline | 1.48 | 289.1 | 7.95 | 0.78 | 0.64 |
| AliTok-XL | CFG | 1.37 | 295.3 | 7.18 | 0.79 | 0.65 |
| **AliTok-XL** | **SoftCFG** | **1.27** | **302.4** | **6.70** | **0.79** | **0.65** |

*表1：SoftCFG與基線和標準CFG在ImageNet 256x256上的性能對比。*

### 視覺質量對比

定性比較也顯示出SoftCFG在視覺質量上的優勢。與標準CFG相比，SoftCFG生成的圖像更加連貫和自然，有效減少了因「過度引導」造成的物體扭曲和偽影。

![生成圖像視覺對比](assets/softcfg_figure2_visual_comparison.png)
*圖4：由LuminaGPT-8B生成的圖像對比。標準CFG（左）生成的圖像中出現了不合理的物體融合（如大象鼻子變成了香蕉），而SoftCFG（右）則生成了更為連貫和合理的圖像。*

![Over-guidance 問題示例](assets/softcfg_figure4_over_guidance.png)
*圖5：過度引導現象的圖示。當引導強度過大時，模型會錯誤地將提示詞中的「香蕉」映射到大象的象牙上，導致語義錯亂。*

---

## 相關研究背景

SoftCFG的提出建立在視覺自回歸模型和引導生成技術的發展之上。

- **視覺自回歸模型（VAR）**: 如**VQGAN**、**Parti**以及最近的**AliTok**和**LuminaGPT**，將圖像建模為一系列離散的token，並以自回歸的方式逐個預測，在圖像生成領域取得了巨大成功。
- **Classifier-Free Guidance (CFG)**: 最初為擴散模型設計，通過結合有條件和無條件的預測來提高生成質量和可控性，已成為生成模型的標準配置。
- **針對AR引導的研究**: 在SoftCFG之前，已有研究試圖解決CFG在AR模型中的局限性。例如，**Condition Contrastive Alignment (CCA)** [1] 提出了一種**訓練時對齊**的方法，通過微調模型來消除對引導的需求，但這犧牲了「即插即用」的靈活性。SoftCFG則代表了另一條技術路線：**在推理時進行智能引導**，無需修改模型本身。

SoftCFG的獨特之處在於，它首次提出利用AR模型自身的**歷史生成信息（即已生成的token）的不確定性**來指導未來的生成，這是一種專為AR架構設計的、內生的引導機制。

---

## 個人評價與意義

SoftCFG是一項非常實用且巧妙的研究，它精準地抓住了CFG在AR模型中水土不服的核心痛點，並提出了一個優雅而高效的解決方案。

**主要優勢**：
1.  **實用性強**：無需訓練的特性使其可以輕鬆應用於任何現有的或未來的AR模型，極大地降低了應用門檻。
2.  **思想深刻**：將「不確定性」作為引導信號來源，是對生成模型內部狀態的一種深刻洞察。這不僅解決了當前的問題，也為未來更精細化的引導策略開闢了新的思路。
3.  **平衡性好**：通過自適應加權和歸一化，SoftCFG在增強引導效果和維持生成穩定性之間取得了出色的平衡，避免了「過猶不及」的陷阱。

**潛在局限**：
- **依賴簡單的置信度代理**：目前使用最大預測概率作為不確定性的度量，雖然有效，但在複雜場景下可能不夠魯棒。未來的研究可以探索更精確的不確定性估計方法。
- **文本對齊能力**：實驗表明，SoftCFG主要增強圖像端的視覺連貫性，但在需要精確文本-圖像對齊的任務（如計數、空間關係）上可能不會帶來提升，甚至略有下降。

總體而言，SoftCFG為如何將引導技術有效應用於自回歸模型提供了一個極具價值的範例。它不僅顯著提升了AR模型的生成質量，更重要的是，它所提出的**「歷史到未來」的自適應引導框架**，為探索下一代智能生成算法提供了寶貴的啟示。

---

## 參考文獻

[1] Chen, H., Su, H., Sun, P., & Zhu, J. (2024). *Toward Guidance-Free AR Visual Generation via Condition Contrastive Alignment*. arXiv preprint arXiv:2410.09347.
