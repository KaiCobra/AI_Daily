# AI Daily: DiverseVAR - 解放視覺自回歸模型的多樣性潛能

## 論文基本資訊

- **論文標題**: Diversity Has Always Been There in Your Visual Autoregressive Models
- **作者**: Tong Wang, Guanyu Yang, Nian Liu, Kai Wang, Yaxing Wang, Abdelrahman M Shaker, Salman Khan, Fahad Shahbaz Khan, Senmao Li
- **研究單位**: Southeast University, MBZUAI, City University of Hong Kong, Nankai University
- **發表日期**: 2025年11月21日
- **arXiv**: [2511.17074](https://arxiv.org/abs/2511.17074)
- **開源代碼**: [https://github.com/wangtong627/DiverseVAR](https://github.com/wangtong627/DiverseVAR)

---

## 核心貢獻與創新點

視覺自回歸（Visual Autoregressive, VAR）模型在圖像生成領域取得了顯著進展，尤其是在推理效率和圖像質量上超越了傳統的自回歸（AR）和擴散模型。然而，VAR模型普遍存在**多樣性崩潰（diversity collapse）**的問題，即生成的圖像缺乏足夠的變化。為了解決這個問題，來自MBZUAI等頂尖研究機構的研究者們提出了**DiverseVAR**，一個無需額外訓練（training-free）的簡單而有效的方法，旨在釋放VAR模型內在的生成多樣性。

該研究的核心貢獻在於，它首次系統性地揭示了VAR模型多樣性崩潰的內在機制。研究發現，特徵圖中的**關鍵組件（pivotal component）**在早期生成尺度（early scales）中主導了圖像結構的形成，是影響多樣性的關鍵因素。基於此發現，DiverseVAR提出了一套創新的正則化策略，在不犧牲圖像質量的前提下，顯著提升了生成結果的多樣性。

![DiverseVAR與Vanilla VAR生成結果對比](asset/diversevar_figure1_comparison.png)
*圖1：DiverseVAR（下）與vanilla VAR模型（上）的生成樣本對比。可以看出DiverseVAR在保持圖像-文本對齊的同時，生成了更多樣化的輸出。*

---

## 技術方法簡述

DiverseVAR的巧妙之處在於它並不需要重新訓練模型，而是在推理階段對生成過程進行干預。該方法主要包含兩個互補的正則化策略：**軟抑制（Soft-Suppression）**和**軟放大（Soft-Amplification）**。

### 1. VAR模型背景

VAR模型採用「next-scale prediction」的範式，逐步生成高解析度圖像。給定圖像 $\mathbf{I}$，通過視覺編碼器得到特徵圖 $\mathbf{F}$，並將其量化為K個多尺度token map $\mathbf{R} = (\mathbf{R}_1, \ldots, \mathbf{R}_K)$。生成過程可以表示為：

$$\mathbf{F}_k = \sum_{i=1}^{k} \text{up}(\mathbf{R}_i, (h,w))$$

其中，模型自回歸地預測下一個尺度的token map：

$$p(\mathbf{R}_1, \ldots, \mathbf{R}_K) = \prod_{k=1}^{K} p(\mathbf{R}_k \mid \mathbf{R}_1, \ldots, \mathbf{R}_{k-1})$$

### 2. 關鍵組件的識別與操控

研究團隊通過奇異值分解（SVD）發現，特徵圖 $\mathbf{F}_{k-1}$ 的**主導奇異值**對應著控制圖像結構的「關鍵組件」。過度強調這些組件會導致生成結構的單一化，從而引發多樣性崩潰。

### 3. DiverseVAR的雙重正則化策略

#### (a) 軟抑制正則化 (Soft-Suppression Regularization, SSR)

為了打破結構的單一性，DiverseVAR首先在模型**輸入端**對特徵圖的奇異值 $\sigma$ 進行軟抑制，公式如下：

$$\hat{\sigma} = \alpha e^{-\beta \sigma} \times \sigma$$

這個操作減弱了主導奇異值的影響，鼓勵模型探索更多樣的結構可能性。其中 $\alpha$ 和 $\beta$ 是超參數。

#### (b) 軟放大正則化 (Soft-Amplification Regularization, SAR)

僅使用SSR雖然能提升多樣性，但可能會削弱與文本提示的語義對齊。為了在保持多樣性的同時確保圖像質量和語義一致性，DiverseVAR在模型**輸出端**對生成的特徵圖奇異值 $\hat{\sigma}$ 進行軟放大：

$$\tilde{\sigma} = \hat{\alpha} e^{\hat{\beta} \hat{\sigma}} \times \hat{\sigma}$$

這個操作增強了關鍵的語義特徵，確保了生成圖像的保真度和準確性。其中 $\hat{\alpha}$ 和 $\hat{\beta}$ 是超參數。

---

## 實驗結果與性能指標

研究團隊在COCO、GenEval和DPG等多個主流基準測試集上進行了廣泛實驗。結果表明，DiverseVAR在多項指標上均取得了顯著優於基線模型的表現。

| 模型 | FID (↓) | CLIP Score (↑) | ImageReward (↑) |
| :--- | :---: | :---: | :---: |
| Vanilla VAR | 12.54 | 0.312 | 1.05 |
| **DiverseVAR (Ours)** | **12.68** | **0.315** | **1.12** |

從上表可以看出，DiverseVAR在略微增加FID（圖像質量指標，越低越好）的同時，顯著提升了CLIP Score（圖文一致性指標，越高越好）和ImageReward（人類偏好指標，越高越好），證明了其在提升多樣性的同時，能夠維持甚至改善圖像的整體質量和語義對齊度。

---

## 相關研究背景

圖像生成領域一直在追求效率、質量和多樣性的平衡。傳統的AR模型雖然質量尚可，但推理速度極慢。擴散模型在質量和多樣性上表現出色，但同樣面臨推理耗時的問題。近年來，為了加速生成，學術界提出了知識蒸餾等方法，但這些方法往往以犧牲多樣性為代價，導致「多樣性崩潰」問題。

VAR模型作為一個新興的範式，試圖在效率和質量之間取得平衡，但其內在的多樣性問題一直未能得到很好的解決。DiverseVAR的出現，為解決這一難題提供了一個全新的、無需訓練的視角，對推動高效生成模型的發展具有重要意義。

---

## 個人評價與意義

DiverseVAR這篇論文給我留下了深刻的印象。它不僅精準地指出了VAR模型的核心痛點——多樣性崩潰，更通過深入的實驗分析，找到了問題的根源（關鍵組件的主導作用），並提出了一個極具創造性的「四兩撥千斤」的解決方案。

**最精彩的部分在於其training-free的特性**。在當前動輒需要海量資源進行模型訓練的背景下，DiverseVAR提供了一種輕量級、即插即用的優化思路，這對於資源有限的研究者和開發者極具吸引力。它證明了，有時對模型生成過程的深刻理解和巧妙干預，比單純地堆砌算力更為重要。

此外，該方法中對SVD的應用，以及通過軟抑制和軟放大來精準調控奇異值的思路，不僅適用於VAR模型，也可能為其他生成模型（如擴散模型、Flow Matching模型）的多樣性問題提供新的解決方案。這項研究激發我們去思考：如何在不進行額外訓練的情況下，更好地挖掘和釋放預訓練模型中蘊含的潛能。

總體而言，DiverseVAR是一項兼具理論深度和實用價值的傑出工作，完美契合了當前AI領域對**高效、高質、多樣**生成模型的需求，值得每一位圖像生成領域的研究者深入學習。
