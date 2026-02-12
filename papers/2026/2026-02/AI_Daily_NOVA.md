# AI Daily: NOVA - 無需訓練，用熵引導VAR模型實現自適應加速

> **論文標題**: Adaptive Visual Autoregressive Acceleration via Dual-Linkage Entropy Analysis
> **作者**: Yu Zhang, Jingyi Liu, Feng Liu, Duoqian Miao, Qi Zhang, Kexue Fu, Changwei Wang, Longbing Cao
> **單位**: 同濟大學, 哈爾濱工業大學, 麥考瑞大學, 中國科學院
> **發表**: arXiv 2026.02
> **論文連結**: [https://arxiv.org/abs/2602.01345](https://arxiv.org/abs/2602.01345)

---

## 核心貢獻：熵驅動的VAR加速新框架

視覺自回歸模型 (Visual Autoregressive, VAR) 在圖像生成領域取得了巨大成功，但其巨大的計算成本（主要來自龐大的token數量）限制了其應用。現有的加速方法通常依賴啟發式規則、非自適應的靜態調度，且加速範圍有限，未能充分挖掘潛力。

為此，來自同濟大學等機構的研究者提出了 **NOVA (adaptive visual autOregressive acceleration via dual-linkage entropy analysis)**，一個**無需訓練 (training-free)** 的VAR模型token reduction加速框架。NOVA的核心思想是利用**熵 (Entropy)** 作為預測不確定性的度量，從而動態地捕捉建模過程的演化，實現自適應的加速策略。

![NOVA生成範例](asset/nova_figure1_generation_example.webp)
*圖1: NOVA在單張NVIDIA RTX 3090 GPU上，僅需3.2秒即可生成2K高解析度圖像，且不會耗盡記憶體。*

NOVA的主要貢獻可以總結為以下三點：
1.  **自適應加速激活 (Adaptive Acceleration Activation)**: NOVA在推理過程中，通過在線監測**尺度熵 (scale entropy)** 增長的**拐點 (inflection point)**，動態地確定何時啟動token reduction，而非採用固定的啟動階段。
2.  **雙重鏈接加速 (Dual-Linkage Acceleration)**: 該框架同時在**尺度級 (scale-level)** 和**層級 (layer-level)** 進行自適應調整。它通過**尺度鏈接比率函數**為每個尺度計算基礎削減率，再通過**層鏈接比率函數**為該尺度下的每一層計算更精細的削減率。
3.  **殘差緩存重用 (Residual Cache Reuse)**: 充分利用VAR模型在預測下一尺度時使用**殘差 (residuals)** 的特性，NOVA重用前一尺度計算出的低熵token的緩存，進一步提升推理速度並保持生成質量。

---

## 技術方法簡述：熵如何引導Token削減？

NOVA的成功關鍵在於它將信息論中的「熵」作為指導原則， elegantly地解決了何時削減、削減哪些、削減多少的問題。

### 1. 理論基礎：熵與信息增益

在信息論中，條件互信息 $I(X; Y | Z) = H(X | Z) - H(X | Y, Z)$ 衡量了在已知 $Z$ 的情況下，觀測到 $Y$ 後對 $X$ 不確定性的減少量。在VAR模型中，每個token的生成都是一次觀測。高熵的token意味著其包含更多的不確定性，對後續生成的影響更大；反之，低熵的token則信息量較少。

因此，NOVA的核心假設是：**保留高熵token可以最大化潛在的信息增益，而剪除低熵token對生成質量的影響很小，卻能顯著降低計算成本。**

### 2. NOVA框架：尺度與層級的雙重奏

NOVA的框架設計精妙，分為尺度級和層級兩個協同工作的模塊。

![NOVA框架圖](asset/nova_figure5_framework.webp)
*圖2: NOVA框架示意圖，展示了其在尺度級和層級的協同工作機制。*

#### a. 尺度級：確定加速的「時機」

NOVA首先需要確定從哪個尺度 $t^*$ 開始進行token reduction。它通過監測**尺度整體熵** $H_t$ 的增長率 $g_t$ 來實現：

$$g_t \triangleq H_t - H_{t-1}$$

當熵增長率的移動平均值 $\delta_t = (g_t + g_{t+1})/2$ 首次低於一個根據早期尺度計算出的基線時，NOVA就認為模型已經捕獲了足夠的全局信息，此時便是啟動加速的最佳時機 $t^*$。

#### b. 雙重鏈接：計算「削減率」

確定了啟動時機後，NOVA需要為每個後續尺度 $t$ 和其中的每一層 $l$ 計算一個動態的token削減率 $Ratio_{t,l}$。

- **尺度鏈接 (Scale-Linkage)**: 首先，它會計算一個基礎削減率 $Ratio_t$，這個比率會隨著尺度的增加而變大（越到後期，細節越多，可削減的冗餘信息越多），同時也會被當前的熵增長率 $g_t$ 所調控（如果熵仍在劇烈變化，就少削減一些）。其公式為：

  $$Ratio_t = \sigma\left(\frac{\rho \cdot (t^* + 1)}{t} - \lambda \cdot \tanh(g_t)\right)$$

  其中 $\sigma$ 是Sigmoid函數，$\rho$ 和 $\lambda$ 是超參數。

- **層鏈接 (Layer-Linkage)**: 接著，在每個尺度內部，NOVA會進一步為每一層微調削減率。它會比較當前層的熵和之前所有層的平均熵，如果當前層的熵相對較高，就減少削減率，反之則增加。

通過這種雙重鏈接機制，NOVA實現了對token削減的精準控制。

![頻率與熵分析](asset/nova_figure4_frequency_entropy_analysis.webp)
*圖3: (a) 不同圖像實例在生成過程中的頻率譜演化。(b) 整體熵在不同尺度下的變化趨勢。(c) 不同Transformer層的熵熱圖。這些分析揭示了熵是捕捉建模動態的有效指標。*

---

## 實驗結果：速度與質量的雙贏

NOVA在多個基準測試中都取得了優異的成績，證明了其有效性。

在**GenEval**和**DPG-Bench**的評估中，將NOVA應用於現有的SOTA方法（如FastVAR和Infinity-2B）後，結果顯示：

- **顯著加速**: 在Infinity-2B模型上，NOVA實現了高達 **2.89倍** 的推理加速，延遲從2.43秒降低到0.84秒。
- **質量保持/提升**: 在大幅加速的同時，生成圖像的質量指標（如GenEval的Overall分數和DPG-Bench的Relation分數）幾乎沒有下降，甚至在某些情況下略有提升。

| Methods      | #Speed↑ | #Latency↓ | #Params | GenEval (Overall) | DPG-Bench (Overall) |
| :----------- | :------ | :-------- | :------ | :---------------- | :------------------ |
| Infinity-2B  | 1.00×   | 2.43s     | 2.0B    | 0.73              | 83.12               |
| **+NOVA**    | **2.89×**   | **0.84s**     | 2.0B    | **0.72**              | **82.66**               |
| FastVAR      | 1.38×   | 0.81s     | 0.7B    | 0.60              | 80.71               |
| **+NOVA**    | **1.66×**   | **0.67s**     | 0.7B    | **0.57**              | **80.56**               |

*表1: NOVA在GenEval和DPG-Bench上的性能比較。結果顯示NOVA在不同模型上均能實現顯著加速，同時保持高生成質量。*

---

## 相關研究背景

VAR模型的加速一直是研究熱點。先前的工作如**FastVAR**通過緩存和剪枝實現加速，而**ToProVAR**也利用了熵的概念，但NOVA的獨特之處在於其**無需訓練**的特性和創新的**雙重鏈接自適應機制**。與通用的token reduction方法相比，NOVA專為VAR模型的層次化、多尺度生成過程設計，使其能夠更精準地利用模型內在的動態特性。

---

## 個人評價與意義

NOVA論文為解決大型生成模型（特別是VAR模型）的效率問題提供了一個非常優雅且實用的思路。它最大的亮點在於**完全無需訓練**，這意味著它可以作為一個即插即用的模塊，輕鬆地應用於各種已有的預訓練VAR模型上，極大地降低了應用門檻。

從理論層面看，該工作深入分析了VAR模型在生成過程中的**動態演化特性**，並創造性地將**信息熵**作為核心度量，為token削減提供了堅實的理論基礎，擺脫了過去方法的啟發式弊病。**雙重鏈接**的設計充分體現了對VAR模型多尺度、多層次結構的深刻理解。

總體而言，NOVA不僅是一個有效的加速工具，更為我們理解和優化自回歸生成模型提供了一個全新的、基於信息論的視角。它完美地平衡了**效率、質量和通用性**，是近期模型壓縮與加速領域中一篇非常值得關注的佳作，有望激發更多關於如何利用模型內在動態進行自適應優化的研究。
