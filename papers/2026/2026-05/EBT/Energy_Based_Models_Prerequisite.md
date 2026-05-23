# 先備知識：能量模型 (Energy-Based Models, EBMs) 核心概念與原理

為了更好地理解 Energy-Based Transformers (EBTs)，我們需要先掌握其基礎——**能量模型 (Energy-Based Models, EBMs)**。本指南將從大概念出發，逐步解析核心數學公式，並介紹近年來基於 EBM 的重要研究。

---

## 1. 大概念：什麼是能量模型？

在機器學習與生成模型中，我們的目標通常是學習數據的機率分佈 $p(x)$。傳統方法（如 VAE、GAN 或自迴歸模型）會直接嘗試預測機率或生成數據。然而，要確保一個函數的輸出是合法的機率分佈（即所有可能情況的機率總和為 1），在數學上非常困難。

**能量模型 (EBMs) 提供了一個截然不同的視角，其靈感來自於統計物理學。**

想像一個物理系統（例如掉落的蘋果或冷卻的金屬），系統總是傾向於處於**能量最低**的狀態。EBM 借用了這個概念：
* 我們不直接預測機率，而是訓練一個神經網路來輸出一個**標量值（能量，Energy）**。
* **低能量**代表該狀態（數據）是**合理、真實、常見**的。
* **高能量**代表該狀態（數據）是**不合理、虛假、罕見**的。

這就像是在高維空間中雕刻地形：我們希望真實數據（如真實的貓咪圖片）位於深深的「山谷」（低能量），而隨機噪聲或不合理的圖片則位於「高山」（高能量）。當我們需要生成新數據時，只需將隨機噪聲順著地形「滾下山」，尋找能量最低的點即可。

---

## 2. 數學公式逐項解析

要將「能量」轉換為合法的「機率」，EBM 使用了統計力學中的**波茲曼分佈 (Boltzmann Distribution)**。

### 核心公式：能量轉機率

$$ p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)} $$

**逐項解釋：**
* **$x$**：輸入數據（例如一張圖片、一段文字）。
* **$\theta$**：神經網路的參數（權重）。
* **$E_\theta(x)$**：能量函數（Energy Function）。這就是我們訓練的神經網路，它接收 $x$ 並輸出一個代表能量的實數。
* **$\exp(-E_\theta(x))$**：指數函數。這一步有兩個目的：
    1. 確保輸出的值永遠是正數。
    2. 將「低能量」轉換為「高數值」，將「高能量」轉換為「低數值」。
* **$Z(\theta)$**：配分函數 (Partition Function)。它的定義是所有可能狀態的未歸一化機率的總和（或積分）：$Z(\theta) = \int \exp(-E_\theta(x)) dx$。
    * **作用**：因為 $\exp(-E_\theta(x))$ 只是未歸一化的值，除以 $Z(\theta)$ 可以確保所有 $x$ 的機率總和為 1，從而使其成為合法的機率密度函數。
    * **痛點**：在連續或高維空間（如圖像）中，計算所有可能圖像的積分是**不可能的 (intractable)**。這正是 EBM 訓練中最困難的地方。

### 訓練公式：分數匹配 (Score Matching) 與分數函數 (Score Function)

因為 $Z(\theta)$ 無法計算，直接使用最大似然估計 (Maximum Likelihood) 訓練 EBM 非常困難。為了解決這個問題，研究人員引入了**分數函數 (Score Function)** 的概念 [1]。

分數函數定義為機率密度對數的梯度：

$$ s(x) = \nabla_x \log p_\theta(x) $$

**為什麼要用分數函數？** 讓我們將 EBM 的公式代入：

$$ s(x) = \nabla_x \log \left( \frac{\exp(-E_\theta(x))}{Z(\theta)} \right) $$
$$ s(x) = \nabla_x \left( -E_\theta(x) - \log Z(\theta) \right) $$

因為 $Z(\theta)$ 是一個常數（對特定的 $\theta$ 而言，不依賴於 $x$），所以它對 $x$ 的梯度為 0。因此：

$$ s(x) = -\nabla_x E_\theta(x) $$

**重大突破：** 透過計算分數（Score），我們**完全避開了無法計算的配分函數 $Z(\theta)$**！分數函數 $s(x)$ 代表了機率密度增加最快的方向，也就是能量下降最快的方向。這為後來的 Score-based models 和 Diffusion models 奠定了理論基礎。

### 生成公式：朗之萬動力學 (Langevin Dynamics)

當我們訓練好能量函數 $E_\theta(x)$ 後，如何生成新數據？我們使用**朗之萬動力學**（一種馬可夫鏈蒙地卡羅 MCMC 方法）[2]：

$$ x_{t+1} = x_t - \frac{\alpha}{2} \nabla_x E_\theta(x_t) + \sqrt{\alpha} \epsilon $$

**逐項解釋：**
* **$x_t$**：當前步驟的數據狀態（從隨機噪聲 $x_0$ 開始）。
* **$\alpha$**：步長（Step size），控制每次更新的幅度。
* **$-\nabla_x E_\theta(x_t)$**：能量函數的負梯度。這引導數據朝著**能量更低（機率更高）**的方向移動（類似於梯度下降）。
* **$\epsilon \sim \mathcal{N}(0, I)$**：標準常態分佈的隨機噪聲。
    * **作用**：如果只做梯度下降，數據會卡在局部最小值（Local minima）。加入隨機噪聲可以讓數據在能量景觀中「抖動」，幫助它跳出局部最優，探索更多樣的真實數據分佈。

---

## 3. 重要應用與相關研究 (2019-2026)

EBM 的概念深刻影響了現代深度學習的發展，以下是幾個具有里程碑意義的應用與研究：

### 1. 隱式生成與泛化 (Implicit Generation and Generalization in EBMs)
* **論文**：*Implicit Generation and Generalization in Energy-Based Models* (NeurIPS 2019, Yilun Du, Igor Mordatch) [3]
* **貢獻**：這篇論文證明了可以直接在連續高維空間（如圖像）上訓練基於神經網路的 EBM，並使用 Langevin Dynamics 進行採樣。他們展示了 EBM 在圖像生成、修復（Inpainting）以及出色的分佈外（OOD）泛化能力。EBT 論文的第一作者群中也包含了 Yilun Du。

### 2. 你的分類器其實是能量模型 (JEM)
* **論文**：*Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One* (ICLR 2020, Will Grathwohl et al.) [4]
* **貢獻**：這項研究提出了一個驚人的觀點：任何帶有 Softmax 輸出的標準判別式分類器，都可以被重新解釋為一個聯合能量模型 (Joint Energy-based Model, JEM)。這意味著我們可以用同一個模型同時進行高準確率的分類和高質量的圖像生成，且具有極強的對抗魯棒性。

### 3. 分數生成模型與擴散模型 (Score-Based Generative Modeling & Diffusion)
* **論文**：*Score-Based Generative Modeling through Stochastic Differential Equations* (NeurIPS 2020, Yang Song et al.) [5]
* **貢獻**：雖然這篇論文通常被視為 Diffusion Models 的基石，但其核心理論「Score Matching」正是為了解決 EBM 中配分函數無法計算的問題而生的。Yang Song 等人將 Score-based models 與 Diffusion models 統一在隨機微分方程 (SDE) 的框架下，徹底改變了生成式 AI 的發展軌跡。

### 4. Yann LeCun 的 JEPA 願景
* **架構**：*Joint-Embedding Predictive Architecture (JEPA)* [6]
* **關聯**：圖靈獎得主 Yann LeCun 近年來極力推廣 JEPA 架構（如 I-JEPA, V-JEPA），這是一種非生成式的自監督學習世界模型。LeCun 將 JEPA 描述為一種**基於能量的模型 (Energy-Based Model)**，其目標是最小化輸入與預測之間的能量（預測誤差），而不是像傳統生成模型那樣重建像素。這與 EBT 論文中「驗證輸入與預測相容性」的思想高度一致。

### 5. 現代 Hopfield 網路 (Modern Hopfield Networks)
* **概念**：Hopfield 網路是早期的一種聯想記憶神經網路，本質上也是一種 EBM（透過最小化能量來回憶記憶）。近年來提出的「現代 Hopfield 網路」引入了新的能量函數，大大提升了記憶容量，並被證明與 Transformer 中的 Self-Attention 機制在數學上是等價的。這進一步印證了 Energy-Based 概念與 Transformer 架構結合的巨大潛力。

---

## 結語

理解了 Energy-Based Models，我們就能明白 **Energy-Based Transformers (EBTs)** 的革命性在於：它將 EBM 靈活的「相容性驗證（能量評估）」能力，與 Transformer 強大的「序列建模與擴展」能力結合在一起。透過將預測過程轉化為**能量最小化的優化過程**，EBTs 成功地在無監督學習中實現了類似人類的 System 2 Thinking（慢思考）。

## 參考文獻
[1] Vizuara AI. (2026). Energy Based Models - Score Matching.
[2] Aliakbarian, S. (2025). Bridging Energy, Score, and Diffusion in Generative Modeling.
[3] Du, Y., & Mordatch, I. (2019). Implicit Generation and Generalization in Energy-Based Models. *NeurIPS 2019*.
[4] Grathwohl, W., et al. (2020). Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One. *ICLR 2020*.
[5] Song, Y., et al. (2020). Score-Based Generative Modeling through Stochastic Differential Equations. *NeurIPS 2020*.
[6] LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.
