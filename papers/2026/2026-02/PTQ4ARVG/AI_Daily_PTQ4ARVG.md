# AI Daily: PTQ4ARVG - 無需訓練，為自回歸視覺生成模型量身打造的量化框架

> **論文標題**: PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models
> **作者**: Xuewen Liu, Zhikai Li, Jing Zhang, Mengjuan Chen, Qingyi Gu
> **機構**: Institute of Automation, Chinese Academy of Sciences
> **會議/期刊**: ICLR 2026
> **arXiv**: [2601.21238](https://arxiv.org/abs/2601.21238)
> **代碼**: [http://github.com/BienLuky/PTQ4ARVG](http://github.com/BienLuky/PTQ4ARVG)

---

## 總結

隨著自回歸視覺生成（ARVG）模型在性能上逐漸追上甚至超越擴散模型，其巨大的模型尺寸和計算開銷成為了部署的瓶頸。來自中科院自動化所的研究團隊提出了 **PTQ4ARVG**，這是首個專為 ARVG 模型設計的**無需訓練的後訓練量化（Post-Training Quantization, PTQ）**框架。該框架通過創新的理論驅動方法，成功將 ARVG 模型量化至 8-bit 甚至 6-bit，同時幾乎不損失性能，為在資源受限設備上高效部署大型視覺生成模型鋪平了道路。

## 核心貢獻與創新點

PTQ4ARVG 框架的核心在於識別並解決了 ARVG 模型量化中的三大獨特挑戰，並為每個挑戰設計了針對性的解決方案：

1.  **識別三大挑戰**: 論文首次系統性地分析了 ARVG 模型量化的難點：
    *   **通道級異常值 (Channel-wise Outliers)**: AdaLN 模組導致的激活值在不同通道間範圍差異巨大。
    *   **權杖級動態性 (Token-wise Dynamics)**: 位置嵌入和條件權杖導致激活值在權杖維度上高度動態且存在「沉沒權杖 (sink tokens)」現象。
    *   **樣本級分佈失配 (Sample-wise Mismatch)**: 不同校準樣本間的激活值分佈高度相似，導致校準信息冗餘。

2.  **理論驅動的縮放策略 (Gain-Projected Scaling, GPS)**: 針對通道級異常值，作者沒有採用傳統的經驗性方法，而是提出了基於數學優化的 **GPS**。該方法通過泰勒展開量化損失，從理論上推導出最優的縮放因子，以最小化激活值和權重的總體量化誤差。這是**首個基於數學優化推導的量化縮放策略**，具有堅實的理論基礎。

3.  **零開銷的靜態權杖量化 (Static Token-Wise Quantization, STWQ)**: 針對權杖級動態性，STWQ 利用 ARVG 模型生成固定長度權杖序列的特性，為每個權杖位置分配**靜態的、細粒度的量化參數**。這避免了動態量化在推理時引入的額外開銷，同時通過百分位校準保證了高精度。

4.  **分佈引導的校準集選擇 (Distribution-Guided Calibration, DGC)**: 為了處理樣本級的分佈失配，DGC 使用馬氏距離（Mahalanobis distance）評估每個樣本對整體分佈熵的貢獻，從而選擇信息量最大、最具代表性的樣本進行校準，提高了校準效率和準確性。

## 技術方法簡述

PTQ4ARVG 框架由 GPS、STWQ 和 DGC 三個核心組件構成，協同解決 ARVG 量化難題。

### 1. Gain-Projected Scaling (GPS)

GPS 的目標是找到一個最優的縮放因子 $\mathbf{s}$，使得縮放後的激活值 $\mathbf{x}\' = \mathbf{x} \oslash \mathbf{s}$ 和權重 $\mathbf{W}\' = \mathbf{W} \otimes \mathbf{s}$ 的總體量化損失最小。論文將縮放帶來的「增益」定義為：

$$\text{Gain} = (E_{\mathbf{x}} - E_{\mathbf{x}}\' ) - (E_{\mathbf{W}}\' - E_{\mathbf{W}})$$

其中，$E_{\mathbf{x}}$ 和 $E_{\mathbf{W}}$ 分別是激活值和權重的量化損失。通過對其進行泰勒展開並近似，論文推導出縮放增益的表達式，並通過對縮放因子 $s_i$ 求導並令其為零，得到最優解：

$$s_i = \sqrt[4]{\frac{\mathbb{E}[\mathbf{W}_{j,i}^2 \Delta x_i^2]}{\mathbb{E}[\Delta \mathbf{W}_{j,i}^2 \mathbf{x}_i^2]}}$$

這個公式直觀地平衡了縮放對激活值量化誤差的減小和對權重量化誤差的增大，找到了最佳平衡點。

### 2. Static Token-Wise Quantization (STWQ)

與 LLM 中常用的動態權杖級量化不同，STWQ 利用了 ARVG 模型權杖長度固定的特性。如下圖所示，不同類別樣本的激活值分佈在權杖維度上表現出一致性（位置不變性）。

![Figure 4: Inputs of AdaLN in RAR-B from different sample class. The distribution remains invariant across samples.](asset/ptq4arvg_figure4_adaln.webp)

基於此觀察，STWQ 可以為每個權杖位置**離線計算並存儲**一組靜態的量化參數（縮放因子和零點），從而在推理過程中完全沒有額外開銷。這巧妙地解決了「沉沒權杖」等動態性問題，同時避免了性能損失。

### 3. Distribution-Guided Calibration (DGC)

DGC 的核心是使用馬氏距離來度量一個樣本 $\mathbf{x}$ 與整個校準集 $(\mathbf{u}, \mathbf{S})$ 的「距離」，這個距離反映了該樣本對分佈熵的貢獻：

$$\rho(\mathbf{x}) = \sqrt{(\mathbf{x} - \mathbf{u})^T \mathbf{S}^{-1} (\mathbf{x} - \mathbf{u})}$$

通過選擇 $\rho(\mathbf{x})$ 最大的樣本，DGC 構建了一個小而精的校準集，確保了量化參數的準確性。

## 實驗結果和性能指標

PTQ4ARVG 在多個主流 ARVG 模型（VAR, RAR, PAR, MAR）上進行了廣泛驗證，並取得了 SOTA 性能。

![Table 1: Comparative results for VAR and RAR models.](asset/ptq4arvg_experiment_table.webp)

從上表可以看出：

-   **W8A8 (8-bit)**: PTQ4ARVG 在 VAR 和 RAR 模型上均取得了與 FP32（全精度）非常接近的性能。例如，在 RAR-XL 上，FID 僅從 1.54 微降至 1.58，幾乎無損。
-   **W6A6 (6-bit)**: 在更具挑戰性的 6-bit 量化下，PTQ4ARVG 的優勢更加明顯。在 VAR-d16 上，其 FID (8.34) 遠優於次優方法 SmoothQuant (18.54) 和 OmniQuant (22.19)，性能提升顯著。

消融實驗也證明了 GPS、STWQ 和 DGC 三個組件的不可或缺性，每個組件都對最終性能有重要貢獻。

## 相關研究背景

-   **自回歸視覺生成 (ARVG)**: 受 LLM 啟發，將圖像建模為離散權杖序列，並通過自回歸方式生成。代表模型有 VAR、PAR、RAR 等，它們在生成質量和可擴展性上展現了巨大潛力。
-   **神經網絡量化**: 將浮點數表示的權重和激活值轉換為低比特整數，以壓縮模型、加速推理。主要分為 QAT（量化感知訓練）和 PTQ（後訓練量化）。PTQ 因其無需重新訓練、數據依賴少而備受青睞。
-   **大模型量化挑戰**: 大模型（包括 LLM 和 ARVG）的量化面臨激活值中存在顯著異常值（Outliers）的挑戰，這也是 PTQ4ARVG 中 GPS 方法重點解決的問題。

## 個人評價和意義

PTQ4ARVG 是一項非常紮實且具有高度實用價值的工作。它不僅是首個系統性解決 ARVG 模型量化問題的框架，更重要的是，其核心組件 **GPS** 提出了一種**理論驅動**而非經驗試錯的縮放因子計算方法，為大模型量化領域提供了新的思路。

這項工作最大的意義在於，它證明了在**無需任何訓練**的情況下，僅通過精巧的數學建模和對模型特性的深刻洞察，就能將數十億參數的視覺生成模型壓縮至 6-bit，同時保持高質量的生成效果。這極大地降低了 ARVG 模型在邊緣設備、移動端等資源受限場景下的部署門檻，有望推動 ARVG 技術的廣泛應用。

對於研究者而言，這篇論文激發了我們去思考：

1.  **模型特有屬性的利用**: 如何更深入地挖掘特定模型架構（如 ARVG 的固定權杖長度）的內在屬性，並將其轉化為算法設計上的優勢？
2.  **理論與實踐的結合**: 在量化、剪枝等模型壓縮領域，如何從經驗驅動轉向更具普適性的理論驅動方法？

總體而言，PTQ4ARVG 是近期模型壓縮領域一篇不可多得的佳作，其提出的方法和思想對所有大模型的量化研究都具有重要的參考價值。

---

**參考文獻**

[1] Liu, X., Li, Z., Zhang, J., Chen, M., & Gu, Q. (2026). *PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models*. arXiv preprint arXiv:2601.21238.
