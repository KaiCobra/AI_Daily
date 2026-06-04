# Markovian Scale Prediction: 視覺自迴歸生成的馬可夫新紀元

隨著自迴歸（Autoregressive, AR）模型在圖像生成領域的復興，**視覺自迴歸（Visual AutoRegressive, VAR）** 憑藉「下一尺度預測（Next-Scale Prediction）」範式，實現了高畫質且穩定的由粗到細生成。然而，VAR 依賴於**全上下文依賴（Full-Context Dependency）**，即在預測當前尺度時需要關注所有歷史尺度。這種設計雖然保證了資訊流的完整性，但也帶來了嚴重的計算瓶頸：隨著解析度提升，Token 數量呈平方級增長，跨尺度的注意力計算導致記憶體與計算開銷呈超線性暴增，極大地限制了 VAR 向高解析度（如 1024×1024）的擴展。

為了解決這一痛點，來自**同濟大學、布里斯托大學與麥考瑞大學**的研究團隊在 **CVPR 2026** 發表了突破性工作：**Markov-VAR (Markovian Scale Prediction)** [1]。該研究將 VAR 重新表述為一個**非全上下文的馬可夫過程**，大膽地拋棄了全上下文依賴，並創新地引入了**滑動窗口歷史補償機制（Sliding-Window History Compensation）**。在 ImageNet 1024×1024 解析度下，Markov-VAR 將峰值記憶體消耗驚人地降低了 **83.8%**（從 117.9GB 降至 19.1GB），同時將 FID 降低了 **10.5%** [1]！這項工作不僅打破了自迴歸模型的計算壁壘，更為高效視覺生成樹立了新的里程碑。

---

## 1. 論文基本信息

*   **論文標題**：Markovian Scale Prediction: A New Era of Visual Autoregressive Generation
*   **作者團隊**：Yu Zhang, Jingyi Liu, Yiwei Shi, Qi Zhang, Duoqian Miao, Changwei Wang, Longbing Cao (同濟大學、布里斯托大學、麥考瑞大學) [1]
*   **發表會議**：**CVPR 2026** [1]
*   **論文連結**：[arXiv:2511.23334](https://arxiv.org/abs/2511.23334)
*   **專案主頁**：[Markov-VAR Page](https://luokairo.github.io/markov-var-page/)

---

## 2. 全上下文依賴的關鍵挑戰

傳統 VAR 雖然強大，但其「全上下文依賴」在實際應用中面臨三大核心挑戰 [1]：

1.  **極高的計算成本**：隨著生成尺度增加，Token 序列長度呈平方級增長。VAR 累積建模所有歷史尺度，導致 Transformer 的 Activation 與 KV Cache 記憶體消耗超線性暴增（如圖 1(a) 所示）。
2.  **持續的誤差累積**：自迴歸的單向因果鏈會導致早期預測誤差不斷傳播與放大。如圖 1(b) 所示，越早注入的擾動對最終生成品質（FID）的破壞越大。VAR 的全上下文機制會反覆利用並放大這些早期誤差。
3.  **跨尺度干擾（Cross-Scale Interference）**：VAR 要求在每個尺度學習獨特的表徵。然而，全上下文注意力會將所有歷史尺度的資訊混雜在一起，導致不同尺度的梯度與特徵在共享空間中競爭與衝突。如圖 1(c) 的**殘差特徵對齊分數（Residual-Feature Alignment, RFA）**所示，極早期尺度對當前尺度的特徵學習往往具有**負向干擾**。

![Markov-VAR Figure 2](../../assets/MarkovVAR_fig2_challenges.png)
> **圖 1：全上下文依賴帶來的挑戰分析** [1]。(a) 記憶體消耗對比；(b) 擾動在不同尺度注入時的誤差積累；(c) 殘差特徵對齊分數（RFA）顯示早期尺度對當前尺度特徵學習的負向干擾。

---

## 3. 技術方法與數學公式解析

Markov-VAR 的核心思想是：**將視覺自迴歸建模為一個馬可夫過程，並透過輕量級的歷史補償來解決非全上下文帶來的資訊遺失。**

### 3.1 傳統 VAR vs Markovian 預報的數學對比

給定 $T$ 個多尺度殘差特徵 $\{R_1, R_2, \dots, R_T\}$，其解析度尺寸為 $\{S_1 \times S_1, S_2 \times S_2, \dots, S_T \times S_T\}$。

*   **傳統 VAR 的自迴歸似然函數** [1]：
    $$p(R_1, R_2, \dots, R_T) = \prod_{t=1}^{T} p(R_t \mid \langle \text{sos} \rangle, R_{<t})$$
    其中 $R_{<t} = \{R_1, R_2, \dots, R_{t-1}\}$ 代表當前尺度 $R_t$ 的完整歷史前綴。這需要模型在預測 $R_t$ 時，注意力必須覆蓋所有 $t-1$ 個歷史尺度，導致 KV Cache 不斷累積。

*   **Markov-VAR 的馬可夫似然函數** [1]：
    $$p(R_1, R_2, \dots, R_T) = \prod_{t=1}^{T} p(R_t \mid M_{t-1})$$
    其中 $M_{t-1}$ 為第 $t-1$ 尺度的**馬可夫狀態（Markovian State）**。在這種表述下，預測當前尺度只依賴於緊鄰的前一個狀態，從而在根本上拋棄了全上下文注意力，實現了**無需 KV Cache** 的高效生成。

---

### 3.2 滑動窗口歷史補償機制（History Compensation Mechanism）

如果直接採用純馬可夫假設（即 $p(R_t \mid R_{t-1})$），由於丟棄了所有更早尺度的原始資訊，必然會面臨嚴重的資訊遺失。為此，Markov-VAR 提出了一個極其優雅且輕量的**滑動窗口歷史補償機制** [1]：

1.  **滑動窗口定義**：設定一個大小為 $N$ 的滑動窗口 $\mathcal{W}_t$，用以儲存最近的 $N$ 個連續尺度特徵：
    $$\mathcal{W}_t = \{E_{t-1}, E_{t-2}, \dots, E_{t-N}\}$$
    其中 $E_t \in \mathbb{R}^{n_t \times d}$ 是由殘差特徵 $R_{t-1}$ 經過詞嵌入（Word Embedding）與上採樣插值（Up-interpolation）得到的嵌入特徵，$n_t$ 為 Token 數量，$d$ 為特徵維度。

2.  **特徵拼接**：將窗口內的所有尺度特徵在序列維度進行拼接，得到拼接序列 $\hat{X}_t$：
    $$\hat{X}_t = \text{Concat}(X_{t-1}, X_{t-2}, \dots, X_{t-N}) \in \mathbb{R}^{(\sum_{i=1}^N n_{t-i}) \times d}$$

3.  **交叉注意力壓縮（Cross-Attention Compression）**：為了避免歷史資訊長度隨著窗口增大而膨脹，Markov-VAR 使用一個**可學習的全局查詢向量（Learnable Query）** $q \in \mathbb{R}^{1 \times d}$，透過交叉注意力將 $\hat{X}_t$ 壓縮為一個固定維度的**歷史補償向量（History Vector）** $h_{t-1}$ [1]：
    $$h_{t-1} = \text{Attn}(q, \hat{X}_t, \hat{X}_t) \in \mathbb{R}^{1 \times d}$$
    這個 $h_{t-1}$ 充當了歷史資訊的**充分統計量（Sufficient Statistic）**，以極低的維度保留了窗口內的關鍵多尺度資訊。

4.  **構建動態代表狀態（Representative Dynamic State）**：將歷史向量 $h_{t-1}$ 廣播（Broadcast）至與當前尺度特徵 $E_{t-1}$ 相同的序列長度 $H_{t-1} = \mathbf{1}_{n_{t-1}} h_{t-1}^\top$，並在通道維度進行拼接，構建出最終的動態馬可夫狀態 $M_{t-1}$ [1]：
    $$M_{t-1} = \text{Concat}(E_{t-1}, H_{t-1}) \in \mathbb{R}^{n_{t-1} \times 2d}$$

這個動態狀態 $M_{t-1}$ 既包含了當前尺度的精細局部特徵，又融合了歷史尺度的全局上下文，隨後送入 Markov-VAR Transformer 中進行下一個尺度的預測。

![Markov-VAR Figure 3](../../assets/MarkovVAR_fig3_method.png)
> **圖 2：VAR 與 Markov-VAR 預報模式對比及 Markov-VAR 整體架構圖** [1]。

---

### 3.3 訓練與推理策略

*   **Teacher-Forcing 訓練**：在訓練階段，Markov-VAR 採用 Teacher-Forcing 模式。由於馬可夫性質，每個尺度的預測在計算上是相互獨立的。這使得模型可以採用**馬可夫注意力遮罩（Markovian Attention Mask）**，限制每個尺度僅關注當前狀態，實現了並行化訓練。
*   **推理無 KV Cache**：在推理生成時，由於模型不再需要關注所有歷史 Token，因此**完全不需要維護龐大的 KV Cache**，這也是其記憶體開銷暴降 80% 以上的根本原因。

---

## 4. 實驗結果與性能指標

研究團隊在 ImageNet-1K 類別條件圖像生成基準上對 Markov-VAR 進行了全面評估。

### 4.1 圖像生成品質對比

如表 1 所示，在 ImageNet 256×256 解析度下，Markov-VAR 在不同模型規模下均顯著超越了基準 VAR [1]：
*   **Markov-VAR-d16** (329M)：FID 從 3.61 降至 **3.23**（提升 10.5%），IS 從 225.6 升至 **256.2**。
*   **Markov-VAR-d24** (1.02B)：FID 達到 **2.15**，IS 達到 **310.9**，超越了相同參數量的 FlexVAR-d24 與 NestAR-H 等變體。

| 模型 | 參數量 | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| VAR-d16 [2] | 310M | 3.61 | 225.6 | 0.81 | 0.52 |
| **Markov-VAR-d16** (Ours) | **329M** | **3.23** | **256.2** | **0.84** | **0.52** |
| VAR-d20 [2] | 600M | 2.67 | 254.4 | 0.81 | 0.57 |
| **Markov-VAR-d20** (Ours) | **623M** | **2.44** | **286.1** | **0.83** | **0.56** |
| VAR-d24 [2] | 1.0B | 2.17 | 271.9 | 0.81 | 0.59 |
| **Markov-VAR-d24** (Ours) | **1.02B** | **2.15** | **310.9** | **0.83** | **0.59** |

> **表 1：ImageNet 256×256 類別條件圖像生成性能對比** [1]。

---

### 4.2 生成效率與記憶體開銷（核心優勢）

如表 2 所示，隨著解析度提升，Markov-VAR 的效率優勢呈指數級放大 [1]：
*   在 **1024×1024** 解析度下，基準 VAR-d24 的峰值記憶體高達 **117.9GB**，而 Markov-VAR-d24 僅需 **19.1GB**，記憶體開銷暴降 **83.8%**！
*   在推理速度上，Markov-VAR 相比 VAR 也實現了約 **10% - 15% 的加速**，且顯著快於 FlexVAR。

| 模型 | 深度 | 解析度 | 推理時間 (s/batch) ↓ | 峰值記憶體 (GB) ↓ | 記憶體節省比例 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| VAR-d24 | 24 | 256 | 0.711 | 12.4 | - |
| **Markov-VAR-d24** | 24 | 256 | **0.608** | **4.7** | **62.1%** |
| VAR-d24 | 24 | 512 | 1.335 | 31.4 | - |
| **Markov-VAR-d24** | 24 | 512 | **1.261** | **8.1** | **74.2%** |
| VAR-d24 | 24 | 1024 | 5.891 | 117.9 | - |
| **Markov-VAR-d24** | 24 | 1024 | **5.322** | **19.1** | **83.8%** |

> **表 2：不同解析度下的推理時間與 GPU 峰值記憶體開銷對比** [1]（Batch Size = 25，單張 H200 GPU）。

---

### 4.3 擴展規律（Scaling Law）與消融實驗

*   **滑動窗口大小消融**：如表 3 所示，當滑動窗口大小 $N=3$ 時，模型在各深度下均取得最佳性能 [1]。這與前文「跨尺度干擾」的分析高度一致：最近的 3 個尺度提供了最關鍵的歷史特徵，而過遠的尺度（如 $N=4$ 或全上下文）反而會引入噪聲與干擾。
*   **Scaling Law 驗證**：研究團隊對參數規模從 19.8M 到 1.02B 的模型進行了擬合，發現 Loss 與 Error Rate 均嚴格遵循冪律分佈（Power-Law），證明了 Markov-VAR 具備極佳的擴展潛力（如圖 3 所示）。

| 窗口大小 $N$ | FID (d16) ↓ | IS (d16) ↑ | FID (d20) ↓ | IS (d20) ↑ |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 3.53 | 237.8 | 2.50 | 267.9 |
| 2 | 3.39 | 248.6 | 2.47 | 281.4 |
| **3 (Ours)** | **3.23** | **256.2** | **2.44** | **286.1** |
| 4 | 3.33 | 252.3 | 2.56 | 278.2 |

> **表 3：滑動窗口大小 $N$ 的消融研究** [1]。

![Markov-VAR Figure 5](../../assets/MarkovVAR_fig5_scaling.png)
> **圖 3：Markov-VAR 的 Scaling Law 冪律擬合曲線** [1]。

---

## 5. 相關研究對比：Markov-VAR (CVPR 2026) vs MVAR (ICLR 2026)

在視覺自迴歸的馬可夫化探索中，讀者可能會聯想到另一篇發表於 **ICLR 2026** 的工作：**MVAR (UESTC)** [3]。雖然兩者都探討了馬可夫性質在 VAR 中的應用，但在設計哲學與實現路徑上存在本質區別，這非常值得深入對比：

1.  **歷史資訊流的處理（核心區別）**：
    *   **MVAR (ICLR 2026)** [3] 採用了**強馬可夫假設**，預測第 $l$ 層時，**直接拋棄**了除緊鄰前一層 $r_{l-1}$ 之外的所有歷史尺度。這種設計極其激進，雖然實現了完全的並行訓練，但不可避免地造成了嚴重的歷史資訊遺失。
    *   **Markov-VAR (CVPR 2026)** [1] 則認識到「純馬可夫」的局限性，提出了**滑動窗口歷史補償機制**。它利用交叉注意力將最近 $N$ 個尺度的資訊壓縮為一個緊湊的歷史向量 $h_{t-1}$。這相當於在馬可夫鏈中引入了「記憶特徵」，既保留了多尺度全局上下文，又避免了全上下文注意力的計算膨脹。
2.  **注意力機制的優化維度**：
    *   **MVAR** 同時關注「尺度馬可夫」與「空間馬可夫」，引入了**空間馬可夫注意力（Spatial-Markov Attention）**，將 Token 的關注範圍限制在局部鄰域 $k$ 內，將空間複雜度從 $O(N^2)$ 降至 $O(Nk)$ [3]。
    *   **Markov-VAR** 則聚焦於「尺度預測模式」的重構。它透過引入歷史補償向量，將馬可夫狀態 $M_{t-1}$ 與當前特徵進行通道拼接，從而讓 Transformer 能夠在不依賴任何 KV Cache 的情況下，自然地學習到更具代表性的動態狀態演化 [1]。
3.  **生成品質與效率的權衡**：
    *   由於 MVAR 丟棄了較多歷史資訊，其生成品質（FID）在很大程度上依賴於空間注意力的局部性約束。
    *   Markov-VAR 則透過優雅的滑動窗口壓縮，不僅在效率上（1024 解析度下記憶體降低 83.8%）達到了與 MVAR 相當甚至更優的水平，而且在生成品質上實現了對基準 VAR 的**全面超越**（FID 降低 10.5%），真正做到了「既要效率，又要品質」 [1]。

---

## 6. 個人評價與未來研究啟發

Markov-VAR 是一項令人拍案叫絕的工作。它用資訊論中的**充分統計量（Sufficient Statistic）**視角，重新審視了自迴歸模型中的上下文冗餘。其提出的「滑動窗口 + 交叉注意力壓縮」機制，本質上是在**狀態空間（State Space）的緊湊性**與**資訊流的完整性**之間找到了一個近乎完美的黃金分割點。

對於希望激發研究靈感的讀者，這項工作與多個前沿方向存在深刻的內在聯繫：

### 6.1 與聯合能量預測架構（JEPA）的潛在融合
Yann LeCun 提倡的 **JEPA (Joint Embedding Predictive Architecture)** 核心在於「在表徵空間而非像素空間進行預測」，並強調摒棄不必要的細節以學習世界模型。
*   **啟發**：Markov-VAR 的歷史補償向量 $h_{t-1}$ 本質上就是對歷史多尺度表徵的一個「最優壓縮表徵」。如果我們將 Markov-VAR 的生成過程與 JEPA 的非對稱預測器（Predictor）相結合，讓模型在編碼器的潛在表徵空間中進行馬可夫狀態演化，是否能構建出一個具備強大物理世界常識推理能力的**視覺世界模型（Visual World Model）**？這將極大地推動無監督視頻預測與機器人操作規劃的發展。

### 6.2 與能量自迴歸模型（Energy-Based Transformer, EBT）的互補
**EBT (Energy-Based Transformer)** 透過定義能量函數來評估生成樣本的合理性，並透過能量最小化（如 MCMC 採樣）來進行自我修正，這在「System 2」慢思考與推理任務中展現了巨大潛力。
*   **啟發**：自迴歸生成（包括 VAR）的最大痛點是「一步走錯，步步皆錯」的誤差累積（如圖 1(b) 所示）。雖然 Markov-VAR 透過非全上下文緩解了早期誤差的反覆放大，但仍無法主動修正誤差。如果我們在 Markov-VAR 的馬可夫狀態轉移 $M_{t-1} \rightarrow M_t$ 之間，引入一個**輕量級的能量正則化器（Energy Regularizer）**，在每個尺度生成後進行微小的能量最小化調整（Training-free alignment），是否能以極低的代碼代價，徹底消除自迴歸模型的幻覺與畸變？

### 6.3 免訓練注意力調製（Training-Free Attention Modulation）
Markov-VAR 發現早期尺度對當前尺度的特徵學習存在負面干擾（RFA Score < 0），這與大語言模型中的 "Attention Sink" 或擴散模型中的 "Feature Drift" 異曲同工。
*   **啟發**：這意味著我們甚至不需要重新訓練模型！對於現有的預訓練 VAR 模型，我們可以設計一種**免訓練的衰減遮罩（Training-Free Decay Mask）**，讓當前 Token 對歷史 Token 的注意力權重隨著尺度跨度呈指數衰減，或者主動剪枝掉 RFA 分數為負的早期尺度 Token。這種純推理端的注意力調製，有望直接提升現有 VAR 模型的生成品質，實現零成本的性能飛躍。

---

## 7. 總結

**Markov-VAR** 證明了視覺自迴歸生成並不需要沉重的「歷史包袱」。透過將全上下文依賴重構為帶有輕量補償的馬可夫過程，它在解析度擴展性上展現了驚人的威力，讓 1024×1024 的自迴歸圖像生成在單張消費級顯卡上成為可能。這項工作不僅是 VAR 技術路線的一次重大勝利，也為未來結合世界模型、能量模型以及更高效的免訓練調製技術開闢了廣闊的想像空間。

---

## 參考文獻

[1] Y. Zhang et al., "Markovian Scale Prediction: A New Era of Visual Autoregressive Generation," *arXiv preprint arXiv:2511.23334*, 2026. [Online]. Available: https://arxiv.org/abs/2511.23334

[2] K. Tian et al., "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction," *arXiv preprint arXiv:2404.02905*, 2024. [Online]. Available: https://arxiv.org/abs/2404.02905

[3] J. Zhang et al., "MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning," *International Conference on Learning Representations (ICLR)*, 2026. [Online]. Available: https://arxiv.org/abs/2505.12742
