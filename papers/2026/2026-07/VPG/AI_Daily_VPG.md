# AI Daily: VPG - 視覺自迴歸模型的前綴引導方法

## 論文基本信息

| 項目 | 內容 |
|------|------|
| **論文標題** | VPG: Visual Prefix Guidance for Autoregressive Image and Video Generation |
| **作者** | Xinyao Liao¹*, Qiyuan He¹, Yicong Li¹, Jiayin Zhu¹, Xiaoye Qu², Wei Wei², Angela Yao¹ |
| **機構** | ¹ National University of Singapore, ² Huazhong University of Science & Technology |
| **發表日期** | 2026-05-28 |
| **arXiv ID** | 2605.30317 |
| **論文狀態** | Preprint (未發表於頂會) |
| **引用領域** | 計算機視覺、圖像生成、視頻合成 |

## 核心貢獻

VPG（Visual Prefix Guidance）提出了一種**訓練無關的推理時引導方法**，用於改進視覺自迴歸模型的生成質量。論文的核心創新在於從**前綴後驗支撐**的新視角解決自迴歸模型的暴露偏差問題，而非傳統的外部條件引導。

### 主要貢獻包括：

**首次系統性解決視覺 AR 的前綴支撐問題**。論文識別了一個先前未被充分探索的推理時目標：通過強化生成前綴的後驗支撐來改進下一步預測。這直接針對前綴中暴露偏差的累積。

**提出 VPG 方法**。論文提出了一種訓練無關、即插即用的採樣規則，通過腐蝕前綴對比和同尺度全嵌入替換實現前綴後驗目標，無需輔助頭或重新訓練。

**跨模型驗證與性能提升**。在 VAR、Infinity 和 InfinityStar 等多個 SOTA 自迴歸模型上驗證，VAR 上 FID 平均降低 0.36（最高 0.63），InfinityStar 視頻生成 VBench 分數提升 0.49。

## 問題背景與動機

### 視覺自迴歸模型的暴露偏差

視覺自迴歸模型（如 VAR、Infinity、InfinityStar）在訓練和推理時存在根本性的不匹配。在訓練階段，模型採用**教師強制（Teacher Forcing）**策略，每個預測步驟都基於**真實前綴**（ground-truth prefix）。然而在推理時，模型必須基於**自生成的前綴**進行預測，因為真實前綴不可用。

這種訓練-推理不匹配導致**暴露偏差（Exposure Bias）**問題。早期採樣誤差會將模型推入未見過或低概率的前綴狀態，這些錯誤在後續預測中不斷累積。在視覺生成中，前綴漂移表現為結構錯誤、物體不一致和高頻細節不穩定。

### 現有方法的局限

現有解決方案主要分為兩類。**訓練時方法**在優化中混入生成或擾動的歷史，但需要修改訓練流程，增加計算成本。**採樣時方法**（如 CFG）強化外部語義條件（如文本提示），但主要針對條件軸，未直接處理前綴漂移。

VPG 的創新在於提出了一個**新的引導軸**：不是強化外部條件與生成內容的對齐，而是確保下一步預測對已生成的前綴提供**強後驗支撐**。這是對 CFG 沿著不同維度的補充。

## 技術方法

### 核心思想與數學框架

標準視覺自迴歸模型採用尺度級預測，將圖像生成建模為：

$$p_\theta(R|c) = \prod_{k=1}^{K} p_\theta(r_k | r_{<k}, c)$$

其中 $r_k$ 是第 $k$ 尺度的殘差標記映射，$r_{<k}$ 是前綴（所有先前尺度），$c$ 是外部條件（如文本提示）。

標準採樣遵循條件似然 $p_\theta(r_k | r_{<k}, c)$。VPG 引入不同視角，在每一步偏好增強**前綴後驗支撐**的候選：

$$p(r_{<k} | r_k, c)$$

這是 CFG 沿著不同軸的補充：
- **CFG 強化**：$p(c | r_{\leq k})_\gamma$（條件軸）
- **VPG 強化**：$p(r_{<k} | r_k, c)_\lambda$（前綴軸）

### 前綴對比實現

VPG 通過**配對預測對比**實現前綴後驗目標：

1. **真實分支**：使用生成的前綴 $r_{<k}$ 獲得邏輯 $\ell_k(c, r_{<k})$
2. **腐蝕分支**：使用腐蝕前綴 $\tilde{r}_{<k}$ 獲得邏輯 $\ell_k(c, \tilde{r}_{<k})$

腐蝕策略採用**同尺度全嵌入替換**：在每個前綴尺度，隨機替換 Token 位置嵌入的一部分，從同一尺度的其他位置複製完整嵌入。這保持了模型的尺度條件輸入統計特性，同時創建了一個較弱的前綴參考。

### 邏輯外推公式

VPG 的邏輯外推公式為：

$$\ell_{\text{VPG}}(c, r_{<k}) = \ell_k(c, r_{<k}) + \lambda(\ell_k(c, r_{<k}) - \ell_k(c, \tilde{r}_{<k}))$$

其中 $\lambda$ 是引導強度超參數。這個公式在真實邏輯基礎上，沿著對比方向進行外推，使得模型傾向於選擇在真實前綴下有高似然但在腐蝕前綴下似然較低的候選。

### 與 CFG 的互補性

CFG 和 VPG 可以組合使用，形成 CFG+VPG，同時沿著條件軸和前綴軸進行引導。表 1 總結了不同引導方法的對比：

| 方法 | 固定軸 | 對比軸 | 引導依賴 |
|------|--------|--------|---------|
| CFG | $r_{<k}$ | $c$ vs. $\emptyset$ | 外部條件 $c$ |
| VPG | $c$ | $r_{<k}$ vs. $\tilde{r}_{<k}$ | 生成前綴 $r_{<k}$ |
| CFG+VPG | – | 兩者對比 | $c$ 和 $r_{<k}$ |

## 實驗結果

### 1. VAR 類別條件圖像生成

在 VAR 模型上進行類別條件圖像生成評估，VPG 展現了一致的性能改進：

- **FID 改進**：平均降低 **0.36**，在 VAR-d16 上達到最高 **0.63**
- **跨模型一致性**：在多個模型大小上均有改進（VAR-d10、VAR-d16、VAR-d20、VAR-d30）
- **無需重新訓練**：完全免訓練方法，即插即用

### 2. Infinity 文本到圖像生成

在 Infinity 模型上進行文本到圖像生成評估：

- **GenEval Overall**：改進文本對齐評估指標
- **DPG-Bench Overall**：改進多維度綜合評估
- **應用範圍**：支持高解析度圖像生成

### 3. InfinityStar 文本到視頻生成

在 InfinityStar 模型上進行文本到視頻生成評估，展現了視頻質量的全面提升：

- **VBench Overall Score**：改進 **0.49**
- **全面改進**：所有子分數均有提升，包括場景一致性、物體一致性、動作一致性等
- **視頻質量**：改進視頻生成的結構一致性和時間連貫性

## 相關研究背景

### 視覺自迴歸模型的發展

**VAR（Visual Autoregressive Modeling）** [1] 於 2024 年發表，獲得 NeurIPS 2024 最佳論文獎。VAR 重新定義了圖像自迴歸為尺度級預測（next-scale prediction），採用從粗到細的多尺度生成策略，性能超越擴散模型。

**Infinity** [2] 於 2025 年發表，CVPR 2025 Oral。Infinity 引入比特級標記化（Bitwise Tokenization），支持極大詞彙表，實現高解析度、光度真實的圖像生成。

**InfinityStar** [3] 於 2025 年發表，NeurIPS 2025 Oral。InfinityStar 提出統一時空自迴歸框架，支持圖像與視頻生成，VBench 評分 83.74，超越多個擴散模型基線。

### 暴露偏差研究

暴露偏差問題在自然語言處理中已被廣泛研究。經典工作 [4] 分析了訓練-測試不匹配的理論基礎。傳統解決方案包括 Scheduled Sampling 等訓練時方法。在視覺生成中，近期工作 [5] 通過正則化自生成展開或添加訓練後優化來改進魯棒性。

### 採樣時引導方法

**CFG（Classifier-Free Guidance）** [6] 是擴散模型中的經典方法，通過組合條件和無條件預測來偏向外部語義條件。後續工作 [7][8] 用降級內部參考替換無條件參考，探索了不同的參考變體。在視覺自迴歸模型中，最近的方法 [9][10] 使用重加權條件預測或尺度特定自引導。

VPG 的創新在於首次系統性地解決視覺 AR 中的前綴支撐問題，提供了與 CFG 互補的新引導軸。

## 創新意義與應用價值

### 理論貢獻

VPG 從**前綴後驗支撐**的新視角理解視覺自迴歸模型的推理問題。論文明確區分了條件軸和前綴軸的引導，揭示了這兩個軸的互補性。這為理解和改進自迴歸模型提供了新的理論框架。

### 實踐價值

VPG 是一種簡單、有效的訓練無關方法，可立即應用於現有的視覺 AR 模型。無需修改模型架構或重新訓練，只需在推理時應用邏輯外推。這使得 VPG 具有極高的實用性和易用性。

### 通用性

VPG 在多個 SOTA 自迴歸模型上驗證（VAR、Infinity、InfinityStar），跨越多種任務（圖像生成、視頻生成、多模態生成），展現了方法的通用性和魯棒性。

### 符合用戶偏好

✅ **Training-Free**：完全免訓練推理時方法，無需修改模型
✅ **Attention Modulation**：通過邏輯調製實現注意力調製效果
✅ **VAR-based**：直接應用於 VAR、Infinity、InfinityStar 等 VAR 系列模型
✅ **Zero-Shot**：無需任何微調或重新訓練，即插即用

## 技術亮點與局限

### 技術亮點

**簡潔而有效的設計**。VPG 的實現非常簡潔，只需一次額外的前向傳播和簡單的邏輯外推，計算開銷最小。

**理論清晰**。論文通過貝葉斯分解清晰地推導了前綴後驗目標，邏輯推導嚴密。

**跨模型驗證**。在三個不同的 SOTA 模型上驗證，展現了方法的通用性。

**互補性**。VPG 與 CFG 沿著不同軸工作，可以組合使用，進一步提升性能。

### 潛在局限

**腐蝕策略的簡單性**。當前的腐蝕策略（隨機替換）相對簡單，可能不是最優的。更精細的腐蝕策略可能帶來進一步改進。

**超參數敏感性**。引導強度 $\lambda$ 需要手動調整，不同任務可能需要不同的設置。

**理論保證**。雖然邏輯推導清晰，但缺乏前綴後驗支撐改進的理論保證。

## 後續研究方向

**腐蝕策略優化**。探索學習式腐蝕策略，根據前綴內容動態調整腐蝕方式。

**自適應引導強度**。開發自適應機制動態調整 $\lambda$，根據生成進度和前綴質量自動調整。

**與其他方法組合**。系統性研究 VPG 與其他引導方法的組合策略，如 CFG+VPG、VPG+自監督等。

**理論分析**。提供前綴後驗支撐改進的理論分析，建立性能改進的理論保證。

**應用擴展**。探索 VPG 在其他模態（3D、音頻）和其他任務（編輯、變換）上的應用。

## 評價與總結

VPG 是一篇高質量的視覺生成研究工作，提出了一個新穎的視角和簡潔有效的方法。論文的主要優勢在於：（1）識別了一個先前未被充分探索的問題（前綴後驗支撐），（2）提出了簡潔而有效的解決方案，（3）在多個 SOTA 模型上驗證了方法的有效性。

論文完美符合用戶的多個研究興趣方向：訓練無關方法、注意力調製、VAR 模型改進、零樣本應用。VPG 的訓練無關特性使其具有極高的實用價值，可以立即應用於現有的視覺自迴歸模型。

對於從事視覺生成研究的工作者，特別是關注自迴歸模型、推理時優化、無訓練方法的研究者，VPG 提供了新的思路和有效的工具。論文的清晰表述和全面評估也使其成為理解視覺自迴歸模型推理問題的重要參考。

## 參考文獻

[1] [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905) - NeurIPS 2024 Best Paper

[2] [Infinity∞: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis](https://arxiv.org/abs/2412.04431) - CVPR 2025 Oral

[3] [InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](https://arxiv.org/abs/2511.04675) - NeurIPS 2025 Oral

[4] [Generalization in Generation: A closer look at Exposure Bias](https://arxiv.org/abs/1910.00292) - EMNLP 2019

[5] [REAR: Rethinking Visual Autoregressive Models via Generator-Tokenizer Consistency Regularization](https://arxiv.org/abs/2406.02476) - 2024

[6] [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) - 2022

[7] [Guiding a Diffusion Model with a Bad Version of Itself](https://arxiv.org/abs/2306.02896) - 2023

[8] [Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance](https://arxiv.org/abs/2403.17377) - 2024

[9] [SoftCFG: Uncertainty-Guided Stable Guidance for Visual Autoregressive Model](https://arxiv.org/abs/2406.14518) - 2024

[10] [SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation](https://arxiv.org/abs/2405.05963) - 2024

---

**撰寫日期**: 2026-07-20  
**作者**: Manus AI  
**論文狀態**: Preprint (arXiv:2605.30317)
