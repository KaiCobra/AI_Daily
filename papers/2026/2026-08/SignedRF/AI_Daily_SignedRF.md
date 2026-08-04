# AI Daily: Signed Rectified Flow - Negativity-Controlled Generation

## 基本資訊
- **論文標題**: Signed Rectified Flow: Negativity-Controlled Generation
- **作者**: Runlong Liao, Baiyu Su, Lizhang Chen, Qiang Liu (UT Austin)
- **發表日期**: 2026-07-20 (arXiv:2607.18516)
- **領域**: Generative Modeling, Flow Matching, Diffusion Models, Alignment, Safe Generation
- **論文連結**: [arXiv:2607.18516](https://arxiv.org/abs/2607.18516)

## 論文核心貢獻和創新點
在條件生成、安全對齊和偏好學習等任務中，我們經常需要**促進（promote）**某個期望的目標分佈 $\pi^+$，同時**抑制（suppress）**某個不期望的目標分佈 $\pi^-$。傳統的擴散模型或流匹配（Flow Matching）通常依賴啟發式的 Classifier-Free Guidance (CFG) 或負向引導，但其底層的採樣分佈缺乏嚴謹的理論保證，難以精確控制哪些區域被促進或抑制。

這篇來自 UT Austin 的論文提出了一種全新的生成框架——**Signed Rectified Flow (Signed RF)**，核心貢獻包括：
1. **引入帶符號測度（Signed Measure）**：將生成目標定義為 $\pi^{\mathtt{sign}} = (1+\alpha)\pi^+ - \alpha\pi^-$，其中 $\alpha > 0$。雖然帶符號測度不能直接採樣，但 Signed RF 構建了一個有效的生成過程，能將機率集中在 $\pi^{\mathtt{sign}}$ 的正區域，同時在數學上保證**完全排斥（provably excluding）**被負面成分主導的區域。
2. **帶電粒子解釋與 Ghost Region 理論**：從連續性方程式（continuity equation）出發，解釋了負質量如何形成排斥屏障（exclusion barriers）。動態過程將空間劃分為可達區域（reachable region）、負區域和未被採樣的「幽靈區域（ghost region）」。
3. **實用的自適應引導演算法**：推導出基於密度比（density ratio）的動態引導公式，可通過訓練輔助分類器或在線 ODE 追蹤（online ODE tracking）來實現，實現了 Training-Free 的推理期引導。
4. **廣泛的應用場景**：在 ImageNet 條件生成中改善了保真度-多樣性權衡（fidelity-diversity trade-off）；在反記憶化（anti-memorization）測試中減少了與訓練集的相似度；在 SD 3.5 中有效抑制了對抗性提示詞產生的裸露內容，同時保持了高 CLIP 和美學分數。

## 技術方法簡述

### 1. 從 Rectified Flow 到 Signed Rectified Flow
標準的 Rectified Flow (RF) 在源分佈 $\pi_0$ 和目標分佈 $\pi_1$ 之間構建線性插值。對於凸混合分佈 $\pi_1^{\mathtt{mix}} = (1-w)\pi_1^+ + w\pi_1^-$ ($w \in [0,1]$)，RF 的速度場是 $v_t^+$ 和 $v_t^-$ 的加權平均。

Signed RF 將權重擴展到負值區間，令 $w = -\alpha$ ($\alpha > 0$)，目標變為帶符號測度 $\pi_1^{\mathtt{sign}}(\bm{x}) = (1+\alpha)\pi_1^+(\bm{x}) - \alpha\pi_1^-(\bm{x})$。
對應的 Signed RF 速度場為：
$$ v_t^{\mathtt{signRF}}(\bm{x}) = \frac{(1+\alpha)\pi_t^+(\bm{x})v_t^+(\bm{x}) - \alpha\pi_t^-(\bm{x})v_t^-(\bm{x})}{(1+\alpha)\pi_t^+(\bm{x}) - \alpha\pi_t^-(\bm{x})} $$

雖然分母 $\pi_t^{\mathtt{sign}}(\bm{x})$ 可能為負，但從源分佈 $\bm{Z}_0 \sim \pi_0$ 初始化的常微分方程式（ODE）軌跡會保持在正區域內，不會穿過奇異的零點集合 $\Omega_t^0 = \{\bm{x}: \pi_t^{\mathtt{sign}}(\bm{x}) = 0\}$。

### 2. 採樣行為與區域分解
Signed RF 的動態將空間劃分為三個區域：
- **可達區域（Reachable Region, $\Omega_t^r$）**：從 $\pi_0$ 初始化的軌跡所能到達的區域。在此區域內，採樣密度 $\pi_t^{\mathtt{signRF}}$ 完全等於帶符號邊際分佈 $\pi_t^{\mathtt{sign}}$。
- **負區域（Negative Region, $\Omega_t^-$）**：$\pi_t^{\mathtt{sign}} < 0$ 的區域，採樣軌跡永遠不會進入。
- **幽靈區域（Ghost Region, $\Omega_t^g$）**：$\pi_t^{\mathtt{sign}} > 0$ 但軌跡無法到達的區域。幽靈區域內的正質量恰好抵消了被排除的負質量。

![Signed RF Region Decomposition](assets/fig_3.png)
*(圖：Signed RF 動態將空間劃分為可達區域、負區域（拒絕）和幽靈區域。採樣法則僅在可達區域內與帶符號密度一致。)*

### 3. 實用引導公式
為了在實踐中應用，Eq.(3) 可改寫為類似引導（guidance）的形式。定義密度比 $r_t(\bm{x}) = \pi_t^-(\bm{x}) / \pi_t^+(\bm{x})$ 以及速度差 $\Delta v_t(\bm{x}) = v_t^+(\bm{x}) - v_t^-(\bm{x})$，則：
$$ v_t^{\mathtt{signRF}}(\bm{x}) = v_t^+(\bm{x}) + \lambda_t^\alpha(\bm{x}) \Delta v_t(\bm{x}), \quad \lambda_t^\alpha(\bm{x}) = \frac{\alpha r_t(\bm{x})}{(1+\alpha) - \alpha r_t(\bm{x})} $$
這種狀態感知的動態引導（State-Aware Guidance）可透過兩種方式估計密度比 $r_t$：
1. **分類器方法**：在加噪狀態上訓練二元分類器。
2. **在線比例追蹤（Online ratio tracking）**：利用散度（divergence）和分數函數（score functions）在 ODE 積分過程中直接追蹤對數密度比。

## 實驗結果和性能指標

### ImageNet 條件生成
在 ImageNet 256x256 的類別條件生成中，Signed RF 被用作狀態感知的 CFG（State-Aware CFG）。與標準的常數 CFG 相比，Signed RF 在各種 NFE（網路函數評估次數）預算下都顯著降低了 FID 分數。例如在 16 NFE 時，FID 從 CFG 的 2.38 降至 1.82。在 Precision-Recall 曲線中，Signed RF 展現了更優的保真度-多樣性權衡。

### 反記憶化（Anti-memorization）
模型經常會記憶訓練數據，導致生成與訓練集高度相似的圖像。Signed RF 將經驗訓練分佈作為負目標 $\pi^-$，構建了 **Data Repulsive Flow**。在壓力測試（針對高風險 seed）中，基礎模型會精確複製訓練樣本（甚至包含浮水印），而 Signed RF 在保持生成品質（FID 2.03 vs 基礎模型 2.07）的同時，顯著增加了生成樣本與最近鄰訓練樣本的距離（SSCD $L_2$ 距離），表現優於先前的 SPELL 方法。

### 概念抑制與安全生成
在 Stable Diffusion 3.5 的裸露內容防護測試中，Signed RF 被用來抑制不安全的生成模式。在 Ring-A-Bell 基準測試中，Signed RF 將攻擊成功率（ASR）從 15.19% 降至 6.33%，毒性率（TR）從 0.180 降至 0.125，同時維持了與基礎模型相當的 CLIP 和美學分數，證明其能在不損害生成品質的前提下提供強大的安全防護。

![Overview](assets/fig1_overview.webp)
*(圖：Signed RF 透過避免負目標 $\pi^-$ 主導的區域，將樣本引導至正目標 $\pi^+$。可用於抑制不期望的模式、增強期望特徵、防止生成受版權保護的數據，以及減輕訓練數據洩露。)*

## 相關研究背景
傳統上，引導生成（Guided Generation）如 Classifier-Free Guidance (CFG) 透過外推無條件和條件預測來提升生成品質，但這本質上是修改了得分函數（score function）或向量場，缺乏對最終採樣分佈的嚴格保證。近期研究如 Momentum Guidance、Safe Latent Diffusion 等試圖解決特定問題，但大多仍是啟發式方法。Signed RF 從基礎理論出發，將 Rectified Flow 推廣至帶符號測度，為這類問題提供了統一且具有數學保證的框架。

## 個人評價和意義
Signed Rectified Flow 是一項非常優雅且具啟發性的工作。它巧妙地解決了生成模型中「如何告訴模型不要做什麼」的問題。透過引入帶符號測度和幽靈區域的概念，作者不僅給出了嚴謹的數學證明，還提供了極具直覺的「帶電粒子」物理圖像。

這項研究對多個領域都有深遠影響：
1. **Alignment 與 Safety**：提供了一種 Training-Free 的安全引導方式，比傳統的 Concept Erasure 更具魯棒性。
2. **版權與隱私保護**：Data Repulsive Flow 展示了在不犧牲生成品質下防止模型吐出訓練數據的潛力，這對於企業級 AI 應用至關重要。
3. **推理期計算（Inference-time Compute）**：這種自適應的動態引導實際上是在推理階段進行了更精細的計算資源分配，與近期強調 System 2 Thinking 的趨勢不謀而合。

對於關注 Energy-based models、Flow Matching 和 Training-free inference 的研究者來說，這篇論文無疑是 2026 年必讀的佳作。
