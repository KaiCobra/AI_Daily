# AI Daily

## TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning
**研讀日期：2026-11-09**

### 論文資訊

- **作者**：Marco Bagatella、Matteo Pirotta、Ahmed Touati、Alessandro Lazaric、Andrea Tirinzoni。
- **研究單位**：FAIR at Meta、ETH Zürich、Max Planck Institute for Intelligent Systems, Tübingen。第一作者的論文註記為部分工作在 Meta 完成。[1] [2]
- **發表狀態**：arXiv:2510.00739 v1 首次提交於 2025-10-01；正式發表於 ICLR 2026，並列為 Oral。[1] [2] [3]
- **研究領域**：JEPA 式潛在表徵預測、離線無監督強化學習、零樣本策略遷移。這不是圖像生成或 VAR 論文；它處理的是從狀態或 RGB 觀測學習可供控制使用的表示。[1] [2]

## 一句話摘要

TD-JEPA 以無獎勵的離線轉移資料，學習「在不同策略條件下，未來狀態的潛在表示會如何累積」；新任務到來時，再將少量 reward-labelled 樣本映射到任務向量，從預訓練策略族中選出對應策略，而不為該任務重新訓練策略。[1] [2]

## 為何值得讀

它把 JEPA 從常見的視覺自監督預測，推進到零樣本控制：模型不只預測下一個 latent，而是用 temporal-difference（TD）目標學習長期、依策略而變的 latent successor features。這使表示學習目標直接連到後續的價值估計與策略選擇。[2] [5]

repo 去重後才選擇此篇。候選中的 **Energy-Based Transformers are Scalable Learners and Thinkers** 已在 AI Daily 的既有文章出現，因此排除；TD-JEPA 則尚無獨立解析文章。這篇也符合你近期關注的 JEPA 與 zero-shot，但要留意它是經過離線訓練的強化學習方法，不是 training-free 推論法。[6]

## 核心貢獻

TD-JEPA 的主要新意，是把 latent prediction 變成零樣本 RL 的核心目標，而不只是附加在另一個 RL loss 上的輔助項。它共同學習狀態 encoder、任務 encoder、策略條件式預測器，以及一組以 latent 任務向量索引的策略。[1] [2]

第二個貢獻是以 TD bootstrap 取代直接收集每個目標策略的長期 rollout。訓練因此可從既有的一步離線轉移資料估計，並讓預測器逼近多個策略的 successor features。第三個貢獻是提供簡化設定下的理論分析，連結 latent prediction、successor-measure factorization 與零樣本政策評估。[2]

## 技術方法：用 TD 目標預測長期 latent

令狀態 encoder 為 $\phi(s)$，任務 encoder 為 $\psi(s)$。任務 encoder 定義可表達的 reward 空間：對近似線性於該表示的 reward，可寫成

$$
r(s) \approx \psi(s)^\top z_r,
$$

其中 $z_r$ 是新任務的 latent 向量。給定以 $z$ 為條件的策略 $\pi_z$，其 successor feature 可表示為

$$
F_{\psi}^{\pi_z}(s,a)
=\mathbb{E}_{s^+\sim M^{\pi_z}(\cdot\mid s,a)}[\psi(s^+)],
\qquad
Q_r^{\pi_z}(s,a)\approx F_{\psi}^{\pi_z}(s,a)^\top z_r.
$$

因此，若能學到不同 $z$ 對應的未來表示，便能在任務到來時用新的 reward 向量重用既有策略，而不必重新學整個控制器。[2]

直接的 Monte Carlo JEPA 損失會讓預測器比對目標策略長期 rollout 的 latent：

$$
\mathcal{L}_{\mathrm{MC}}=
\mathbb{E}_{(s,a),z,s^+}
\left[\left\|T_\phi(\phi(s),a,z)-\operatorname{sg}(\psi(s^+))\right\|_2^2\right].
$$

其中 $s^+$ 是依照 $\pi_z$ 前進多步後的狀態，$\operatorname{sg}$ 表示 stop-gradient。這個形式需要為感興趣的策略收集相應 rollout，對離線資料並不方便。[2]

TD-JEPA 改用 Bellman bootstrap。以非對稱版本的一側為例，令 $(s,a,s')$ 取自離線資料，再從目標策略於 $s'$ 取樣 $a'$：

$$
\mathcal{L}_{\mathrm{TD}}^{\phi}=
\mathbb{E}_{(s,a,s')\sim\mathcal{D},\,z\sim\mathcal{Z},\,a'\sim\pi_z(\cdot\mid s')}
\left[
\left\|T_\phi(\phi(s),a,z)
-\operatorname{sg}(\psi(s'))
-\gamma\operatorname{sg}\!\left(T_\phi(\phi(s'),a',z)\right)
\right\|_2^2
\right].
$$

這是 successor-feature Bellman equation 在 latent space 的形式：目前的預測要等於一步後的 task representation，加上折扣後的未來預測。與 Monte Carlo 版本相比，它只需離線的一步轉移及由目標策略產生的下一個動作，無須在資料集中事先保存完整長軌跡。實際方法另訓練反向配對的 $T_\psi$，交換 $\phi$ 與 $\psi$ 的角色；target networks 與 covariance regularization 用於穩定訓練、避免表示塌縮。[2]

策略以預測 successor feature 與任務向量的內積為目標。部署新 reward 時，論文先用帶有 reward 的樣本 $\mathcal{D}_{\mathrm{rwd}}=\{(s,r)\}$ 做線性回歸：

$$
z_r=\arg\min_z\;\mathbb{E}_{(s,r)\sim\mathcal{D}_{\mathrm{rwd}}}
\left[(r-\psi(s)^\top z)^2\right],
\qquad
\pi_{z_r}(\phi(s))\approx\arg\max_a T_\phi(\phi(s),a,z_r)^\top z_r.
$$

所以這裡的 **zero-shot** 指不針對新任務重新做策略梯度訓練；它不代表完全不需要新任務訊號。實作仍須取得 reward-labelled 樣本來估計 $z_r$，而 reward 若落在 encoder 所表示的任務空間之外，線性投影也可能失真。[2]

理論分析在 tabular state space 與線性 predictor 等理想化設定下，證明 latent-predictive loss 與 successor-measure 的 TD 近似目標有梯度上的對應；在指定初始化與條件下，表示 covariance 可保持不變，並避免理論模型塌縮。部分結果要求表示具單位協方差、狀態分布均勻及轉移核對稱等條件，因此不能解讀成對任意深度網路訓練都提供無條件保證。[2]

## 實驗結果：像素設定有優勢，但不是全面第一

作者在 ExoRL 與 OGBench 的 13 個資料集、65 個任務上評估 locomotion、navigation 與 manipulation，並同時使用 proprioceptive state 與 RGB observation。主要比較包含 Laplacian、ICVF、HILP、Forward-Backward（FB）、RLDP、BYOL 及 BYOL-$\gamma$。DMC 報告平均 return（部分設定以 1000 為最大值正規化）；OGBench 報告成功率。表格是跨 seed 的 mean ± standard error，DMC 使用 5 個 seed、OGBench 使用 10 個 seed。[2]

- **DMC，RGB observation**：TD-JEPA 平均為 **628.8 ± 5.5**，高於表中最佳非 TD-JEPA 基線 BYOL-$\gamma$ 的 **582.4 ± 9.8**。
- **DMC，proprioception**：TD-JEPA 為 **661.2 ± 6.3**；FB 為 **648.2 ± 4.1**。平均值較高，但論文標示的信賴區間有重疊，不宜宣稱明確大幅勝出。
- **OGBench，RGB observation**：TD-JEPA 為 **41.34 ± 0.45**，略低於 BYOL-$\gamma$ 的 **41.58 ± 0.64**，整體接近但不是最高。
- **OGBench，proprioception**：TD-JEPA 為 **37.98 ± 0.77**，低於 FB 的 **39.04 ± 0.66**；各 manipulation/navigation 子任務則有不同勝負。[2]

這些結果支持「像素輸入下表現具競爭力」，但不支持「所有資料集與輸入都全面最佳」。值得注意的是，OGBench 包含高品質但覆蓋範圍窄的離線軌跡；作者在附錄與審查回覆中補上更低品質、低覆蓋資料的分析，指出在最小資料集上加入 FlowQ 式或行為克隆正則可改善約 10–20%，也顯示資料分布仍會影響方法效果。[2] [4]

## 相關研究：JEPA、Forward-Backward 與 successor features

I-JEPA 的基本想法是，從同一張圖像的 context block 預測其他 target block 的表示，而不必重建像素。TD-JEPA 沿用「預測 latent」而非像素的路線，但把預測單位改成由策略 $\pi_z$ 決定的長期狀態特徵，並透過 TD bootstrap 接上離線控制。[5] [2]

在零樣本 RL 脈絡中，Forward-Backward（FB）方法以近似雙線性分解的方式表達不同策略的 successor measure，常以對比式目標學習任務與未來特徵。TD-JEPA 改以共享狀態表示、policy-conditioned latent predictor 與非對比式 TD loss 逼近同一類長期資訊。BYOL-$\gamma$ 則可視為依行為策略預測折扣後 latent 的 Monte Carlo 路線；TD-JEPA 多了任務條件，並把預測改成離線 TD 更新。[2]

## 可延伸的研究問題

1. **把 EBT 式相容性評分接到 JEPA 控制。** 目前 TD-JEPA 的核心訊號是向量預測誤差；可以另學一個 scalar energy，評估 $(s,a,z,\hat{s}')$ 是否與資料支持的策略轉移相容。高能量或不確定區域可觸發額外模擬或較保守的動作選擇。這是延伸假說，不是 TD-JEPA 的論文貢獻。
2. **檢驗 reward 向量的可表達範圍。** 將 reward 樣本數、狀態覆蓋與 reward 非線性程度分開控制，量測 $z_r$ 線性投影誤差如何傳導到策略成功率；再比較 kernelized 或非線性 reward encoder 是否能擴大可遷移任務範圍。
3. **把 VAR 的尺度式表徵引入視覺控制。** 可研究以 coarse-to-fine visual tokens 形成任務 encoder $\psi$，讓低解析度尺度代表布局或目標位置，高解析度尺度保留操作細節，再用 TD-JEPA 學習策略條件下的長期 successor features。需要與單一尺度 encoder 在資料效率、遮擋和動態場景上做匹配比較。
4. **從模擬環境走向真實機器人。** 論文主要驗證 ExoRL 與 OGBench。下一步可在含多種機器人與相機分布的大型離線資料上測試少量 reward 標註、domain shift 及安全失敗率，並同步報告策略推論成本。[2]

## 限制與個人評價

TD-JEPA 最有價值的部分，是將 JEPA 的表示預測、TD 學習與 successor-feature 零樣本控制連成單一訓練目標。方法有清楚的數學橋樑，不只是「用 JEPA 當 encoder」；官方程式庫也提供從 ExoRL 與 OGBench 資料訓練及重現表格的程式。[2] [6]

但讀結果時應保留三個邊界。第一，「zero-shot」是預先學好策略族後，對新 reward 估計任務向量並選擇策略，不是免訓練，也不是不需 reward 樣本。第二，主表呈現的是依資料集而變的優勢：DMC RGB 明顯較強，OGBench 兩種輸入的平均值則略低於某些基線。第三，理論保證基於理想化模型假設，現有實驗仍以模擬 benchmark 為主；作者也將大型真實機器人資料集列為後續方向。[2] [4]

我的評價是，這篇適合作為「JEPA 如何從表徵學習走向可重用決策」的研究起點，而非圖像生成方向的直接突破。最值得追問的是：若讓任務 latent 同時具備能量式可信度分數與尺度式視覺結構，能否在少量 reward 標註和低覆蓋離線資料下，仍可靠地挑出未見任務策略？這個問題把你關注的 JEPA、Energy-based Transformer、VAR 與 zero-shot 串在一起，也能用可量測的外部任務與失敗率驗證，而不只是概念類比。

## References

[1]: https://arxiv.org/abs/2510.00739 "TD-JEPA arXiv record and version history"
[2]: https://proceedings.iclr.cc/paper_files/paper/2026/file/3d158f054ff0cb83397367234899db07-Paper-Conference.pdf "TD-JEPA ICLR 2026 conference paper"
[3]: https://iclr.cc/virtual/2026/oral/10009367 "TD-JEPA ICLR 2026 Oral program page"
[4]: https://openreview.net/forum?id=SzXDuBN8M1 "TD-JEPA OpenReview discussion and reviews"
[5]: https://arxiv.org/abs/2301.08243 "I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
[6]: https://github.com/facebookresearch/td_jepa "Official TD-JEPA implementation"
