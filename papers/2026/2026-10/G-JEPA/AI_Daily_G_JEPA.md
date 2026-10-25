# AI Daily

> 報告整理：Manus AI
> 日期：2026-10-25

## 今日唯一深讀

### Exploring Temporal Context-Aware Physical Grounding in Latent World Models

- **作者**：Yilei Man、Yang Chen、Jingcai Guo
- **機構**：Lumen Lab、Department of Computing、Department of Land Surveying and Geospatial Science，The Hong Kong Polytechnic University
- **狀態**：arXiv v1、cs.AI；2026-10-20 16:30:27 UTC 提交。arXiv 紀錄未列期刊、會議或正式出版資訊，因此本文標示為預印本，不稱為正式會議論文。[1]
- **原始連結**：[arXiv 摘要與版本紀錄][1]、[PDF][2]、[作者程式庫][8]

## 為何選這篇

本日候選中，VANE 最接近 JEPA 表徵學習：它以 next-frame embedding 取代 masked completion，並做 matched ViT-L 消融，但沒有 action planning、物理狀態 grounding 或 zero-shot 評估。AEON 的 arXiv 註記稱其獲 NeurIPS 2026 接收，將影片 latent 解碼為可渲染 4D Gaussian Splatting；不過主要證據是 VBench、重建品質和 novel trajectory，尚未驗證 action-conditioned control 或物理規劃。[9] Looking Ahead 研究 nonconsecutive future-chunk supervision，對 VAR 與 JEPA 有啟發，但沒有閉環規劃。[10]

ALLOT 與 JAM 更直接命中 training-free attention modulation：ALLOT 以輸出敏感度配置 layer-step sparsity；JAM 用 latent gradient 和 cross-attention KL 控制版面，再以第二個 pass 修復外觀。[11][12] 兩者解的是影片生成效率或版面控制，不是 world-model learning。G-JEPA 把 JEPA 表徵連到物理 state、action distribution 與 CEM 規劃，對 Energy-based transformer、multi-horizon JEPA、training-free proposal 和 zero-shot transfer 留下較集中的研究接口。它仍是預印本；選擇理由是主題契合與機制證據，不是正式會議等級。

## 核心貢獻與方法

LeWM 是本文的 JEPA 基線：pixel encoder 產生 latent，action-conditioned predictor 預測下一個 embedding，並以 SIGReg 避免表徵塌縮。G-JEPA 保留這個骨架，再加入三個部件：

1. **Temporal physical grounding**：觀測 latent 與 action-conditioned 預測 latent 各自累積 causal S4D 記憶；兩支路共享 grounding module，並預測物理狀態。
2. **Nuisance invariance**：對亮度、對比、飽和度與高斯雜訊做 clip-level augmentation，要求原圖與增強圖的 latent 方向相近。
3. **Action prior 與 mixed CEM**：凍結 world model 後，以 current latent、goal latent 和兩者差值訓練 25-action diffusion prior。規劃時將 learned proposals、Gaussian samples 與保留的 mean candidate 放入同一候選池，以 world-model latent cost 選 shared elite，再更新 Gaussian 分布。

### JEPA 預測與時間物理 grounding

基礎 JEPA 目標為

$$
\mathcal L_{\mathrm{JEPA}}=\frac{1}{N}\sum_{t=0}^{N-1}\lVert \hat z_{t+1}-z_{t+1}\rVert_2^2+\lambda_{\mathrm{reg}}\operatorname{SIGReg}(Z).
$$

$N$ 是訓練窗中的 transition 數，$t$ 是時間索引，$z_{t+1}$ 是 target encoder latent，$\hat z_{t+1}$ 是 predictor latent，$Z$ 是 latent batch，$\lambda_{\mathrm{reg}}$ 是 SIGReg 權重。[2]

觀測支路以

$$
(\hat s_t^{\mathrm{obs}},h_t^{\mathrm{obs}})=Q_\psi(z_t,h_{t-1}^{\mathrm{obs}}),\qquad h_{-1}^{\mathrm{obs}}=0
$$

更新物理狀態預測與時間記憶。$Q_\psi$ 是共享 grounding module，$\hat s_t^{\mathrm{obs}}$ 是預測 physical state，$h_t^{\mathrm{obs}}$ 是 S4D memory，$z_t$ 是當前 latent，$\psi$ 是 grounding 參數。S4D 記憶的 recurrence 為

$$
h_t=\alpha\odot h_{t-1}+Bv_t,
$$

其中 $h_t$ 是記憶狀態，$v_t$ 是輸入序列，$B$ 是輸入投影，$\odot$ 是逐元素乘法，$\alpha$ 是固定衰減向量。其第 $i$ 個元素為

$$
\alpha_i=\exp[-\Delta(i+\tfrac12)],\qquad i=0,\ldots,N_\alpha-1.
$$

$N_\alpha$ 是 S4D memory 維度，主設定為 64；$\Delta$ 是離散衰減步長，主設定為 0.1。[2]

觀測 state loss 為

$$
\mathcal L_{\mathrm{state}}=\frac{1}{T_Cd_s}\sum_{t=0}^{T_C-1}\lVert\hat s_t^{\mathrm{obs}}-\bar s_t\rVert_2^2,
$$

其中 $T_C$ 是 context 長度，$d_s$ 是 physical-state 維度，$\bar s_t$ 是標註 state。預測 latent 使用另一個從零開始的 predicted memory：

$$
(\hat s_k^{\mathrm{pred}},h_k^{\mathrm{pred}})=Q_\psi(\hat z_k,h_{k-1}^{\mathrm{pred}}),\qquad k=1,\ldots,T_C,
$$

並以

$$
\mathcal L_{\mathrm{pred-state}}=\frac{1}{T_Cd_s}\sum_{k=1}^{T_C}\lVert\hat s_k^{\mathrm{pred}}-\bar s_k\rVert_2^2
$$

約束 predicted latent 的物理意義。$\hat z_k$ 是 world-model rollout latent，其餘符號與前式相同。[2]

### 外觀不變性與完整 loss

外觀不變性項為

$$
\mathcal L_{\mathrm{inv}}=\frac{1}{T_C+1}\sum_{t=0}^{T_C}[1-\cos(z_t,\operatorname{sg}(z'_t))],
$$

其中 $z'_t$ 是 brightness、color、saturation 和 noise augmentation 後的 latent，$\cos$ 是 cosine similarity，$\operatorname{sg}$ 是 stop-gradient。完整 loss 為

$$
\mathcal L=\mathcal L_{\mathrm{JEPA}}+\lambda_s\mathcal L_{\mathrm{state}}+\lambda_p\mathcal L_{\mathrm{pred-state}}+\lambda_i\mathcal L_{\mathrm{inv}}.
$$

$\lambda_s$、$\lambda_p$、$\lambda_i$ 分別控制 observed-state、predicted-state 和 invariance loss；作者設定 $\lambda_{\mathrm{reg}}=\lambda_s=0.09$、$\lambda_p=\lambda_i=0.05$。[2]

### Action prior 與閉環規劃

規劃條件為

$$
c_{t,g}=[\tilde z_t,\tilde z_g,\tilde z_g-\tilde z_t],
$$

其中 $\tilde z_t$ 與 $\tilde z_g$ 是依 policy-training embeddings 的 mean/std standardize 後的 current/goal latent；$[\cdot]$ 表示串接，差值項直接表示 goal discrepancy。候選 action sequence $A$ 的 CEM cost 為

$$
J(A)=\sum_{k=1}^{K}w_k\operatorname{MSE}(\hat z_{t+k\mid t},z_g),\qquad w_k\ge 0.
$$

$K$ 是預測 horizon，$w_k$ 是各步權重，$\hat z_{t+k\mid t}$ 是從當前時刻 rollout 的 latent，$z_g$ 是 goal latent，MSE 是 mean squared error。最低 cost 候選形成 shared elite set，並用來更新 Gaussian mean 和 variance。action prior 的 diffusion noise-prediction loss 在正文只有文字描述，因此本文不自行補寫未編號公式。[2]

## 實驗設定與結果

模型使用 ViT-tiny、192 維 latent；grounding module 是兩層 S4D、每層輸出 128 維、$N_\alpha=64$、$\Delta=0.1$。PushT physical state 為 7 維，OGBench Cube 為 18 維，TwoRooms 為 2 維。appearance augmentation 機率為 0.9，brightness、contrast、saturation factor 取 [0.5,1.5]，clip 共用同一增強；Gaussian noise 標準差為 0.05。world model 使用 AdamW，learning rate $5\times10^{-5}$、weight decay $10^{-3}$；PushT 訓練 20 epochs、每 epoch 10,000 mini-batches、batch 32；Cube 訓練 10 epochs、batch 128。

action prior 是 25-action diffusion model，width 512、三個 FiLM residual MLP blocks、100 diffusion steps、batch 2048、50 epochs。標準規劃使用 $K=5$，每個 model step 對應五個 environment actions，主實驗 proposal ratio $\rho=0.2$。這些數字來自論文附錄與作者 README；該系統需要離線訓練，不是 training-free 或 zero-shot。[2][8]

### 主要規劃結果

- **PushT-Ada**：256 tasks、25 environment steps、$K=5$、CEM 100 candidates／10 elites／10 iterations、20 px 與 20° threshold、3 個 planning seeds。G-JEPA 為 **87.11±1.17%**，移除 action prior 的 w/o AD 為 79.43±1.58%；LeWM 為 56.90±2.35%，TD-JEPA 為 56.64±2.71%，PSG-JEPA 為 17.32±1.58%。完整 G-JEPA 比 LeWM 高 30.21 個百分點；w/o AD 已有明顯增益，表示 action prior 不是短距離任務中全部收益的來源。
- **OGBench Cube**：25 steps、$K=5$、CEM 300／30／10、4 cm threshold、10 個 planning seeds。G-JEPA 為 **100.00±0.00%**，w/o AD 為 78.60±4.12%；LeWM 為 72.80±2.53%，TD-JEPA 為 78.00±3.13%，PSG-JEPA 為 75.40±2.50%。結果接近 direct-policy ceiling，不能外推到困難任務；完整 G-JEPA w/o CEM 為 99.80±0.63%。
- **TwoRooms**：100 validation episodes、25 steps、$K=5$、CEM 300／30／30、16 px threshold、3 個 planning seeds。論文沒有評估完整 G-JEPA；G-JEPA w/o AD 為 97.00±1.00%，低於 LeWM 的 99.00±0.00%。這是重要反例，不能說方法全面提升。
- **PushT-Distance**：使用 shared recorded-trajectory goal offsets，$K=5/10/15$、CEM 300／30／30，每次 replan 執行 15 actions。G-JEPA 在 25／50／75 steps 的成功率為 **91.80±1.17／85.55±0.39／55.21±0.90%**；LeWM 為 90.49±0.81／52.99±0.60／16.54±0.60%；w/o AD 為 96.09±1.17／80.34±0.81／47.53±1.63%。75-step 結果比 LeWM 高 38.67 個百分點、比 w/o AD 高 7.68 個百分點；但在 25 steps，加入 prior 反而比 w/o AD 低 4.29 個百分點。
- **PushT-Challenging**：保留的 25／50／75 steps 分別有 126／209／240 tasks。G-JEPA 成功率為 83.33±2.38／82.30±0.48／52.22±0.96%；LeWM 為 80.69±1.65／42.42±0.73／10.97±0.64%；w/o AD 為 92.06±2.38／75.92±1.00／44.03±1.73%。75-step 結果比 LeWM 高 41.25 個百分點、比 w/o AD 高 8.19 個百分點，但各距離的 task 數不同。[2]

上述 ± 多數是 planning seeds 的 sample standard deviation，不是跨 world-model training rerun 的信賴區間；主要配置每個 benchmark 只有一個 training seed。[2]

### 消融是否支持作者的機制解釋

- **Grounding loss**：PushT-Ada 從無 S/I/P 的 56.90±2.35%，到 S only 55.73±1.58%、S+I 79.43±0.90%、S+I+P 79.43±1.58%；Cube 則為 72.80±2.53%、75.40±1.65%、76.20±2.57%、78.60±4.12%。S+I 是 PushT 的主要躍升；P 對 Cube 較有幫助，不能宣稱每項 loss 在每個任務都增益。
- **Temporal module**：PushT-Ada 在相近參數量下，S4D 為 55.431K grounding parameters、78.12±3.34%；GRU 為 54.279K、69.14±14.89%；causal Transformer 為 55.607K、61.46±10.61%；attention-only 為 55.555K、53.39±34.02%。這支持 S4D 在本 protocol 的優勢，但 attention-only 的變異很大，且此消融只有 3 個 training replicates。[2]
- **Proposal/CEM 組合**：PushT-Ada 的 G-JEPA w/o AD 為 79.43±1.58%，G-JEPA w/o CEM 僅 25.91±1.37%，完整模型為 87.11±1.17%；$\rho=1$ 時降至 61.33%，作者 sweep 後選 $\rho=0.2$。結果支持把 learned prior 當 proposal source，而不是直接取代 world-model scoring 和 search。[2]

## 相關研究

**LeWorldModel** 是直接 backbone：以 next-embedding MSE 加 SIGReg 學 latent dynamics，再用 latent goal matching 與 Gaussian CEM 規劃。G-JEPA 的新增量是 temporal state grounding、appearance invariance 和 learned action proposal，不是重新發明 JEPA。[3]

**PSG-JEPA** 也在訓練期加入 physical state grounding，但偏向 static state 與 latent-pair multi-horizon joint-angle-change。G-JEPA 的差異是對觀測與 predicted latent sequence 使用參數共享的 causal S4D，並將 action prior 接入 mixed CEM。兩者 planning protocol 不同，分數不可直接混算。[4]

**DINO-WM** 是 latent planning 的正式 ICML 2025／PMLR 前身：凍結 DINOv2 features、學 action-conditioned dynamics，再以 latent cost 和 CEM 做 zero-shot goal reaching。G-JEPA 改成 raw-pixel end-to-end encoder並加入 physical supervision；但本文沒有做 zero-shot protocol，不能把 DINO-WM 的標籤轉移過來。[5]

**Diffusion Policy** 提供 conditional action diffusion 與 receding-horizon execution 的生成式 action prior 背景；G-JEPA 的 prior 只產生候選，最終仍由 frozen world-model cost 和 CEM 選擇。[6] **AdaJEPA** 則以 closed-loop transition 做 test-time adaptation；G-JEPA 不做 test-time gradient adaptation，而是固定 world model、學習 offline prior。[7]

## 局限與不能過度宣稱之處

1. 主要 benchmark 的 world-model training 只有一個 seed；只有 temporal-module comparison 做 3 個 training replicates，穩定性證據有限。
2. 實驗限於 PushT、OGBench Cube、TwoRooms 的離線視覺任務，沒有實機、真實相機 domain shift、跨物體／跨任務 zero-shot 或 policy success 評估。
3. physical-state labels 和 domain-specific normalization 需要額外 supervision；換到沒有 state labels 的新 domain 時，S4D grounding 的成本與可行性尚未驗證。
4. action prior 由同域 offline trajectories 訓練，不是無資料 zero-shot planner；$\rho=0.2$ 是在既定 protocol 上 sweep 選出。
5. Appendix 指出 learned proposals 會增加 sampling cost，但沒有完整 wall-clock、GPU memory 或端到端 energy 表；不能只看 success rate。
6. physical-state head 只在 training 使用，實際 planning cost 仍是 latent goal MSE；論文未報 open-loop latent error 或 state estimation error 隨 horizon 的增長，尚未證明可長期模擬物理。
7. TwoRooms 未評估完整 G-JEPA，且 G-JEPA w/o AD 低於 LeWM；Cube 的 100% 也接近 ceiling。適當結論是部分長距離 manipulation protocol 有提升，而非全面勝出。
8. 截至研究日，本文沒有正式 venue 接收資訊，只能標為 arXiv v1 預印本。

## 個人評價

我給 **novelty 4/5、evidence 3/5、主題契合 5/5**。新意在於把 physical grounding 與 action prior／CEM 的耦合做成可拆解實驗，而不只是替 JEPA 換一個 backbone；證據弱點是單一主要 training seed、模擬 benchmark 和缺少長期或實機驗證。最值得深讀的不是單看 87.11%，而是 75-step PushT-Distance 中完整方法 55.21±0.90% 對 LeWM 16.54±0.60% 的差距，以及 25-step 時 prior 反而降低表現的非單調性。這些結果提醒我們分開分析 representation quality、proposal diversity、CEM search 和 horizon compounding，而不是把 action prior 當成無條件增益。[2]

## 可操作的新研究構想

1. **Energy-based transformer planner**：把固定 latent-MSE cost 換成輸入 current／goal latent、predicted latent trajectory 與 action chunk 的 energy scorer；以成功 future 為 positive、時間打亂或違反接觸幾何的 trajectory 為 hard negative。固定 CEM budget，在 PushT-Distance 75-step 比較 success、energy-ranking Spearman、calibration 和 latency。
2. **Multi-horizon JEPA**：除一步 target 外，同時預測 1、3、5 個 model-step 的 stop-gradient target，使用 horizon-weighted loss；報各 horizon latent error、CEM success 與 rollout drift，並做 one-step-only、multi-horizon-only、合併三組 matched ablation。
3. **Training-free proposal**：凍結 encoder、predictor 和 prior，不重新訓練；改以 offline action chunks 的 nearest-neighbor retrieval 或 empirical mixture 產生 proposals，再與 Gaussian CEM 共用 elite。固定候選數後比較 prior training 需求、diversity、success 和端到端 latency。
4. **Attention modulation**：以 normalized goal discrepancy $\tilde z_g-\tilde z_t$ 在 S4D、action-conditioning 或 AdaLN 做 inference-time channel/head gate。固定 world model 與 CEM budget，測試能否縮小 $\rho=1$ 在 PushT-Ada 的退化，又不損害 Cube 結果。
5. **VAR 式 action hierarchy**：把 25-action prior 改成先生成粗粒度 5-action chunks，再生成 chunk 內細粒度 actions；固定總候選數與 wall-clock，測長 horizon sampling cost、proposal coverage 和 PushT-Distance success。
6. **Zero-shot stress test**：凍結完整模型，測試未見物體、外觀、相機或 dynamics shift，分開報 latent/state error、prior proposal quality、CEM success 與 calibration；與同域 prior 和 Gaussian-only 對照。只有在沒有 task-specific fine-tuning 仍穩定時，才能支持 physical grounding 有助於 zero-shot transfer。

## 結論

G-JEPA 是今天最適合作為 JEPA／world-model 深讀的候選：它把預測表徵、物理狀態、生成式 action proposal 和閉環搜尋放在同一可核查框架，對 Energy-based transformer、JEPA、training-free、attention modulation、VAR 和 zero-shot 留下具體實驗接口。閱讀時應保留兩個邊界：它是 arXiv 預印本，不是正式接收論文；它證明的是部分模擬規劃任務的有效組合，不是已完成的物理世界模型。

## References

[1]: https://arxiv.org/abs/2610.28483 "Exploring Temporal Context-Aware Physical Grounding in Latent World Models"
[2]: https://arxiv.org/pdf/2610.28483v1 "Exploring Temporal Context-Aware Physical Grounding in Latent World Models PDF"
[3]: https://arxiv.org/abs/2603.19312 "LeWorldModel Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
[4]: https://arxiv.org/abs/2608.06799 "Is Forward Prediction Enough Physical State Grounding for JEPA World Models"
[5]: https://proceedings.mlr.press/v267/zhou25t.html "DINO-WM World Models on Pre-trained Visual Features Enable Zero-Shot Planning"
[6]: https://arxiv.org/abs/2303.04137 "Diffusion Policy Visuomotor Policy Learning via Action Diffusion"
[7]: https://arxiv.org/abs/2606.32026 "AdaJEPA An Adaptive Latent World Model"
[8]: https://github.com/myl11045/G-JEPA "G-JEPA official code and project repository"
[9]: https://arxiv.org/html/2610.28631v1 "AEON Unifying Video and 3D World Models"
[10]: https://arxiv.org/abs/2610.29989 "Looking Ahead Modeling Future Evolution in Autoregressive Video Generation"
[11]: https://arxiv.org/html/2610.29441v1 "Sparsity Should Be ALLOTted by Output Training-Free Allocation of Block-Sparse Attention for Video Diffusion Transformers"
[12]: https://arxiv.org/html/2610.30390v1 "Latent Space Jam Layout Guided Video Synthesis"
