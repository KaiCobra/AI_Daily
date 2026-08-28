# AI Daily

> **日期：2026-08-28**　**作者：Manus AI**

## 今日精選

### Tether the Subject, Release the Scene: Query-Aware Memory Routing for Long-Horizon Autoregressive Video Generation

這篇由 **華中科技大學（HUST）與小米 MiLM Plus** 合作的研究，提出 **TetherMem**：一個不更新生成器參數、直接在推理時改變歷史記憶讀取方式的 query-aware spatiotemporal memory router。論文的核心觀察很精準：長影片的歷史記憶確實能保住主體身份，但如果所有 query 都以相同政策讀取歷史，背景、視角與場景狀態也會被鎖在過去，形成作者稱為 **memory-anchored scene under-progression** 的失敗模式。[1]

我選擇這篇論文，是因為它同時命中本期偏好的 **Visual Autoregressive、training-free、attention modulation、zero-shot inference intervention** 四個方向，而且切入點不是再增加一個更大的生成器，而是重新思考「同一份 KV memory 是否應該被不同種類的 query 以相同方式讀取」。這個觀點也很容易延伸到 Energy-based Transformer、JEPA predictive state 與 VAR 的 scale-wise memory routing。

| 項目 | 內容 |
|---|---|
| 論文標題 | Tether the Subject, Release the Scene: Query-Aware Memory Routing for Long-Horizon Autoregressive Video Generation |
| 作者 | Chen Li, Peng Zhang, Hanyu Zhou, Jialong Zuo, Fei Wang, Daiguo Zhou, Nong Sang, Changxin Gao |
| 研究單位 | Huazhong University of Science and Technology；MiLM Plus, Xiaomi Inc. |
| 發表狀態 | arXiv 預印本，cs.CV，v1 於 2026-08-27 提交；尚未標示頂會接收狀態 [1] |
| 論文連結 | [arXiv HTML][1]；[官方 project page][2] |
| Repository 去重 | 已檢查 `KaiCobra/AI_Daily` 既有 146 篇索引；未發現 TetherMem 或 arXiv:2608.26902，亦未發現同一核心方法的既有文章 |
| 關鍵標籤 | Autoregressive video；KV memory；training-free；query-aware routing；attention logit modulation；long-horizon generation |

## 一句話摘要

**TetherMem 不刪掉整份歷史記憶，也不在 attention 輸出後粗暴縮放 Value，而是把「主體 query」與「場景 query」分開，將 region-aware 與 recency-aware prior 加到 softmax 前的 attention logits，讓主體保留身份歷史、背景則優先讀取近期場景狀態。**

## 1. 背景：長影片真正卡住的不是「有沒有記憶」

Streaming autoregressive video generator 通常將長影片切成連續 chunk。第 $n$ 個 chunk 由文字 prompt $P$、隨機噪聲 $\epsilon_n$ 與歷史記憶 $M_n$ 共同生成：

$$
X_n = G_\theta(\epsilon_n, P, M_n).
$$

這種設計具有兩個重要優點。第一，模型可以因果地逐段輸出，支援可變長度與串流生成；第二，先前生成的內容可以透過 KV cache、retrieval memory 或 attention sink 提供跨 chunk 的身份與結構訊息。然而，歷史也會把模型綁在已經生成過的背景與視角上。若 prompt 要求人物向前走、鏡頭退後、城市逐步展開，模型可能仍然維持人物不變，卻只重複原來的背景，或把局部紋理運動誤當成真正的 scene progression。[1]

這帶來一個比「身份漂移」更隱蔽的 failure mode：**subject identity 穩定、local motion 仍存在，但背景、視角與空間關係沒有朝 prompt 要求的方向發展。** 因而，單獨使用 subject consistency、dynamic degree 或短片影像品質，未必能揭露長時段場景停滯；TetherMem 的人評協議特別把 spatial/scene progression 與 prompt-directed progression 分開評估。

## 2. 方法總覽：把「歷史重要性」改成 query-dependent

標準 scaled dot-product attention 對目前 query $q$ 與 context key $i$ 定義：

$$
s_{qi}=\frac{Q_qK_i^\top}{\sqrt d},
\qquad
A_{qi}=\frac{\exp(s_{qi})}{\sum_{j\in\mathcal C_n}\exp(s_{qj})},
\qquad
O_q=\sum_{i\in\mathcal C_n} A_{qi}V_i.
$$

其中 $\mathcal C_n$ 是目前 chunk 能讀取的 context，$\mathcal I_n\subset\mathcal C_n$ 是歷史 token，$(K_i,V_i)$ 是 token 的 key/value。標準 attention 讓 query 自己依據內容相似度選擇歷史，但沒有明確表達「這個 query 是在維持主體，還是在推進背景」的角色差異。

### 2.1 為何不直接做 Value Reweighting？

一個直覺作法是在 attention 完成之後，對不同歷史區域的 Value 乘上保留權重 $w_i$：

$$
O_q^{\mathrm{val}}=\sum_{i\in\mathcal C_n} A_{qi}w_iV_i.
$$

令

$$
g_q=\sum_iA_{qi}w_i,
\qquad
\bar A_{qi}=\frac{A_{qi}w_i}{g_q},
$$

則可以改寫成

$$
O_q^{\mathrm{val}}=g_q\sum_i\bar A_{qi}V_i.
$$

這表示 post-attention Value Reweighting 不只重新分配相對權重，還會引入 query-dependent 的輸出幅度 $g_q$。在長影片 rollout 中，這個幅度擾動會進入後續的 hidden state 與歷史 memory，可能逐步累積；而且它不能改變原本 attention 已經做出的選擇，只能在選擇之後縮放結果。[1]

### 2.2 TetherMem 的 normalized routing

TetherMem 將正值 routing prior $\pi_n(q,i)>0$ 放入 softmax：

$$
\widetilde A_{qi}
=
\frac{\pi_n(q,i)\exp(s_{qi})}
{\sum_{j\in\mathcal C_n}\pi_n(q,j)\exp(s_{qj})}
=
\operatorname{softmax}_i\left(s_{qi}+\log\pi_n(q,i)\right).
$$

輸出則是

$$
\widetilde O_q=\sum_{i\in\mathcal C_n}\widetilde A_{qi}V_i.
$$

由於 $\sum_i\widetilde A_{qi}=1$，TetherMem 會改變「query 讀取哪些歷史 token」而不改變 Value 本身，也不引入額外 output gate。換句話說，$\log\pi_n(q,i)$ 是一個加到 attention logit 的 bias；$\pi_n$ 越大，該 query 讀到此 token 的機率越高，$\pi_n$ 越小，該路徑被 softmax 正規化抑制。

這個形式也是本文最值得借鑑的抽象：**inference-time control 不一定要直接改 activation；只要把一個可解釋的先驗寫成 logit-space energy/bias，就能在維持 normalized attention 的同時重排資訊流。**

## 3. Region-aware 與 age-aware memory routing

### 3.1 Subject prior 的建立

對有指定主體的 prompt，作者先以相同 generator 做一個 Full-Memory reference rollout，再使用 SAM 2 Hiera-Large、以 box prompt 追蹤一個主體。每一幀的二值 mask 會縮放到 $30\times52$ 的 latent-token grid，並施加 $5\times5$ dilation；沒有主體的 P05 prompt 則使用固定中心區域作為預設分割。這是一个 offline two-pass pipeline，並非完全不需要額外模型或額外計算。[1] [8]

令目前 query token 是否位於主體區域為 $m_q(q)\in\{0,1\}$，令歷史 key 是否位於其來源幀的主體區域為 $m_k(i)\in\{0,1\}$。其中 1 表示 subject，0 表示 background。需要注意，這個 mask 是 routing scaffold，不是重新訓練 generator 的 supervision label。

### 3.2 Area-calibrated regional prior

對歷史 token，若 query 與 key 屬於同一區域，作者給予單位 prior；若是跨區域讀取，則使用 $\gamma_n\in\mathbb{R}_{>0},\ \gamma_n\le1$ 抑制：

$$
\pi_{\mathrm{reg}}(q,i)=
\begin{cases}
1, & m_q(q)=m_k(i),\\
\gamma_n, & m_q(q)\ne m_k(i).
\end{cases}
$$

$\gamma_n$ 並非完全固定，而是依目前 query frame 的 subject-token fraction $r_n$ 與固定 release budget $\alpha=0.25$ 做 area calibration：

$$
\bar\gamma_n=
\begin{cases}
\dfrac{\alpha-r_n}{1-r_n}, & 1-r_n>10^{-6},\\
\alpha, & \text{otherwise},
\end{cases}
$$

$$
\gamma_n=\operatorname{clip}_{[10^{-9},1]}\left(\max(0,\bar\gamma_n)\right).
$$

這個設定的直覺是讓 background query 不要一直讀取歷史 subject，而 subject query 仍然保有跨區域資訊的柔性通路；$r_n$ 變動時，跨區域 prior 也會依可用面積調整。對 local context 與 persistent sink token，$\pi_n(q,i)=1$，因此 TetherMem 不干預這些非歷史路徑。

### 3.3 Recency-aware background routing

僅做 region routing 仍然可能讓 background query 讀到「很像現在、但其實已經過時」的舊場景。於是作者在 background-to-background 路徑上再加入 recency prior。令歷史 token $i$ 位於 memory pool 的 source-frame position $p_i$，並定義

$$
A_{\max}=\max\left(1,\min(N_{\mathrm{pool}},120)\right),
\qquad
\tau_i=\frac{p_i}{A_{\max}},
\qquad
\rho_i=\max(\tau_i,\rho_{\min}),
$$

其中 $\rho_{\min}=0.05$。越靠近目前時間的背景歷史，其 $\tau_i$ 越大；最舊的背景仍保留至少 0.05 的路徑權重，避免完全硬刪除遠期 context。

完整 routing prior 可寫成：

$$
\pi_n(q,i)=
\begin{cases}
1, & m_q(q)=1,\ m_k(i)=1,\\
\gamma_n, & m_q(q)\ne m_k(i),\\
\rho_i, & m_q(q)=0,\ m_k(i)=0,\\
1, & i\notin\mathcal I_n\quad\text{(local/sink)}.
\end{cases}
$$

因此，主體 query 主要保留 long-range identity path；背景 query 主要讀取背景歷史，並偏向近期 scene state。這不是把整個 memory pool 分成「保留」與「丟棄」，而是讓同一份 memory 對不同 query 呈現不同的讀取能量。

| Current query | Historical key | Prior | 作用 |
|---|---|---:|---|
| Subject | Subject | $1$ | 保留身份與結構證據 |
| Subject | Background | $\gamma_n$ | 降低背景干擾但不完全切斷 |
| Background | Subject | $\gamma_n$ | 防止背景被主體歷史鎖住 |
| Background | Background | $\rho_i$ | 偏好近期場景狀態 |
| Any | Local/sink | $1$ | 不改動非歷史 context |

### 3.4 實作流程與 computational implication

實作上，作者把 subject queries 與 background queries 拆開，對同一份完整 KV context 做兩次 attention call，再把結果 scatter 回原 token 位置。相同 routing bias 施加於所有 heads、所有 denoising steps，以及 30 個 causal self-attention modules；不縮放 Q、K、V 或 attention output，也不修改歷史 KV 的內容。[1]

這個設計的代價是它不是「零額外計算」。在一張 NVIDIA H200 上、七支約 30 秒影片的 profiling 中，reference rollout 平均為 $72.6\pm7.5$ 秒，SAM 2 extraction 為 $24.9\pm1.0$ 秒，controlled rollout 為 $201.9\pm3.8$ 秒，完整兩階段 pipeline 為 $299.4\pm10.0$ 秒，峰值記憶體為 30.5 GB。Controlled rollout 約為 single Full-Memory pass 的 $2.81\pm0.27\times$，完整 reference–segmentation–controlled pipeline 約為 $4.15\pm0.30\times$。[1]

## 4. 實驗設計：把「進步」與「穩定」拆開量測

作者使用 Wan2.1-T2V-1.3B 作為 5 秒 base-model reference，並以 LongLive-RAG 作為主要 host。評估包含十個 prompts、三個 seeds、832×480、16 fps、約 30 秒影片，共 270 支影片；另以 42、52 與 120 秒 rollout 作為長時段定性展示。對 subject-present prompts，Full-Memory reference 經 SAM 2 取得 frame-aligned subject track；TetherMem 與 Full-Memory 共用 generator、checkpoint、denoising schedule、context、attention sink 與 retrieval budget，只改變歷史 routing。[1]

| 評估面向 | 定義與解讀 |
|---|---|
| Img5 | 前 5 秒的 VBench Imaging Quality，檢查短時段畫質是否被破壞 |
| Ovr. EP | 完整 30 秒影片的人類 Overall expected preference |
| Prog. EP | Spatial/scene progression 與 prompt-directed progression 的人評 EP 平均 |
| ID EP | Subject identity and continuity 的人評 EP；subject-free prompt 不適用 |
| L1 EP | Technical quality、artifact、late-stage collapse 的人評 EP |
| nT | 對完整影片 dense optical flow 做時間平均後再取向量範數的 net-translation diagnostic |
| Subj./Back. | Subject/background consistency proxy |
| Smooth./Dyn. | Motion smoothness 與 dynamic degree；使用統一 8 秒 trim |

人評共有 **2,400 個 blinded pairwise judgments、10 位獨立 annotators**，比較八個 streaming long-video baselines。作者使用可處理 tie 的 Davidson–Bradley–Terry model，把不同 pair 的結果放到共同 EP scale；此外以 2,000 次 crossed annotator–prompt-seed cluster bootstrap 估計信賴區間。[1]

## 5. 實驗結果

### 5.1 主比較

下表整理論文 Table 1 的主要欄位。Human EP 越高越好；$nT$ 只代表長時段淨方向性移動，並不等於 prompt fulfillment 或身份穩定。

| Method | Img5 ↑ | Ovr. EP ↑ | Prog. EP ↑ | ID EP ↑ | L1 EP ↑ | nT ↑ | Subj. | Back. | Smooth. | Dyn. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LongLive-RAG | 71.0 | 0.615 | 0.608 | 0.507 | 0.587 | 0.868 | 0.861 | 0.897 | 0.973 | 0.867 |
| Self Forcing | 70.5 | 0.320 | 0.356 | 0.437 | 0.273 | 0.192 | 0.885 | 0.907 | 0.983 | 0.767 |
| Rolling Forcing | 72.2 | 0.553 | 0.429 | 0.633 | 0.651 | 0.040 | 0.933 | 0.935 | 0.985 | 0.433 |
| Deep Forcing | 70.5 | 0.526 | 0.495 | 0.531 | 0.508 | 0.350 | 0.899 | 0.916 | 0.983 | 0.800 |
| Causal Forcing | 70.2 | 0.210 | 0.440 | 0.285 | 0.158 | 1.226 | 0.824 | 0.873 | 0.973 | 0.867 |
| Reward Forcing | 69.3 | 0.573 | 0.530 | 0.487 | 0.620 | 0.366 | 0.903 | 0.927 | 0.985 | 0.733 |
| CausVid | 65.9 | 0.283 | 0.295 | 0.461 | 0.328 | 0.190 | 0.879 | 0.896 | 0.982 | 0.767 |
| MemRoPE | 72.1 | 0.640 | 0.578 | 0.559 | 0.666 | 0.551 | 0.892 | 0.908 | 0.986 | 0.700 |
| **TetherMem** | **71.1** | **0.780** | **0.769** | 0.600 | **0.708** | **1.289** | 0.855 | 0.895 | 0.973 | 0.867 |

TetherMem 的 Overall EP 為 **0.780**，高於 MemRoPE 的 0.640；Progression EP 為 **0.769**，高於 LongLive-RAG 的 0.608。Identity EP 為 0.600，略低於 Rolling Forcing 的 0.633，但仍維持第二高區間；這正好反映作者想要的 trade-off：不是讓身份指標單獨最大，而是在不犧牲身份的前提下把 scene progression 拉上來。TetherMem 的 nT=1.289 也是表中最高，但應把它視為輔助 motion diagnostic，不要誤稱為「影片語義正確率」。[1]

前 5 秒 Img5=71.1，與 LongLive-RAG 的 71.0 幾乎相同，表示 routing 主要改善 long-horizon behavior，而不是破壞短片初始畫質。

### 5.2 統計可靠性

| Criterion | TetherMem EP (95% CI) | Strongest baseline | Baseline EP | Difference (95% CI) |
|---|---:|---|---:|---:|
| Overall | 0.780 [0.684, 0.868] | MemRoPE | 0.640 | 0.140 [0.004, 0.273] |
| Progression | 0.769 [0.666, 0.861] | LongLive-RAG | 0.608 | 0.161 [0.076, 0.240] |
| Identity | 0.600 [0.504, 0.699] | Rolling Forcing | 0.633 | −0.033 [−0.182, 0.111] |
| L1 | 0.708 [0.625, 0.794] | MemRoPE | 0.666 | 0.042 [−0.086, 0.182] |

在 crossed annotator–prompt-seed clustering 下，Overall 與 Progression 的 margin 95% CI 都排除零；但是較嚴格的 prompt-level clustering 後，Progression interval 仍排除零，而 Overall interval 變成包含零。這是合理且重要的統計語氣：**Progression 的證據較穩，Overall 的優勢則對 prompt-level clustering 更敏感。**

### 5.3 機制消融：region 與 age 必須一起看

第一個消融 panel 以 Full Memory 為中心，比較兩個單因子版本與完整 TetherMem。由於這些 EP 是在獨立小 panel 中分別 fitted，不能把不同 panel 的絕對 EP 當作同一 pool 的直接排名，但方向仍有參考價值。

| Variant | Ovr. | Prog. | ID | L1 | nT |
|---|---:|---:|---:|---:|---:|
| Full Memory | 0.426 | 0.391 | 0.472 | 0.457 | 0.980 |
| Routing w/o Region | 0.461 | 0.472 | 0.489 | 0.505 | 1.232 |
| Routing w/o Age | 0.476 | 0.428 | 0.505 | 0.490 | 1.083 |
| **TetherMem** | **0.638** | **0.709** | **0.534** | **0.548** | **1.458** |

另一個 design comparison 直接測 post-attention Value Reweighting。結果顯示 Value Reweighting 的 nT=1.405 接近 TetherMem 的 1.468，但 Overall/Progression/Identity/L1 EP 僅為 0.174/0.303/0.108/0.230，而 normalized routing 為 0.812/0.714/0.758/0.713。這支持作者的機制論點：**同樣想釋放背景，放在 attention output 之後的縮放可能比放在 softmax 前的 route selection 更容易傷害身份與整體品質。** 但這個 panel 的樣本數與主比較不同，應保留定性解讀而非把數字視為同一實驗的絕對分數。[1]

### 5.4 跨 autoregressive host 的 transfer

為了檢驗 TetherMem 是否只對 LongLive-RAG 有效，作者把同一 routing 方法套到 Deep Forcing host。Deep Forcing 的 Overall/Progression/ID/L1 EP 為 0.404/0.360/0.470/0.439；加入 TetherMem 後變成 0.596/0.640/0.530/0.561。Progression difference 為 **0.281**，crossed-cluster 95% CI 為 [0.033, 0.522]。[1]

這個結果不代表所有 host 都會得到同樣幅度的提升，但它增加了方法不是「LongLive-RAG 特例」的可信度。更精確的說法是：**TetherMem 可以作為 host memory/sink policy 之上的 query-conditioned routing layer。**

### 5.5 Prior quality 與 late-stage drift

Subject prior 並不完美。Reference mask 與 controlled rollout 的 proxy agreement 在完整 rollout 為 0.431，20 秒後的 late window 降至 0.275；這代表 reference 與 controlled trajectory 越走越遠時，離線 mask 會出現 drift。人類 mask comparison 在 28 秒的 21 個 subject-present frames 上，reference prior 的 raw semantic IoU median 為 0.202，而 controlled-output re-extraction 為 0.520；因此，TetherMem 依賴的 prior 可能在後期變得不精確。[1]

在 prior approximation sensitivity 中，原始 SAM 2、低解析 coarse prior 與 bounding-box prior 的結果如下：

| Prior | nT ↑ | Artifact veto ↓ | Tail subject ↑ |
|---|---:|---:|---:|
| Original SAM 2 | 1.490 | 0.158 | 0.934 |
| Low-res coarse | 1.463 | 0.107 | 0.932 |
| Bounding box | 1.664 | 0.207 | 0.878 |

Coarse prior 幾乎保留原始結果，但 bounding box 雖提高 nT，卻讓 tail subject consistency 下降、artifact veto 上升。這說明 TetherMem 對 prior 的要求不是像素級完美，而是需要避免把大量背景誤標成 subject。

## 6. 相關研究：TetherMem 放在什麼位置？

| 研究 | 核心解法 | 是否需要訓練/改 generator | 與 TetherMem 的關係 |
|---|---|---|---|
| LongLive-RAG [3] | 將 self-generated latent history 變成可檢索的 content-addressable memory，並以 Window Temporal Delta Loss 讓 query embedding 捕捉 temporal change | 需要其訓練/檢索框架；提供 TetherMem 的主要 host | LongLive-RAG 決定「哪些歷史被 retrieval」，TetherMem 決定「不同 query 如何讀取已取得的歷史」 |
| Self Forcing [4] | 在訓練時以 self-generated autoregressive rollout、KV cache 與 video-level loss 直接處理 exposure bias | 需要訓練；改變 train-time objective | Self Forcing 處理 train-test gap，TetherMem 是 frozen generator 的 inference-time intervention，理論上可疊加 |
| Deep Forcing [6] | Deep Sink 重新對齊 temporal RoPE，Participative Compression 保留近期 attention actively participating 的 KV | Training-free；改變 sink/cache 管理方式 | TetherMem 透過 query/region/age bias 讀取 host memory，可疊加於不同 sink/cache policy；論文也測了 Deep Forcing transfer |
| MemRoPE [5] | 以 EMA 壓縮 long-/short-term memory tokens，並在 attention 時動態施加 RoPE | Training-free；改變 memory aggregation 與 positional handling | MemRoPE 解決「如何壓縮與定位記憶」，TetherMem 解決「誰應該讀哪些記憶」 |
| VBench [7] | Video generative model 的多維自動評估套件 | 評估工具 | TetherMem 顯示 consistency 與 dynamic degree 與人類 scene progression 並不等價 |
| SAM 2 [8] | 通用影像/影片分割與追蹤 | 額外 offline model | TetherMem 目前用它取得 routing scaffold，不是用來生成內容 |

LongLive-RAG 將過去 latents 視為動態 searchable history，並以 retrieval 減少 sliding-window 導致的不可逆退化；TetherMem 沿著另一條軸線前進，主張即使 memory pool 已經取得，**subject 與 scene query 仍不應共享同一個 access policy**。[3]

Self Forcing 則從訓練角度處理 autoregressive exposure bias：訓練期間就讓模型使用自己產生的結果，並用 holistic video-level loss 監督整支 sequence。[4] 相較之下，TetherMem 不重新訓練、不改動 frozen generator，換取較低的參數修改門檻，但把困難移到 offline reference rollout、SAM 2 prior 與額外 attention calls。

Deep Forcing 以 training-free Deep Sink 與 Participative Compression 讓 5 秒訓練分佈可以外推到 60 秒以上；MemRoPE 則以 dual-stream EMA memory 與 online RoPE indexing 支援固定大小 cache 的長時段生成。[5] [6] TetherMem 的新意不在於再造一個 sink，而在於對同一歷史建立不同的 query-conditioned path。

## 7. 個人評價：真正有價值的是「route selection」這個抽象

### 7.1 優點

**第一，問題定義比單純追求 motion score 更成熟。** 長影片品質不是「動得越多越好」。Causal Forcing 的 nT 可以很高，卻不一定有高的人類 progression EP；Rolling Forcing 的 subject/background consistency 很高，也可能犧牲場景進展。TetherMem 把 stability、motion、spatial progression 與 prompt-directed progression 拆開，讓研究者看到不同目標之間的真實衝突。[1]

**第二，方法把 attention modulation 放到正確的座標系。** 若目標是改變資訊選擇，直接對 post-attention Value 做縮放不是最乾淨的操作；$\log\pi$ 作為 softmax 前 bias 的形式，同時具有概率、能量與可組合先驗的解釋。它也不需要為每個 host 重新訓練 controller，符合 training-free/zero-shot intervention 的精神。

**第三，消融與 transfer 都在回答機制問題。** 作者沒有只報一個總分，而是分別拿掉 region、age，再與 Value Reweighting 比較；此外又將 router 套到 Deep Forcing host。這使得「為什麼有效」比單純的 end-to-end score 更容易被檢驗。

### 7.2 限制與需要保留的語氣

**第一，training-free 不等於 zero-cost。** 目前 pipeline 有 reference generation、SAM 2 extraction、two-call attention 與 30.5 GB H200 peak memory；完整兩階段 runtime 約為一個 Full-Memory rollout 的 4.15 倍。[1]

**第二，mask prior 的依賴是真正的 deployment bottleneck。** 目前只追蹤一個 subject，且 reference mask 在長 rollout 後會 drift；多主體、快速遮擋、主體離開畫面、鏡頭劇烈旋轉時，binary subject/background partition 可能不再足夠。

**第三，實驗仍集中於一個 foundation family。** 主比較是 Wan2.1-T2V-1.3B/LongLive-RAG，跨 host transfer 也在 Wan2.1 family 內的 Deep Forcing。這是有價值的 transfer，但尚不能宣稱對所有 AR video generator、不同 latent grid、不同 memory layout 都普遍成立。[1]

**第四，人評設計雖然比單一自動指標可信，仍需要更大且更跨文化的驗證。** 主實驗有 2,400 judgments 與 10 位 annotators，這已優於只報短片自動分數；但嚴格 prompt-level clustering 讓 Overall 的 CI 包含零，表示總體優勢仍可能依 prompt 分佈而改變。[1]

## 8. 對 Energy-based Transformer、JEPA 與 VAR 的啟發

### 8.1 Energy-based Transformer：把 routing prior 當作可學的局部能量

若把原始 attention logit 寫成負能量 $s_{qi}=-E_{\mathrm{attn}}(q,i)$，則 TetherMem 的 controlled attention 可表示為

$$
\widetilde A_{qi}
=\operatorname{softmax}_i\left(-E_{\mathrm{attn}}(q,i)+\log\pi_n(q,i)\right)
=\operatorname{softmax}_i\left(-\underbrace{\left[E_{\mathrm{attn}}(q,i)-\log\pi_n(q,i)\right]}_{E_{\mathrm{effective}}(q,i)}\right).
$$

因此，$-\log\pi_n(q,i)$ 可以看作 query/key pair 的額外 routing energy。未來可以將固定的 region/age prior 改成一個由 predictive state、uncertainty 或 scene-progress score 產生的 energy：當背景 query 預測「此歷史狀態會讓場景停滯」時，提高其 energy；當 subject query 判斷某歷史 token 對身份辨識有用時，降低其 energy。這是一個研究假說，不是 TetherMem 已經實作的內容，但數學接口非常直接。

### 8.2 JEPA：以 predictive disagreement 取代離線 binary mask

TetherMem 的 subject/scene role 目前來自 SAM 2 mask。另一條路是讓 frozen 或輕量 JEPA predictor 預測下一個 latent state：若某一 token 對未來 latent 的預測較穩定、且與當前 subject identity state 對齊，可以給 subject path 較低的 routing energy；若某歷史背景對未來 scene state 的預測誤差高、或會使多個 future hypotheses collapse 到同一舊場景，則 background query 應降低對該歷史的依賴。

可以考慮以 predictive disagreement 定義資料驅動的 prior：

$$
\pi_{\mathrm{JEPA}}(q,i)
=\exp\left(-\lambda\,D_{\mathrm{pred}}(q,i)\right),
$$

其中 $D_{\mathrm{pred}}$ 可是多個 predictor 對未來 latent 的 disagreement，或 current/future embedding 的 energy distance。這會把 TetherMem 的「query role」從人工/外部 segmentation 逐步移向 latent predictive semantics，也與 JEPA 擅長的「預測可預測狀態、忽略像素細節」相容。

### 8.3 VAR：從時間 memory routing 推向 scale-wise token routing

在 Visual Autoregressive Model 中，coarse scale 通常決定全局 layout、主體配置與語義骨架，fine scale 則補足局部紋理。TetherMem 的 region/age routing 可以轉寫成 scale-aware prior $\pi_s(q,i)$：

$$
\widetilde A_{qi}^{(s)}
=\operatorname{softmax}_i\left(s_{qi}^{(s)}+\log\pi_s(q,i)\right).
$$

coarse scale 可以保留較長的 identity/layout prefix；scene query 則在中後期 scale 提高對新 token、新區域與近期 state 的權重。fine scale 不應只複製高頻歷史，而可以額外抑制對 stale background 的 attention。這比單純剪枝更有彈性，因為同一 token 對不同 query 或不同 scale 可以擁有不同的有效能量。

### 8.4 Training-free 與 zero-shot：真正可推廣的是「先驗介面」

TetherMem 最值得移植的不是某個固定的 $\alpha=0.25$ 或 $\rho_{\min}=0.05$，而是 **prior → logit bias → normalized attention** 這個介面。未來可比較固定 heuristic、由 prompt/scene graph 產生的 prior、由 JEPA predictive state 產生的 prior，以及由 Energy-based scorer 產生的 prior。只要它們都輸出正值 $\pi(q,i)$，就能以相同 attention API 進行公平 ablation。

## 9. 我會怎樣延伸這篇工作

若要將這篇研究發展成下一個可投稿方向，我會優先做一個 **Predictive Energy Memory Router**。第一階段保留 TetherMem 的 normalized routing，不改 generator；第二階段以小型 frozen/linear JEPA predictor 對每個 memory token 產生 future-state compatibility；第三階段把 compatibility、subject identity、recency 與 scene-progress disagreement 合併成

$$
\log\pi(q,i)
=\beta_{\mathrm{id}}\,b_{\mathrm{id}}(q,i)
+\beta_{\mathrm{scene}}\,b_{\mathrm{scene}}(q,i)
-\beta_{\mathrm{stale}}\,b_{\mathrm{stale}}(i)
-\beta_{\mathrm{unc}}\,D_{\mathrm{pred}}(q,i).
$$

評估上不能只看 FVD 或 Dynamic Degree，應同時測 subject identity、scene progression、prompt-directed event、late-stage artifact，以及 routing cost。最關鍵的實驗應是：在不增加 generator training 的情況下，是否能用 predictive energy 取代 SAM 2 binary mask，並在多主體與遮擋場景中比固定區域 prior 更穩定。

## 10. 結論

TetherMem 將長影片生成的一個常見但容易被忽略的問題重新命名並量化：**歷史記憶可能同時是 identity 的保護傘，也是 scene progression 的枷鎖。** 它的答案不是把 memory 全部放掉，而是讓 subject query 與 scene query 透過不同的 region/age-conditioned prior 讀取相同歷史，並把控制放在 normalized attention 的 logit space。

在論文設定下，TetherMem 以 Overall EP=0.780、Progression EP=0.769、nT=1.289 領先主比較，且在 Deep Forcing host 上仍提升 progression EP；但它需要 offline reference rollout、SAM 2 mask 與額外 attention 計算，且仍受到 subject-prior drift、單一 foundation family 與有限 prompt/annotator 規模限制。我的總評是：**這不是一篇「再造一個影片生成 backbone」的論文，而是一個很適合被移植到 Energy-based Transformer、JEPA predictive state 與 VAR scale-wise memory 的控制抽象。**

## 定性結果圖

下圖是從論文 PDF 擷取的定性比較圖，只保留論文中的方法/軌跡 figure，不包含完整 PDF 頁面或人評介面。可觀察 Full Memory、單獨 routing、完整 TetherMem 以及跨方法長時段 trajectories 的差異；TetherMem 的主張不是讓人物消失，而是讓背景、視角與場景狀態在保持主體可辨識的前提下繼續展開。

![TetherMem qualitative comparison: Full Memory, routing ablations, TetherMem and cross-method long-horizon trajectories](../../../../asset/TetherMem_fig1_qualitative_comparison.png)

> **圖像來源與處理說明：** 圖像取自 TetherMem arXiv PDF 的論文 figure，透過本 repository 的 PDF image extraction workflow 提取，並只保留必要的定性比較素材。原始論文與作者資訊請見 [1] 與 [2]。

## References

[1]: https://arxiv.org/html/2608.26902v1 "Tether the Subject, Release the Scene: Query-Aware Memory Routing for Long-Horizon Autoregressive Video Generation"
[2]: https://lichen1015.github.io/tethermem/ "TetherMem official project page"
[3]: https://arxiv.org/abs/2606.02553 "LongLive-RAG: A General Retrieval-Augmented Framework for Long Video Generation"
[4]: https://arxiv.org/abs/2506.08009 "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"
[5]: https://arxiv.org/abs/2603.12513 "MemRoPE: Training-Free Infinite Video Generation via Evolving Memory Tokens"
[6]: https://arxiv.org/abs/2512.05081 "Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression"
[7]: https://arxiv.org/abs/2311.17982 "VBench: Comprehensive Benchmark Suite for Video Generative Models"
[8]: https://arxiv.org/abs/2408.00714 "SAM 2: Segment Anything in Images and Videos"

---

**研究狀態：** 本報告依 2026-08-28 可取得的 arXiv v1、作者 project page 與相關研究摘要整理；TetherMem 目前在 arXiv 上標示為預印本，不能寫成已通過 ICCV/CVPR/ICML/NeurIPS 等頂會審查的論文。
