# AI Daily

## 2026-11-04 — What Remains in Sight? Autoregressive Video Decoding as Representation-Guided Context Rewriting

**選中論文**：*What Remains in Sight? Autoregressive Video Decoding as Representation-Guided Context Rewriting*（[arXiv:2611.04164][1]）。作者為 Lixuan He、Jihyeon Je、Yang You、Leonidas Guibas，論文 TeX 將四人均標為 Stanford University；官方 NeurIPS 頁核實作者與題名，但未另列機構。[1] [2] 版本是 arXiv v1，首次提交與更新均為 2026-11-03；截至 2026-11-04，NeurIPS 官方資料列為 **NeurIPS 2026 正式收錄／poster**，但 arXiv metadata 仍未列 proceedings DOI 或 journal reference，故應同時寫成「NeurIPS 2026 論文、arXiv v1 預印本」，不把它誇大成已核實的 camera-ready 版本。

### 為何今天選它

依已讀 repo index，本篇 arXiv ID、標題與版本未與 AI Daily 既有收錄重複。就本批候選而言，FlowTranslator 著重 in-image translation，RSF 著重高解析度 DiT 語義，Z-Cache 著重 feature cache，Flat Features 著重 channel-energy detail；本篇則把**長時域 AR video 的可見 context 配置**本身提升為研究對象。它同時具備 training-free inference、attention/context modulation、VAR 式視覺自回歸的直接啟發，以及官方 NeurIPS 收錄、七個 backbone family、matched visible-token budget 和機制診斷等較完整證據。[1] [2] 這比只因貼合 energy-based 或 zero-shot 關鍵字而選一篇預印本更穩妥；代價是它並非 JEPA 或 energy model，本篇對那些方向的價值主要在可遷移的 context-selection 抽象。

未選 Flat Features、RSF 或 Z-Cache，不代表它們不值得讀：前者最貼近 energy-based feature modulation，後兩者在 DiT 推理控制與 cache correction 上也有漂亮結果；但本篇在正式 venue、跨 backbone matched protocol、長片機制診斷與相關基線覆蓋上更平衡。這是「今日唯一深讀」的品質取捨，而非對其他候選的否定。

### 核心貢獻與方法

作者將長片生成重新寫成：歷史已經很長，但 decoder 每個 block 只能看固定數量的 visible context，問題因此不是「保留最近幾幀」而是「哪些歷史證據值得被看見」。ReCR（Representation-guided Context Rewriting）在 decoder clean context pass 取得每個 latent frame/unit 的內部表徵，依全局相容性、近期冗餘、早期／近期邊界覆蓋與選取集合的相似度，挑出稀疏歷史；必要時做一次 spatial-range exchange，再將被選單元按原時間順序壓到 compact causal axis，重新套用 RoPE。這是 inference-time 的 context policy，不改 generator 參數，也不另加訓練目標；但會增加選擇、KV bank 與重編碼等工程成本，不能等同於免費加速。[1] [3]

令第 j 個歷史 unit 的 layer-ℓ token 平均向量為 $z_{j,m}^{(\ell)}$，其正規化表徵為

$$
u_j=\operatorname{Norm}\left(\frac{1}{n_j}\sum_m z_{j,m}^{(\ell)}\right),\qquad
s_t(j)=\kappa\langle u_j,g_t\rangle-\lambda\log\left(\varepsilon+\exp(\kappa\langle u_j,l_t\rangle)\right).$$

其中 $\operatorname{Norm}(v)=v/\|v\|_2$；$g_t$ 是歷史的 early/global 摘要，$l_t$ 是偏近期的摘要；$\langle\cdot,\cdot\rangle$ 是 cosine-like 內積；$\kappa$ 控制相容性尺度，$\lambda$ 控制對近期重複證據的懲罰，$\varepsilon$ 避免數值問題。直覺是留下「能代表整段 rollout、又不是只重複最近內容」的 unit。實作再以 boundary coverage $\Phi_t$ 保證早期與近期都有人被選，以 pairwise redundancy $\Psi_t$ 抑制相似 unit，最後以 greedy 近似 subset objective；因此這個 utility 很像一個手工設計的 energy，而論文並未宣稱它是 energy-based model。當歷史不超過 budget 時直接全保留；超過 budget 時才重寫 context。

### 實驗與證據

主實驗使用 MovieGen prompt suite 的 128 個 prompts，分別 rollout 30 秒與 60 秒，評估 VBench-Long 的 Total Score、Dynamic Degree、subject/background consistency、motion smoothness、aesthetic 與 imaging quality。方法疊加到 Self-Forcing、Self-Forcing++、LongLive、Causal-Forcing、HiAR、Deep Forcing、Infinity-RoPE 七個 backbone family；同一 family 內固定 text condition、seed、sampling schedule、causal protocol 與 visible-token budget。作者報告平均 Total Score 增益為 30 秒 +1.41、60 秒 +1.94；Dynamic Degree 的最高增幅分別為 Self-Forcing++ 的 +17.92 與 Causal-Forcing 的 +23.65。[1]

這些數字不是所有 metric 都支配。以 Causal-Forcing 為例，30 秒 Total 由 80.77 升至 82.50、Dynamic Degree 由 62.64 升至 77.97，但 subject、background 與 temporal flickering 並非每項同步改善。Deep Forcing 的 30 秒 drift diagnostic 則顯示 TA overall 25.69→27.08、aesthetic drift 3.56→1.76；Infinity-RoPE 也由 25.92→26.59、3.30→2.01，但個別 tail metric 仍可能變差。[1] 另有 24 個 controlled prompt families，60 秒影片切成五段並測 $K=0\ldots4$ 次 directive switch；圖示 ReCR 隨切換次數增加的 alignment 下降較緩，但正文沒有提供完整數值表，所以只能採趨勢解讀。

效率資料也很重要：單張 H200、batch size 1、60 秒 rollout，Self-Forcing 的 block latency 728→712 ms，LongLive 685→659 ms，Causal-Forcing 737→711 ms，但 Infinity-RoPE 667→710 ms，說明它不是普遍降低 wall-clock 的方法。作者的 full-history KV bank 仍需保留所有過去 unit；README 估算 240 latent frames、30 layers、1560 tokens/frame、12 heads、head dimension 128、fp16 約 69 GB。[3] 因此它壓縮的是 decoder「看見什麼」，不是整個歷史儲存量。

證據完整度方面，本篇比單一 qualitative demo 更扎實：有主表、drift 分析、context perturbation、prompt switching 與 latency，且同一 backbone 內採 matched budget；但證據仍不是無缺口。正文沒有主要結果的完整 seed 數、信賴區間或顯著性檢定，部分診斷圖沒有清楚標示資料集，消融也未完整掃描 utility 權重。官方程式庫雖提供核心選擇與測試元件，卻要求讀者自行整合特定 backbone；因此目前最可靠的結論是「在作者測試的長片 AR 系統與 budget 下，ReCR 有一致的動態—穩定性折衷優勢」，而不是「對任何 VAR、影片生成器或 zero-shot 部署都保證有效」。[1] [3]

### 相關研究與差異

**Self Forcing** 以 self-generated context、rolling KV cache 和 video-level loss 處理 exposure bias，是 training-based 的 train-test gap 解法；ReCR 不重訓 decoder，而是在推理時改變可見歷史，兩者可疊加。[4] **Deep Forcing** 以 Deep Sink 與 participative KV-cache pruning 穩定長 rollout，主要管理 cache 與 sink；ReCR 以 representation-guided、frame-level evidence selection 和 compact reindex 改變 context composition，且仍保留 full-history bank。[5] **Infinity-RoPE** 著重 Block-Relativistic RoPE、KV Flush 與 scene cut；ReCR 的位置處理是選完歷史後按 compact causal axis 重排，故不是同一種 positional extrapolation。[6] **TokenTrim** 以 latent representation 找出可能 corrupted 的 unstable tokens 後刪除，粒度是 token；ReCR 選整個 frame/unit，並同時處理全局相容、近期非冗餘與時間邊界。[7] **PaFu-KV** 用 teacher distillation 得到 learned salience head 壓縮 KV cache；ReCR 不加訓練頭，採 decoder 自身表徵與手工 utility，但也因此更依賴 heuristic。[8]

### 限制與個人評價

論文最清楚的限制是：mean-pooled frame representation 看不到 object、region 或 motion track；每一步只做 myopic selection，未預測數段後的證據需求；固定的 $\kappa,\lambda,\mu,\gamma,\beta$ 與 greedy selection 缺乏完整敏感度和顯著性分析。主要 benchmark 只有 128 prompts、30/60 秒，300 秒僅作定性展示；不同 backbone 的分數也不能直接排名。官方 repo 提供選擇、KV bank、reindex 與 toy tests，但 backbone-specific integration 不包含在內，必須能存取 pre-RoPE K/V、block-wise causal decoding 和內部 layer output。[3] 所以「training-free」不等於 universal zero-shot plug-in。

我的評價是：這篇值得深讀，首要原因不是它最像某一個熱門關鍵字，而是它有正式 NeurIPS 證據、跨 backbone 的 matched-budget 實驗，以及把「context 到底留下什麼」和「長片品質為何漂移」連成可檢驗的方法假說。方法新意屬中高：不是首次做 cache policy 或 token pruning，但把 frame-level representation、時間邊界、非冗餘、空間範圍與 compact RoPE 組成一個清楚的 inference-time context rewriting 介面。我的保留是 full-history memory 讓系統成本仍高，且手工 utility 未必能代表真正的未來生成需求；它比較像一個很好的 VAR/AR context policy 基線，而非已證明的通用長片解法。

### 研究延伸想法

1. **研究延伸想法｜Energy-based context selector。** 將目前的 $-\sum_jm_{t,j}s_t(j)-\mu\Phi_t+\gamma\Psi_t$ 正式定義為 context energy，加入下一個 block 的 prediction error、長期 drift 或 contrastive negative history，測試 energy 是否比 cosine/softplus heuristic 更能預測保留價值。這是我的延伸，不是論文原創貢獻。

2. **研究延伸想法｜接上 JEPA 式未來表徵。** 以 predictor 預測下一段 latent 的抽象 embedding，讓 $C_t$ 選擇能最小化 future representation prediction loss 的歷史，而不只依賴當前 decoder representation；可保留 training-free 版本作 frozen probe，也可另測小型 selector 訓練版，嚴格比較 zero-shot 與 learned policy 的差距。

3. **研究延伸想法｜VAR 的分層 context rewriting。** 將 ReCR 從整 frame/unit 擴展到 coarse frame、object region、patch token 三層 budget，對 scene return、物件重現和 prompt switching 做長期 retrieval；同時測真正不同 VAR backbone 的 adapter-free transfer，避免把目前的「可插拔」只理解為需要大量 backbone-specific engineering。

### References

[1]: https://arxiv.org/abs/2611.04164 "What Remains in Sight? Autoregressive Video Decoding as Representation-Guided Context Rewriting"
[2]: https://neurips.cc/virtual/2026/poster/149388 "NeurIPS 2026 official poster page"
[3]: https://github.com/lixuan27/ReCR "Official ReCR repository"
[4]: https://arxiv.org/abs/2506.08009 "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"
[5]: https://arxiv.org/abs/2512.05081 "Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression"
[6]: https://cvpr.thecvf.com/virtual/2026/poster/37370 "Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout"
[7]: https://arxiv.org/abs/2602.00268 "TokenTrim: Inference-Time Token Pruning for Autoregressive Long Video Generation"
[8]: https://arxiv.org/abs/2601.21896 "Past- and Future-Informed KV Cache Policy with Salience Estimation in Autoregressive Video Diffusion"
