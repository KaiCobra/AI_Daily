# AI Daily

## AcFlow：把文字條件變成可積分的 activation flow，讓 frozen DiT 具有連續、可泛化的控制軸

**研究日期：2026-09-15**　　**作者：Manus AI**

## 今日結論

本期選擇 **AcFlow: Controlling Text-to-Image Diffusion Transformers via Learned Conditional Activation Flow**。論文的核心觀點是：文字到圖像模型的控制不必只發生在 prompt、LoRA 權重或固定 steering vector 上，也可以發生在去噪 Transformer 的中間 activation space。AcFlow 在 frozen Diffusion Transformer（DiT）的一個 single-stream block 只修改 image-token residual，並以一個由概念文字條件化的 velocity field 搬運 activation。積分時間的終點 $T$ 直接成為連續控制旋鈕，$T=0$ 回到原始生成，較大的 $T$ 則逐步增強風格或抑制概念。[1]

這個想法特別值得研究，是因為它把三種通常分開的能力放在同一個介面：**token-varying update、activation-dependent direction，以及 unseen concept generalization**。在 FLUX.1-dev 的 style control 實驗中，AcFlow 在高風格對齊區域取得最佳的 style–content trade-off；在整個 style family 被排除於訓練之外時，仍能對未見風格產生可調強度的控制。[1] 不過，這不是嚴格意義的 training-free 方法：frozen 的是原始生成器，控制器本身仍需離線學習。因此比較準確的分類是 **frozen-backbone + learned controller + inference-time continuous control**。

本次選題也排除了儲存庫內已存在的 EBT、DIAL、VISTA、Logit Refiner 與其他 JEPA／VAR／training-free 文章。AcFlow 尚沒有獨立文章收錄，且它能把你近期關注的 **attention modulation、flow matching、zero-shot concept transfer** 連成一個具體的 activation-level 研究問題。

## 一、論文基本資訊

| 欄位 | 內容 |
|---|---|
| 論文標題 | *AcFlow: Controlling Text-to-Image Diffusion Transformers via Learned Conditional Activation Flow* |
| 作者 | Junran Wang、Zehao Jin、Tianyu Luan、Xinjie Shen |
| 研究單位 | South China University of Technology、Georgia Institute of Technology（論文作者資訊） |
| 發布狀態 | arXiv:2609.10723v1，2026-09-09 發布；論文頁未標示已接收的頂會 venue，因此本期將它標為近期預印本，而非已確認的 ICCV/CVPR/ICML 論文。[1] |
| 研究任務 | 連續 style control、concept suppression，以及在未見 style family／concept 上的條件泛化 |
| 基礎生成器 | frozen FLUX.1-dev；另以 Z-Image 檢驗跨 backbone portability |
| 主要介入位置 | FLUX single-stream block 16 的 image-token residual；Z-Image style 在 layer 19、suppression 在 layer 25 |
| 官方程式碼 | [Nove1yst/AcFlow](https://github.com/Nove1yst/AcFlow)；檢查時公開儲存庫主要包含 README 與 Apache-2.0 license，尚未看到完整訓練程式或權重。[3] |

## 二、為什麼選它

| 候選方向 | 與本期偏好的關係 | 儲存庫排除結果 |
|---|---|---|
| AcFlow | 近期 DiT activation modulation；以 learned conditional flow 支援 continuous control 與 unseen concepts | **選入**，尚無獨立收錄 |
| Energy-Based Transformers | 直接符合 energy-based transformer 偏好，但已在 2026-07 與 2026-05 收錄 | 排除重複 |
| DIAL | 近期 training-free attention control，但已在 2026-09-10 收錄 | 排除重複 |
| Logit Refiner | 新近 VAR 研究，但已在 2026-09-12 收錄 | 排除重複 |
| VISTA | VAR 的 test-time compositional alignment，但已在 2026-09-01 收錄 | 排除重複 |
| RAR | ICCV 2025 的 randomized autoregressive generation，方法重要但與既有 VAR 文章重疊較高 | 作為背景對照，不作本期主文 |

AcFlow 的選擇理由不是它已經是最終的 SOTA，而是它提供了一個很容易被移植、拆解與重新設計的控制介面：先固定 backbone，再學習「中間表徵應該如何沿著概念條件移動」。這個介面可以自然地接上能量函數、JEPA critic 或 VAR 的 scale-wise token state。

## 三、核心貢獻與創新點

### 3.1 把控制寫成 activation-space flow

傳統 activation steering 常使用固定方向 $d(c)$，或使用一個 scalar 只調整固定方向的強度。AcFlow 則學習一個速度場 $v_\phi(h,t,c;\sigma)$。因此，更新方向可以同時依賴目前 activation $h$、概念條件 $c$、flow time $t$ 與 DiT 的去噪 noise level $\sigma$。同一個概念不必對所有 image token 使用相同向量，也不必在所有去噪階段使用相同方向。[1]

### 3.2 用一個共享 field 處理多個概念

Style field 不是針對每一個 style 個別擬合一個 LoRA；它在多個 style family 上共享參數，並以自然語言的 fine-grained style description 作為條件。Suppression field 則以 `erase <concept>` 作為條件，從多個移除 pair 中學習概念抑制。這使作者能測試「訓練時沒有看過整個 style family 或 target concept」的 transfer。

### 3.3 flow horizon 是可解釋的連續控制旋鈕

AcFlow 將積分終點 $T$ 直接用作 intervention strength。$T=0$ 不改變 activation；增加 $T$ 通常會提高 style alignment 或 suppression strength，但也可能降低 content alignment、改變物體位置或改變構圖。這比只用 `weak`、`strong` 等語言修飾詞更容易校準與掃描。

### 3.4 以 token 與 noise-level 分析證明它不是固定向量

作者將 endpoint displacement 拆成 leading rank-one component 與 residual component，並觀察 off-axis energy fraction。結果顯示中段去噪時 token-varying directions 的比例顯著升高，晚期又降低。這支持 AcFlow 的主要主張：它不是把同一個方向廣播到所有 token，再隨時間做 scalar scaling，而是在生成軌跡中調整更新幾何。[1]

## 四、技術方法詳解

### 4.1 生成器與 activation state

令 $G_\theta$ 表示 frozen 的 DiT-based denoising velocity predictor，hidden dimension 為 $d$。在一個 single-stream block 的輸出位置，將 text token 與 image token 串成

$$
h=[h^{\mathrm{txt}};h^{\mathrm{img}}]\in\mathbb{R}^{S\times d},
$$

其中 $S=S_{\mathrm{txt}}+S_{\mathrm{img}}$。給定 source prompt $p$ 與概念描述 $c$，目標是產生一個 activation $h'$，使後續生成接近 target prompt 的效果，同時盡量保留 source content。[1]

作者只介入 image-token state：

$$
h'=[h^{\mathrm{txt}};\varphi_T^{\mathrm{img}}(h^{\mathrm{img}})],
$$

也就是保留 text-token slice，將 image-token slice 經過 flow 後寫回 residual stream。其理由是 image state 更直接位於 output head 的 residual path；text state 對輸出的影響較間接，且兩種 token 的 activation distribution 不一定共享同一個幾何。

### 4.2 Activation flow 與 Euler integration

在固定的 denoising noise level $\sigma\in[0,1]$ 和概念條件 $c$ 下，定義 activation flow

$$
\varphi_t(\cdot;c,\sigma):\mathbb{R}^{S_{\mathrm{img}}\times d}
\rightarrow \mathbb{R}^{S_{\mathrm{img}}\times d}.
$$

flow time $t$ 與生成器的 noise level $\sigma$ 是兩個不同的時間變數。每一個 denoising step 都會從該 step 的 block output 重新開始一段 activation-flow integration。

速度場滿足常微分方程：

$$
\frac{d}{dt}\varphi_t(h)=v_\phi(\varphi_t(h),t,c;\sigma),
\qquad \varphi_0(h)=h. \tag{1}
$$

因此 intervention endpoint 為

$$
 h'=\varphi_T(h)=h+
 \int_0^T v_\phi(\varphi_t(h),t,c;\sigma)\,dt. \tag{2}
$$

實作時使用 $N$ 步 forward Euler：

$$
 h^{(k+1)}=h^{(k)}+\frac{T}{N}
 v_\phi\left(h^{(k)},\frac{kT}{N},c;\sigma\right),
 \qquad k=0,\ldots,N-1. \tag{3}
$$

初始值為 $h^{(0)}=h$，最後將 $h^{(N)}\approx\varphi_T(h)$ 寫回 DiT residual stream。主要 FLUX 設定使用 28 個 denoising steps、$N=3$ 個 Euler steps，因此每張圖大約需要 $28\times3=84$ 次 FlowBlock evaluation；論文沒有提供完整端到端 latency table，因此不應將它宣稱為低延遲方法。

這個定義也清楚展示 AcFlow 與固定 steering vector 的差異。若限制

$$
v_\phi(h,t,c)=d(c),
$$

就退化成一般的 additive intervention：$h_i'=h_i+T d(c)$。若速度只依賴 mean-pooled activation，所有 token 會收到共享的 velocity。AcFlow 的完整版本沒有這些限制，token 的方向與幅度都可隨 activation 改變。

### 4.3 Backbone-matched FlowBlock

AcFlow 以一個符合 backbone 形式的 Transformer-style FlowBlock 實作 $v_\phi$。令概念文字 token features 為

$$
C=E_{\mathrm{txt}}(c)\in\mathbb{R}^{S_c\times d},
$$

若 generator 有 pooled text feature，另令 $\bar c=E_{\mathrm{pool}}(c)$。FlowBlock 先用 cross-attention 取出概念資訊。$e_{\mathrm{flow}}(t)$ 是 flow time embedding，$e_c(\bar c)$ 是 pooled concept projection。經過 backbone-matched 的 adaptive normalization 與 residual gate：

$$
(x_t,g_t)=\mathrm{AdaNorm}_\phi
\left(h^{\mathrm{img}};e_{\mathrm{flow}}(t)+e_c(\bar c)\right),
$$

$$
\tilde h=h^{\mathrm{img}}+g_t\odot
\mathrm{CrossAttn}_\phi(Q=x_t,K=C,V=C). \tag{5--6}
$$

接著使用一個 native-style Transformer block：

$$
 h_{\mathrm{out}}=\mathrm{TransformerBlock}_\phi
 \left(\tilde h;e_{\mathrm{model}}(\sigma)+e_c(\bar c)\right),
$$

$$
 v_\phi(h^{\mathrm{img}},t,c;\sigma)=h_{\mathrm{out}}-h^{\mathrm{img}}. \tag{7}
$$

關鍵設計是：概念 cross-attention 使用 flow time $t$，而 self-attention／FFN 的 native conditioning 仍使用 denoising noise level $\sigma$。因此 field 能在一次 activation transport 內改變自身方向，同時知道 generator 目前處於哪個 denoising stage。

### 4.4 Velocity distillation

訓練資料由 $(z_0,p_{\mathrm{src}},p_{\mathrm{tgt}},c)\sim\mathcal{D}$ 組成，其中 $z_0$ 是 target image 的 VAE latent。取噪聲 $\epsilon\sim\mathcal{N}(0,I)$ 與 noise level $\sigma$，建立 noisy latent：

$$
 z_\sigma=(1-\sigma)z_0+\sigma\epsilon. \tag{8}
$$

Teacher 是在 target prompt 下、沒有 AcFlow 的 frozen generator：

$$
 u_{\mathrm{tgt}}=G_\theta(z_\sigma,\sigma,p_{\mathrm{tgt}}). \tag{9}
$$

Student 使用 source prompt，但在 block $b$ 啟用 AcFlow：

$$
 \hat u=G_{\theta,\phi}
 (z_\sigma,\sigma,p_{\mathrm{src}},c,T). \tag{10}
$$

Teacher 與 student 共用同一個 $z_\sigma$、$\sigma$ 與 guidance setting。訓練時令 $T=1$，以 velocity matching objective 最小化：

$$
 \mathcal{L}_{\mathrm{vel}}(\phi)=
 \mathbb{E}_{\mathcal{D},\epsilon,\sigma}
 \left[\mathrm{MSE}\left(\hat u,
 \mathrm{stopgrad}(u_{\mathrm{tgt}})\right)\right]. \tag{11}
$$

這裡的「zero-shot」不是完全不訓練，而是指同一個已學到的 concept-conditioned field 可以在 task family 內處理訓練時沒看過的 target concept。Style field 與 suppression field 仍是分開訓練的；換 backbone 時也要重新訓練 backbone-matched field。

## 五、實驗結果與性能指標

### 5.1 實驗設定

| 設定 | FLUX style / suppression | Z-Image portability |
|---|---:|---:|
| 解析度 | $512\times512$ | $512\times512$ |
| Denoising steps | 28 | 28 |
| Guidance | 3.5 | CFG 3.5，scheduler shift 6 |
| 介入位置 | single-stream block 16 | style layer 19；suppression layer 25 |
| Flow Euler steps | $N=3$ | $N=3$ |
| 訓練 horizon | $T=1$ | $T=1$ |
| 評估 horizon | 主要使用 $T=2$ | 掃描 $T\in\{0,0.4,0.8,1.2,1.6,2.0\}$ |
| FLUX field 訓練預算 | 20k updates，effective batch 16 | — |
| Z-Image field 訓練預算 | — | 10k updates |

Style alignment 是生成圖與 MegaStyle reference image 的 fine-tuned style-sensitive encoder cosine similarity；content alignment 是生成圖與 content prompt 的 CLIP similarity。這兩個指標分別偏向風格與語義內容，不等同於完整的人類偏好或構圖保真度。[1]

### 5.2 Continuous style control

作者將 MegaStyle description 按 style family 分組，訓練時使用 39 個 family，另保留 6 個完整未見 family 作 held-out evaluation。主要表格結果如下，數字依論文 Table 2 四捨五入：[1]

| 方法 | Held-in Style ↑ | Held-in Content ↑ | Held-out Style ↑ | Held-out Content ↑ |
|---|---:|---:|---:|---:|
| Styled prompt reference | 0.548 | 0.294 | 0.511 | 0.287 |
| Concept Sliders | 0.511 | 0.279 | — | — |
| Text Slider | 0.157 | 0.306 | — | — |
| Linear-AcT | 0.271 | 0.202 | — | — |
| Mean-AcT | 0.440 | 0.268 | — | — |
| ActAdd (block 16) | 0.300 | 0.278 | — | — |
| SHIFT | 0.221 | 0.279 | — | — |
| **AcFlow** | **0.537** | **0.286** | **0.442** | **0.281** |

在論文摘要所報告的固定 operating point 上，AcFlow 的 style/content alignment 為 **0.5365/0.2860**；style alignment 最高的 activation-intervention baseline Mean-AcT 為 **0.4397/0.2684**，Concept Sliders 則為 **0.5109/0.2786**。[1] 這個比較支持 AcFlow 在高風格區域具有較好的 trade-off，但不能解讀成它在所有 style/content operating point 都同時優於 styled prompt reference。

![AcFlow 連續風格控制：同一張人像在未控制與四種細粒度 pop-art 描述下的結果。圖像由論文 PDF 透過 `pdf-image-extractor` 擷取，未改動原圖。](../../../../asset/AcFlow_style_control.png)

*圖一。AcFlow 將同一個 content prompt 與初始噪聲沿著不同 concept description 與 flow horizon 生成。視覺結果顯示它能從 unsteered portrait 逐步移向 screen-print、glossy reflective、digital illustration 與 paper-cutout 等風格；這是定性示意，不是新的模型評估。來源：作者 Figure 5，論文 [1]。*

### 5.3 未見 style family 的泛化

在訓練中完整排除 style family，而不是只換一個熟悉 family 的新描述後，AcFlow 在 held-out family 上達到 **0.442 style alignment / 0.281 content alignment**（$T=2$）。作者也展示 $T$ 對 held-in 與 held-out family 都能形成連續 response。這是本文較有價值的 zero-shot claim，但應使用嚴格定義：它是 **within-task-family、same-backbone、learned-field 的 concept transfer**，不是跨任務、跨模型或無訓練 zero-shot。

### 5.4 Flow horizon 與 content drift

Field 在 $T=1$ 訓練，測試時可外推到 $T=2$ 甚至更高。風格強度通常隨 $T$ 增加而上升，之後逐漸飽和；content alignment 則會下降，較大的 horizon 還可能改變 composition。這個現象把 $T$ 變成一個可繪製的 Pareto curve，而不是只有一個最佳 checkpoint。

### 5.5 Euler steps 與 intervention block 消融

| 變因 | 設定 | Held-in Style ↑ | Held-in Content ↑ | Held-out Style ↑ | Held-out Content ↑ |
|---|---:|---:|---:|---:|---:|
| Block | 12, $N=3$ | 0.541 | 0.279 | 0.475 | 0.281 |
| Block | 20, $N=3$ | 0.508 | 0.282 | 0.464 | 0.282 |
| Euler steps | Block 16, $N=1$ | 0.432 | 0.284 | 0.375 | 0.285 |
| Euler steps | Block 16, $N=3$ | 0.510 | 0.279 | 0.442 | 0.281 |
| Euler steps | Block 16, $N=5$ | 0.485 | 0.280 | 0.433 | 0.281 |
| Euler steps | Block 16, $N=10$ | 0.493 | 0.283 | 0.439 | 0.286 |

Block 12、16、20 都能工作，表示方法不完全依賴單一神奇位置。Euler-3 在 style alignment 上是相對精簡且強的設定；增加到 5 或 10 步沒有單調提升，說明 integration accuracy 與控制效果並非簡單正相關。[1]

### 5.6 Concept suppression 與安全應用

在一般 concept suppression 中，作者使用從 InstructPix2Pix mining 的 removal pairs，訓練資料涵蓋 1,122 個 concept。held-in 例子包括 fog、lighthouse、river；held-out 例子包括 peonies、feathered hat、daisies。較大的 $T$ 可以逐步縮小 lighthouse、river 或花朵，但也可能影響 appearance 與 composition。這部分主要是定性結果，論文沒有提供與所有 suppression baseline 對齊的單一 generic-concept 指標表。[1]

附錄另測試 NSFW suppression。使用一個 separate field，在 I2P sexual-content subset 的 931 prompts 上，NudeNet total detections 由原始 FLUX 的 **402** 降至 AcFlow $s=2$ 的 **73**，再降至 $s=3$ 的 **38**。但是 DINO-I2I similarity 由 **0.612** 降至 **0.534**，LPIPS 由 **0.509** 升至 **0.563**，顯示較強 suppression 主要以內容漂移換取。[1] 這項安全實驗不應被誤述成不改變內容的 concept erasure。

### 5.7 Flow field 幾何分析

令每個 image token 的 endpoint displacement 組成 $\Delta\in\mathbb{R}^{S_{\mathrm{img}}\times d}$。作者用 leading right-singular vector $u_1$ 拆出 rank-one component $Q_1$ 與 residual $R$：

$$
q_i=\langle\Delta_i,u_1\rangle u_1,
\qquad r_i=\Delta_i-q_i,
\qquad \Delta=Q_1+R. \tag{12}
$$

再定義 off-axis energy fraction：

$$
\rho(\sigma)=
\frac{\|R(\sigma)\|_F^2}{\|\Delta(\sigma)\|_F^2}.
$$

Held-in／held-out style 的 $\rho$ 起始約為 **12.3%/14.3%**，在中段 noise level 達到 **38.3%/41.4%** 的最大值，接近生成結尾時下降至 **3.2%/3.8%**。因此，若假設 $\Delta(\sigma)=a(\sigma)\Delta_{\mathrm{ref}}$、只對固定方向做 scalar scaling，$\rho$ 應保持不變；觀察到的曲線則否定了這個簡化模型。這是本文最能支持「learned field 不是 fixed steering vector」的量化分析。

## 六、相關研究背景與定位

| 研究 | 核心操作 | 與 AcFlow 的關係 |
|---|---|---|
| DiT | 用 Transformer 取代 latent diffusion 的 U-Net，並以模型規模／Gflops 研究可擴展性 | AcFlow 以 frozen DiT 作為 activation control 的載體；[5] |
| Flow Matching | 對固定 conditional probability path 的 vector field 做 regression，避免直接模擬 CNF | AcFlow 借用 velocity-field / ODE 語言，但它 transport 的對象是中間 activation，不是直接學 image data distribution；[4] |
| Concept Sliders | 為每個概念微調 LoRA，再以 inference scale 控制屬性 | AcFlow 將 per-concept adapter 改成 task-family shared field，但代價是需要 field training；[6] |
| AcT | 以 optimal transport map 在 source／target activation distributions 間搬運 | AcFlow 延續 activation transport 想法，但從較固定的 transport map 推進到 concept-conditioned、state-dependent velocity field；[7] |
| SHIFT | 在 flow transformer 的 hidden intermediates 做 steering，著重介入位置與方向縮放 | AcFlow 更強調 activation-dependent direction、token-varying update 與 unseen concept transfer；[8] |
| DIAL | 直接從 DiT attention 讀出 subject-aware map，部分流程可 inference-time training-free | DIAL 與 AcFlow 都利用 frozen generator 內部訊號，但 DIAL Phase I 是 inference-only，AcFlow 的 controller 需要訓練；[9] |

因此 AcFlow 的新意不是「第一次在中間層改 activation」，而是將控制更新的幾何提升為一個 **可積分、可條件化、可連續調整、可依 token 狀態改變方向** 的 learned field。

## 七、批判性評估

### 優點

第一，方法介面清楚。原始 DiT 權重保持 frozen，概念條件走 generator 原本的文字路徑，介入位置只需指定一個 block 與 image-token slice，因此容易做 controlled ablation。第二，$T$ 提供了比 prompt wording 更直接的 strength coordinate，可以用同一個 field 產生完整的 style–content Pareto curve。第三，作者沒有只展示漂亮圖片，也用 token-swap diagnostic、rank-one residual decomposition 與 noise-level analysis 檢查方法是否真的需要 token-specific nonlinear update。第四，style family-disjoint split 與 concept-disjoint suppression split 使「zero-shot」至少有可驗證的 protocol，而非只用新 prompt 當成泛化證據。

### 限制

第一，AcFlow 不是 training-free。FLUX style field 需要 20k updates，Z-Image field 需要 10k updates，suppression 也使用另一套資料與 field。第二，zero-shot 泛化仍受任務 family 與 backbone 限制；換到 Z-Image 時作者重新建立 backbone-matched field，而不是不訓練即可轉移。第三，論文只保證 activation control，不保證 composition preservation。作者明確承認 intervention 可能改變 object position、pose 與 scene layout。第四，style alignment 使用 fine-tuned style-sensitive encoder，content alignment 使用 CLIP；兩者都不能完整反映人類對構圖、身份、物體數量或細節的判斷。第五，generic concept suppression 主要是定性展示，NSFW quantitative result 使用 separate field，不能直接推廣成一個普遍的 erasure module。第六，官方 GitHub 在本次檢查時尚未提供完整可重現的訓練程式或權重，復現成本與結果獨立性仍待後續驗證。[3]

## 八、對 EBT、JEPA、VAR、training-free 與 zero-shot 的研究啟發

### 8.1 AcFlow × Energy-Based Transformer：把 velocity field 改寫成可校準 energy landscape

AcFlow 學的是 $v_\phi(h,c,\sigma)$，而 Energy-Based Transformer（EBT）學的是 input–candidate compatibility energy。可以研究一個混合版本：

$$
E_\psi(h,c,\sigma),
\qquad
v_\psi(h,c,\sigma)=-\nabla_hE_\psi(h,c,\sigma).
$$

AcFlow 的 Euler transport 便成為在 activation space 沿著低能量方向移動：

$$
 h^{(k+1)}=h^{(k)}-
 \eta_k\nabla_hE_\psi(h^{(k)},c,\sigma).
$$

它比單純把 AcFlow 名稱改成 energy-based 更嚴格，因為可檢查 path independence、局部 Hessian、能量 monotonicity 與不同 candidate state 的排序。研究問題是：若 EBT 的 energy 是更可靠的 compatibility verifier，是否能用 energy drop 決定哪些 denoising step 才需要啟動 controller？

### 8.2 AcFlow × JEPA：用 predictive consistency 自動決定 intervention strength

AcFlow 的主要失敗模式是 style／suppression 變強時 content 或 composition drift。可加入 frozen JEPA encoder $f_{\mathrm{JEPA}}$ 作為內容預測 critic：

$$
\mathcal{L}_{\mathrm{preserve}}(T)=
1-\cos\left(f_{\mathrm{JEPA}}(x_T),
 f_{\mathrm{JEPA}}(x_0)\right),
$$

其中 $x_0$ 是 unsteered output，$x_T$ 是控制後 output。推理時以 style gain 與 predictive consistency 共同選擇最大可接受 horizon：

$$
T^*=\max_T\left\{
S_{\mathrm{style}}(T)-\lambda
\mathcal{L}_{\mathrm{preserve}}(T)\geq\tau
\right\}.
$$

這會把「固定 $T$」改成 **JEPA-gated adaptive control**。更進一步，可以不用 pixel reconstruction，而是在 latent prediction space 中比較 object identity、spatial relation 或 future state consistency。

### 8.3 AcFlow × VAR：從 denoising block transport 改成 next-scale token transport

VAR 在每一個 scale $s$ 預測下一層 visual tokens。可令當前 scale state 為 $r_s$，並學習

$$
 r_s'=\varphi_{T,s}(r_s;c),
$$

其中 field condition 同時包含 concept token 與 scale embedding。早期 coarse scales 負責 layout／object allocation，後期 fine scales 負責 texture，因此可令

$$
T_s=T_{\mathrm{layout}}\mathbf{1}[s\leq s_0]
+T_{\mathrm{detail}}\mathbf{1}[s>s_0].
$$

這能測試 AcFlow 的 token-wise adaptive transport 是否比 VISTA、SynVAR 或固定 attention bias 更適合 VAR 的離散 next-scale interface。關鍵評估不能只看 FID，還要看 attribute binding、position、object count、source preservation 與每個 scale 的 error accumulation。

### 8.4 Learned controller 與真正 training-free attention modulation 的分界

AcFlow 的 field 是 learned controller；DIAL Phase I、RTD、部分 attention steering 則比較接近 inference-only。後續研究應把方法分成三類：

| 類別 | 是否更新 backbone | 是否需要離線控制器訓練 | 例子 |
|---|---:|---:|---|
| Inference-only | 否 | 否 | attention/logit/latent modulation 類方法 |
| Frozen backbone + learned controller | 否 | 是 | **AcFlow** |
| Backbone 或 adapter fine-tuning | 是或部分是 | 是 | Concept Sliders、某些 erasure adapter |

一個有價值的 baseline 是：保留 AcFlow 的 image-token intervention site，但不用 learned $v_\phi$，改從每一步 cross-attention map、token norm、JEPA disagreement 或 EBT energy gradient 建立 deterministic update。這樣可以量化「learned nonlinear geometry」相對於「training-free diagnostic signal」到底貢獻多少。

### 8.5 嚴格 zero-shot protocol

對這類論文，zero-shot 應至少拆成四層：新 prompt、訓練 family 內的新描述、完全未見 concept、以及新 backbone／新 task family。AcFlow 主要證明前兩層與部分第三層；它尚未證明第四層。未來若要把結果宣稱為更強的 zero-shot，應同時報告：不重新訓練的 cross-backbone transfer、不同 resolution、不同 text encoder，以及 controller 是否能在新 task 上保持 content preservation。

## 九、我的評價與研究意義

我認為 AcFlow 的研究價值高於它目前的 benchmark 絕對數字。它把「生成控制」從 prompt engineering 重新定位為一個 activation-space dynamical system，並用 $T$、$t$、$\sigma$ 三個可分離變數描述控制強度、控制積分時間與原始生成時間。這使後續研究能明確問：控制方向是否由 energy gradient 產生？哪些 image tokens 應被改動？在何種 noise level 控制最有效？怎樣用 JEPA 保護語義與構圖？

另一方面，報告不應把 AcFlow 包裝成免訓練或普遍 zero-shot 方法。它真正的定位是 **一個可共享的、條件化的、可積分的 learned activation controller**。它的實驗已證明 concept-shared field 可以在同一 task family 內轉移，卻仍需要大量離線 field training、backbone-specific adaptation 與更完整的 latency／reproducibility 評估。若將它與 EBT 的 energy verifier、JEPA 的 predictive consistency、VAR 的 scale-wise token state 及 training-free attention signal 結合，便可能形成一條更有研究張力的路線：**用能量判斷是否應控制，用 JEPA 判斷控制是否傷害內容，用 VAR 或 DiT 的 token geometry 決定控制位置，最後只在必要的步驟啟動 intervention。**

## References

[1]: https://arxiv.org/html/2609.10723v1 "AcFlow: Controlling Text-to-Image Diffusion Transformers via Learned Conditional Activation Flow"
[2]: https://arxiv.org/abs/2609.10723 "AcFlow arXiv abstract page"
[3]: https://github.com/Nove1yst/AcFlow "Official AcFlow code repository"
[4]: https://arxiv.org/abs/2210.02747 "Flow Matching for Generative Modeling"
[5]: https://arxiv.org/abs/2212.09748 "Scalable Diffusion Models with Transformers"
[6]: https://arxiv.org/abs/2311.12092 "Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models"
[7]: https://arxiv.org/abs/2410.23054 "Controlling Language and Diffusion Models by Transporting Activations"
[8]: https://arxiv.org/html/2604.09213v1 "SHIFT: Steering Hidden Intermediates in Flow Transformers"
[9]: https://arxiv.org/html/2609.11507v1 "Harnessing Intrinsic Subject-Aware Attention for Controllable Multi-Subject Video Generation"
