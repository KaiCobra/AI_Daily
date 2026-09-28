# AI Daily

## 2026-09-28｜I Act Therefore I Am：動作條件何時足以讓 JEPA 學到因果機制？

> **一句話結論**：A-JEPA 把「能不能預測下一個 latent」和「latent 是否對應到可遷移的因果狀態」分開，指出真正能消除 representation mixing 的訊號，不只是 anti-collapse regularization，而是足夠豐富的 **action-induced variation**。

## 論文基本資訊

| 欄位 | 資訊 |
|---|---|
| 論文標題 | *I Act Therefore I Am: When Is JEPA's Action-Conditioning Enough to Learn Causal Mechanisms?* |
| 作者 | Yuhang Liu、Zhuo Huang、Javen Qinfeng Shi |
| 研究單位 | Responsible AI Research Centre, Australia；Australian Institute for Machine Learning, Adelaide University [3] |
| 發表狀態 | arXiv v1 預印本，2026-09-25 提交；截至本報告日沒有可核實的會議接收或期刊發表資訊 [4] |
| 領域 | cs.LG；JEPA、causal representation learning、action-conditioned world models |
| 論文頁面 | [arXiv abstract page][1]；[HTML 全文][2]；[PDF][3] |
| Repo 重複檢查 | `2609.31161`、完整標題與 `A-JEPA` 均未在本 repo 出現；與既有 [AD-WM](../AD-WM/AI_Daily_AD-WM.md)、[Causal-JEPA](../Causal-JEPA/AI_Daily_Causal_JEPA.md)、[Physically Grounded JEPA](../PhysicallyGroundedJEPA/AI_Daily_PhysicallyGroundedJEPA.md) 是研究脈絡相鄰，但不是同一篇論文。 |

作者背景也符合這篇工作的理論取向。Yuhang Liu 是 Adelaide University 的 Research Fellow，研究目標包含從 correlation 走向 causality、representation learning 與 latent causal model；Javen Qinfeng Shi 是 University of Adelaide 教授，也是 Causal AI Group 的 Founding Director 與 Responsible AI Research Centre 的 Interim Director [5] [6]。這不是單純把 JEPA 套進另一個 benchmark，而是由具因果表示學習背景的研究團隊直接問「action 到底提供了多少 identifiability 訊號」。

## 核心問題：預測正確，為什麼仍可能學不到因果狀態？

World model 通常把影像 $x_t$ 編碼成低維表示 $z_t$，再學習

$$
\hat z_{t+1}=F_\phi(z_t,a_t)。
$$

如果預測誤差很低，我們很容易說模型學會了世界動力學。但低 prediction error 只代表表示中保留了「足以預測」的資訊，不代表每一個 learned coordinate 都對應到真實世界中的一個 state factor。模型可能把多個因子混成一個可逆但難以解釋的座標：

$$
\mathbf z=T(\mathbf s),
$$

其中 $T$ 是可逆的混合變換。只要 $T$ 沒有丟失資訊，joint-state decoder 仍可能以很高的 $R^2$ 還原整體狀態；可是單一座標不再對應單一因果因素，當 robot morphology、環境或 transition mechanism 改變時，這種混合表示不一定能穩定遷移 [2]。

論文把這個問題拆成兩層：

1. **Level I：information-loss collapse。** 表示把不同狀態壓成近似同一點，模型因此失去世界資訊。I-JEPA、VICReg、LeWM 的 SIGReg 或 entropy regularization 主要處理這一層。
2. **Level II：information-preserving representation mixing。** 表示仍保留大部分狀態資訊，卻用旋轉或其他可逆變換把多個 causal factors 混在一起。單純最大化 entropy 或讓表示接近 isotropic Gaussian，無法排除這種 ambiguity [2]。

A-JEPA 的主張是：**要解第二層問題，必須觀察 action 改變 transition mechanism 的方式。** Action 不是只用來幫 predictor 做條件化，它本身可以成為拆解 latent factors 的 identifying signal。

## 核心貢獻與創新點

- **提出 JEPA 的 information-theoretic objective。** 以 conditional likelihood 代表 transition predictiveness，再以 entropy maximization 代表 information preservation，將「預測」和「不丟資訊」放在同一個理論框架中 [2]。
- **給出 JEPA 內的 component-wise identifiability 結果。** 在 smooth invertible observation mapping 與足夠 action-induced variation 下，表示只剩 permutation 加上每一維的 invertible scalar transform，而不是任意的跨維度混合。
- **將抽象條件具體化為 A-JEPA。** 以 action-modulated Gaussian additive-noise transition 同時讓 conditional mean 與 variance 受 action 調制，再用 coordinate-wise Gaussian transition 與 minibatch negative 的 contrastive surrogate 實作。
- **用「定理—合成資料—視覺控制—跨 robot transfer」串成完整證據鏈。** Synthetic experiment 驗證 action 數量、$\alpha$ trade-off、observation invertibility 與 noise misspecification；visual benchmark 檢查物理 state factor recovery；RoboNet 則測試沒有 adaptation 的 held-out robot rollout。

## 技術方法與數學細節

### 1. Latent causal generative model

令真實 latent state 為 $\mathbf s_t\in\mathbb R^d$，觀測為高維影像 $\mathbf x_t$，action 為 $\mathbf a_t$。每一個 latent component 只依賴前一時間點的部分 parent set $\mathrm{pa}_i$：

$$
 s_{t,i}=f_i\!\left(\mathbf s_{t-1,\mathrm{pa}_i},\mathbf a_{t-1},\epsilon_{t,i}\right),
 \qquad i=1,\ldots,d,
$$

$$
\mathbf x_t=\mathbf g(\mathbf s_t)。
$$

在獨立 noise 的 temporal SCM 假設下，transition factorize 成

$$
 p(\mathbf s_t\mid\mathbf s_{t-1},\mathbf a_{t-1})
 =\prod_{i=1}^{d}p\!\left(s_{t,i}\mid \mathbf s_{t-1,\mathrm{pa}_i},\mathbf a_{t-1}\right)。
$$

這個設定的關鍵不是假設影像本身可直接分解，而是要求 action-conditioned transition 在 latent space 中具有可分析的因果結構 [2]。

### 2. 從 prediction 加上 information preservation

令 encoder 產生

$$
\mathbf z_t=\mathbf h(\mathbf x_t)，
$$

並令 $p_\phi(\mathbf z_t\mid\mathbf z_{t-1},\mathbf a_{t-1})$ 是 representation-space transition model。論文提出的一般目標為

$$
\mathcal L_\lambda(\mathbf h,\phi)
 =-\mathbb E\left[\log p_\phi(\mathbf z_t\mid\mathbf z_{t-1},\mathbf a_{t-1})\right]
 -\lambda H(\mathbf z_t)。
$$

第一項要求 learned transition 能描述下一個 representation；第二項則鼓勵表示保留足夠資訊，而不是透過常數或過度壓縮解決 prediction。

論文的 Theorem 2.1 表示，在 $\lambda\ge 1$、global optimum 且能達到理論 lower bound 的理想條件下，learned representation 保留真實 latent transition 所需的 mutual information，且 learned transition 會等於 encoder 所誘導的 representation-space transition distribution。這仍然只是「保留 transition-relevant information」的結果，尚未排除 component mixing [2]。

### 3. Identifiability 的兩個條件

**Assumption 3.1：觀測映射可逆。** $\mathbf g$ 必須 smooth 且 invertible。若兩個不同 latent states 產生同一個 observation，任何方法都不可能從影像中精確恢復它們。

**Assumption 3.2：action-induced variation 足夠豐富。** 設

$$
q_i=\log p\!\left(s_{t,i}\mid \mathbf s_{t-1,\mathrm{pa}_i},\mathbf a_{t-1}\right)，
$$

$$
\mathbf w(\mathbf s_t,\mathbf s_{t-1},\mathbf a_{t-1})
=\left(
\frac{\partial q_1}{\partial s_{t,1}},\ldots,
\frac{\partial q_d}{\partial s_{t,d}},
\frac{\partial^2 q_1}{\partial s_{t,1}^2},\ldots,
\frac{\partial^2 q_d}{\partial s_{t,d}^2}
\right)^\top。
$$

對同一個 state，若存在 $2d+1$ 個 actions，使得

$$
\mathbf L=\left[
\mathbf w(\mathbf a^{(1)})-\mathbf w(\mathbf a^{(0)}),\ldots,
\mathbf w(\mathbf a^{(2d)})-\mathbf w(\mathbf a^{(0)})
\right]
$$

滿足 $\operatorname{rank}(\mathbf L)=2d$，不同 action 就會讓 transition mechanism 產生足以區分各 latent component 的獨特變化。直觀地說，action 不能只改變一個幾乎固定的方向；它必須在 conditional log-density 的一階與二階結構中提供足夠多樣的 variation。

### 4. Theorem 3.3：任意混合被縮減成逐維變換

在上述兩個條件下，若 encoder 是 Eq. (4) 的 global minimizer 且 $\lambda>1$，則 learned representation 與真實 latent state 的關係為

$$
\mathbf z_t=
\left(
 r_1(s_{t,\pi(1)}),\ldots,
 r_d(s_{t,\pi(d)})
\right)，
$$

其中 $\pi$ 是 permutation，而 $r_i$ 是一維 invertible transformation [2]。

這個結果的意義是：模型不需要恢復真實座標的絕對尺度，也不需要知道每一維的命名；但是它不能再任意地把 $s_1$ 與 $s_2$ 混合成同一個 coordinate。對 causal representation learning 而言，這比「整體 state 可被 decoder 還原」更強，也更接近可遷移的 world model。

### 5. A-JEPA：可訓練的 Gaussian instantiation

A-JEPA 用 action-modulated Gaussian additive-noise model 具體實作上述條件：

$$
 s_{t,i}
 =f_i\!\left(\mathbf s_{t-1,\mathrm{pa}_i},\mathbf a_{t-1}\right)
 +\sigma_i(\mathbf a_{t-1})\epsilon_{t,i},
 \qquad \epsilon_{t,i}\sim\mathcal N(0,1)。
$$

對 Gaussian conditional log-density，

$$
\frac{\partial q_i}{\partial s_{t,i}}
 =-\frac{s_{t,i}-f_i}{\sigma_i^2},
 \qquad
\frac{\partial^2 q_i}{\partial s_{t,i}^2}
 =-\frac{1}{\sigma_i^2}。
$$

因此 action 對 conditional mean $f_i$ 的改變會影響一階 derivative，對 variance $\sigma_i^2$ 的改變則同時影響一階與二階 derivative。A-JEPA 的實際 transition model 對 representation 也採用 factorized Gaussian：

$$
 p_\phi(\mathbf z_t\mid\mathbf z_{t-1},\mathbf a_{t-1})
 =\prod_{i=1}^{d}
 \mathcal N\!\left(
 z_{t,i};
 \mu_{\phi,i}(\mathbf z_{t-1,\mathrm{pa}_i},\mathbf a_{t-1}),
 \sigma_{\phi,i}^{2}(\mathbf a_{t-1})
 \right)。
$$

對應的 negative log-likelihood，省略與優化無關的常數，是

$$
\operatorname{NLL}_\phi
 =\frac12\sum_{i=1}^{d}
 \left[
 \frac{\left(z_{t,i}-\mu_{\phi,i}\right)^2}{\sigma_{\phi,i}^{2}}
 +\log\sigma_{\phi,i}^{2}
 \right]。
$$

高維 differential entropy 不容易直接估計，所以論文以 minibatch 中其他 transition 的 future representations 作為 negatives，使用下列 practical objective：

$$
\mathcal L
=\mathbb E\left[
 \alpha\,\operatorname{NLL}_\phi(\mathbf z_t\mid\mathbf z_{t-1},\mathbf a_{t-1})
 +(1-\alpha)
 \log\frac1K\sum_{k=1}^{K}
 \exp\left(
 -\frac{\operatorname{NLL}_\phi(\mathbf z_t^{(k)}\mid\mathbf z_{t-1},\mathbf a_{t-1})}{\tau}
 \right)
\right]。
$$

其中 $\alpha$ 控制 prediction 與 information preservation 的權重，$\tau$ 是 temperature。第一項讓正確 future 的 NLL 變小；第二項避免同一個 transition model 對一批錯誤 future 也給出同樣高的 likelihood [2]。

在 visual benchmark 中，encoder 是四個 channel width 為 $32,64,128,256$ 的 convolution blocks，加上 global average pooling 與兩層 MLP；action 先經過 action encoder。A-JEPA 使用 predefined latent order，讓第 $i$ 個 coordinate 的 conditional mean 依賴前面 coordinate 與 action，variance 則由另一個 action-conditioned head 預測。這個 order 是對應到理論中 permutation indeterminacy 的實作選擇，而不是額外聲稱真實因果順序已知 [2]。

## 實驗結果與性能指標

### Synthetic：action variation 是可觀察的識別訊號

Synthetic 系統直接按照 Gaussian ANM 產生，並以 ground-truth latent states 與 temporal DAG 作為評估標準。主要指標是：

- **MCC（component-wise Spearman mean correlation coefficient）**：計算 learned 與 true latent coordinates 的 absolute Spearman correlation，再用 Hungarian matching 對齊。
- **Graph F1 / precision / recall / SHD**：先按照相同 coordinate matching 對齊，再評估 temporal DAG recovery。

合成訓練使用 80% transitions、20% test split，常規設定為 $\alpha=0.51$、$\tau=1.0$、Adam learning rate $10^{-4}$、weight decay $10^{-5}$、batch size 512、200 epochs [2]。

#### $\alpha$ 的 trade-off 不是裝飾性超參數

當 information-preservation 權重太低時，模型容易落入 collapse；當 prediction likelihood 權重太高時，表示又可能只保留容易預測的混合訊號。圖中 $\alpha\approx0.51$ 的 stable 區間達到最高 MCC，約為 $0.95$；$\alpha<0.50$ 時 MCC 約停留在 $0.2$ 左右 [2]。

![A-JEPA 在 transition predictiveness 與 information preservation 之間的 $\alpha$ trade-off。圖片取自論文 Figure 1(a)。][alpha-figure]

#### action settings 從 5 增加到 50，MCC 從 0.8566 升到 0.9588

![A-JEPA 的 action-induced variation 實驗；episode/action settings 增加時，component-wise MCC 上升並趨於飽和。圖片取自論文 Figure 1(b)。][action-figure]

論文 Figure 1(b) 顯示，action-setting/episode number 從 5 增至 50 時，MCC 由 $0.8566\pm0.0185$ 上升至 $0.9588\pm0.0019$；中間在 10、20、30、40 個 settings 的結果約為 $0.9457$、$0.9415$、$0.9545$、$0.9577$。這不是「action 越多，資料量越大」的普通 scaling claim，而是直接對應 Assumption 3.2：transition mechanism 被不同 action 激發得越豐富，latent components 越容易被區分 [2]。

#### Assumption violation 與 graph recovery

- 逐步移除 observation coordinates，MCC 從 $0.9584$ 降到 $0.5312$，符合 observation mapping 不再接近 invertible 時 recovery 變難的預期。
- 只讓 action 調制 conditional mean 或只調制 variance，都會降低 latent recovery；mean 與 variance 的雙重 variation 比單一路徑更有辨識力。
- 最佳 parent-gate sparsity 區間的 graph edge F1 約為 $0.70$；latent MCC 約維持在 $0.96$，但維度升高後 SHD 增加，表示「恢復每個 factor」比「完整恢復 DAG」容易。
- 將 Gaussian noise 換成 Laplace 或 Student-$t$ 時仍相對穩定；Uniform noise 造成較明顯退化。這支持方法對中等程度 model misspecification 有韌性，但不等於定理對所有 noise family 都成立 [2]。

### Visual control：高 joint-state $R^2$ 不等於 component-wise recovery

A-JEPA 與 LeWM、PLDM 使用相同 compact CNN、latent dimension、frame skip、optimizer 與訓練 budget，評估四個 visual environments：TwoRoom、OGBench Cube、Reacher、PushT。作者同時報告 MCC、nonlinear state $R^2$ 與 recursive rollout $R^2$，以避免只看一種指標 [2]。

最有啟發性的對照是 OGBench Cube：PLDM 的 nonlinear state $R^2$ 可以達到 $0.989$，但 MCC 只有 $0.485$。也就是說，joint decoder 幾乎能還原整體 state，不代表 representation coordinates 已經對齊到可解釋的物理 factors。A-JEPA 的優勢正是出現在這個更嚴格的 component-wise metric；同時它仍保留 state information 與 action-conditioned predictive dynamics。

在 dimension-selection run 中，論文報告：

- Reacher：$d_z=6$ 時 MCC $=0.926$、$R^2_{h=5}=0.973$。
- PushT：$d_z=10$ 時 MCC $=0.620$、$R^2_{h=5}=0.919$。
- OGBench Cube：$d_z=14$ 時 MCC $=0.865$、$R^2_{h=5}=0.999$；$h=10$ 的 rollout $R^2=0.9940\pm0.0051$。

這些數字來自 latent dimension sweep 與三個 random seeds 的 visual protocol，不能直接解讀成「A-JEPA 在所有 control task 都達到 state-of-the-art」；比較穩妥的結論是：它在保持 joint information 的同時，讓 coordinate-wise recovery 更可靠 [2]。

### RoboNet：未見 robot、無 adaptation 的長期 rollout

RoboNet 實驗在 Baxter、Kuka、WidowX 上訓練，測試時換成完全未見過的 Franka 與 Sawyer，不做 adaptation。資料包含 9,300 個 transition samples；A-JEPA 使用 latent dimension 8，訓練 5 epochs，batch size 64，三個 random seeds。測試時固定所有參數，從 held-out robot 的 encoded state 出發，沿著真實後續 action sequence recursive rollout 20 步 [2]。

![RoboNet held-out robot transfer：A-JEPA 在 Franka 與 Sawyer 上的 latent rollout $R^2$ 隨 horizon 保持穩定。圖片取自論文 Figure 5。][robonet-figure]

在 $h=20$：

| Held-out robot | A-JEPA | LeWM | PLDM |
|---|---:|---:|---:|
| Franka | $0.9099\pm0.0167$ | $0.0119\pm0.1864$ | $-8.7251\pm6.7523$ |
| Sawyer | $0.9028\pm0.0159$ | $0.5155\pm0.0351$ | $-13.3202\pm8.2978$ |

A-JEPA 的 transfer 結果很強，但必須精確描述：這是 **held-out robot 的 no-adaptation latent rollout transfer**，不是 zero-shot image generation，也不是「完全沒有訓練」。它支持 action-conditioned dynamics 跨 morphology 與 visual appearance 變化較穩定，卻不能單獨證明模型已經在真實世界中做出無條件 causal discovery [2]。

## 相關研究與差異

### LeJEPA：從 anti-collapse 與 Gaussian regularization 走向 identifiability

LeJEPA 論證在 stationary、additive-noise 的特定世界中，alignment 加 Gaussian regularization 可以讓 learned representation 線性恢復世界 latent；其核心保證是 linear/orthogonal identifiability [7]。A-JEPA 的差異在於，它想處理更直接的 **component-wise** recovery：允許每一維做任意一維 invertible transform，但不允許 arbitrary orthogonal mixing。它把辨識訊號從 marginal Gaussian geometry 推到 action 改變 transition mechanism 的方式。

A-JEPA 的 discussion 也把 LeWM 視為其一般 objective 的受限版本：在 fixed isotropic covariance 下，Gaussian NLL 會退化成 MSE；SIGReg 則可視為在固定 mean/covariance 下逼近 maximum-entropy Gaussian。然而 isotropic Gaussian 對 orthogonal rotation 不敏感，所以它能避免 Level-I information-loss collapse，卻不必然消除 Level-II representation mixing [2]。

### C-JEPA：object-level masking 的因果 inductive bias

C-JEPA 把 masked joint-embedding prediction 從 image patches 推到 object-centric latents，透過遮蔽物件迫使模型利用 surrounding context 推理 interaction-dependent dynamics；其重點是控制 observability、避免 shortcut，並報告 counterfactual reasoning 與 planning efficiency 改善 [8]。

兩者的分工很清楚：C-JEPA 用 **object-level masking** 施加結構性 inductive bias；A-JEPA 用 **action-induced transition variation** 建立可識別性理論。C-JEPA 的 masking 不等同於 A-JEPA 的 theorem condition，而 A-JEPA 也沒有主張 object-centric representation 已被完整恢復。未來可以測試：對 C-JEPA 的 object latent 再加入 action-modulated mean/variance transition，是否能把 heuristic interaction bias 推進到 component-wise identifiability。

### V-JEPA 2：大規模 video pretraining 與 zero-shot robot planning

V-JEPA 2 先用超大規模 internet video 做 action-free pretraining，再以少量 robot video post-train V-JEPA 2-AC，最後在兩個 Franka 實驗室環境中做不收集當地 robot data 的 planning [9]。它展示的是「大規模視覺預訓練如何轉成可用 world model」；A-JEPA 關心的是「representation coordinates 何時具備因果可識別性」。

因此 A-JEPA 本身不是 zero-shot robot planning 方法。即使 RoboNet 使用 held-out robot without adaptation，也應稱為 **no-adaptation transfer evaluation**，不要把它誤寫成 zero-shot generation 或 zero-shot world-model deployment。

### 與本 repo 已有方向的關係

本 repo 已有 EBT、LeJEPA、Causal-JEPA、AD-WM、Physically Grounded JEPA、VAR、training-free attention modulation 等文章。A-JEPA 的新增價值不是再做一個「JEPA benchmark comparison」，而是把以下缺口寫成可驗證命題：

> **預測得準** 解決不了 coordinate mixing；**action-induced variation** 才可能把可逆混合縮小成 permutation 加 component-wise transform。

它也不是 Energy-Based Transformer、VAR、flow matching 或 inference-only attention modulation。這些方向可以由 A-JEPA 的 transition likelihood、action disagreement 與 identifiability signal 延伸出新方法，但不應當成原論文貢獻。

## 個人評價與可延伸研究想法

### 我認為最重要的 insight

這篇論文真正有價值的地方，不是把 loss 從 MSE 換成 Gaussian NLL，而是重新定義 action 的角色。很多 world model 把 action 當作 predictor 的另一個 input；A-JEPA 則把 action 視為一組可用來「照亮不同 causal mechanisms」的 probes。這讓它與一般 anti-collapse 技巧產生清楚的理論分工：entropy 解決「有沒有保存資訊」，action variation 解決「保存的資訊是否按照 factor 組織」。

### 主要限制

1. **Theorem 依賴強假設。** $\mathbf g$ smooth 且 invertible、存在滿 rank 的 action-induced variation、objective 達到 global optimum，這些條件在真實影像與 robot data 上很難直接驗證。
2. **理論 objective 與實作 surrogate 不完全相同。** 理論使用 differential entropy；實作使用 minibatch negative 的 contrastive surrogate，再加 predefined latent order、有限 epochs 與 Gaussian transition。實驗支持關聯，但不等於有限 optimization 已經達到 Theorem 3.3 的 global optimum。
3. **Visual baselines 並非同一個 learning objective。** A-JEPA、LeWM、PLDM 雖然使用 matched encoder 與訓練設定，但 transition structure 與 regularizer 不同；MCC、nonlinear state $R^2$、rollout $R^2$ 衡量的也不是同一件事。
4. **Transfer 仍是 latent prediction。** RoboNet 的 $R^2$ 很有說服力，但沒有直接測試 downstream planning success、counterfactual intervention 或真實控制閉環成功率。
5. **它不是圖像生成論文。** 對使用者近期關注的 image generation、VAR、flow matching 與 training-free inference，A-JEPA 提供的是表示與識別的理論零件，而非可直接替換 diffusion/VAR sampler 的方法。

### 可以激發後續研究的四條路徑

- **Energy-Gated JEPA。** 把 action-conditioned negative log-likelihood 視為 compatibility energy，定義不同 action 下的 energy curvature 或 transition disagreement；在 planner 中優先保留能最大幅度區分 causal factors 的 action，而不是只選 prediction error 最小的 action。
- **EBT × JEPA。** 讓 Energy-Based Transformer 的 iterative energy minimization 產生 latent proposal，再用 A-JEPA 的 action-induced variation 作為辨識正則，測試 energy descent 是否能同時提升 rollout 與 component-wise MCC。
- **VAR 的 scale-wise causal factorization。** 在 visual autoregressive model 中，將 next-scale latent 的 conditional distribution 寫成 action-conditioned factorized transition，觀察不同 scale 的 action-induced variation 是否能降低早期 token mixing；這會把 A-JEPA 的「按 coordinate 識別」推向「按 scale 與 token 識別」。
- **Training-free attention modulation 的 uncertainty proxy。** 不重新訓練 backbone 的前提下，用多個候選 action 或 latent intervention 的 transition energy disagreement，對 attention heads 做 inference-time modulation；嚴格 protocol 應同時報告 FLOPs、無 intervention baseline、held-out mechanism 與 no-adaptation transfer，避免把 training-free 寫成 zero-cost。

## 結論

A-JEPA 的核心訊息可以濃縮成一句話：**一個不混淆因果因素的 JEPA，需要的不只是更好的 predictor，而是能讓 transition mechanisms 產生足夠差異的 action。**

它目前仍是 arXiv v1 預印本，不能寫成頂會已接收論文；但就研究問題而言，它比單純再提升一點 rollout score 更值得追蹤。對 EBT、JEPA、VAR、training-free 與 attention modulation 的交叉研究，最值得帶走的不是 A-JEPA 的 Gaussian implementation，而是它提供了一個可操作的問題定義：**我們要如何證明 learned latent 的資訊不是只被保存，而是被組織成能在新 action、新 mechanism、新 robot 上維持意義的 coordinates？**

## References

[1]: https://arxiv.org/abs/2609.31161 "I Act Therefore I Am: When Is JEPA's Action-Conditioning Enough to Learn Causal Mechanisms? — arXiv abstract"
[2]: https://arxiv.org/html/2609.31161 "I Act Therefore I Am: When Is JEPA's Action-Conditioning Enough to Learn Causal Mechanisms? — arXiv HTML full text"
[3]: https://arxiv.org/pdf/2609.31161v1 "I Act Therefore I Am: When Is JEPA's Action-Conditioning Enough to Learn Causal Mechanisms? — arXiv PDF"
[4]: https://export.arxiv.org/api/query?id_list=2609.31161 "arXiv API record for arXiv:2609.31161"
[5]: https://researchers.adelaide.edu.au/profile/yuhang.liu01 "University of Adelaide profile — Dr Yuhang Liu"
[6]: https://researchers.adelaide.edu.au/profile/javen.shi "University of Adelaide profile — Prof Javen Qinfeng Shi"
[7]: https://arxiv.org/abs/2605.26379 "LeJEPA: Efficient & Scalable Video Pretraining without the Heuristics — arXiv abstract"
[8]: https://arxiv.org/abs/2602.11389 "C-JEPA: Learning World Models through Object-Level Latent Masking — arXiv abstract"
[9]: https://arxiv.org/abs/2506.09985 "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning — arXiv abstract"

[alpha-figure]: ../../../../asset/ajepa_alpha_tradeoff.png "A-JEPA alpha trade-off figure"
[action-figure]: ../../../../asset/ajepa_action_variability.png "A-JEPA action-induced variation figure"
[robonet-figure]: ../../../../asset/ajepa_robonet_transfer.png "A-JEPA RoboNet held-out robot transfer figure"
