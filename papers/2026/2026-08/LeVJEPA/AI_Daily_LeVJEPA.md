# AI Daily

## 今日精選：LeVJEPA——把 JEPA 的「反崩潰理論」搬進影片，並讓因果注意力成為編碼器本身的屬性

**研究日期：2026-08-29**　　**作者：Manus AI**

今天選出的論文是 **LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics**。它不是另一個圖像生成器，而是一篇針對**影片自監督表徵學習、視覺世界模型與串流推理**的基礎方法研究；之所以值得放在今日的 AI Daily，是因為它同時觸及你近期關注的三個核心問題：如何用更少的訓練成本學到可預測的 latent state、如何把時間因果性直接寫入 attention topology，以及如何用顯式分佈約束取代大量經驗性 training heuristics。 [1] [2]

> **一句話摘要：** LeVJEPA 將 LeJEPA 的 SIGReg 反崩潰目標延伸至影片，以單一 encoder 加 projector 訓練 global/local view invariance；搭配 95% uniform random token dropping 與 block-causal attention，模型在相同總 FLOPs 下於 ImageNet-1K 超越最強影片基線 7.6 個百分點，同時保留對動態內容的優勢。[1]

## 一、論文基本資訊

| 欄位 | 內容 |
|---|---|
| 論文標題 | **LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics** |
| 作者 | Lukas Kuhn、Lucas Maes、Giuseppe Serra、Quentin Le Lidec、Yann LeCun、Randall Balestriero、Florian Buettner |
| 研究單位 | German Cancer Research Center、German Cancer Consortium、Goethe University Frankfurt、Mila、Université de Montréal、Brown University、Courant Institute at NYU、Advanced Machine Intelligence (AMI Labs) |
| 發表狀態 | **arXiv preprint**，2026-08-27 提交；論文頁目前沒有列出已接受的頂會版本。[1] |
| 分類 | Computer Vision and Pattern Recognition；Artificial Intelligence |
| 論文連結 | [arXiv:2608.27395][1]；[HTML 全文][10] |
| 程式碼 | 論文頁目前未列出公開程式碼連結；以下結果均指論文報告數值。 |
| 選文理由 | 最新投稿、Yann LeCun 與 Randall Balestriero 參與、直接延伸 LeJEPA，並把 JEPA 與 block-causal streaming representation 結合。 |

### 為什麼不是今天選一篇圖像生成論文？

本次搜尋的候選中，部分 training-free VAR／diffusion 論文與儲存庫既有文章重疊，另一些雖然更靠近圖像生成，卻主要是局部控制或推理加速。LeVJEPA 則提供一個更底層、也更容易激發後續研究的視角：**把「可預測的視覺狀態」與「不崩潰的表示分佈」分開處理，再讓時間因果性由 encoder 原生承擔。** 這使它可以作為後續 Energy-based Transformer、VAR latent planning、training-free attention modulation 與 zero-shot world modeling 的共同表徵底座。這是一個有意識的取捨：它的 venue 尚未確認，且不是 image generation paper；但在研究新穎性、作者背景與對你目前研究方向的啟發性之間，整體價值最高。

## 二、核心貢獻與創新點

LeVJEPA 的第一個貢獻，是把 **LeJEPA** 的 collapse-free objective 從靜態影像移植到影片。傳統 BYOL、DINO、V-JEPA 類方法通常需要 EMA target encoder、stop-gradient、predictor 或 masked-token prediction 來避免 representation collapse；LeVJEPA 則讓同一個 encoder 處理 global 與 local views，再用 SIGReg 直接約束 embedding distribution，將 trainable architecture 簡化成 **一個影片 encoder 加一個 projector**。[1][2]

第二個貢獻，是把極高比例的 token dropping 從「為了完成 masked prediction 而設計的遮罩」改成「encoder 的稀疏觀測 augmentation」。預設丟棄 95% patch tokens，剩下的 token 仍須足以支撐 clip-level embedding。這個設計不只降低每次 forward 的成本，還迫使模型從稀疏、隨機分布的時空觀測中恢復穩定的語義表示；在 ImageNet probe 上，token dropping 反而帶來準確率提升。[1]

第三個貢獻，是在預訓練階段直接使用 **block-causal attention**：同一幀內可以雙向互看，跨幀則只能看目前與過去幀。因而每一幀的 representation 不需要未來資訊，也不必等到預訓練完成後再額外擬合 temporal model。這讓 encoder 天然適合串流輸入、增量更新與 autoregressive world model；值得注意的是，論文在 ImageNet attentive probing 上觀察到 block-causal 版本為 51.2%，反而略高於 bidirectional 版本的 50.7%。[1]

| 創新設計 | 取代的既有做法 | 直接效果 | 研究意義 |
|---|---|---|---|
| SIGReg + invariance | EMA teacher、stop-gradient、predictor | 單一 encoder 與 projector 即可訓練 | 以顯式分佈約束取代架構性反崩潰技巧 |
| 95% uniform random token dropping | 影片 masked prediction 的 tube mask | 降低 token 計算量，同時提高 appearance representation | 稀疏觀測本身成為時空 augmentation |
| Block-causal attention | 全雙向影片 encoder + 後置 temporal model | 表示天然尊重時間順序，可逐幀延伸 | 直接對接 streaming perception 與 autoregressive planning |
| 只監督 [CLS] | 額外 patch-level reconstruction／dense loss | patch token 仍自發出現語義組織 | 暗示 clip-level predictive objective 可誘發 dense structure |

## 三、技術方法：從 LeJEPA 到 LeVJEPA

### 3.1 Global/local views 與 latent 表示

給定一段 16 幀的影片 clip，方法建立一個完整的 global view \(x_0\)，以及 \(V\) 個經過空間裁切和 photometric augmentation 的 local views \(x_1,\ldots,x_V\)。所有 views 使用相同的 temporal window，但 local view 覆蓋較小的空間區域。對每個 view，影片 ViT encoder \(E_\theta\) 先輸出 [CLS] 表示，再經過小型 projector \(h_\phi\)：

$$
z_v = h_\phi\!\left(E_\theta(x_v)_{[\mathrm{CLS}] }\right)\in\mathbb{R}^{K},\qquad v\in\{0,1,\ldots,V\}.
$$

projector 的存在不是為了增加表徵能力，而是因為 encoder 最後的 layer normalization 會限制 [CLS] 表示的幾何形狀；SIGReg 需要在一個可以調整分佈的 embedding space 中作用。預訓練完成後 projector 被丟棄，下游任務只使用 encoder representation。[1]

### 3.2 Invariance loss：保持同一段影片的視覺語義一致

global view 被視為覆蓋範圍最大、未經 photometric alteration 的參考 view。LeVJEPA 讓每個 local embedding 接近 global embedding，使用 mean squared error：

$$
\mathcal{L}_{\mathrm{inv}}
=\frac{1}{V+1}\sum_{v=0}^{V}
\left\|z_0-z_v\right\|_2^2.
$$

與 BYOL／V-JEPA 類方法不同，這裡對 \(z_0\) 和 \(z_v\) **都傳遞梯度**，沒有 stop-gradient，也沒有另一個 EMA target network。單獨最小化這個式子當然會允許所有輸入被映射到同一常數向量；LeVJEPA 的關鍵在於下一個 SIGReg 項。

### 3.3 SIGReg：以隨機投影逼近各向同性高斯分佈

LeJEPA 的理論核心是：在 JEPA 的下游 predictive risk 分析中，各向同性 Gaussian embedding distribution 是一個理想目標；SIGReg（Sketched Isotropic Gaussian Regularization）把這個高維分佈約束轉化成多個一維 goodness-of-fit tests。[2]

每一步從單位球面均勻抽取 \(M\) 個方向 \(a_m\in\mathbb{S}^{K-1}\)，並把第 \(i\) 個 embedding 投影為 \(u_{i,m}=\langle z_i,a_m\rangle\)。如果整體 embedding 真的是 \(\mathcal{N}(0,I_K)\)，那麼每個方向上的投影都應該接近標準常態 \(\mathcal{N}(0,1)\)。LeVJEPA 以 empirical characteristic function 與 Epps–Pulley normality statistic 寫成：

$$
\mathcal{L}_{\mathrm{SIGReg}}
=\frac{1}{M}\sum_{m=1}^{M}
\int
\left|
\frac{1}{n}\sum_{i=1}^{n}e^{\,\mathrm{i}t\langle z_i,a_m\rangle}
-e^{-t^2/2}
\right|^2e^{-t^2/2}\,dt.
$$

其中第一項是 batch 在方向 \(a_m\) 上的 empirical characteristic function，第二項 \(e^{-t^2/2}\) 是標準 Gaussian 的 characteristic function；積分衡量兩者的差異。論文使用 1,024 個隨機方向，並以 17 個 \(t\)-knots 於 \([0,3]\) 上做 trapezoidal quadrature。[1]

Cramér–Wold theorem 提供這個做法的直覺：一個高維分佈若在所有一維投影上都符合標準 Gaussian，就可被視為符合對應的各向同性 Gaussian 結構。反過來，collapsed embedding 在某些方向上方差為零，投影後會變成接近 delta distribution，不可能同時通過 Gaussian normality test。因此 SIGReg 不是「希望模型不要 collapse」的經驗正則，而是把 collapse 轉成明確可測量的分佈偏差。

最後的 LeVJEPA 訓練目標為：

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{inv}}
+\lambda\mathcal{L}_{\mathrm{SIGReg}},
\qquad \lambda=0.02.
$$

\(\lambda\) 是整個目標唯一的 trade-off hyperparameter，論文在所有訓練中固定使用 0.02，沒有針對實驗另外調參。[1]

### 3.4 Token dropping 與 block-causal attention

影片以空間 patch size \(16\times16\) tokenization，預設 temporal extent \(\tau=1\)，即每個 patch token 對應單一幀；對每個 view 隨機丟棄比例 \(\rho=0.95\) 的 patch tokens，僅保留 token 序列進入 Transformer。這種做法在全域 view 與 local views 上都使用，且 [CLS] token 永不丟棄。[1]

attention mask 採 block-causal 結構。幀內 token 可互相注意，跨幀 token 只可注意當前幀與過去幀；[CLS] 可以讀取所有 token，但 patch tokens 不回讀 [CLS]。因此，第 \(t\) 幀的表徵可寫成：

$$
h_t=f_\theta(x_{\leq t}),
$$

而不是依賴完整序列 \(x_{1:T}\)。當新幀 \(x_{T+1}\) 到來時，理想情況下只需要增量處理新幀，而不必重算所有歷史幀。這一點把 causal representation 從「訓練後另加的 predictor」提前變成 encoder 的結構性保證。

![LeVJEPA Figure 1：global/local views 經過 95% token dropping，由共享 encoder 產生 embedding，再以 MSE + SIGReg 訓練。圖像取自論文 PDF 的 Figure 1，僅裁切方法示意圖。][11]

## 四、實驗結果與性能指標

### 4.1 主要比較：相同資料與相同 FLOPs

作者使用 K710（Kinetics-400/600/700，移除 validation overlap）的 class-balanced 20% subsample 做受控比較，並以 frozen attentive probing 評估 ImageNet-1K、Something-Something-v2（SSv2）及 Kinetics-400（K400）。這種設定不等同於網路規模的 V-JEPA 2 預訓練，但對方法本身的計算效率比較較為公平。 [1] [3]

| FLOP-matched ViT-B | ImageNet-1K top-1 | SSv2 top-1 | K400 top-1 |
|---|---:|---:|---:|
| VideoMAEv2 | 53.4% | **43.6%** | 37.4% |
| V-JEPA 2 | 51.6% | 42.5% | 40.7% |
| **LeVJEPA** | **61.0%** | 40.4% | **44.6%** |

在相同總 FLOPs 下，LeVJEPA 比最強 ImageNet 影片基線高 **7.6 個百分點**，比 V-JEPA 2 高 9.4 個百分點；在 K400 也最高。代價是 SSv2 仍落後 VideoMAEv2 3.2 個百分點，說明極端稀疏觀測對需要跨幀細粒度 motion correspondence 的任務仍可能不利。[1]

### 4.2 與 image pretraining 的比較

作者另以相同影片資料與相同總 FLOPs 訓練 DINOv2 的 frame-based baseline。DINOv2 在 ImageNet-1K 為 53.8%，SSv2 為 16.9%；LeVJEPA 為 50.7% 與 30.4%。這代表 LeVJEPA 在靜態外觀任務仍落後 image-pretrained encoder 3.1 個百分點，但在 motion-centric SSv2 幾乎達到其兩倍。 [1] [8]

這個結果的重要性不在於宣稱影片預訓練全面勝過圖像預訓練，而在於重新提出資料型態的 trade-off：如果影片能以足夠低的成本預訓練，影片提供的額外時間結構可能值得承擔少量 appearance transfer 損失。

### 4.3 Token dropping、attention topology 與硬體效率消融

| 消融設定 | ImageNet-1K top-1 | 觀察 |
|---|---:|---|
| 無 token dropping | 33.9% | 完整 token 計算反而較弱 |
| \(\rho=0.90\) | 47.4% | 已有大幅提升 |
| **\(\rho=0.95\)** | **47.6%** | 預設，計算量最低且準確率最高 |
| Uniform random dropping | 50.7% | 時空分布完整，保留可辨識內容 |
| Tube dropping | 39.6% | 固定遮住相同空間區域，形成跨幀盲區 |
| Bidirectional attention | 50.7% | 無因果限制 |
| **Block-causal attention** | **51.2%** | 無可測的精度懲罰，且支援串流 |

token dropping 的解釋需要小心：它在 ImageNet appearance probe 上呈現單調改善，但在短訓練 schedule 的 SSv2 上，drop ratio 超過 0.3 可能降低 motion accuracy；增加訓練時間可以部分彌補。因此「95% dropping 是普遍最佳」不是論文可以支持的結論，較準確的說法是：它在這個 objective 與 appearance-heavy probe 下最有效，motion task 仍需要更好的 correspondence-preserving sparse sampling。[1]

硬體實驗也具有可及性意義。ViT-Tiny 在單張 16 GB RTX 5080 上以 12 小時、約 620K frames 的 Walking Tours 無標註影片訓練，ImageNet accuracy 從初始化的 8.9% 提升至 25.2%；同一張卡上 LeVJEPA batch size 可達 128 且使用不到 8 GB，V-JEPA 配置則在 batch size 28 左右便飽和顯存。[1]

### 4.4 跨資料規模與自發 dense structure

在更大的組合資料上，ViT-L/16 以 K710、SSv2、Walking Tours 與 Perception Encoder video dataset 預訓練 100 epochs，在 frozen probing 上得到 ImageNet-1K 69.5%、SSv2 55.0%。這些數字不應直接與使用超過一百萬小時網路影片的 V-JEPA 2 比成「誰的 foundation model 更強」；它們更像是 scaling headroom 證據：即使使用遠小於 V-JEPA 2 的資料，objective 仍隨資料擴張而改善。 [1] [3]

另一個有趣現象是，訓練只監督 clip-level [CLS]，patch tokens 卻在 PCA 與 query-patch cosine similarity 視覺化中呈現物件—背景分離及局部語義組織。作者指出，這種 dense structure 並未透過額外 patch-level loss 強行指定；它可能源自 global/local invariance 與隨機稀疏觀測共同施加的資訊瓶頸。[1]

## 五、相關研究與定位

### 5.1 LeJEPA：理論上的直接前身

LeJEPA 由 Balestriero 與 LeCun 提出，主張 JEPAs 的 embedding distribution 應以各向同性 Gaussian 為目標，並以 SIGReg 取代 stop-gradient、teacher–student、scheduler 等常見 heuristic。其 abstract 報告在 10+ datasets 與 60+ architectures 上進行驗證，且 ViT-H/14 在 ImageNet-1K frozen linear evaluation 達 79%。[2]

LeVJEPA 的真正新意不是重新發明 SIGReg，而是回答「這套 collapse-free recipe 能否在時間維度更昂貴的影片上工作」：作者加入 view construction、稀疏 token observation、block-causal topology，並以 FLOP-matched protocol 驗證影片預訓練效率。這也是為何本文應被視為**影片 JEPA 的方法學延伸**，而非單純換資料集。

### 5.2 I-JEPA、V-JEPA 與 V-JEPA 2：從表徵到世界模型

I-JEPA 在 CVPR 2023 將 joint-embedding prediction 應用於影像，核心思想是預測抽象 representation 而非重建像素；V-JEPA 將此方向推進到影片，透過 latent video prediction 學習時間結構。 [5] [6] V-JEPA 2 則以超過一百萬小時網路影片預訓練，並在少量機器人影片上 post-train action-conditioned world model，展示理解、預測與 zero-shot robot planning。 [3] [4]

LeVJEPA 與 V-JEPA 2 的差異值得明確保留。V-JEPA 2 的重點是 web-scale foundation model 與 robot planning pipeline；LeVJEPA 的重點是用更簡潔的 objective 與稀疏計算，在受控資料上讓 encoder 本身變成 causal、streamable。前者較接近完整世界模型產品路線，後者較像可被其他 world model 或 VAR planner 重用的表徵訓練原語。

### 5.3 VideoMAE、DINOv2 與「影片是否值得」問題

VideoMAE／VideoMAEv2 以高比例時空遮罩與 pixel reconstruction 學習影片表示，tube mask 的必要性來自「避免模型從相鄰幀抄答案」。[7] LeVJEPA 不重建被遮住的內容，因此 uniform random dropping 反而優於 tube dropping：模型不是完成 imputation，而是在有限觀測下形成穩定 clip embedding。DINOv2 則是強大的 image-pretraining baseline；LeVJEPA 的 matched-compute 結果顯示，影片可以用較低成本接近 appearance transfer，同時保留明顯 motion advantage。[8]

## 六、我的評價：這篇論文真正改變的是研究問題的拆法

我認為 LeVJEPA 最有價值的地方，不是「95% token dropping」這個單一數字，而是它把影片表徵學習拆成三個可以獨立研究的軸：**表徵不崩潰由 SIGReg 負責、視覺語義一致性由 invariance loss 負責、時間順序由 attention topology 負責。** 在 V-JEPA 類模型中，這些功能往往分散在 target encoder、predictor、mask design 和 post-training temporal model 之間；LeVJEPA 讓每一個元件的責任更容易被測量。

但它並非沒有代價。第一，論文目前是 2026-08-27 的 arXiv preprint，尚未有頂會接收資訊，因此 61.0% ImageNet 或 5.6–20.8× compute saving 仍應視為作者自己的 controlled evaluation，而不是已經由外部 benchmark protocol 廣泛確認的 SOTA。第二，主要比較使用 K710 的 20% 子集與 frozen probes，和 V-JEPA 2 的百萬小時規模並不對稱。第三，95% dropping 對 motion-centric SSv2 在短 schedule 會造成損失，且 dense prediction、tracking、action-conditioned control 尚未充分驗證。第四，SIGReg 的理論 target 是 embedding distribution，而不是能直接代表物理世界 transition 的 energy landscape；它可以降低 collapse，卻不保證 latent dynamics 已經可規劃。

因此，我會把 LeVJEPA 定位成：**一個適合拿來建造下一層 world model、VAR latent planner 或 causal visual memory 的高效率 representation substrate，而不是現成的生成模型或機器人控制器。**

## 七、可激發後續研究的方向

### 7.1 Energy-based Transformer × LeVJEPA

可以把 LeVJEPA 的 latent embedding 當作 state，另外學習 transition energy：

$$
E_\psi(z_{t+1}\mid z_t,a_t)
=\left\|g_\psi(z_t,a_t)-z_{t+1}\right\|_2^2
+\beta\,\Omega_\psi(z_{t+1}),
$$

其中第一項度量 action-conditioned predictive compatibility，第二項可加入 SIGReg 或資料分佈的 energy prior。這樣可以測試一個清楚的假設：**SIGReg 負責讓 state space 不崩潰，energy model 負責讓 transition 可排序、可規劃。** 不應把兩者混為同一個 objective；更有價值的問題是研究它們在 scale、長期 rollout 和 out-of-distribution action 上是否互補。

### 7.2 LeVJEPA × VAR：讓多尺度 token 具備因果 latent

VAR 通常依 coarse-to-fine scale 生成 visual tokens。可以在各 scale 的 token state 上加入 LeVJEPA-style local/global predictive objective，讓 coarse prefix 對 fine-scale future embedding 形成 JEPA prediction；同時使用 block-causal mask，使 scale order 變成 encoder topology。若再以 SIGReg 約束每一個 scale 的 latent distribution，可能減少 VAR 早期 prefix collapse 與後期細節 drift，而不必在每一步引入額外 diffusion guidance。

### 7.3 Training-free attention modulation：以 predictive uncertainty 決定介入位置

LeVJEPA 的 causal embedding 可以提供每一幀的 state consistency score。對新觀測 \(x_t\)，可用其與歷史 predictive embedding 的 distance 或 projected-normality deviation 定義 uncertainty：

$$
\delta_t=\left\|z_t-\hat z_t\right\|_2^2,
\qquad
b_{ij}=b_{ij}^{(0)}-\eta\,\delta_i\,\mathbf{1}[j<t].
$$

在不更新 backbone 的條件下，將 \(b_{ij}\) 作為 attention logit modulation，只在 predictive disagreement 高的時間點放大或縮小歷史讀取範圍。這會把「何時應介入 attention」從固定 timestep heuristic 改成由 causal JEPA state 觸發，適合測試 training-free video editing、長影片生成與 memory routing。

### 7.4 Zero-shot world modeling 與規劃

V-JEPA 2 已展示 action-conditioned post-training 與 zero-shot robot planning 的方向；LeVJEPA 提供更便宜且原生 causal 的 encoder。下一步可比較：同一個 action-conditioned predictor 分別接在 bidirectional V-JEPA、LeVJEPA 和 image-pretrained DINOv2 上，並在 unseen embodiment、長 horizon、partial observability 及 action perturbation 下測試 planning regret。這比單純比較 ImageNet linear probe 更能回答 LeVJEPA 是否真的學到對世界模型有用的時間狀態。

## 八、結論

LeVJEPA 的核心訊息可以濃縮為：**不要用更多 heuristic 來掩蓋表示學習的基本問題；先用顯式分佈約束避免 collapse，再用稀疏觀測與因果 topology 讓每一個 token 都服務於可預測的視覺狀態。** 它目前尚未證明自己是最好的影片 foundation model，也還沒有完成從 representation 到 action-conditioned generation 的閉環；但它提出了一個簡潔、可拆解、可擴展的基線，特別適合拿來連接你關注的 Energy-based Transformer、JEPA、VAR、attention modulation 與 zero-shot planning。

對今日研究最值得記住的不是 95% 這個數字，而是以下研究問題：**如果 representation distribution、temporal causality 與 transition energy 可以被分別控制，我們是否能用一個更小、更可解釋的 causal visual substrate，取代部分巨大 world model 的訓練與推理成本？**

## References

[1] [LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics, arXiv:2608.27395][1]

[2] [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics, arXiv:2511.08544][2]

[3] [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning, arXiv:2506.09985][3]

[4] [Meta AI — Introducing V-JEPA 2][4]

[5] [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture, CVPR 2023][5]

[6] [V-JEPA: Latent Video Prediction for Visual Representation Learning][6]

[7] [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training, NeurIPS 2022][7]

[8] [DINOv2: Learning Robust Visual Features without Supervision, TMLR][8]

[9] [LeVJEPA Figure 1 source: arXiv PDF][9]

[10]: https://arxiv.org/html/2608.27395v1 "LeVJEPA HTML full text"
[11]: ../../../../asset/levjepa_figure1_method.png "Cropped LeVJEPA Figure 1 method diagram"

[1]: https://arxiv.org/abs/2608.27395 "LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics"
[2]: https://arxiv.org/abs/2511.08544 "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
[3]: https://arxiv.org/abs/2506.09985 "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
[4]: https://ai.meta.com/research/vjepa/ "Introducing V-JEPA 2"
[5]: https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html "I-JEPA CVPR 2023"
[6]: https://arxiv.org/abs/2404.08471 "V-JEPA: Latent Video Prediction for Visual Representation Learning"
[7]: https://arxiv.org/abs/2203.12602 "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
[8]: https://arxiv.org/abs/2304.07193 "DINOv2: Learning Robust Visual Features without Supervision"
[9]: https://arxiv.org/pdf/2608.27395 "LeVJEPA PDF source"
