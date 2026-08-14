# AI Daily

## V-RAE：以凍結視覺表徵重思影片生成的潛在空間

**研究日期：** 2026-08-14

> **一句話摘要：** V-RAE 不再把影片潛在空間只當作「重建像素的壓縮碼」，而是以凍結的視覺基礎模型特徵作為語義座標，透過可學習的時間注意力池化與 3D RoPE 影片解碼器，讓同一潛在空間同時服務重建、類別條件影片生成與未來預測；其 V-JEPA 2.1 變體在 Kinetics-600 取得 **rFVD 2.13**、**gFVD 19.16**，且作者報告在受控設定下可達最高 **6×** 更快收斂。[1]

| 欄位 | 資訊 |
|---|---|
| **論文** | *V-RAE: Rethinking Video Latent Spaces for Generation* |
| **作者** | Minghui Guo、Shengqiong Wu、Hao Fei |
| **研究單位** | National University of Singapore、University of Oxford |
| **發表狀態** | **arXiv 預印本 v1**；截至 2026-08-14，原始頁面未列正式會議／期刊錄用資訊。[1] |
| **發布日期** | 2026-08-13（UTC）[1] |
| **論文來源** | [arXiv:2608.13556](https://arxiv.org/abs/2608.13556)；[專案頁](https://v-rae.github.io/) |
| **關鍵詞** | Video Generation、Representation Autoencoder、V-JEPA、DiT、Rectified Flow、Temporal Attention、World Model |
| **AI Daily 選題理由** | 這是把 **JEPA／視覺表徵** 直接變成生成模型工作座標的近期嘗試；它把「語義、時間平滑度與可生成性」放到同一個 latent-space 問題中，對 JEPA、attention modulation、影片 VAR 與世界模型都有高遷移價值。 |

---

## 為何值得今天讀

現代影片生成通常先使用 VAE 或 tokenizer 把影片壓縮，再讓 diffusion 或自迴歸模型在該壓縮碼上建模。然而，若 autoencoder 的唯一目標是逐像素重建，它未必會留下對動作、物體身份、場景結構最友善的幾何；這正是 V-RAE 追問的核心問題。作者主張，**重建最佳的潛在碼，不必然是最容易生成或預測的潛在碼**。[1]

這個問題與近來的 Representation Autoencoder（RAE）脈絡一致：RAE 使用凍結的視覺表徵編碼器和可訓練解碼器，直接在高維語義特徵中做生成。它已在文字生圖擴展中顯示相對 VAE 更快、更穩的訓練，但影片面臨額外的時間冗餘、動態一致性與長序列建模難題。[4] V-RAE 的貢獻不是把圖像 RAE 直接逐幀套用，而是明確把**時間壓縮、時間幾何與時空解碼**納入設計與評估。[1]

---

## 核心貢獻與創新點

| 貢獻 | 具體內容 | 為何重要 |
|---|---|---|
| **語義型影片潛在空間** | 以凍結 DINOv3、SigLIP2、EUPE 或 V-JEPA 2.1 編碼器特徵，而非傳統 VAE latent，作為影片生成器的直接目標。 | 讓生成器從已具語義結構的座標開始學習，而非同時重新發現語義與動態。 |
| **時間注意力池化** | 對固定空間位置的相鄰時間特徵做 content-adaptive pooling；初始化時退化為均值池化，訓練後學得各時間片的重要性。 | 在四倍時間壓縮下移除冗餘，同時保住動作與物體語義。[1] |
| **時空解碼器** | 解碼端採 3D RoPE；圖像 encoder 版本使用 chunk-wise causal attention，V-JEPA 2.1 版本則使用全時空 attention。 | 將語義特徵重新渲染為連續動作，避免逐幀 RAE 常見的 flicker。 |
| **tFVD 診斷** | 不只測 encode–decode 重建，而是以相鄰 latent 的中點破壞原始路徑後解碼，量測時間插值是否仍自然。 | 直接測試預測／生成時最重要的「偏離真實 latent 軌跡後，decoder 是否仍穩定」。 |
| **受控下游驗證** | 在相同 token 預算、DiT 訓練與採樣設定下，比較多種 tokenizer 的類別條件生成；另固定未來預測 DiT，只替換 latent space。 | 使「latent 的貢獻」比僅比較不同大型系統更容易歸因。[1] |

---

## 技術方法：由視覺表徵到可生成的影片狀態

### 1. 凍結表徵、可訓練壓縮與解碼

令輸入影片為 $\mathbf{X}\in\mathbb{R}^{T\times3\times H\times W}$。凍結的視覺基礎模型編碼器 $\mathcal{E}$ 產生稠密特徵：

$$
\mathbf{F}=\mathcal{E}(\mathbf{X})\in\mathbb{R}^{T_E\times C\times h\times w}.
$$

圖像模型（DINOv3、SigLIP2、EUPE）沒有內建時間下採樣，故 $r_E=T/T_E=1$；V-JEPA 2.1 的 tubelet 設定則有 $r_E=2$。接著 temporal pooler $\mathcal{P}$ 以額外壓縮率 $r_P$ 聚合相鄰特徵，產生生成器實際建模的 latent：

$$
\mathbf{Z}=\mathcal{P}(\mathbf{F})=[\mathbf{z}_0,\ldots,\mathbf{z}_{T_Z-1}],
\qquad T_Z=\frac{T_E}{r_P},
\qquad r_{\mathrm{all}}=r_Er_P=\frac{T}{T_Z}.
$$

作者令所有主實驗達到 $4\times$ 的總時間壓縮。關鍵在於 pooling **不跨空間位置混合**：每個位置 $p$ 僅在自己的時間窗口中決定該保留哪個動態訊號。這保留了空間 token 的對齊性，亦避免把時間濾波誤當成空間物體融合。[1]

### 2. 時間注意力池化：可學的「保留什麼、丟棄什麼」

對某一時間窗口的第 $i$ 個 encoder token $\mathbf f_i$，第 $m$ 個 head 先投影 key/value：

$$
\mathbf{k}_{i}^{(m)}=\mathbf W_K^{(m)}\mathbf f_i,
\qquad
\mathbf{v}_{i}^{(m)}=\mathbf W_V^{(m)}\mathbf f_i.
$$

共享、可學習的時間 query $\mathbf q^{(m)}$ 與局部相對位移 bias $\beta_i^{(m)}$ 產生注意力權重：

$$
\alpha_i^{(m)}=
\operatorname{Softmax}_i\!\left(
\frac{(\mathbf q^{(m)})^\top\mathbf k_i^{(m)}}{\sqrt d}+\beta_i^{(m)}
\right),
\qquad
\mathbf u^{(m)}=\sum_{i=0}^{r_P-1}\alpha_i^{(m)}\mathbf v_i^{(m)}.
$$

最後將 heads 串接投影並正規化為 $\mathbf z_{t,p}$。因為 query 與 bias 以零初始化，初始時 $\alpha_i^{(m)}=1/r_P$，模型先是安全的平均池化，再漸進學會依內容調制時間權重。[1] 這提供了一個很直接的 attention-modulation 範例：不是加更多 token，而是讓每一個空間位置決定哪一段動態最有資訊。

### 3. 影片解碼與重建目標

解碼器 $\mathcal D$ 是配備 3D RoPE 的輕量 Transformer，將壓縮後的 $\mathbf Z$ 還原成 $\widehat{\mathbf X}=\mathcal D(\mathbf Z)$。圖像 encoder 的版本採 **chunk-wise causal attention**：同一時間 chunk 內可雙向互動，後續 chunk 只能看自己與歷史；V-JEPA 2.1 特徵本身來自非因果全時空 context，對應 decoder 亦採 full attention。每個 latent time step 會一次 unpatchify 成 $r_{\mathrm{all}}$ 張連續影格。[1]

僅訓練 pooler 與 decoder，整體重建目標是：

$$
\mathcal L_{\mathrm{recon}}
=\lambda_1\mathcal L_1
+\lambda_{\mathrm{lpips}}\mathcal L_{\mathrm{LPIPS}}
+\lambda_{\mathrm{gan}}\mathcal L_{\mathrm{GAN}}
+\lambda_{\mathrm{gram}}\mathcal L_{\mathrm{Gram}}.
$$

這個選擇很務實：凍結 encoder 使其原有語義幾何不被「為了重建像素而改寫」，而 L1、感知、對抗與 Gram losses 則把語義 feature 接回可視像素域。[1]

### 4. 在高維影片表徵上做 Rectified Flow

凍結 V-RAE 後，作者以 DiT 在 $\mathbf Z$ 上訓練 rectified flow。由於有效維度隨時間長度、空間 token 與通道數一起增大，先把 $\tau\sim\operatorname{LogitNormal}(0,1)$ 轉為維度感知的時間：

$$
\widehat\tau=\frac{s\tau}{1+(s-1)\tau},
\qquad
s=\sqrt{\frac{T_ZhwC}{n}},\quad n=4096,
$$

並建立噪聲路徑 $\mathbf Z_{\widehat\tau}=(1-\widehat\tau)\mathbf Z+\widehat\tau\boldsymbol\epsilon$。DiT 直接預測 clean latent $\widehat{\mathbf Z}_\theta$，再轉成速度：

$$
\widehat{\mathbf v}_\theta=
\frac{\mathbf Z_{\widehat\tau}-\widehat{\mathbf Z}_\theta}
{\max(\widehat\tau,t_\epsilon)},
\qquad t_\epsilon=0.05.
$$

作者同時在第 8 個 Transformer block 放一個 auxiliary clean-latent head，以雙頭 rectified-flow loss 訓練，並在採樣時保留該支路作 internal guidance。[1] 其精神與大尺度 RAE 的經驗一致：高維語義 latent 的**噪聲排程**需要隨有效維度調整；否則生成器面對的 SNR 幾何會失衡。[4]

### 5. tFVD：不要只問「能否重建」，要問「偏一點還能否解碼」

對時間有序 latent $\mathbf Z=[\mathbf z_0,\ldots,\mathbf z_{L+1}]$，tFVD 以兩側鄰點中值取代每個內部點：

$$
\mathbf z_t'=\frac{1}{2}(\mathbf z_{t-1}+\mathbf z_{t+1}),
\qquad t=1,\ldots,L.
$$

將被擾動序列 $\mathbf Z'$ 解碼，並與真實影片分佈做 FVD：

$$
\operatorname{tFVD}=
\operatorname{FVD}\bigl(\{\mathbf X\},\{\mathcal D(\mathbf Z')\}\bigr).
$$

因此 tFVD 並非完整 generator 評測，而是對 **latent 軌跡局部幾何與 decoder 穩定性**的 stress test。若中點已脫離能解碼成自然動作的流形，便會產生 ghosting、flicker 或突變；較低 tFVD 表示模型較能容忍未來預測或去噪過程中不可避免的小偏差。[1]

---

## 實驗結果與性能指標

### 影片重建與語義保留

V-RAE 使用 UCF101 與 Kinetics-600（K600）評估重建；所有版本都採 $4\times$ 時間壓縮。K600 上的 V-JEPA 2.1-L 變體達到 **rFVD 2.13**，低於作者比較的最佳大型影片 VAE（Wan2.1 VAE，3.58）。值得注意的是，V-RAE 的 LPIPS／PSNR／SSIM 未必優於專為低階重建優化的 VAE；這正支持作者的論點：逐幀失真不能完全代表時空分佈品質。[1]

| Tokenizer／latent | UCF101 rFVD $\downarrow$ | K600 rFVD $\downarrow$ | 解讀 |
|---|---:|---:|---|
| Wan2.1 VAE | **6.05** | 3.58 | UCF101 的強重建基線。 |
| Wan2.2 VAE | 12.20 | 4.76 | 大型 VAE 對照。 |
| V-RAE（DINOv3-L） | 6.12 | 2.76 | UCF101 幾乎追平最佳 VAE，K600 則優於所有 VAE 對照。 |
| V-RAE（EUPE-B） | 8.05 | 3.36 | 顯示優勢不依賴單一 encoder。 |
| V-RAE（V-JEPA 2.1-L） | 6.65 | **2.13** | K600 最佳；相對 3.58 約低 **40.5%**。 |

在語義 probe 中，$4\times$ 壓縮的 V-RAE latent 仍大幅保留凍結 encoder 的辨識能力。最好的 V-RAE 結果分別為 UCF101 **90.92%**、Something-Something V2 **72.91%**、Kinetics-400 **83.12%**；作者所列最強 VAE token baseline 對應為 30.83%、45.05%、53.27%。這不表示 V-RAE 已在所有視覺任務取代 VFM，而是證明「可解碼、可生成」不必以完全犧牲可線性讀取的語義為代價。[1] [2]

### 受控類別條件生成：相同 token budget 下的 latent 比較

下表全部以 **1,280 tokens**、$256\times256$ 解析度與一致的 100-step Euler 採樣協定比較；$20\to17$ 表示先生成 20 幀後裁成 17 幀評估。因此表格支持的是**在作者設定下，latent space 使 DiT 更容易學**，而非宣稱已擊敗所有開放式文字生影片系統。[1]

| Latent space | UCF101 gFVD $\downarrow$ | K600 gFVD $\downarrow$ | 相對最強非 V-RAE 對照的改善 |
|---|---:|---:|---|
| Wan2.1 VAE | 148.20 | 53.75 | 對照。 |
| Cosmos VAE | 152.70 | 41.66 | K600 上最佳非 V-RAE 結果。 |
| AToken | **143.00** | 46.74 | UCF101 上最佳非 V-RAE 結果。 |
| V-RAE（DINOv3-L） | 131.40 | 30.09 | 跨資料集均優於 VAE tokenizer。 |
| V-RAE（EUPE-B） | 125.98 | 24.77 | 跨 encoder 仍有穩定優勢。 |
| V-RAE（V-JEPA 2.1-L） | **117.86** | **19.16** | 相對 143.00 / 41.66 分別降低 **25.14 / 22.50**。 |

收斂曲線同樣值得注意。作者報告在 UCF101，V-JEPA 2.1 變體約 30K updates 即達 Wan2.2 VAE 150K updates 的 gFVD，約 **5×**；在 K600，EUPE-B 約 30K updates 比上 Wan2.2 VAE 180K updates，約 **6×**。這是「預訓練語義將學習問題重新參數化」的直接實驗訊號，而不只是末端分數改善。[1]

### tFVD 與未來影片預測：最反直覺但最關鍵的結果

作者在 UCF101／K600 中比較 rFVD 和 tFVD 對下游 gFVD 的 Pearson 相關：rFVD 分別只有 **0.200 / 0.473**，tFVD 則升為 **0.621 / 0.919**。這不是 tFVD 已成為通用標準的證明，但它強烈提醒：若目標是 generation 或 world-model rollout，測試「encoder 輸出的點能不能重建」太容易；真正應測試的是「模型可能走到附近時，decoder 還能不能穩定」。[1]

在 Cityscapes 未來預測，兩者共用 conditional DiT、token budget、訓練排程與採樣設定，僅替換 latent space。V-RAE（EUPE-B）雖有**較差 rFVD**，卻有較低 tFVD、gFID 與 gFVD：

| Latent space | rFVD $\downarrow$ | tFVD $\downarrow$ | gFID $\downarrow$ | gFVD $\downarrow$ |
|---|---:|---:|---:|---:|
| Wan2.2 VAE | **7.0256** | 319.0233 | 15.02 | 144.47 |
| V-RAE（EUPE-B） | 29.2931 | **224.6040** | **11.52** | **111.36** |

這張表是全文最有價值的反例：**更好 reconstruction metric 不等於更好 future-state metric**。若把 latent 看作世界模型的狀態，預測器需要的是平滑、可外插且仍可 render 的狀態流形，而不是只在觀測點附近達到最小重建誤差。[1]

---

## 相關研究背景與定位

| 脈絡 | 代表研究 | V-RAE 的差異與連結 |
|---|---|---|
| **JEPA／自監督影片表徵** | V-JEPA 2.1 以 dense predictive loss、深層自監督與圖像—影片共同 tokenizer 學得稠密、時空一致的特徵。[3] | V-RAE 不重訓 JEPA predictor；它詢問的是：這些為理解與預測學出的特徵，能否成為可生成、可解碼的 latent。 |
| **RAE 與高維語義 latent diffusion** | Scale-RAE 在文字生圖中發現 frozen representation encoder + trained decoder 能較 VAE 快速收斂；維度感知 noise shift 仍是關鍵。[4] | V-RAE 將該想法推到影片，增加時間池化、3D RoPE 與「時間插值幾何」測試；其噪聲 shift 也顯式納入 $T_Z$。 |
| **以 foundation model 指導影片 diffusion** | VideoREPA 透過 Token Relation Distillation，把 VFM 的時空關係作為 T2V finetuning 的軟監督，提升物理合理性；已收錄 NeurIPS 2025。[5] | VideoREPA 在**generator 中間特徵**加入對齊 loss；V-RAE 則在**輸入 latent space**層級換座標。兩者可視為互補：前者校正生成動力學，後者重整生成所需的狀態空間。 |
| **傳統影片 VAE／tokenizer** | Wan、HunyuanVideo、CogVideoX、Cosmos 等潛在空間優先追求高效、低階視覺重建與壓縮。[1] | V-RAE 的實驗並不否定 VAE；它量化了為何「重建導向」評估與「生成／預測導向」評估會出現反向排序。 |

由於 V-RAE 僅在 2026-08-13 公開，尚不應用引用量或「後續影響」判斷其成敗。更合理的評估方式是檢查其與既有 RAE／JEPA 主張的連續性、實驗控制是否支持 latent-space 歸因，以及未覆蓋的開放式 setting。[1] [3] [4]

---

## 個人評價、限制與可延伸的研究想法

**評價。** 這篇論文最成熟的部分不只是把 V-JEPA 特徵送進 decoder，而是建立了可被反駁的判準：如果語義 latent 真能幫助生成，則它應在固定 token budget 與 DiT 下改善 gFVD；如果它真更像可預測狀態，則它應在固定預測器下減少未來 rollout drift。兩組實驗均提供正向證據。[1] 而 tFVD 的價值在於它把「模型能否處理 off-manifold 誤差」這個常被忽略的生成問題，變成便宜的 autoencoder-level 診斷。

**限制。** 結果目前仍是 arXiv v1，且核心生成實驗為 UCF101/K600 的類別條件 $256\times256$ 短影片，不能外推為其已優於開放域文字生影片或長時間自迴歸系統。不同 V-RAE encoder 也採不同 decoder attention（V-JEPA 為 full attention），使跨 encoder 的性能差異不完全只由 encoder 預訓練決定。最後，tFVD 僅檢驗二階鄰點中值；它很有診斷性，但尚未證明可以預測大幅度分佈外推、相機控制或多物體物理交互的品質。[1]

| 與使用者關注方向的連結 | 可操作的研究假說 |
|---|---|
| **JEPA／世界模型** | 將 V-JEPA 的 dense predictive feature 當成「可 render 的 state」後，可比較 JEPA predictor 的 latent rollout 與 DiT／flow predictor 的 rollout；假說是較低 tFVD 的 state space 會降低長期 identity drift。 |
| **Energy-based Transformer** | tFVD 可被重新解讀為局部 energy smoothness 的 proxy：若相鄰狀態的中點有高能量、解碼就崩壞。可在 Transformer latent 上直接估計 $E((z_{t-1}+z_{t+1})/2)-\tfrac12[E(z_{t-1})+E(z_{t+1})]$，檢查它與 tFVD／rollout 失敗的關聯。 |
| **Attention modulation** | 目前 $\mathbf q^{(m)}$ 是全局共享 query。可改成由 motion magnitude、JEPA prediction uncertainty 或 prompt token 動態產生 $\mathbf q_{t,p}^{(m)}$；關鍵對照是是否在相同壓縮率下改善 tFVD，而非只改善重建。 |
| **VAR／視覺自迴歸** | 將「下一尺度」或「下一時間 chunk」的 AR token 放在 V-RAE 類語義空間，可能降低 decoder 對小 AR 誤差的敏感度。應同時報告 token cross-entropy、tFVD 和長 rollout consistency，避免只以單步 likelihood 判斷。 |
| **Training-free／zero-shot** | 本文不是 training-free，也未證明 zero-shot T2V；但可做不重訓的 diagnostic：對既有 VAE latent 抽樣計算 tFVD，作為選擇 sampler、cache 或 guidance 強度的風險指標。這是待驗證假說，不能視為論文既有結論。 |

> **我的結論：** V-RAE 最值得帶走的不是「JEPA 一定比 VAE 好」，而是設計生成 latent 時應同時問三件事：它是否保有語義、其時間局部幾何是否平滑、以及 decoder 能否承受預測誤差。把這三者拆開測量，將比單純追逐 reconstruction FID 更容易催生真正可長期 rollout 的影片世界模型。

---

## 參考文獻

[1]: https://arxiv.org/abs/2608.13556 "Guo, Wu, and Fei, V-RAE: Rethinking Video Latent Spaces for Generation, 2026"
[2]: https://v-rae.github.io/ "V-RAE Project Page"
[3]: https://arxiv.org/abs/2603.14482 "Mur-Labadia et al., V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning, 2026"
[4]: https://arxiv.org/abs/2601.16208 "Tong et al., Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders, 2026"
[5]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/b1d4973f5d21708abef3cd6f17d842c8-Abstract-Conference.html "Zhang et al., VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models, NeurIPS 2025"

---

*本報告由 AI Daily 研究流程整理。所有數值均依原論文及其專案頁在對應設定下的報告值，非獨立復現結果；其中 AI Daily 的評價、限制與研究假說均明確與作者結論區分。*
