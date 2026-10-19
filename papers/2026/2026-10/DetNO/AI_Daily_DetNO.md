# AI Daily: DetNO——以行列式最大化做「整批多樣性」最佳化

**日期：2026-10-19**　**方向：training-free、Diffusion、初始噪聲最佳化、生成多樣性**

本輪比較未收錄的近期論文後，選讀 DetNO。它把「同一個 prompt 生成的一批圖片是否真的涵蓋不同語義模式」當成整體集合問題，而不是逐張或逐對推開；核心做法是在不訓練模型、不改採樣器的前提下，對初始噪聲做少量最佳化。這篇對研究想法最有價值之處，是把 DPP／RKHS 的集合體積目標搬進 frozen diffusion 的推論流程。查過 AI_Daily 既有索引與文章後，未找到相同標題或 arXiv ID 2610.22267。

## 論文資訊

- **標題：** *Beyond Pairwise Repulsion: Collective Diversity via Determinant Maximization in Diffusion Sampling*
- **作者：** Kwanyoung Lee、SeungJu Cha、Yebin Ahn、Sungho Koh、Dong-Jin Kim；作者機構為 Hanyang University。[1] [3]
- **論文與日期：** arXiv:2610.22267v1，2026-10-15，cs.CV。[1] [2]
- **會議狀態：** arXiv 紀錄的作者註記標示「ACCV 2026, oral presentation」，作者程式庫亦以 ACCV 2026 標示本文。[2] [3] ACCV 官方網站列出的會議日期為 2026-12-14 至 18；因此本文目前應描述為「作者標示已獲 ACCV 2026 口頭報告」，而不是已經舉行並出版的 proceedings 論文。[4]
- **資源：** [論文全文](https://arxiv.org/html/2610.22267v1)｜[作者程式庫](https://github.com/mobled37/DetNO)。[1] [3]

## 核心貢獻與方法

強 classifier-free guidance（CFG）通常會令同一 prompt 的多次抽樣集中到條件分布中較常見的模式。DetNO 的出發點是：在確定性採樣器下，最初的噪聲決定了之後的生成軌跡，因此可以先在初始噪聲端調整一批候選，再照常執行原模型。它沿用 CNO 的「在 Tweedie 預測空間處理初始噪聲」路徑，將其成對對比目標換成集合層級的 log-determinant。[1] [5]

擴散模型在時間步 $t$ 的 clean latent 估計為：

$$
\hat z_{0|t}(z_t,t,c)=\frac{z_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(z_t,t,c)}{\sqrt{\bar\alpha_t}}.
$$

對一個 prompt 的 $B$ 個初始噪聲 $\{z_T^i\}_{i=1}^B$，先各自計算 $t=T$ 的一步 Tweedie 預測 $\hat z_{0|T}^i$。作者以 average pooling 將預測降採樣，再做 $L_2$ 正規化：

$$
f_i=\frac{\operatorname{AvgPool}(\hat z_{0|T}^i;w)}{\|\operatorname{AvgPool}(\hat z_{0|T}^i;w)\|_2}.
$$

這會壓低高頻變化，讓後續比較偏向粗略語義與構圖，而非 latent 的像素級細節。接著用 RBF 核建立批次相似度矩陣：

$$
S_{ij}=\exp\!\left(-\frac{\|f_i-f_j\|_2^2}{2\sigma^2}\right),\qquad
\sigma=\operatorname{median}_{i<j}\|f_i^{\mathrm{ref}}-f_j^{\mathrm{ref}}\|_2.
$$

$\sigma$ 由未最佳化的參考預測以 median heuristic 決定。最終目標是：

$$
\mathcal L=-\log\det(S+\epsilon I)
+\lambda\sum_{i=1}^{B}\|\hat z_{0|T}^i-\hat z_{0|T}^{i,\mathrm{ref}}\|_2^2.
$$

第一項最大化所有樣本在 RBF 特徵空間共同張成的體積；第二項則將每一個預測限制在自己的初始參考附近，以免只追求分散而跑出 prompt 的合理區域。$\epsilon$ 用於數值穩定，$\lambda$ 控制多樣性與保真度的取捨。演算法對 $z_T^i$ 做 $N_{\mathrm{opt}}$ 次梯度更新，之後把最佳化後的噪聲送入原本的 sampler；模型權重與後續去噪步驟不變。作者也對計算 Tweedie 預測的 denoiser 輸出使用 stop-gradient，以避免儲存完整模型反向傳播 activation。[1]

![DetNO 原論文 Figure 2 聚焦裁圖。圖中文字包括 “A guy holding a racket in a game of tennis”、Mode Covering、Mode Collapse、Initial Data Point、Optimized Data Point、Initial Data Volume、Optimized Data Volume、Diversity Term、Fidelity Term，以及 “Overview in Tweedie Space” 和 “Volume Change with DetNO”；左側示意 CFG 模式集中，右側示意批次 RKHS 體積變化。](../../../../asset/2026-10-19-DetNO/figure2_overview.png)

*圖 1。裁自原論文 Figure 2，只保留概念圖與子圖標籤；原文圖說指出，多樣性項擴大整批 Tweedie 預測共同張成的體積，保真項則把各樣本拉回未最佳化參考附近。[1]*

為何選行列式而非單純 pairwise repulsion？若 $S$ 的特徵值是 $\mu_k$，則：

$$
\nabla_S[-\log\det(S+\epsilon I)]=-(S+\epsilon I)^{-1}
=-U\,\operatorname{diag}\!\left(\frac{1}{\mu_k+\epsilon}\right)U^\top.
$$

因此，批次中體積不足、特徵值偏小的方向會得到較大的修正權重；隨著樣本分散，修正會自行減弱。論文另比較 log-determinant 與 pairwise RBF 的矩陣梯度範數，並報告其 noise-space gradient amplification 與該分析的相關係數為 $r=0.999$。這是論文在其設定下的理論與實驗支持，不代表所有資料與特徵空間都必然有相同結果。[1]

## 實驗結果

主要文字條件生成測試使用 MS-COCO validation prompts：SD1.5 與 SD3 各採每個 prompt 五張圖，SDXL 每個 prompt 三張圖；SD1.5／SDXL 使用 50-step DDIM，SD3 使用 28-step Euler。作者將 PG、SPELL、DiversityPrompt 與 CNO 放在相同 prompt、seed、CFG 和硬體設定下比較。Vendi Score 是由樣本相似度矩陣特徵值的 Shannon entropy 指數化而來的無參考多樣性指標，分數越高代表有效多樣性越高；MSS 則越低表示同批樣本平均越不相似。[1] [8]

- **SD1.5：** DDIM 的 Vendi Score／MSS 為 4.7352／0.1509；DetNO 為 4.8024／0.1229。CNO 為 4.7679／0.1383。DetNO 的 CLIPScore 由 30.9096 升至 31.1396，但 Precision 由 0.7005 降至 0.6744，顯示分布覆蓋與真實資料重疊並非同一目標。[1]
- **SDXL：** Vendi Score 由 2.8560 升至 2.8920，MSS 由 0.2031 降至 0.1720；PickScore 與 ImageReward 較 DDIM 略低。[1]
- **SD3：** Vendi Score 由 4.2517 升至 4.3988，MSS 由 0.2944 降至 0.2542；PickScore 由 22.5719 降至 22.4967，ImageReward 由 1.0503 降至 1.0109。[1]
- **控制聚合器消融：** 在 SD1.5、$B=5$ 且其他設定固定時，DetNO 的 Vendi Score 為 4.8024，略高於相同 RBF 核的 pairwise sum 4.8006，也高於 CNO InfoNCE 的 4.7679；1-shot Pets/CUB 準確率分別是 54.2%／26.3%，pairwise RBF 是 53.7%／25.8%，CNO 是 51.7%／24.5%。論文本身提醒，$B=5$ 時 log-det 與 pairwise RBF 的多樣性差距很小；相對 CNO 的大部分差異來自共同使用的 RBF kernel 與 fidelity 設計，不應把全部增益都歸功於行列式聚合。[1]
- **下游 few-shot 分類：** 每類加入 16 張合成圖，以預訓練 ResNet-18 微調，結果是 10 個 split 的平均值。Pets 1-shot 為 DetNO 54.2%、DDIM 52.2%、CNO 51.7%；論文標示 DetNO 相對 CNO 達 Welch $t$ 檢定 $p<0.01$，但未標示相對 DDIM 顯著。CUB 1-shot 為 26.3%、24.6%、24.5%，DetNO 對兩者皆標示 $p<0.01$；8-shot 則沒有顯著性標記。這支持低樣本資料增強的潛力，但證據仍限於 Pets 與 CUB 兩個細粒度資料集。[1]
- **玩具模式實驗：** 在兩個同心環、共 16 個模式的合成分布中，CFG 遺漏少數外環模式；CNO 找回外環但扭曲原本的內環分布；DetNO 覆蓋全部 16 個模式。這是機制示範，不等於真實影像品質的獨立驗證。[1]

## 相關研究脈絡

- **Contrastive Noise Optimization（CNO）** 是最直接的前作：它同樣在 Tweedie 預測空間、去噪開始前最佳化整批初始噪聲，並以 InfoNCE 將每個樣本錨定到自己、同時排斥其他樣本。DetNO 保留這個初始噪聲介面，改以 log-determinant 聚合 RBF 相似度，主張可處理 pairwise 距離難以表達的高階冗餘。[5]
- **GDPP** 在 ICML 2019 以 determinantal point process kernel 鼓勵 GAN／VAE 訓練時的多樣性。DetNO 延續「Gram determinant 衡量集合體積」的幾何想法，但把目標放在 frozen diffusion 的推論期 Tweedie features，並直接最佳化噪聲，不做 DPP 抽樣。[6]
- **Particle Guidance** 在整段 diffusion trajectory 中，以 joint-particle time-evolving potential 讓多個樣本彼此互動；DetNO 則把控制搬到起始噪聲，避免每個去噪步驟都加入多樣性引導。[7]
- **Vendi Score** 是 DetNO 的評估指標之一，而不是其訓練或最佳化目標。它把相似度矩陣特徵值轉成有效多樣性的熵量測，因此能在沒有外部參考分布時比較同批樣本，但仍需配合品質或語義對齊指標判讀。[8]

## 評價與研究靈感

我認為 DetNO 的重要性不只是「圖像更不一樣」，而是它把生成多樣性定義成整個 sample set 的幾何量，並給出一個可直接測試的訓練免費推論介面。尤其 pairwise-RBF 消融只帶來小幅多樣性差距，提醒我們必須區分「新聚合形式」與「改用 RBF kernel／保真約束」各自的貢獻。另一方面，保真項不是裝飾：作者的消融指出只最大化行列式會把噪聲推到分布外，產生卡通化、色彩錯亂等結果。[1]

適合接續測試的方向如下；以下是研究假設，不是原論文已證明的結果。

1. **Energy-based batch control：** 以凍結的 energy-based model 為每個候選加上條件相容性能量，改成
   $$
   \mathcal L_{\mathrm{EBT}}=-\log\det(K+\epsilon I)+\lambda\sum_i\|\hat z_i-\hat z_i^{\mathrm{ref}}\|^2+\gamma\sum_i E_\theta(\hat x_i,c).
   $$
   可測試低能量約束能否在維持 prompt fidelity 的同時，降低 DetNO 造成的品質損失；另比較全域固定 $\lambda$ 與依能量自適應的 $\lambda_i$。
2. **JEPA 語義核與不確定度：** 將 pooled latent 的 $f_i$ 換成 frozen JEPA encoder 對 clean estimate 或快速解碼結果的語義表示，建立 $K_{ij}=\exp(-\|\phi_{\mathrm{JEPA}}(x_i)-\phi_{\mathrm{JEPA}}(x_j)\|^2/\tau)$。再以 JEPA predictor 的預測分歧調節 fidelity 權重：語義已穩定時允許更多多樣性，預測不確定時加強 anchor。這可分辨「低階外觀不同」與「真正語義模式不同」。
3. **VAR 與 attention modulation：** 對 Visual Autoregressive Model，不直接移植連續 $z_T$，而是在每個 next-scale 產生多個候選 token block，對候選的區域／語義 embedding 最大化集合 log-det，並以 frozen-model likelihood 或能量限制保真。可先做 zero-shot reranking；若 reranking 有效，再測試把 log-det 的低特徵值方向轉成 scale-wise attention bias，讓注意力資源流向同批候選尚未涵蓋的區域。評估應同時控制生成張數與額外 forward 次數。

最小驗證實驗可固定同一批 prompts、seed 與候選數，直接比較 latent-RBF、DINO／JEPA 語義核及 EBT energy gate；在 $B\in\{2,5,8,16\}$ 下同時報 Vendi Score、MSS、CLIP／PickScore、低樣本分類與推論時間。特別應檢查 $B=2$ 時行列式與 pairwise 目標差異有限的情況，以及多樣性提升是否真的轉化成語義覆蓋和下游效益。

另有兩點重現注意：作者公開腳本將 SD1.5 與 SD3 的 MS-COCO 範例各設為 200 prompts，且公開腳本目錄列出 SD1.5／SD3、未見 SDXL 腳本；論文主表則描述 2,000 prompts 並包含 SDXL。公開範例中 SD1.5 使用 $N_{\mathrm{opt}}=3$，SD3 使用 $N_{\mathrm{opt}}=5$。因此「training-free」不等於零額外計算：初始噪聲最佳化仍需要多次 denoiser forward；完整論文主實驗的重現範圍也需進一步釐清。[3] [9] [10]

## References

[1]: https://arxiv.org/html/2610.22267v1 "Beyond Pairwise Repulsion: Collective Diversity via Determinant Maximization in Diffusion Sampling"
[2]: https://export.arxiv.org/api/query?id_list=2610.22267 "arXiv record for 2610.22267"
[3]: https://github.com/mobled37/DetNO "Official DetNO repository"
[4]: https://accv2026.org/ "ACCV 2026 official website"
[5]: https://openreview.net/forum?id=EVRMnAREc3 "Diverse Text-to-Image Generation via Contrastive Noise Optimization"
[6]: https://proceedings.mlr.press/v97/elfeki19a.html "GDPP: Learning Diverse Generations using Determinantal Point Processes"
[7]: https://arxiv.org/abs/2310.13102 "Particle Guidance: Non-IID Diverse Sampling with Diffusion Models"
[8]: https://arxiv.org/abs/2210.02410 "The Vendi Score: A Diversity Evaluation Metric for Machine Learning"
[9]: https://raw.githubusercontent.com/mobled37/DetNO/main/scripts/sd15_mscoco.sh "DetNO SD1.5 evaluation script"
[10]: https://raw.githubusercontent.com/mobled37/DetNO/main/scripts/sd3_mscoco.sh "DetNO SD3 evaluation script"
