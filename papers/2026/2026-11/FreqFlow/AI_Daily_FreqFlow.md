# AI Daily

## Frequency-Aware Flow Matching for High-Quality Image Generation

- **報告日期：** 2026-11-03
- **作者與研究單位：** Sucheng Ren（Johns Hopkins University）；Qihang Yu、Ju He、Xiaohui Shen、Liang-Chieh Chen（ByteDance）。[1]
- **發表狀態：** CVPR 2026 論文；arXiv:2604.15521。論文與程式碼可由 [CVPR 論文頁](https://cvpr.thecvf.com/virtual/2026/poster/39662) 和 [官方程式庫](https://github.com/OliverRensu/FreqFlow) 取得。[2] [3] [4]
- **主題：** Flow matching、頻率分解、類別條件影像生成。

### 候選篩選與選題理由

本次先廣泛檢索近期 arXiv、Hugging Face Trending Papers 與 CVPR／ICLR／ICML 2026 的論文，再將題目與 arXiv 編號對照 repo 的 README、INDEX 和既有文章。Energy-Based Transformers are Scalable Learners and Thinkers 雖是 ICLR 2026 口頭論文，但 repo 已有多篇解析；較新的 Transformer Energy-Based Model（TEM，arXiv:2610.31889）也已於 2026 年 10 月收錄，因此兩者都排除。FreqFlow 則未出現在現有索引中，而且是 CVPR 2026 的圖像生成論文，聚焦 flow matching 的頻率建模，符合這次的研究偏好。[1] [2]

這篇的切入點很明確：flow matching 不只是把樣本從噪聲連續地搬到影像，也有「先形成整體形狀、再補細節」的頻率結構。FreqFlow 把這種生成順序變成明確的模型路徑與訓練訊號，因而提供一個可以延伸到 attention modulation、動態計算分配及 VAR coarse-to-fine 生成的研究介面。[1]

### 核心貢獻

第一，作者觀察到標準 flow matching 在去噪過程中通常先恢復低頻成分，例如輪廓、布局與大塊色彩，再逐步恢復高頻紋理與邊緣；但一般模型只在空間域預測，沒有明確區分這兩種訊號。論文量測的頻率誤差中，SiT 的低頻／高頻誤差為 0.08／0.69，FreqFlow 降至 0.06／0.48，顯示改善主要也涵蓋容易模糊的高頻細節。[1]

第二，FreqFlow 採用頻率分支與空間分支。頻率分支分開處理低、高頻資訊，再由時間條件化的權重決定兩者如何融合；空間分支則依據融合後的頻率特徵預測完整影像流場。這讓全局結構和細節紋理可以透過不同路徑協作，而不是要求單一空間域網路隱式學會兩種尺度的恢復規律。[1]

第三，訓練時同時監督空間域和頻率域的預測，避免只讓低、高頻特徵作為額外條件，卻沒有直接約束頻譜結果。作者在 ImageNet-256 上以 FID-50K 評估，FreqFlow-H 的 class-conditional FID 為 1.38。[1] [2]

### 技術方法與數學形式

令乾淨影像或其 latent 為 $x$，噪聲為 $\epsilon\sim\mathcal{N}(0,I)$。論文採用的 flow matching 路徑為

$$x_t=(1-t)x+t\epsilon,\qquad t\in[0,1].$$

因此 $t=0$ 對應乾淨影像，$t=1$ 對應純噪聲；沿此路徑的目標速度為

$$v_t=\frac{dx_t}{dt}=\epsilon-x.$$

模型 $f_\theta$ 以均方誤差學習速度場：

$$\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{x,\epsilon,t}\left[\left\|f_\theta(x_t,t,c)-(\epsilon-x)\right\|_2^2\right],$$

其中 $c$ 是類別條件。推論時從噪聲端沿 ODE 積分回資料端。這是標準 flow matching 的基礎；FreqFlow 的改動在於讓速度預測顯式感知頻率。[1]

對每個帶噪影像 $x_t$，模型先計算離散傅立葉轉換 $F_t=\mathcal{F}(x_t)$，再用低通與高通 Gaussian mask 分離頻段。令 $r^2=(u-H/2)^2+(v-W/2)^2$，則

$$M_L(u,v)=\exp\left(-\frac{r^2}{2\sigma_L^2}\right),\qquad M_H(u,v)=1-\exp\left(-\frac{r^2}{2\sigma_H^2}\right),$$

$$L_t=\mathcal{F}^{-1}(M_L\odot F_t),\qquad H_t=\mathcal{F}^{-1}(M_H\odot F_t).$$

$L_t$ 保留較慢變化的全局結構，$H_t$ 強調邊緣、紋理等較快速變化的細節。實作上，作者用一個 ViT 頻率分支處理兩類成分並產生其特徵。網路再透過時間條件化的 gate 混合低、高頻特徵：

$$\omega_t=\operatorname{sigmoid}\!\left(\operatorname{MLP}(h_t^L,h_t^H,t)\right),\qquad h_t=\omega_t\odot h_t^L+(1-\omega_t)\odot h_t^H.$$

這裡 $\omega_t$ 不是手寫的固定 schedule，而是由模型依據頻率特徵和時間步學得。生成早期可以較重視低頻結構，接近資料端時再增加高頻細節的影響。[1]

空間分支以 ConvNeXt 實作，接收帶噪輸入與融合特徵的逐元素相加，預測完整速度場：

$$\hat v_t=f_{\mathrm{spatial}}(x_t+h_t,t,c).$$

作者選擇簡單相加，而不是跨注意力或 channel concatenation。這個設計將頻率分支定位為空間生成器的條件訊號；頻率分支並非獨立產生最後影像。[1]

為同時要求空間結果和頻譜結果接近目標，論文定義

$$\mathcal{L}_s(a,b)=\|a-b\|_2^2,\qquad \mathcal{L}_f(a,b)=\|\mathcal{F}(a)-\mathcal{F}(b)\|_2^2,$$

並以全速度場和低、高頻分量組成總損失：

$$\begin{aligned}\mathcal{L}={}&\mathcal{L}_s(\hat v_t,v_t)+\mathcal{L}_f(\hat v_t,v_t)\\
&+\alpha\sum_{b\in\{L,H\}}\left[\mathcal{L}_s(\hat v_t^b,v_t^b)+\mathcal{L}_f(\hat v_t^b,v_t^b)\right],\end{aligned}$$

其中 $\alpha=0.5$ 為預設值。前兩項監督最終空間輸出，後兩組項目則要求頻率分支對各自的頻段預測負責。[1]

### 實驗結果

主要結果是在 ImageNet-256 類別條件生成上，以 50,000 張樣本計算 FID-50K；以下數字來自論文的 classifier-free guidance（CFG）表格，FID 越低越好。[1]

- **FreqFlow-L** 有 507M 參數，FID 為 1.54、Inception Score 為 295.6；**FreqFlow-H** 有 1.08B 參數，FID 為 1.38、Inception Score 為 298.5。作為參考，DiT-XL/2（675M）FID 為 2.27，SiT-XL/2（675M）為 2.06。FreqFlow-L 用較少參數仍優於這兩個基線；FreqFlow-H 則在該表格取得最低 FID。[1]
- 在 ImageNet-512 上，FreqFlow-L（507M）報告 FID 2.02、Inception Score 285.3。論文另報告不使用 CFG 時，FreqFlow-L／H 的 ImageNet-256 FID 為 3.12／2.45。這兩組數字的 guidance 設定不同，不能混在同一排名中比較。[1]
- 在 134M 模型的 400-epoch 消融中，未加入頻率資訊的 FID 為 3.86；只加入低頻後為 3.55，只加入高頻後為 3.12，同時加入兩者為 2.95。這支持低頻和高頻提供互補訊號，而不只是增加網路容量。[1]
- 相同消融中，空間分支採逐元素相加時 FID 為 2.95，優於 channel concatenation 的 3.46 和 cross-attention 的 3.95；拿掉頻率分支專屬損失，FID 從 2.95 惡化至 4.67。這是論文設計選擇中較有說服力的 ablation。[1]

代價也需要一併看。主要模型訓練 800 epochs、batch size 2,048，論文附錄列出的 ODE sampling 為 250 步；官方訓練範例使用 32 個 process、4 台機器。這不是免訓練的推論技巧，而是需要額外模型分支與昂貴訓練的生成架構。[1] [4]

### 相關研究

**DiT** 將 latent diffusion 的 U-Net 主幹替換成 Transformer，並顯示增加計算量與模型規模可降低 ImageNet FID。**SiT** 延續 DiT 架構，以 interpolant／dynamical transport 方式研究生成路徑和速度場，包含 flow matching 設定。FreqFlow 沒有只更換 ODE 求解器，而是把頻率分解、時間自適應融合和頻譜損失接到生成主幹上。[5] [6]

較新的 **WaiT** 也沿著頻率階層重設生成過程，但採用不同路線：它以可逆 Haar wavelet 分解低、高頻帶，先讓高頻維持噪聲，等低頻結構形成後再加入聯合去噪；因此主要改變的是各頻帶的 noise schedule，而不是增加 FreqFlow 式的雙分支頻率網路。WaiT 的 Meta Research 介紹報告 ImageNet-512 pixel-space FID 1.43，並稱最高可減少 50% sampling compute；這是較新的 arXiv 工作，且它與 FreqFlow 在解析度、pixel／latent 設定和評估指標上不同，不能直接用數字判定優劣。[7] [8]

### 個人評價與研究意義

我認為 FreqFlow 的價值在於把「全局先行、細節後補」從生成結果的經驗觀察，轉成可以診斷、控制和消融的模型變數。最值得借鑑的不只是雙分支，而是它同時回答三個問題：各頻帶如何分開、何時融合、如何讓損失確實約束頻域品質。消融結果也顯示，簡單的 feature addition 和直接的頻帶監督，比更複雜的融合方式更有效。[1]

限制是，目前證據集中在 ImageNet 類別條件生成，尚未證明這些增益可直接轉移到開放詞彙文字生成、編輯或其他影像分布；模型規模也只到約 1B 參數。另一方面，FreqFlow 需 800 epochs 和多機訓練，所以雖然頻率 gate 可啟發 training-free 控制，它本身並不是 training-free 或 zero-shot 方法。[1] [4]

這篇可以延伸成幾個可驗證的研究問題：

1. **Energy-gated 頻率計算：** 訓練一個對條件與候選輸出打分的 verifier，估計目前樣本在全局結構或局部細節哪一側較不一致，再動態調整高／低頻更新步數。比較固定 250 步、等量增加步數與 energy-gated 分配，並同時報告 FID、頻譜誤差、條件遵從和 wall-clock。
2. **JEPA 語義守門：** 以凍結的 JEPA 表徵保護低頻布局和物件關係，讓高頻分支負責紋理變化；測試是否能減少增強細節時的語義漂移。需和只用 class/text condition 的 FreqFlow 分開比較，並計入 JEPA 額外成本。
3. **Training-free attention modulation：** 在既有 DiT／flow Transformer 中，依時間步與頻帶預測調制 cross-attention 或 spatial attention，而不更新權重。這能檢驗 FreqFlow 的收益是否來自「顯式頻率表徵」，還是必須靠重新訓練整個頻率分支。
4. **VAR 的頻率—尺度預算：** 將低頻結構對應到較早的 coarse scales，高頻細節對應到後續細尺度 token；讓預測不確定度決定每一層需要多少 token 或 refinement。評估時除 FID 外，應加入頻譜誤差、細節品質、推理延遲與顯存，避免單一生成指標掩蓋品質—成本取捨。

### References

[1]: https://openaccess.thecvf.com/content/CVPR2026/papers/Ren_Frequency-Aware_Flow_Matching_for_High-Quality_Image_Generation_CVPR_2026_paper.pdf "Frequency-Aware Flow Matching for High-Quality Image Generation at CVPR 2026"
[2]: https://arxiv.org/abs/2604.15521 "Frequency-Aware Flow Matching for High-Quality Image Generation on arXiv"
[3]: https://cvpr.thecvf.com/virtual/2026/poster/39662 "CVPR 2026 poster listing for FreqFlow"
[4]: https://github.com/OliverRensu/FreqFlow "Official FreqFlow implementation"
[5]: https://arxiv.org/abs/2212.09748 "Scalable Diffusion Models with Transformers"
[6]: https://arxiv.org/abs/2401.08740 "Scalable Interpolant Transformers"
[7]: https://arxiv.org/html/2607.28760v1 "WaiT for the Signal: Simple Frequency-Aware Flow-Matching"
[8]: https://ai.meta.com/research/publications/wait-for-the-signal-simple-frequency-aware-flow-matching/ "Meta Research overview of WaiT"
