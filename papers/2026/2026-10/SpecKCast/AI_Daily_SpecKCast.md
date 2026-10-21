# AI Daily: SpecKCast——用頻譜 Kalman 預測替擴散模型分配計算

**日期：2026-10-21**　**方向：Diffusion／DiT、training-free inference、頻域預測、Kalman filtering**

今日選讀 **SpecKCast**。它不快取 DiT 的中間特徵，而是直接預測 sampler 下一步會用到的 denoiser output。作者先將輸出拆成不同空間頻帶，再為各頻帶建立 Kalman forecast；當預測共變異數顯示風險升高時，才重新執行完整 Transformer。這讓「何時跳過模型」不再只靠固定步距或特徵差距，而可以由可量化的不確定度控制。[1] [2]

## 論文資訊

- **標題：** *SpecKCast: Spectral Kalman Forecasting for Diffusion Acceleration*
- **作者與機構：** Javad Rajabi、Hakki Can Karaimer、David B. Lindell、Babak Taati、Michael S. Brown；論文列出的機構包括 University of Toronto、Vector Institute、AI Center–Toronto 與 Samsung Electronics。[1] [2]
- **日期與領域：** arXiv:2610.26837v1，arXiv 記錄時間為 2026-10-19 17:58 UTC；換算台灣時間為 10 月 20 日凌晨。本文報告日期為 10 月 21 日。[2]
- **會議／期刊狀態：** 截至 2026-10-21，arXiv 記錄未列正式會議或期刊接收資訊，因此本文標示為 **arXiv 預印本**，不稱為已發表或 peer-reviewed 論文。[1] [2]
- **論文連結：** [HTML 原文](https://arxiv.org/html/2610.26837)｜[PDF](https://arxiv.org/pdf/2610.26837v1)｜[arXiv 摘要](https://arxiv.org/abs/2610.26837v1)。

## 核心貢獻與方法

擴散模型在相鄰去噪步之間有時間冗餘，因此許多加速法會重用中間特徵，或預測未來特徵。SpecKCast 改變預測對象：它預測的是 sampler 真正需要的完整 denoiser output，例如 noise、velocity 或其他模型輸出，而不是網路內部 activation。若以 $x_j$ 表示第 $j$ 步 latent，則原模型可寫成

$$
x_{j+1}=S_j(x_j,Y_j;\sigma_j),
$$

其中 $Y_j$ 是完整模型在該步產生的輸出，$S_j$ 是原有 scheduler 更新。SpecKCast 在部分步驟估計 $\widehat{Y}_j$，其餘時間才呼叫完整 DiT。[1] [3]

**1. 將輸出分成頻帶。** 對每個 channel 的 $Y_j$ 做二維離散餘弦轉換（DCT），再依徑向頻率劃分成 $L$ 個不重疊區域。令 $M_\ell$ 為第 $\ell$ 個頻帶的遮罩，則

$$
Z_j^{(\ell)}=M_\ell\operatorname{DCT}_2(Y_j).
$$

作者預設使用 $L=7$ 個頻帶。這樣做的動機是低頻結構和高頻細節沿去噪時間演化的速度不同；若直接對整張輸出套同一個時間模型，較快變化的成分可能拖累整體預測。[1] [3]

**2. 依頻帶校正時間軸。** 作者從校準軌跡估計每個頻帶的功率 $p_\ell$，再把原始 noise schedule 的時間轉成頻帶自己的 forecast 座標：

$$
\lambda_j=2\log\frac{1-\sigma_j}{\sigma_j},\qquad
s_\ell(j)=\operatorname{Rescale}_{[-1,1]}\left[
\operatorname{sigmoid}\left(\frac{\lambda_j+\log p_\ell}{T_w}\right)
\right].
$$

$T_w$ 控制 time warp 的平滑程度。這一步讓 Kalman filter 追蹤的是不同頻率各自較合適的變化尺度，而不只是原始去噪步索引。[1] [3]

**3. 以二階 Kalman 狀態預測每個頻帶。** 對頻帶中的係數 $i$，令狀態包含目前估計值 $m_{i,j}$ 與變化率 $\nu_{i,j}$：

$$
\mathbf h_{i,j}=\begin{bmatrix}m_{i,j}\\\nu_{i,j}\end{bmatrix},\qquad
\Delta_{\ell,j}=s_\ell(j)-s_\ell(j-1),\qquad
\rho_{\ell,j}=\exp\left(-\frac{\Delta_{\ell,j}}{\tau_\ell}\right).
$$

其線性轉移矩陣與 process covariance 寫成

$$
A_{\ell,j}=\begin{bmatrix}1&\Delta_{\ell,j}\\0&\rho_{\ell,j}\end{bmatrix},\qquad
Q_{\ell,j}=q_\ell\begin{bmatrix}
\Delta_{\ell,j}^3/3&\Delta_{\ell,j}^2/2\\
\Delta_{\ell,j}^2/2&\Delta_{\ell,j}
\end{bmatrix}.
$$

$\tau_\ell$ 控制變化率衰減，$q_\ell$ 控制過程雜訊。每次預測後，協方差依標準 Kalman 形式向前傳播：

$$
P^-_{\ell,j}=A_{\ell,j}P_{\ell,j-1}A_{\ell,j}^{\top}+Q_{\ell,j}.
$$

完整模型被呼叫時，實際 denoiser output 就是觀測值，可用 Kalman innovation 更新狀態；跳過的步驟則使用預測值重建頻帶，再經 inverse DCT 還原成 $\widehat{Y}_j$。[1] [3]

**4. 由不確定度決定何時重算。** 作者把每個頻帶的預測變異數加總成風險分數，並以時間權重調整：

$$
R(j)=\omega(j)\sum_{\ell=1}^{L}d_\ell\,[P^-_{\ell,j}]_{11},\qquad
\omega(j)=\exp\left(-\frac{\beta j}{N-1}\right),
$$

其中 $d_\ell$ 是該頻帶係數數量，$N$ 是去噪步數。當 $R(j)>\epsilon$，系統執行完整 Transformer；否則接受 Kalman forecast。$\epsilon$、$\beta$ 和各頻帶參數需依模型與 scheduler 校準。方法不更新模型權重，但並非完全沒有校準成本。[1] [3]

若使用 classifier-free guidance，標準組合可寫為

$$
G(Y_j^c,Y_j^u)=Y_j^u+w(Y_j^c-Y_j^u),
$$

其中 $Y^c,Y^u$ 分別是條件與無條件輸出，$w$ 是 guidance scale。作者指出，Qwen-Image 的 nonlinear norm rescaling 會破壞分支預測與先組合再預測之間的線性等價；因此套用到含非線性 CFG 後處理的模型時，不能直接假設兩種 forecast 做法完全相同。[1] [3]

## 實驗結果

實驗涵蓋兩個文字生圖模型與兩個文字生影片模型：FLUX.1-dev（1024×1024）、Qwen-Image（1328×1328）、Wan2.1-14B 與 HunyuanVideo。作者以 50 步完整推論作為參考，在 NVIDIA H100 上測量延遲，並比較 ImageReward、CLIP、VBench，以及相同 prompt／seed 下的 PSNR、SSIM、LPIPS。下列 NFE 與數值沿用原文表格設定，不另外換算 classifier-free guidance 的條件與無條件 forward 次數。[1] [3]

- **文字生圖、約 8 NFE：** FLUX.1-dev 達 **5.54×** speedup，PSNR／SSIM／LPIPS 為 19.81 dB／0.683／0.353，額外顯存 0.005 GiB；Spectrum 在相近設定為 5.43×、18.71 dB／0.632／0.397、0.73 GiB。Qwen-Image 達 **6.00×**，為 20.44 dB／0.740／0.267、額外顯存 0.01 GiB；Spectrum 為 5.48×、16.91 dB／0.604／0.527、1.89 GiB。[1] [3]
- **文字生圖、約 14 NFE：** FLUX.1-dev 為 3.38×，Qwen-Image 為 3.48×。在兩個模型上，SpecKCast 的 PSNR 均高於 Spectrum；Qwen-Image 的 LPIPS 為 0.134，Spectrum 為 0.135，差距很小。[1] [3]
- **文字生影片、約 14 NFE：** Wan2.1-14B 達 **3.52×**，PSNR／SSIM／LPIPS 為 24.47 dB／0.813／0.140，VBench 80.86；Spectrum 為 3.35×、22.93 dB／0.789／0.136、VBench 80.76。這裡 LPIPS 略差，提醒不同品質指標不應只挑有利的一項。HunyuanVideo 為 **3.33×**、29.72 dB／0.884／0.107、VBench 79.68；SeaCache 為 2.57×、27.59 dB／0.864／0.121、VBench 79.53。[1] [3]
- **模組消融：** FLUX.1-dev 的完整設定 PSNR 為 24.28 dB。移除頻率分解後降至 21.19 dB；把 frequency-aware time warp 改為 raw log-SNR 或 uniform step index，則分別為 21.03 dB 和 19.54 dB。固定 Kalman schedule 為 22.09 dB；使用 uncertainty-aware schedule 後提升約 2.2 dB，支持「用預測變異數排計算」是方法重要組件，而非只靠 DCT 轉換。[1] [3]

整體而言，作者在其指定模型與設定中報告約 3.3×–6.0× speedup，額外顯存低於 0.05 GiB。這些是特定模型、步數、scheduler、H100 和校準參數下的結果，不應外推成所有擴散或 flow-matching 系統都能得到同樣加速。[1] [3]

## 相關研究脈絡

- **TaylorSeer** 是 training-free 的局部特徵 forecasting 方法，利用相鄰步的變化趨勢近似未來 DiT feature；它在 ICCV 2025 發表。SpecKCast 的主要差別是預測 sampler 使用的 denoiser output，並在頻域以 Kalman covariance 控制完整計算。[4]
- **Spectrum** 對整體 feature trajectory 做 global forecasting，以 Chebyshev basis 和 ridge regression 預測特徵；作者專案頁標示 CVPR 2026。SpecKCast 也利用較長時間的軌跡資訊，但把預測對象改成分頻後的模型輸出，並用 covariance threshold 決定何時重算。[5]
- **SeaCache** 以頻譜演化資訊判斷哪些中間特徵可以重用，並作為影片擴散加速基線；其重點是頻譜感知的 cache policy，SpecKCast 則明確 forecast 連續步的 denoiser output。[6]
- **MeanCache** 是 ICLR 2026 的 flow-matching 推論加速研究，以平均 velocity 和 trajectory scheduling 減少計算。它與 SpecKCast 都利用相鄰推論步的時間冗餘，但 MeanCache 研究的是 flow interval 的 average velocity，不能當成 SpecKCast 已驗證支援 flow matching 的證據。[7]

## 評價與研究靈感

我認為 SpecKCast 最有價值的觀點，是把加速的預測目標從「網路中間特徵」移到「sampler 真正使用的輸出」，再把不確定度變成可操作的計算閘門。頻帶化讓模型不必假設低頻布局與高頻細節遵循同一時間動態；Kalman covariance 則給出何時該停止猜測、重新計算的明確條件。這種拆解有助於把品質—速度的 trade-off 從固定 skip schedule 改成由模型輸出風險驅動。[1] [3]

目前仍有四項限制。第一，方法利用的是**時間冗餘**，每個真正計算的步驟仍處理完整 token grid，尚未利用 Transformer 內部的 spatial/token redundancy。第二，$p_\ell$、$\tau_\ell$、$q_\ell$、$T_w$、$\beta$ 與閾值都需模型／scheduler 特定校準，所以 training-free 不等於免校準，也不等於 strict zero-shot。第三，主要實驗集中在四個生成模型與 H100；目前沒有證據顯示跨硬體、跨 scheduler 或未校準新模型仍能保持相同表現。第四，論文目前是 arXiv 預印本，結論仍需後續同行審查與獨立重現。[1] [2] [3]

以下是我認為值得延伸、但**不是 SpecKCast 原論文結果**的研究問題：

1. **Energy-based risk gate：** 若定義較低 energy 代表更高條件相容性，可在 covariance risk 外加上生成狀態的 energy 惡化量：
   $$
   R_{\text{joint}}(j)=R(j)+\lambda\,[E_\theta(\hat{x}_j,c)-E_\theta(\hat{x}_{j-1},c)]_+.
   $$
   這會測試「統計上的 forecast 不確定」和「語義條件可能偏離」是否互補。Energy-Based Transformers 可作概念背景，但本文沒有使用 EBT 或做 energy-guided inference。[8]
2. **JEPA-style predictive uncertainty：** 可在 latent trajectory 上訓練 predictor，將預測 latent 與 forecasted image latent 的 Mahalanobis disagreement 加進 $R(j)$。關鍵實驗是比較 Kalman-only、predictive-disagreement-only 與兩者融合，在相同 NFE／latency 下是否降低 prompt 遺漏或細節漂移。這需要獨立設計與訓練 latent predictor，不代表現成 JEPA 可直接替換 Kalman。[9]
3. **延伸到 VAR：** VAR 生成的是離散、多尺度 token，不是連續 denoiser output，因此不宜照搬 DCT Kalman。較合理的假設是以 scale-wise logit entropy、top-1 margin 或不同 predictor 的 disagreement 估計下一尺度風險，再決定該尺度是否需要完整 autoregressive computation；其基線可從 next-scale generation 開始。[10]
4. **結合 attention modulation 與 zero-shot 評估：** 先量出每個頻帶風險與 attention head／block 對輸出的關係，再只對高風險區域提高計算或保留稀疏 attention。實驗應固定校準參數，在未見 prompts 上評估；若要聲稱 zero-shot transfer，還要另測未校準 scheduler 或模型，不能把「不更新權重」本身當成 zero-shot 證據。

一個可執行的最小 follow-up，是固定 prompt、seed、硬體與品質目標，比較 full inference、SpecKCast、移除 DCT、固定 skip schedule、Energy gate 與 JEPA disagreement gate；同時記錄 calibration time、實際模型 forward 次數、端到端 latency、PSNR／LPIPS、語義評分與跨 prompt 的閾值穩定性。這能辨認加速究竟來自較好的預測，還是較多的校準與模型特定調參。

## References

[1]: https://arxiv.org/html/2610.26837 "SpecKCast: Spectral Kalman Forecasting for Diffusion Acceleration"
[2]: https://arxiv.org/abs/2610.26837v1 "arXiv record for SpecKCast"
[3]: https://arxiv.org/pdf/2610.26837v1 "SpecKCast paper PDF"
[4]: https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_From_Reusing_to_Forecasting_Accelerating_Diffusion_Models_with_TaylorSeers_ICCV_2025_paper.pdf "TaylorSeer at ICCV 2025"
[5]: https://hanjq17.github.io/Spectrum/ "Spectrum project page"
[6]: https://jiwoogit.github.io/SeaCache/ "SeaCache project page"
[7]: https://iclr.cc/virtual/2026/poster/10010502 "MeanCache at ICLR 2026"
[8]: https://arxiv.org/abs/2507.02092 "Energy-Based Transformers are Scalable Learners and Thinkers"
[9]: https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/ "I-JEPA background from Meta AI"
[10]: https://neurips.cc/virtual/2024/poster/94115 "Visual Autoregressive Modeling at NeurIPS 2024"
