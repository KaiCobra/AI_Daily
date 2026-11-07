# AI Daily

## 今日選讀：把快取誤差變成可回看的生成軌跡

### 論文資訊

**Z-Cache: Accelerating Diffusion Transformers via Self-Reflection**

作者：Zegang Cheng、Zhikai Wang、Jiacheng Liu、Xiaobing Tu、Chang Zou、Junjie Chen、Yue Ma、Zhiyuan Ma、Jinkui Ren、Likai Zou、Linfeng Zhang。論文署名機構包括 Shanghai Jiao Tong University、Southern University of Science and Technology、Alibaba Cloud、Huazhong University of Science and Technology（此為論文署名機構，不代表作者現職）。arXiv v1：2026-11-01；AI Daily：2026-11-07。論文目前是 **arXiv v1 預印本**，但 arXiv 註記明載已獲 **NeurIPS 2026 poster** 接受，NeurIPS 官方 virtual poster 與 Downloads 頁也列有同一題名與 poster ID 156017；因此可稱「正式錄取、NeurIPS 2026 海報」，不能進一步寫成已出版的 proceedings 論文。[1] [3] [4] [5] arXiv PDF 將第五位作者更正為 **Chang Zou**，官方 poster metadata 仍顯示 Chang Zhou；本文依 PDF 更正，並保留這項可追溯的 metadata 差異。[3] [4]

## 一句話摘要

Z-Cache 是一個 **training-free、predictor-agnostic 的 cache-then-check 包裝器**：先用過去 full-computation anchor 暫存一段未來軌跡，走到原排程中的下一個 full anchor 後，再用這個「未來檢查點」回頭反射並重播窗口，讓後續去噪從修正後的 latent 繼續。它不是對已生成圖片做事後平滑，而是在生成迴圈中改變後續 latent trajectory。[1] [2]

## 為何值得今天讀

現有 feature caching 的核心取捨很直接：少做 full Transformer evaluation，就會讓 hidden feature error 在 cache window 內累積。Z-Cache 的洞見不是再發明一個 predictor，而是接受這個 provisional window，等排程本來就要做的下一次 full step 取得未來資訊，再回頭修正已走過的窗口。這使它成為可以疊加在 TaylorSeer、SpeCa、MagCache 等 predictor 之上的 wrapper，而非取代它們的單一 cache policy。[2] [3]

這個定位也很符合近期對 training-free 與 attention/feature modulation 的興趣：方法不訓練新 predictor，能與少步蒸餾、8-bit quantization、U-Net 或 DiT backbone 組合；但它不是自動等於 zero-shot，論文也沒有把 zero-shot 當作正式術語或普遍能力主張。

## 方法：cache、check、reflect、replay

令第 $i$ 個 sampling transition 的抽象形式為

$$x_{i+1}=F_i(x_i,\mathcal H_i),$$

其中 $x_i$ 是 latent，$\mathcal H_i$ 是 DiT 各 block 的 hidden-feature bank。對相鄰兩個排程 full index $a<b$，中間 $a<i<b$ 是 cache window。既有 predictor $C$ 依目前可用的 anchor bank 估計

$$\hat{\mathcal H}_i=C(\mathcal A_i,i).$$

**第一階段：provisional cache。** 方法固定左端的過去 anchor bank $B^p$，以 $\hat{\mathcal H}_i^p=C(B^p,i)$ 執行暫時 transition，得到 provisional latent $\bar x_b$。到達 $b$ 時，照原本 schedule 做一次 FullStep，並在 $\bar x_b$ 上取得未來 full feature $\mathcal H_b$。這不是額外插入的 full step，而是原本就排定的 anchor。

**第二階段：雙錨點反射。** 將過去 bank 與 $\mathcal H_b$ 合併成 $B^f$，得到未來 anchor 參考下的 predictor $\hat{\mathcal H}_i^f$。在窗口內融合兩種預測：

$$\tilde{\mathcal H}_{i}=(1-w_i)\hat{\mathcal H}^{p}_{i}+w_i\hat{\mathcal H}^{f}_{i}, \qquad w_i=\frac{i-a}{b-a},\quad a<i<b.$$

因此窗口前段較信任過去 anchor，接近右端點時逐步轉向未來 anchor；在右端點，權重到達 1。

**第三階段：active-window replay。** 丟棄 provisional window 內的 latent，固定左邊界 $\tilde x_a=x_a$，重新執行

$$\tilde{x}_{i+1}=F_i(\tilde{x}_i,\tilde{\mathcal H}_i), \qquad a\le i<b.$$

重播後的 $\tilde x_{b+1}$ 取代 provisional endpoint，並以 reconciled predictor state 開啟下一個窗口。附錄偽代碼指出 replay 不前進共享 scheduler cursor，也沒有額外 endpoint FULL call；沒有右側 full anchor 的 cache tail 則沿用 base predictor。[2] [3]

### 誤差界告訴我們什麼

令 $\delta_i$ 是 replay latent 與 Full 軌跡的距離，令 $\epsilon_i^p$、$\epsilon_i^f$ 分別是過去與未來 anchor predictor 的 hidden error。論文給出的局部遞迴是

$$\delta_{i+1}\le L_i^x\delta_i+L_i^H\epsilon_i,$$

而雙錨點融合滿足

$$\epsilon_i^z =\|\tilde{\mathcal H}_i-\mathcal H_i^\star\| \le (1-w_i)\epsilon_i^p+w_i\epsilon_i^f.$$

這裡的 $L_i^x,L_i^H$ 是對 latent 與 hidden 的局部 Lipschitz 常數。推導的直觀是：若未來 anchor 對窗口後段的誤差小於單靠過去 anchor 的預測，則線性反射會壓低後段的 hidden error，並透過 transition 遞迴收緊局部 latent error。這是**條件式局部上界**，不是普遍收斂定理，也不保證所有 sample 或所有 predictor 都改善。[2] [3]

## 主要實驗：速度數字必須連同分母閱讀

### ImageNet：最高 6.22× 不是 6.22 倍少 NFE

DiT-XL/2 在 ImageNet class-to-image 使用 DDIM-50、256×256、50,000 張影像與 NVIDIA H20。Z-Cache 的 N 是 cache/activation interval，不是總 NFE；F-Speed 以 50-step Full Transformer 的 FLOPs 為基準。N=5 時為 4.53×、FID 2.46；N=6 為 4.96×、FID 2.68；N=7 為 **6.22×、FID 3.25**。因此「6.22×」正確的說法是相對 50-step schedule 的 FLOPs/F-Speed，不是只用七個 NFE，也不能與 DDIM-10、DDIM-12 或 L2C 的 NFE 直接混稱。[2] [3]

### FLUX.1-dev：品質回復，亦揭示 FLOPs 與 wall-clock 的落差

在 DrawBench 200 prompts、1024×1024 的 FLUX.1-dev 實驗中，TaylorSeer N=8 的 6.23× F-Speed、PSNR 28.46、SSIM 0.5703、LPIPS 0.5190、ImageReward 0.9292，加入 Z-Cache 後為 6.22×、PSNR 28.71、SSIM 0.6002、LPIPS 0.4692、ImageReward 0.9451。這支持「以極低額外 Transformer FLOPs 換回品質」的定位。[2]

但同一表格也提醒不能只看 FLOPs：TeaCache N=8 從 1.87 秒變成 4.64 秒，peak memory 從 35.77 GiB 變成 39.99 GiB。作者因此明確說 Z-Cache 是 FLOPs-light 的 quality-recovery wrapper，而不是 universally latency-free。這是本文最重要的工程 caveat 之一。[2] [3]

### 影片、少步模型與編輯任務

HunyuanVideo 的 VBench 實驗包含 4,720 段、每段 65 frames、640×480；相對原始 50-step 模型，N=7/O=1 時 TaylorSeer 為 6.21×、VBench 79.09，Z-Cache 為 6.20×、VBench 79.60。這仍是相對 50-step 原始模型的 FLOPs/latency speedup，不是七個 NFE。[2]

在 Qwen-Image-Lightning 的 8-step sampler，TaylorSeer(2) 的 PSNR/SSIM/LPIPS 為 30.222/0.6734/0.2984，Z-Cache(2) 為 30.440/0.6785/0.2910。FLUX-Kontext-dev 的 Gedit-bench 也顯示 source-structure/edited-background fidelity 提升：TaylorSeer(8) 的 29.014/0.4338/0.4448，Z-Cache(8) 為 29.511/0.5114/0.3694。這些任務的指標不可與自由生成 FID、VBench 混作同一品質分數。[2]

### 軌跡消融：不是每一個 sample 都變好

在 32 個 frozen DrawBench prompts、每 prompt 2 seeds 的 trajectory control 中，global latent AUC 相對 TaylorSeer 在 N=6 由 0.212977 降至 0.198117，下降 6.98%；N=8 由 0.256287 降至 0.235652，下降 8.05%。但作者也報告 sample-wise final L2 只在 N=6 的 60/64 與 N=8 的 63/64 個 sample 改善，故不能把 prompt-mean 改善寫成每個樣本都改善。[2] [3]

## 相關研究：它到底新在哪裡

相對 FORA、TeaCache、ToCa、DuCa、TaylorSeer、SpeCa、MagCache 等方法，Z-Cache 的差異在於它不是重新設計 predictor，而是把「右端 full anchor」變成窗口內的反射訊號。相對 DDIM、DPM-Solver、rectified flow、consistency/distillation 等減少 sampling timesteps 的路線，它維持既有 schedule 中的 full anchor；相對 quantization、pruning、token merging，它主要改變 hidden-feature 的近似軌跡，而不是模型權重或 token 數量。[2]

它也不是 post-hoc smoothing：post-hoc 方法只修飾已完成的表示，Z-Cache 會從 replay endpoint 改變後續 latent。這個區分很關鍵，因為它讓「品質恢復」與「未來軌跡誤差」可以在同一個 sampling loop 中被測量；代價則是 scheduler、predictor state、window replay 與 hooks 的整合複雜度。

## 研究延伸：把 Z-Cache 接上使用者偏好的方向

以下都是**可檢驗延伸，不是論文已提出的主張**：

1. **Energy-based transformer。** 將 anchor feature 差異定義為跨時間 consistency energy，例如 $E_i=\|\hat{\mathcal H}_i^p-\hat{\mathcal H}_i^f\|$，用 energy threshold 動態選擇窗口長度與 $w_i$，比較固定 N、energy-adaptive N 與實際 hidden AUC、LPIPS、FLOPs、wall-clock。這能測試「何時應該 full check」是否可由誤差能量而非固定週期決定。

2. **JEPA 作 future-anchor estimator。** 以 frozen JEPA/latent predictor 預測未來語義 representation，作為 future-anchor gating 或與 DiT past-anchor hidden feature 融合；在 DrawBench 與 VBench 比較 Taylor extrapolation、JEPA prediction 與實際右端 full feature 的 drift、temporal consistency 和 prompt fidelity。這不是 Z-Cache 使用 JEPA，而是把它的「未來檢查」概念化成 representation prediction 問題。

3. **VAR 的 cache-then-check。** 在 visual autoregressive 的 scale/token window 先 speculative decode，再由後續尺度或 verifier state 作 future anchor，重播同一 token window。需報告 token NLL、FID、composition、decode latency，並正面處理離散 token replay 與 diffusion latent replay 的差異。

4. **Attention modulation。** 論文實際融合的是 block hidden feature bank，沒有主張調制 attention。後續可將雙錨點融合移到 attention output、KV 或 logits，讓 $w_i$ 變成 token-wise/region-wise；在 image editing 檢查 edited region/background trade-off，在 video 檢查跨 frame attention drift。

5. **嚴格 zero-shot 測試。** Z-Cache 是 training-free inference wrapper，但論文沒有把它命名為 zero-shot。可固定 predictor 與 cache interval，測試未見 prompt、不同 backbone、不同解析度與新任務，分開報告 zero-shot transfer、FLOPs、wall-clock、memory 和失敗率，避免把「無再訓練」誇大成普遍 zero-shot 能力。

## 限制與可重現性警告

第一，方法的理論保證依賴局部 Lipschitz 與 hidden-error 假設；它說明「若反射真的降低 hidden error，局部上界會更緊」，不等於普遍 convergence guarantee。第二，future $\mathcal H_b$ 是在 provisional $\bar x_b$ 上做 scheduled full，replay 時重用這個 feature bank，而不是對 replayed input 再做一次 full evaluation；這正是效率來源，也是真實 approximation error 的來源。[2] [3]

第三，論文與附錄沒有明確揭露 CFG、unconditional branch 或 guidance scale，因此本報告不把任何數字標為 CFG 或無 CFG；同理，N=5/6/7/8 是 cache interval/activation schedule，不是 NFE。第四，SenCache/D2Cache 組合是 audited clean-room FLUX ports，不是作者官方實作；D2Cache 指標混合，沒有 quality confidence intervals 或 human-preference study，不能把 point estimate 寫成統計顯著或 uniform dominance。[2] [3]

第五，能源主張只有 inference efficiency 的推論與 impact discussion，沒有 energy-per-sample、功耗或碳排直接測量。Impact Statement 同時提醒更快生成可能放大 misinformation、deceptive/non-consensual synthetic media、spam、bias 與 copyright 風險，並提出 access control、rate limiting、moderation/filtering、provenance 等治理方向。[3]

## 個人評價

我會把 Z-Cache 評為今日最值得讀的候選：它把 feature caching 從「預測下一個 hidden state」推進到「用未來 full computation 回看並修正已暫存窗口」，概念簡潔、可疊加、且在圖像、影片、編輯、U-Net/DiT、量化與少步模型上都有實驗接口。更重要的是，論文對 6.22× 的分母、FLOPs/wall-clock 差異、NFE 誤讀、sample-wise 非普遍改善與 CFG 未揭露都保留了必要的界線；這使它比單報一個最大 speedup 更適合作為研究起點。

但若要採用，我不會只複製 headline。第一個重現實驗應固定同一 GPU、同一 predictor、同一 scheduler，將 Transformer FLOPs、端到端 latency、peak memory、品質與 anchor/replay overhead 分開列出；第二個應測試 right-anchor 是否在不同 predictor 上真的降低 hidden error；第三個才是上述 energy-adaptive、JEPA future-anchor 或 attention-level reflection。若這三步成立，Z-Cache 才可能從漂亮的 cache wrapper 變成更一般的 inference-time trajectory control 原語。

## References

[1]: https://arxiv.org/abs/2611.01846 "arXiv 摘要頁：Z-Cache: Accelerating Diffusion Transformers via Self-Reflection"
[2]: https://arxiv.org/html/2611.01846v1 "arXiv HTML 原文 v1"
[3]: https://arxiv.org/pdf/2611.01846v1 "arXiv PDF 原文 v1"
[4]: https://neurips.cc/virtual/2026/poster/156017 "NeurIPS 2026 官方 poster 頁 156017"
[5]: https://neurips.cc/Downloads/2026 "NeurIPS 2026 官方 Downloads"
