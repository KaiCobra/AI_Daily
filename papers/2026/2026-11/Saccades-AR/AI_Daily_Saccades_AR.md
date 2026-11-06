# AI Daily

**日期：2026-11-06**

**作者：Manus AI**

## 精選論文

**Context-Aware Autoregressive Image Generation for Emerging Reasoning Properties**

作者：Jixuan Ying、Haoyu Liu、Timing Yang、Tingyu Zhu、Predrag Neskovic、Anand Bhattad、Cihang Xie、Zeyu Zheng、Alan Yuille、Feng Wang。作者機構包括 Johns Hopkins University、University of California, Berkeley、University of California, Santa Cruz，以及 Office of Naval Research。[1] [2]

論文於 2026 年 11 月 4 日公開，已列入 NeurIPS 2026 官方 poster 議程，並非僅有預印本或仍在審稿中的投稿。[1] [3] 作者程式碼已公開。[4]

**一句話摘要：**SAR 把「下一個生成哪個位置」從固定或隨機順序，改成由生成條件與目前影像表徵共同決定；在不改變部分既有 AR 模型權重的情況下，推理期間優先生成與整體條件最相關的 patch 或高解析度尺度位置。[2]

## 問題與核心貢獻

視覺自回歸生成通常採用預先指定的順序。Masked AR 模型常依 confidence 逐步解除遮罩，VAR 類模型則由粗到細預測尺度。這些策略不一定會在最適合的位置先補上資訊。SAR 的主張是，生成順序可以依當前部分影像及文字或類別條件動態改變，以支援物件數量、相對位置與屬性綁定等組合需求。[2]

論文把同一個選擇原則套用到兩種生成方式：patch-wise 的 MaskGen／MAR，以及 scale-wise 的 VAR／Infinity。每一步先得到尚未完成位置的隱藏表徵，再計算它們與條件 anchor 的相關性，提交一批分數最佳的位置。它不是新的 tokenizer 或生成 backbone，而是重新安排既有模型的生成軌跡。[2]

## 技術方法與公式

自回歸模型將圖像 token 序列分解成逐步條件分布。對 patch-wise MAR，可寫為

$$
p_\theta(z\mid c)=\prod_{t=1}^{T}p_\theta\!\left(z_{S_t}\mid z_{S_{<t}},c\right),
$$

其中 $c$ 是文字或類別條件，$S_t$ 是第 $t$ 步提交的 token 集合。VAR 類 scale-wise 模型則可寫成

$$
p_\theta(z\mid c)=\prod_{k=1}^{K}p_\theta\!\left(z^{(k)}\mid z^{(<k)},c\right),
$$

其中 $z^{(k)}$ 代表第 $k$ 個解析度尺度。SAR 不只決定下一個尺度，也能在特定模型中進一步決定同一尺度內哪些位置先生成。[2] [5]

在 patch-wise 設定，令 $h_i^{(t)}$ 為目前第 $t$ 步、位置 $i$ 的表徵，$A^{(t)}$ 為條件 anchor 表徵集合。SAR 以平均 cosine similarity 評分：

$$
s_i^{(t)}=\frac{1}{|A^{(t)}|}\sum_{a\in A^{(t)}}\frac{\langle h_i^{(t)},a\rangle}{\|h_i^{(t)}\|\,\|a\|}.
$$

分數較高的位置較符合條件，因此先生成。對 scale-wise 模型的最高解析度尺度，論文改用與 anchor 的平方距離：

$$
s_i^{(m)}=\|h_i^{(m)}-a\|_2^2,
$$

此時分數較低代表表徵較接近 anchor，故選距離最小的位置。兩者評分方向不同，是因為前者使用相似度、後者使用距離；不能把它們簡化成同一種「分數越大越好」規則。[2]

對 MaskGen，SAR 在每個步驟為尚未生成的 patch 打分，再解除一批遮罩位置。對 VAR／Infinity，論文保留先前的粗尺度，並把最高解析度尺度拆成多個子步驟，逐批選擇下一個要生成的位置。這種方法的主要開銷來自多次前向推理；在同樣的多步預算下，Random order 與 SAR 的平均延遲相近，但相較只用單一步驟會更慢。[2]

## 實驗結果

**GenEval 組合生成評估。**每個 prompt 生成 4 張圖，採 3 個 seed 平均。Random → SAR 的 GenEval pass rate 為：MaskGen-L 53.94% → 57.49%；MaskGen-XL 56.22% → 59.99%；Infinity-2B 71.79% → 72.57%；Infinity-8B 80.63% → 81.34%。這支持在多個既有模型上調整生成順序可提高該 benchmark 的組合生成分數。[2]

**DPG-Bench。**Overall 分數由 MaskGen-L 的 70.11 升至 70.51、MaskGen-XL 的 71.00 升至 71.57、Infinity-2B 的 82.71 升至 82.76、Infinity-8B 的 86.09 升至 86.27。增幅並非每個子類別都一致，因此較準確的結論是整體分數小幅提升，而非所有語義維度全面改善。[2]

**MJHQ-30K 品質指標。**結果大致持平：MaskGen-L 的 FID 為 6.79 → 6.79，MaskGen-XL 為 6.75 → 6.74，Infinity-8B 為 9.332 → 9.317；Infinity-2B 則由 10.867 變為 10.970。這表示組合能力提升沒有穩定轉化為較低 FID；論文把多數差異視為近似持平。[2]

**受控 MaskGIT 比較。**在同一 MaskGIT-Base checkpoint、相同解碼步數和 CFG 設定下，只替換位置選擇器，ImageNet 256×256 的 FID 從 6.48 降至 5.95，Recall 從 0.41 升至 0.43。這是較直接隔離「選擇順序」效果的實驗。[2] [7]

作者另建 2,000 個長 prompt 評測集，涵蓋數量、空間關係、屬性綁定、存在性與視角一致性，並由 Gemini 2.5 Pro 依問題評分。MaskGen-XL 平均分為 2.42 → 2.92，Infinity-2B 為 2.62 → 3.12，Infinity-8B 為 2.92 → 3.42。這是模型評審分數，不是人類評測或標準公共 benchmark，解讀時需保留這項限制。[2]

**VAR ImageNet 與推理成本。**論文報告 class-conditional ImageNet 256×256 上，VAR-d16 FID 3.56 → 3.48、VAR-d20 2.77 → 2.73；另一些結果使用 order-agnostic 或 SAR-order retrained checkpoint，含訓練設定變化，不能將所有增益歸因於純推理時重排。單張 H100 上，VAR-d16 從 one-step 的 0.073 秒／張增加到 matched four-step 的 0.093 秒／張；因此 SAR 的價值是用額外生成步驟換取品質與組合能力，而不是免費加速。[2]

## 相關研究

**VAR：Scalable Image Generation via Next-Scale Prediction。**VAR 以粗到細的尺度分解取代傳統逐 token raster 順序。SAR 延續其多尺度生成骨架，進一步在最高解析度尺度內安排位置順序；因此兩者的差異在於 SAR 的動態選擇器，而不是重新提出 next-scale generation。[5]

**Infinity：Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis。**Infinity 是論文的 scale-wise 文字生圖載體之一。SAR 保留 Infinity 的既有尺度生成流程，再以條件 anchor 排列最高解析度尺度的生成位置。[2] [6]

**MaskGIT。**MaskGIT 透過反覆解除遮罩生成圖像，典型策略依 token confidence 選位置。SAR 的 MaskGIT 控制實驗只替換 selector，因而能比較 confidence-based 選擇與條件相似度選擇的差異。[2] [7]

## 評價與研究啟發

這篇論文最值得延伸的觀點，是把「生成順序」當作推理時可調整的計算資源分配策略，而不是固定的模型細節。它與 VAR 的關聯直接，也能延伸到不同生成式模型的 test-time routing。不過，證據支持的是特定 benchmark 上的組合能力與部分品質增益，不是一般視覺推理已被解決。尤其長 prompt 結果仰賴單一 VLM judge；部分 VAR ImageNet 結果又涉及重新訓練，兩者都應與純 inference-time 結果分開看。[2]

對 EBT、JEPA 與 zero-shot 方向，可提出三個可檢驗的後續問題。第一，把 anchor similarity 與條件能量分數結合：相似度衡量語義相關性，能量項則偏好與既有上下文相容的位置，再以固定步數的 Random、SAR 與 Energy-SAR 做公平比較。第二，讓 JEPA predictor 預測下一尺度的 latent target，以預測一致性或不確定度決定生成順序；這可檢驗「最相關的位置」是否也等於「最能降低後續錯誤的位置」。第三，建立嚴格的 zero-shot 評估，將物件數量、空間關係與 prompt 模板留作訓練外組合，並同時報告生成成功率、FID／人類偏好及每張圖延遲。這些是基於論文提出的研究假設，不是原論文已驗證的結果。

## References

[1]: https://arxiv.org/abs/2611.06838 "Context-Aware Autoregressive Image Generation for Emerging Reasoning Properties — arXiv"
[2]: https://arxiv.org/html/2611.06838 "Context-Aware Autoregressive Image Generation for Emerging Reasoning Properties — Full Text"
[3]: https://neurips.cc/virtual/2026/poster/151846 "NeurIPS 2026 Official Poster"
[4]: https://github.com/Hpesojyjx/Saccades-AR "Saccades-AR Official Repository"
[5]: https://arxiv.org/html/2404.02905 "Scalable Image Generation via Next-Scale Prediction"
[6]: https://arxiv.org/abs/2412.04431 "Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis"
[7]: https://arxiv.org/abs/2202.04200 "MaskGIT: Masked Generative Image Transformer"
