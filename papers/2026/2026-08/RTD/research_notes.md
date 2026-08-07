# RTD 研究筆記

## 論文基本資訊
- 標題：Rectify Then Diffuse: Disentangling Concepts Before Denoising Trajectory Unfolds
- arXiv：2608.03135
- 提交日期：2026-08-04
- 作者：Ning Zhu, An Chen, Mengfei Zhao, Juntao Xu, Jingze Liang, Boyuan Gu, Liang-Jian Deng
- 類別：cs.CV, cs.AI
- 核心方向：training-free、attention modulation、多概念 compositional text-to-image generation

## 為何選這篇
- 非常符合使用者近期偏好：training-free、attention modulation、zero-shot compositional generation。
- 核心視角新穎：將多概念生成失敗重新定義為「去噪前的邊界條件問題」，而不是傳統的「去噪過程中反覆控制軌跡」。
- 方法極簡：只在初始 latent 上做一次修正，之後完全回到原本 sampler，不需持續干預。
- 效率高：只增加一次 forward + backward，根據論文主文只增加 6.3% 推理開銷，且比 CO3 快 2.3 倍。

## 方法主線
### 1. Pilot Attention Extraction
- 在高噪聲 timestep t_pilot 做一次 diagnostic pass。
- 擷取每個概念 token 對應的 cross-attention map，聚合不同 head/layer 後得到概念級 attention map A^(k)。
- 觀察：即使圖像尚未成形，attention 已有初步空間組織；若多個概念在初期重疊，之後容易出現 omission / fusion。

### 2. Soft-Overlap Disentanglement (SOD)
- 先對每個概念 attention map 做 max-min normalization，得 soft occupancy map M^(k)。
- 概念 i, j 的 soft overlap：
  O_ij = <M^(i), M^(j)> / (||M^(i)||_1 + ||M^(j)||_1 - <M^(i), M^(j)> + epsilon)
- 分離目標：
  S(x_T) = 1 - (2 / (K(K-1))) * sum_{i<j} O_ij
- 直觀上：最大化 S，就是讓不同概念在初始空間支持區域更分離。

### 3. Isotropic Gradient Rectification (IGR)
- 對初始 latent x_T 反傳 S 的梯度 g = ∇_{x_T} S。
- 用 normalized gradient 做單步修正：
  g_hat = g / max(||g||_2, epsilon)
  x_T' = x_T + rho ||x_T||_2 g_hat
- 核心優點：用相對 latent norm 控制步長，使不同 prompt / seed 下的修正幅度更穩定。

## 關鍵公式
1. RTD 插入初始 latent 修正器：
   x_T' = R_theta(x_T, C), x_0' = Phi_theta(x_T', C)
2. Cross-attention：
   A = softmax(Q K_C^T / sqrt(d))
3. Soft overlap：
   O_ij = <M^(i), M^(j)> / (||M^(i)||_1 + ||M^(j)||_1 - <M^(i), M^(j)> + epsilon)
4. Separation objective：
   S(x_T) = 1 - (2 / (K(K-1))) sum_{i<j} O_ij
5. IGR update：
   x_T' = x_T + rho ||x_T||_2 * g / max(||g||_2, epsilon)

## 圖片觀察
### Figure 1（extracted_images/figures/figure_1_1.png）
- 主要是 qualitative comparison。
- 與 SDXL、CO3 相比，RTD 在多個 prompt 上更能同時保留兩個物體/屬性，而不是只生成其中一個、混成一個，或屬性錯配。
- 特別適合在報告中當作「動機 + 效果總覽」圖。

### Figure 3（extracted_images/figures/figure_3_2.png）
- 方法概覽圖，非常適合放進 AI Daily。
- 上半部：Vanilla SDXL 在 prompt「a red backpack and an orange glasses」中，兩個概念的 attention 長時間重疊，最後只剩 backpack。
- 中間：SOD 將概念 map 正規化後計算 pairwise overlap；IGR 以 normalized gradient 對初始噪聲做單步 rectification。
- 下半部：Rectified latent 後，attention 更早分開，最終成功生成「人 + 紅背包 + 橘色眼鏡」。
- 圖中直接標示平均 overlap 降低 29.5%（Vanilla SDXL 0.269 → RTD 0.170）。

### Figure 5（extracted_images/figures/figure_5_3.png）
- CO3 與 RTD 的定性對比。
- RTD 對以下 prompt 呈現更穩定：
  - shiny black shoes + worn brown boots
  - black keyboard on top of white desk
  - rectangular mirror next to blue sink
  - sunset behind skyscraper and park
  - green lettuce next to red tomatoes
  - white shirt on top of black pants
  - silver necklace on white silk
  - brown leather couch next to plush white pillow
- CO3 常見問題：只生成單一主物件、關係不對、屬性被吞掉或空間布局混亂。

### Figure 6（extracted_images/figures/figure_6_4.png）
- 顯示 attention evolution。
- 在 white shirt / black pants、brown couch / white pillow 的案例中，RTD 的 concept maps 從更早 step 就開始分離，最後兩個物件都保留。
- CO3 則在後期仍有明顯 coupling，導致只剩 shirt 或只剩 couch。
- 這張圖很有說服力，直接證明「早期 allocation bottleneck」假說。

### Figure 7（extracted_images/figures/figure_7_5.png）
- Sensitivity analysis。
- 最佳相對修正比例 rho 大約在 0.02 左右；太大（0.16）會顯著破壞表現。
- 最佳 pilot timestep 在高噪聲區域，主文預設 t_pilot=980 表現最好。
- 支持作者核心主張：應在 very early / pre-structure 階段處理 allocation conflict。

## 主要定量結果（主文）
### AE-Bench
- RTD 在 O-O subset 達到 BLIP-VQA 0.7503、ImageReward 1.2144。
- 相較 CO3：BLIP-VQA +45.8%，ImageReward +19.6%。
- 整體 early overlap（S-IoU_5）最低：RTD 0.2113，優於 CO3 0.2396。

### T2I-CompBench
- RTD：IR 0.4661、HE 0.6615、S-IoU_5 0.2146。
- 全面優於 CO3：IR 0.4406、HE 0.6278、S-IoU_5 0.2369。

### RareBench
- RTD 在 concat / relations / complex 都維持最佳或最強整體結果。
- early overlap 也最低：0.2078（CO3 為 0.2317）。

## 個人評價方向
- 這篇論文最值得記住的不只是效果，而是「問題重新表述」：把 compositional generation failure 看成是初始條件錯配，而非後續去噪軌跡中每一步都要救火。
- 對使用者偏好的 attention modulation / training-free 研究尤其有啟發：
  1. 只改 initial latent，而非改 model weights。
  2. 只做 one-shot correction，而非 step-wise intervention。
  3. 用 attention overlap 作為 proxy，讓空間分配可微且可優化。
- 對未來可延伸方向：
  - 與 VAR / next-scale prediction 結合，將「概念分離」從 latent diffusion 擴展到 coarse-to-fine token generation。
  - 結合 JEPA / energy-based view，把 early allocation 看成 latent energy landscape 的 boundary shaping。
  - 可思考是否能把 overlap objective 替換成更高階的 topology / relation-aware objective。

## 待補資料
- 需要再補 related work，至少包括 CO3、InitNO、Attend-and-Excite、SynGen、ToMe。
- 需要產生最終 AI Daily Markdown。
- 需要挑兩到三張圖複製到 repo asset/ 或該文章資料夾下，並在 README 補上條目。

## 來源
- arXiv abstract: https://arxiv.org/abs/2608.03135
- arXiv HTML: https://arxiv.org/html/2608.03135v1
- PDF: /home/ubuntu/AI_Daily/papers/2026/2026-08/RTD/RTD_paper.pdf
- Extracted figures directory: /home/ubuntu/AI_Daily/papers/2026/2026-08/RTD/extracted_images/figures/

