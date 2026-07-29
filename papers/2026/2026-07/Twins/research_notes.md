# Twins 論文研究筆記

## 選題理由
- 論文：**Twins: Learn to Predict Unified Representations with Focal Loss**
- arXiv: 2607.22531，提交日期 2026-07-24
- 會議：**ICML 2026**
- 團隊：Tencent Hunyuan 相關作者群，方向與使用者偏好的統一視覺表示、圖像生成、flow matching 高度相關。
- 與 repo 已有文章比對後，尚未收錄 Twins。

## 為何值得寫
- 不只是提出新模型，而是直指「**理解表徵**」與「**生成表徵**」長期割裂的核心問題。
- 將 **ViT semantic features** 與 **VAE latents** 在同一 token grid 上做 channel-wise concatenation，避免增加 token 長度，因此不增加 attention 的平方級 token 成本。
- 真正的貢獻不是單純拼接，而是指出 DiT 在聯合建模時會出現明顯的 **optimization imbalance**：先學會低頻、低 intrinsic dimension、與條件高度對齊的 ViT 特徵，卻學不好高頻、較高 intrinsic dimension 的 VAE 特徵。
- 用一個很簡單但有洞察力的 **feature-level focal regression** 去修補這個不平衡，對使用者偏好的「激發想法」很有價值。

## 已讀到的核心技術點
1. **Twins 表徵定義**
   - 設影像為 \(I\)，ViT encoder 輸出 \(f_{vit}(I) \in \mathbb{R}^{L \times d_{vit}}\)
   - VAE encoder 輸出 \(f_{vae}(I) \in \mathbb{R}^{L \times d_{vae}}\)
   - 兩者共享相同 token grid，故可做
     \[
     \mathbf{z} = [f_{vit}(I), f_{vae}(I)]
     \]
   - 得到 \(\mathbf{z} \in \mathbb{R}^{L \times (d_{vit}+d_{vae})}\)

2. **Flow Matching 基本式**
   - 中間狀態：
     \[
     \mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1
     \]
   - 標準 MSE velocity matching：
     \[
     \mathcal{L}(\theta)=\mathbb{E}[\|\mathbf{v} - \mathbf{v}_\theta(\mathbf{x}_t,t)\|^2]
     \]
   - 其中 \(\mathbf{v}=\mathbf{x}_1-\mathbf{x}_0\)

3. **Focal regression for VAE channels**
   - 原始 VAE channel loss：
     \[
     \mathcal{L}_{mse}=\frac{1}{d_{vae}}\sum_{i\in D}(v_i-v_\theta(z,t)_i)^2
     \]
   - 權重：
     \[
     w_i = |v_i - v_\theta(z,t)_i|^{2\gamma}
     \]
   - 最終：
     \[
     \mathcal{L}=\frac{1}{d_{vae}}\sum_{i\in D} w_i (v_i-v_\theta(z,t)_i)^2
     \]
   - 論文設定 \(\gamma = 0.5\)

4. **作者對 imbalance 的三個診斷**
   - 頻率偏置：SigLIP 偏低頻，VAE 含較多高頻細節
   - 內在維度：SigLIP 單類 intrinsic dimension 約 15，VAE 約 35，更難學
   - 條件依賴：SigLIP 在條件下更接近 deterministic，VAE 保留更多 condition-independent uncertainty

## 目前抓到的實驗數字
### 重建
- Twins: **PSNR 31.46 / SSIM 0.90 / rFID 0.11**
- RAE: PSNR 18.83 / rFID 0.57
- 說明 Twins 明顯補上 semantic-only representation 在高頻重建上的不足

### 理解能力
- Twins (SigLIP2 + Flux.2 VAE, Qwen2.5-7B):
  - POPE 87.82
  - GQA 64.93
  - TQA 58.89
  - MMB 77.00
  - MME-S 1971.0
  - MME-P 1588.8
- 相比只用 SigLIP2，GQA / TQA 等細粒度任務更好

### 圖像生成
- 論文摘要稱：在 ImageNet 上，相比 naive MSE，**最高提升 10.57 gFID**（without classifier-free guidance）
- 256x256 實驗：Twins + Focal Loss 在多組設定下都顯著優於 Twins + MSE
- 512x512 實驗：Twins + Focal Loss with guidance 可達 **gFID 1.79**

## 圖片檢視筆記
- `extracted_images/figures/figure_1_1.png`
  - 很適合放在報告中作為核心導讀圖。
  - 內容是「Understanding / Reconstruction / Generation」三角張力圖，Twins 位在中央，表示同時滿足三者。
- 其他抽出的部分圖片是生成樣例 patch，不一定適合作為報告主圖。
- 若需要第二張圖，可能更適合直接從 PDF 再擷取方法總覽或實驗圖，而不是用目前被拆分的小 patch。

## 潛在延伸思考
- 這篇其實不是單純 tokenizer paper，而是在問：**生成與理解是否應該共享同一個 continuous latent space？**
- Focal weighting 是否可進一步做成：
  - 頻率感知 weighting
  - uncertainty-aware weighting
  - layer-wise / timestep-wise adaptive weighting
- 與使用者偏好的 energy-based / JEPA 聯想：
  - 是否可把 Twins 視為 joint latent space，再用 JEPA 或 energy-based objective 取代部分 flow matching 監督？
  - 是否能在 unified latent 中同時做 prediction 與 generation，而非只做 denoising / flow matching？

## 待補資料
- 作者機構與 GitHub/project page 的更完整背景
- 相關研究段落：RAE、UniFlow、UniLip、Show-o2、Janus-Pro、TokenFlow
- 如可行，補一張方法示意圖或實驗圖（非碎片 patch）
- 最終報告檔名與 README 更新格式需對齊 repo 既有風格
