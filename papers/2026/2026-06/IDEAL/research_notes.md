# IDEAL 論文研究筆記

## 基本資訊
- 論文：IDEAL: In-DEpth ALignment Makes A Discrete Representation AutoEncoder
- arXiv：2606.11096
- 提交日期：2026-06-09
- 作者：Yitong Chen, Zijie Diao, Junke Wang, Lingyu Kong, Yixuan Ren, Bo He, Yu-Gang Jiang, Zuxuan Wu
- 機構：Fudan University, Shanghai Innovation Institute, University of Maryland
- 類別：cs.CV
- 程式碼：https://github.com/Row11n/IDEAL

## 為何值得選
1. 非常新（2026-06-09）
2. 與使用者偏好高度相關：AR image generation、semantic tokenizer、zero-shot semantic preservation、VFM/RAE 路線
3. repo 尚未收錄 IDEAL
4. 在 ImageNet 256x256 上達到 autoregressive image generation 新 SOTA：gFID 1.89

## 核心問題
- 既有 RAE / VFM-based tokenizer 使用深層 VFM 特徵，語義強但細節重建差
- 離散化後低階視覺訊息更難恢復
- 淺層特徵有較好的外觀/結構細節，深層特徵有較好的語義，兩者存在深度上的互補性

## 核心方法
- 從 frozen VFM 取 shallow feature f^(s) 與 deep feature f^(d)
- 用單層 cross-attention 做 AttnFuse，以 deep 為 query、shallow 為 key/value
- 融合後表示 z 再做 vector quantization
- feature decoder 解碼後，用兩個 head 分別重建 shallow 與 deep 特徵
- reconstructed deep feature 再送入 pixel decoder 做影像重建
- loss = AE + VQ + deep alignment + shallow alignment

## 重要公式
- z = AttnFuse(f^(d), f^(s))
- L_deep = ||fhat^(d)-f^(d)||_2^2 + (1-cos(fhat^(d),f^(d)))
- L_shallow = ||fhat^(s)-f^(s)||_2^2 + (1-cos(fhat^(s),f^(s)))
- L = L_AE + L_VQ + L_deep + L_shallow
- AR likelihood: p_theta(y|c) = Π_t p_theta(y_t | y_<t, c)

## 主要結果
- rFID = 0.61（比前最佳好 0.28）
- rIS = 230.4
- zero-shot ImageNet Top-1 = 80.89%，Top-5 = 96.40%
- AR generation gFID = 1.89（3B model）
- Table 5 關鍵值：
  - Ideal-B: gFID 3.38
  - Ideal-L: gFID 2.26
  - Ideal-XXL: gFID 1.95
  - Ideal-3B: gFID 1.89

## 圖片/頁面對應
- PDF 第2頁：Figure 1，顯示 shallow/deep trade-off 與 PCA depth transition
- PDF 第4頁：Figure 2，整體架構圖；同頁 Table 1，layer-wise probing
- PDF 第7頁之後：應有 Table 2/3/4/5 與 Figure 3/4，待後續確認

## 已提取圖片
- 使用 pdf-image-extractor 從 PDF 提取至 assets/
- figures 數量約 137 張，需後續挑選最合適者作為報告插圖

## 初步評價
- 本文不是直接修改 Transformer block，而是從 tokenizer/representation 層面提升 AR generation 上限
- 最大價值在於把「語義保持」與「高保真重建」從二選一，改成透過深淺特徵對齊共同保留
- 對後續 JEPA / world model / semantic tokenization / next-scale AR 都有啟發性
- 特別適合用來思考：是否可把不同深度、不同模態、不同時間尺度的表徵，在量化前先做結構化融合

## 後續待做
1. 確認 Table 2/3/5 的完整實驗數字與對比 baseline
2. 讀取論文後半部與 appendix，找限制與 failure cases
3. 查找相關研究：RAE、VFMTok、VQRAE、VAR、OmniGen-AR
4. 撰寫 AI Daily markdown
5. 更新 README 與 push 到 GitHub

## 視覺檢視補充（來自 PDF 頁面）

### PDF 第7頁：Figure 3 + Table 2
Figure 3 展示重建結果，整體觀感顯示 IDEAL 能保留主要形狀、紋理與色彩分佈，但在人臉與細粒度內容上仍有少量模糊。Table 2 中，IDEAL 在系統層級重建上取得最優 rFID 0.61 與 rIS 230.4，且 codebook usage 為 100%。這比 VFMTok 的 0.89 rFID 與 215.4 rIS 更進一步，也顯著優於 LlamaGen 384 版本的 0.95 rFID 與 197.3 rIS。

### PDF 第8頁：Table 3 + Table 4
Table 3 顯示 IDEAL 的 decoded feature 可直接支援 zero-shot ImageNet 分類，Top-1/Top-5 分別為 80.89/96.40，接近原始 SigLIP2 的 83.23/97.11。這很重要，因為大多數離散 tokenizer 並不保留與文字嵌入對齊的語義空間。Table 4 進一步顯示 IDEAL 作為視覺編碼器在多模態理解 benchmark 上具有競爭力，例如 RealWorldQA 52.68、OKVQA 61.06、SEED 68.02、MME 1878，普遍優於 DINOv2 與 SigLIP2。

### PDF 第9頁：Table 5
Table 5 是論文最重要的 generation 結果。以 ImageNet 256x256 class-conditional generation 為例，IDEAL-B/L/XXL/3B 的 gFID 分別為 3.38 / 2.26 / 1.95 / 1.89。對照同尺度 baseline：VFMTok-B/L/XXL/3B 為 3.43 / 2.75 / 2.19 / 2.07；LlamaGen-B/L/XXL/3B 為 6.09 / 3.07 / 2.34 / 2.19。也就是說，IDEAL 在每個模型尺度都優於既有 AR tokenizer 路線，尤其在 Large 以上優勢很明顯。

### PDF 第10頁：Figure 4 + Table 6 + Table 7
Figure 4 展示 class-conditional generation 樣本，整體樣本已具有良好物體結構與畫面自然度。Table 6 做 256x256 controlled comparison：在相同 111M AR 模型下，IDEAL-B 的 gFID 為 3.43，優於 VFMTok-B 的 3.61 與 LlamaGen-B 的 5.46，同時重建品質也更好（rFID 0.98，rIS 220.0）。Table 7 ablation 顯示 attention fusion 優於 linear / no fusion；加入 shallow alignment loss 可從 0.66 rFID 改善至 0.61；不同 backbone 中 DINOv3 的 rFID 最低到 0.54，但作者仍選 SigLIP2 作為主幹，因其保留較強語義與 text-compatible 特性。

## 圖片素材候選
- page-02 / Figure 1：最能說明本文核心洞見（shallow/deep trade-off）
- page-04 / Figure 2：最適合作為方法架構圖
- page-09 / Table 5：最適合作為核心性能證據
- page-10 / Figure 4：可作為生成質量展示圖
