# AI Daily: VideoAR - 結合Next-Frame與Next-Scale預測的自回歸影片生成新篇章

> 論文標題：VideoAR: Autoregressive Video Generation via Next-Frame & Scale Prediction
> 
> 發表單位：ERNIE Team, Baidu
> 
> 發表時間：2026年1月9日
> 
> 論文連結：[https://arxiv.org/abs/2601.05966](https://arxiv.org/abs/2601.05966)
> 
> 關鍵詞：Visual Autoregressive (VAR), Video Generation, Next-Frame Prediction, Next-Scale Prediction, Training-Free

---

## 總結

近期，影片生成領域由Diffusion與Flow-Matching模型主導，雖然品質優異，但計算成本高昂且難以擴展。為此，百度ERNIE團隊提出了**VideoAR**，這是首個結合**多尺度Next-Frame預測**與**自回歸（Autoregressive, AR）建模**的大規模視覺自回歸（Visual Autoregressive, VAR）影片生成框架。VideoAR透過創新的架構，成功地在影片生成的品質與效率之間取得了新的平衡，不僅在AR模型中達到SOTA水準，其性能更能與大一個數量級的Diffusion模型相媲美，為影片生成領域提供了一個具備高擴展性、高效率且時間一致性的新研究方向。

![VideoAR生成範例](asset/videoar_page1_title.webp)
*圖1: VideoAR從文本提示生成高保真且時間一致的影片* 

---

## 核心貢獻

VideoAR框架的核心貢獻可歸納為以下三點：

1.  **創新的混合建模範式**：首次將VAR的「Next-Scale」預測能力與傳統影片生成中的「Next-Frame」預測相結合，有效解耦了影片中的空間紋理細節與時間動態，讓模型能更高效地學習。 

2.  **解決AR模型誤差傳播的關鍵技術**：提出了**Multi-scale Temporal RoPE**、**Cross-Frame Error Correction**與**Random Frame Mask**等技術，顯著緩解了AR模型在生成長影片時常見的誤差累積與內容漂移問題，大幅提升了時間一致性。

3.  **卓越的性能與效率**：在標準測試集上，VideoAR不僅在生成品質（FVD指標）上超越了先前的AR模型，更將推理速度提升了超過10倍。其VBench分數也達到了與頂尖Diffusion模型競爭的水平，證明了AR模型在影片生成領域的巨大潛力。

---

## 技術方法詳解

VideoAR的成功源於其精巧的架構設計，它將影片生成任務分解為幾個關鍵模組：

![VideoAR框架圖](asset/videoar_framework_figure2.webp)
*圖2: VideoAR整體框架圖。影片首先被3D Tokenizer壓縮為時空tokens，再由Transformer以自回歸方式預測，最後由解碼器重建。*

### 1. 3D多尺度視覺Tokenizer

為了高效處理時空資訊，VideoAR設計了一個3D多尺度Tokenizer。它能將影片幀壓縮成極為緊湊的潛在表示（latent token grid），其主要特點是：

-   **高壓縮率**：採用16倍的空間壓縮，相比於MAGVIT等模型的8倍壓縮，序列長度縮短了4倍，大幅降低了後續Transformer的計算負擔。
-   **高品質重建**：儘管壓縮率極高，其重建影片的品質（rFVD指標為61）依然能與MAGVIT（58）等頂尖模型持平，證明了其在保留關鍵時空結構上的高效性。

### 2. 混合式自回歸影片建模

VideoAR巧妙地結合了兩種自回歸策略：

-   **幀內 (Intra-Frame) - Next-Scale預測**：在單一影片幀內部，採用VAR的由粗到細（coarse-to-fine）的多尺度預測方式，高效生成高解析度的空間細節。
-   **幀間 (Inter-Frame) - Next-Frame預測**：在不同影片幀之間，採用傳統的因果預測（causal prediction），即根據前面所有幀來預測下一幀，確保影片的時間流暢性。

這種混合策略讓模型可以專注於各自的任務，空間生成與時間演進互不干擾，從而提升整體性能。

### 3. 提升時間一致性的關鍵技術

為了解決AR模型在長影片生成中「忘記」前面內容的頑疾，VideoAR引入了幾項創新：

-   **Multi-scale Temporal RoPE**：這是一種改進版的位置編碼，能讓模型更清晰地感知到跨越多個尺度的時間關係，提升了對時間動態的建模能力。
-   **Cross-Frame Error Correction**：在訓練過程中，模型會逐漸增加對預測錯誤的容忍度，並將這種「糾錯經驗」在幀與幀之間傳遞。這使得模型在推理時即使出現微小偏差，也能自我修正，避免誤差滾雪球式地放大。

![Cross-Frame Error Correction](asset/videoar_cross_frame_error_correction.webp)
*圖3: Cross-Frame Error Correction機制示意圖，透過在訓練中模擬誤差並繼承，提升模型的魯棒性。*

-   **Random Frame Mask**：訓練時隨機遮蔽某些歷史幀，強迫模型不過度依賴緊鄰的前一幀，而是學會從更廣泛的上下文中理解影片內容，從而緩解內容過度記憶的問題。

### 4. 數學公式解析

VideoAR的基礎建立在VAR模型之上。一個影格 $\mathbf{I}$ 首先被編碼為特徵圖 $\mathbf{F}$：

$$\mathbf{F} = \mathcal{E}(\mathbf{I})$$

然後，Quantizer $\mathcal{Q}$ 將其分解為K個多尺度殘差圖 $\mathbf{R}_{1:K}$。Transformer的核心任務是自回歸地預測下一個尺度的殘差：

$$p(\mathbf{R}_k | \mathbf{R}_{1:k-1}, \Psi)$$

其中 $\Psi$ 是文本提示。為了預測 $\mathbf{R}_k$，模型會聚合所有先前生成的殘差，並通過上採樣與下採樣操作來匹配目標解析度：

$$\hat{\mathbf{F}}_{k-1} = \text{down}(\sum_{i=1}^{k-1} \text{up}(\mathbf{R}_i, (H,W)), (H_k, W_k))$$

這種由粗到細的生成過程是VAR高效生成高品質圖像的關鍵。

---

## 實驗結果

VideoAR在多項基準測試中都取得了令人矚目的成績，充分證明了其設計的有效性。

![性能比較表格](asset/videoar_performance_tables.webp)
*表1, 2, 3: VideoAR在VBench、Tokenizer性能和UCF-101上的比較結果。*

-   **UCF-101數據集**：VideoAR-XL（2B參數）模型取得了**88.6**的FVD分數，顯著優於先前的SOTA AR模型PAR-4x（99.5）。更重要的是，其推理步數減少了超過10倍，生成單個影片僅需0.86秒，速度提升13倍以上。
-   **VBench綜合評測**：VideoAR的總分達到了**81.74**，特別是在**語義一致性（Semantic Score）**上取得了**77.15**的最高分，這表明其生成的影片內容與文本提示高度吻合，超越了所有對比模型。
-   **真實世界影片生成**：模型能夠生成384x672解析度、長達4秒的高保真影片，且表現出清晰的規模效應（scaling behavior）——即模型越大，效果越好。

![生成結果](asset/videoar_generation_results.webp)
*圖4: VideoAR在VBench和UCF-101數據集上的生成結果，展現了多樣的場景與動作。*

![I2V與V2V擴展](asset/videoar_image_to_video_extension.webp)
*圖5: VideoAR在Image-to-Video和Video-to-Video擴展任務上的視覺化結果。*

---

## 相關研究背景

VideoAR的出現並非偶然，它建立在近年來自回歸模型在視覺領域飛速發展的基礎之上。從早期的PixelRNN/CNN到後來的VQ-VAE，再到近期引領風潮的VAR模型（如LlamaGen、MAGVIT-v2），研究者們一直在探索如何將AR模型強大的序列建模能力應用於高維的視覺數據。相較於在NLP領域的巨大成功，AR在視覺上的應用始終面臨著計算量大、空間相關性建模難、誤差易累積等挑戰。VideoAR正是站在這些巨人的肩膀上，針對影片生成的獨特性質，提出了一個兼具效率與品質的解決方案。

---

## 個人評價與意義

VideoAR無疑是影片生成領域一個里程碑式的工作。它打破了Diffusion模型在SOTA性能上的壟斷地位，並以極高的效率證明了AR模型作為另一條技術路線的巨大潛力。這項研究最大的啟示在於**「解耦」**的思想：將複雜的時空生成任務分解為相對獨立的空間紋理生成和時間動態演進，並為各自設計最適合的模型範式。這種「分而治之」的策略，不僅提升了模型性能，也極大地增強了可擴展性。

對於追求高效、可控影片生成的研究者和開發者而言，VideoAR提供了一個極具吸引力的替代方案。它的成功可能會激發更多基於AR的影片生成、編輯甚至模擬的研究，特別是在需要與大型語言模型基礎設施無縫集成的應用場景中。未來，我們或許會看到一個AR與Diffusion模型並駕齊驅、甚至在某些領域AR模型更佔優勢的影片生成新時代。
