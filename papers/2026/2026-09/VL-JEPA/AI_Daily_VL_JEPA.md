# AI Daily

## VL-JEPA：以語義嵌入預測取代 token 生成，讓 Vision-Language 模型具備選擇性解碼能力

> **一句話結論：** VL-JEPA 將 vision-language learning 的主要監督訊號從離散文字 token 移到連續語義嵌入，讓同一個模型在共享 latent space 中支援 zero-shot 分類、影片檢索、判別式 VQA 與文字生成；它最有價值的不是宣稱全面取代生成式 VLM，而是把「先預測語義、需要時才解碼」變成一個可測量的模型介面，為 JEPA、Energy-based Transformer、attention modulation 與低成本線上視覺理解提供了清楚的研究切入點。

## 一、論文基本資料與選稿理由

| 欄位 | 資訊 |
|---|---|
| 論文標題 | **VL-JEPA: Joint Embedding Predictive Architecture for Vision-language** |
| 正式發表 | **ICLR 2026 conference paper** |
| arXiv | [arXiv:2512.10942](https://arxiv.org/abs/2512.10942)，本文主要閱讀 v2，2026-02-02；正式 camera-ready PDF 另核對作者與會議資訊 [1] [2] |
| 正式版本作者 | Delong Chen、Mustafa Shukor、Théo Moutakanni、Willy Chung、Jade Yu、Tejaswi Kasarla、Allen Bolourchi、Yann LeCun、Pascale Fung |
| 研究單位 | Meta FAIR、HKUST、Sorbonne Université、NYU；camera-ready PDF 亦列出 University of Amsterdam 與 USC 的作者 affiliation |
| 研究主題 | JEPA、vision-language model、latent embedding prediction、zero-shot classification、video retrieval、selective decoding |
| 報告日期 | 2026-09-14 |
| Repo 排除結果 | 在本 repo 中未找到以 VL-JEPA、arXiv:2512.10942 或同名為主題的獨立 AI Daily 文章；既有內容可能在 related work 中提到它，因此本文表述為「未有獨立收錄」，而非「從未被提及」。 |

這篇論文符合本日追蹤方向中的 **JEPA、zero-shot 與 attention/推理效率**，也能自然連接到 Energy-based Transformer 與 Visual Autoregressive Model。候選篩選時，我同時檢查了 NRGPT、FreqFlow、Low-Pass Flow Matching 與 VarKD 等未收錄工作。VL-JEPA 最適合本日深讀的原因，是它同時具備正式 ICLR 發表、完整數學定義、跨任務實驗與公開的架構圖，而且能把「非生成式表徵預測」從影像或影片 encoder 擴展到一般 vision-language 任務。

需要先界定論文的結論範圍。VL-JEPA **不是 training-free 方法**，因為它需要大規模預訓練與 supervised fine-tuning；它也不是 Energy-based Transformer、VAR 或完整的 action-conditioned world model。它的 zero-shot 結果主要是 classification 與 retrieval，而不是 zero-shot open-ended generation。Selective decoding 的 2.85 倍是解碼操作數減少，不應直接改寫成端到端 latency 或吞吐量提升。

## 二、問題背景：VLM 為何需要 latent-space prediction？

傳統生成式 Vision-Language Model 接收視覺輸入 $X_V$ 與文字查詢 $X_Q$，在 token space 中自回歸產生目標文字 $Y$。其訓練目標可以抽象寫成

$$
\mathcal{L}_{\mathrm{VLM}} = D(\hat{Y},Y),
$$

其中 $\hat{Y}$ 是模型生成的 token 序列。這個目標對需要完整文字輸出的 captioning 或 open-ended VQA 很自然，但同一個視覺狀態可能有多個語義等價的答案。例如「燈被關掉」與「房間會變暗」都可能是正確回答。在 one-hot token space 中，兩個句子可能幾乎沒有共享 token，因此會被視為相距很遠的目標。模型便被迫同時學習任務語義、詞彙選擇、句型、風格與表面 paraphrase。

VL-JEPA 的核心觀察是：對許多線上視覺任務而言，模型真正需要先得到的是「目前輸入代表什麼」，而不是立刻完成完整的 token-by-token 文字生成。如果先在一個能保留語義、又能將不同表述映射到鄰近位置的 latent space 預測目標，再把文字 decoder 作為按需 readout，就有機會把昂貴的文字生成從常駐監控迴路中移除。

因此論文將問題改寫為：

$$
(X_V, X_Q) \longmapsto S_Y,
$$

而非直接學習 $(X_V,X_Q)\mapsto Y$。這個改寫具有兩個效果。第一，目標分佈從稀疏、離散且可能多峰的 token space，轉為較平滑的連續語義 space。第二，視覺串流可以持續產生語義 embedding，只有在 embedding 發生顯著變化時，才呼叫文字 decoder。

## 三、核心架構與數學方法

### 3.1 四個組件：X-Encoder、Predictor、Y-Encoder、Y-Decoder

VL-JEPA 以訓練三元組 $\langle X_V,X_Q,Y\rangle$ 為輸入，包含以下四個組件：

| 組件 | 映射 | 作用 |
|---|---|---|
| X-Encoder | $X_V\mapsto S_V$ | 將影像或影片壓縮為連續視覺 embedding 序列 |
| Predictor | $(S_V,X_Q)\mapsto \hat S_Y$ | 根據視覺內容與文字 query，預測目標語義 embedding |
| Y-Encoder | $Y\mapsto S_Y$ | 將 ground-truth 文字目標轉為訓練 target embedding |
| Y-Decoder | $\hat S_Y\mapsto \hat Y$ | 只在需要人類可讀文字時，把預測 embedding 解碼回文字 |

架構的最小數學描述是

$$
S_V=f_{\mathrm{X}}(X_V),\qquad S_Y=f_{\mathrm{Y}}(Y),
$$

$$
\hat S_Y=g_{\theta}(S_V,X_Q),
$$

並以 embedding-space loss 取代 token-space cross-entropy：

$$
\mathcal{L}_{\mathrm{VL\text{-}JEPA}}
= D(\hat S_Y,S_Y).
$$

![VL-JEPA 核心四組件：視覺與查詢共同預測目標語義 embedding](../../../../asset/2026-09-14-vl-jepa/vl-jepa-core-formulation.png)

*圖 1。從 camera-ready PDF 擷取的核心 formulation。$Y$ 經 Y-Encoder 得到 $S_Y$，作為 Predictor 輸出 $\hat S_Y$ 的訓練目標；Y-Decoder 不是主要訓練路徑，而是按需 readout。圖片由本文從論文 PDF 擷取，原始論文見 [2]。*

### 3.2 為何使用雙向 InfoNCE？

本文採用雙向 InfoNCE，因為它同時提供 **alignment** 與 **uniformity**。為了清楚說明，可將 batch 中第 $i$ 個樣本的預測與 target 正規化為

$$
\tilde p_i=\frac{\hat S_{Y,i}}{\|\hat S_{Y,i}\|_2},
\qquad
\tilde y_i=\frac{S_{Y,i}}{\|S_{Y,i}\|_2}.
$$

若 batch size 是 $B$、溫度參數是 $\tau$，prediction-to-target 方向可寫成

$$
\mathcal{L}_{p\rightarrow y}
= -\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp(\tilde p_i^{\top}\tilde y_i/\tau)}
{\sum_{j=1}^{B}\exp(\tilde p_i^{\top}\tilde y_j/\tau)}.
$$

反向的 target-to-prediction 損失為

$$
\mathcal{L}_{y\rightarrow p}
= -\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp(\tilde y_i^{\top}\tilde p_i/\tau)}
{\sum_{j=1}^{B}\exp(\tilde y_i^{\top}\tilde p_j/\tau)}.
$$

最終可表示為

$$
\mathcal{L}_{\mathrm{bi\text{-}InfoNCE}}
=\frac{1}{2}\left(
\mathcal{L}_{p\rightarrow y}+
\mathcal{L}_{y\rightarrow p}
\right).
$$

第一項使正確 prediction-target pair 的相似度上升，第二項使 target embedding 也能反向辨識對應 prediction。batch 內的負樣本與 softmax 分母提供分散表示的壓力，降低所有 embedding 塌縮到同一點的風險。論文也指出，VICReg、SIGReg、EMA target encoder 或 frozen Y-Encoder 都是可能的替代 anti-collapse 設計，但沒有在本文中完整展開 [2]。

這一點很值得和 Energy-Based Transformer 對照。若把

$$
E(\hat S_Y,S_Y)=-\frac{\tilde p^{\top}\tilde y}{\tau}
$$

視為 pairwise compatibility energy，InfoNCE 就可以看成以 batch negatives 形成的相對能量排序。然而，VL-JEPA **沒有宣稱自己是 explicit EBM**，也沒有建立 partition function、MCMC negative phase 或可證明的 energy descent。這個 energy 解讀是可用於後續研究的統一視角，不應冒充本文已完成的理論結果。

### 3.3 實際模型配置

本文的主要實作如下：

| 模組 | 實作與規模 | 重要設定 |
|---|---|---|
| X-Encoder | frozen V-JEPA 2 ViT-L，304M parameters | 影片均勻取樣；視覺輸出再投影為 visual embeddings |
| Predictor | Llama-3.2-1B 的最後 8 層 Transformer，約 490M trainable parameters | 使用文字 query embedding；取消 causal mask，使 visual/query token 可雙向共同注意 |
| Y-Encoder | EmbeddingGemma-300M 初始化 | 最高 512 context tokens；學習率 multiplier 設為 0.05 |
| Shared space | Predictor 與 Y-Encoder 的 projection head | 1536 維共享語義空間 |
| Y-Decoder | 輕量文字 readout | 不參與主要 VL-JEPA 預訓練；需要輸出文字時才使用 |

Predictor 不是單純的 visual encoder connector。它先讓 visual embeddings 與 query embeddings 進入 Transformer，再對非 padding token 做平均池化，最後投影到 Y-Encoder 的共享空間。這個設計使同一個預測 embedding 能以不同方式被使用：若任務是分類，就和候選標籤的 Y-Encoder embedding 比相似度；若任務是 retrieval，就和文字 retrieval query 比相似度；若任務需要自然語言回答，才呼叫 Y-Decoder。

![VL-JEPA 詳細架構與三種任務介面](../../../../asset/2026-09-14-vl-jepa/vl-jepa-architecture-and-tasks.png)

*圖 2。論文架構圖顯示 Predictor、Y-Encoder 與三種 downstream interface：selective decoding、判別式 VQA/分類，以及 text-to-video retrieval。圖片由本文從論文 PDF 擷取，原始論文見 [2]。*

### 3.4 統一多任務介面

對於 open-vocabulary classification 或 discriminative VQA，模型先從視覺與 query 得到 $\hat S_Y$，再將候選答案或類別文字 $c$ 透過 Y-Encoder 得到 $S_c$，最後選擇距離最小者：

$$
\hat c=\arg\min_{c\in\mathcal{C}}
D\left(\hat S_Y,f_Y(c)\right).
$$

若使用 cosine similarity，等價於選取與 $\hat S_Y$ 最相近的候選 embedding。這使 VQA 的一部分評測更接近「在候選答案空間做 semantic matching」，而不是自由生成答案。這個差異很重要：論文的 VQA 結果不能直接解讀為模型已經具備與大型 generative VLM 相同的開放式 reasoning 能力。

對 text-to-video retrieval，對每支候選影片 $v$ 計算

$$
S_v=g_{\theta}(f_X(v),X_Q^{\mathrm{retrieval}}),
$$

再依照 $S_v$ 與 retrieval query embedding 的相似度排序。這種介面結合了 CLIP 式的 embedding retrieval 與 JEPA 式的 conditional prediction。CLIP 直接把影像與文字獨立映射到共享空間，並透過對比學習支援 zero-shot classification；VL-JEPA 則先根據視覺輸入與 query 預測 target embedding，因此具有條件化的語義預測路徑 [3]。

## 四、訓練流程與計算規模

VL-JEPA 使用兩階段訓練。第一階段是 query-free 的大規模 caption pretraining，目標是建立穩定的 vision-language alignment。影像資料來自 DataComp 與 YFCC-100M，影片資料使用 Action100M。作者先以單影格影像、有效 batch size 24,000 訓練 100k iterations，模型看過約 2B samples；之後加入 8-frame 與 32-frame video training。整個 pretraining 約使用 24 個節點、每個節點 8 張 NVIDIA H200，耗時約四週。

第二階段是 query-conditioned supervised fine-tuning，資料包含約 25M VQA、2.8M captioning、1.8M classification samples，以及為避免 catastrophic forgetting 而保留的部分 pretraining data。SFT 訓練約 83k steps、batch size 3,072。第一階段得到的模型稱為 **VL-JEPA BASE**，主要評估 zero-shot classification 與 retrieval；第二階段得到 **VL-JEPA SFT**，主要評估 VQA、action anticipation 與其它需要 domain supervision 的任務。

因此，論文「效率」的主要意義不是訓練成本很低。相反地，模型訓練規模仍然很大。它的效率主張是：在相同或相近 supervision budget 下，latent target 比 token target 更有效率；在部署時，embedding stream 可以讓 decoder 不必每一個時間點都執行。

## 五、實驗結果與性能指標

### 5.1 Zero-shot 影片分類與 text-to-video retrieval

在 8 個影片分類與 8 個影片檢索資料集上，VL-JEPA BASE 的平均結果如下。這裡的 zero-shot 是指下游資料集不再對模型做任務特定訓練，並不表示整個模型從未經過大規模預訓練。

| 模型 | 參數量 | Seen samples | 分類平均 Top-1 | Retrieval 平均 Recall@1 |
|---|---:|---:|---:|---:|
| CLIP ViT-L | 389M | 12.8B | 30.7 | 35.9 |
| SigLIP2 ViT-g | 1.9B | 40B | 39.8 | 38.9 |
| PE-Core ViT-G | 2.3B | 86B | 44.7 | 58.1 |
| **VL-JEPA BASE ViT-L** | **1.6B** | **3.3B** | **52.5** | **63.7** |
| VL-JEPA SFT ViT-L | 1.6B | 3.6B | 75.4 | 63.8 |

VL-JEPA BASE 的平均分類準確率比表中最佳 generalist baseline PE-Core 高 7.8 個百分點，平均 retrieval Recall@1 高 5.6 個百分點。這個結果值得注意，但不能只看數字而忽略資料量差異：PE-Core 約看過 86B samples，VL-JEPA BASE 約看過 3.3B samples。VL-JEPA 特別擅長 SSv2、EK-100、EgoExo4D，以及 COIN/CrossTask 的 step-recognition 等 motion-centric benchmark；在 appearance-centric 的 Kinetics-400 與部分 task-recognition benchmark 相對弱一些。

### 5.2 VQA 與 WorldPrediction-WM

VL-JEPA SFT 在四個 VQA benchmark 的 accuracy 為：GQA **61.5**、TallyQA **69.9**、POPE **85.7**、POPEv2 **86.3**。這些數字與多個 7B 至 72B 的生成式 VLM 相比具有競爭力，而 VL-JEPA 本身標示的規模為 1.6B parameters。不過這組評測採用候選答案 embedding matching，主要測量視覺感知、計數與 hallucination resistance，不應直接外推到自由形式的多步 reasoning、tool use 或 agentic behavior。

WorldPrediction-WM 的任務是給定初始與最終世界狀態影像，再從四個候選影片中挑出能解釋狀態變化的動作。VL-JEPA BASE 為 **63.9%**，SFT 為 **65.7%**，論文稱 SFT 達到該 benchmark 的新高 [2]。但這不是完整的 action-conditioned latent rollout。它是將初末狀態與候選動作轉成 embedding 後做判別式匹配，因此更精確的描述是「world-state transition candidate selection」，而不是已經完成可長時域規劃的世界模型。

### 5.3 與 token prediction 的嚴格對照

作者用 Perception Encoder 作為共同 frozen visual backbone，固定影像解析度、影格數、資料、batch size、learning-rate scheduler、訓練迭代與評估 checkpoint。唯一主要差異是 prediction target：VL-JEPA 用約 0.5B predictor 預測 target embedding；token VLM baseline 用約 1B LLM 進行 next-token cross-entropy。

| Seen samples | VL-JEPA caption CIDEr | Token VLM caption CIDEr | VL-JEPA classification Top-5 | Token VLM classification Top-5 |
|---:|---:|---:|---:|---:|
| 500K | 1.23 | 1.35 | 14.9% | 14.0% |
| 5M | 14.7 | — | 35.3% | — |
| 15M | **14.8** | **7.1** | **41.0%** | **27.2%** |

在 500k samples 時兩者相近，但 VL-JEPA 的 learning curve 在後續上升更快。15M samples 時，VL-JEPA 在 caption CIDEr 約為 token baseline 的 2.08 倍，classification Top-5 高 13.8 個百分點。這是本文最能支持「latent prediction 改善 sample efficiency」的實驗。

然而，這個 controlled comparison 應被理解為「共同 backbone 與訓練設定下，改變 prediction objective 的效應」，而不是完全消除所有 implementation 差異。Predictor 初始化、decoder 路徑、embedding target 的品質與各模型的最佳超參數仍可能影響結果。因此，更嚴謹的結論是：在作者設計的 matched setting 下，embedding prediction 顯示出較好的學習曲線與最終分數；仍需要不同資料、模型規模、seed 與獨立實作驗證。

### 5.4 Selective decoding：2.85 倍究竟代表什麼？

對長影片串流，VL-JEPA 會持續產生 embedding stream。作者以 temporal connectivity constraint 的 agglomerative clustering，將 embedding sequence 分成語義相對一致的 segments，再在每個 segment 的 midpoint 解碼。average pooling 也可先對 segment 內的 embedding 去噪，再送入 decoder。

![VL-JEPA selective decoding：以 embedding stream 決定何時呼叫文字 decoder](../../../../asset/2026-09-14-vl-jepa/vl-jepa-selective-decoding.png)

*圖 3。論文 selective decoding 圖。實驗以 EgoExo4D validation set 的 218 支影片、平均每支約 6 分鐘、每支約 143 個 atomic action annotations 評估。圖片由本文從論文 PDF 擷取，原始論文見 [2]。*

在平均解碼頻率 sweep 中，uniform decoding 以固定間隔呼叫 decoder，selective decoding 則依照 embedding stream 的變化決定 segment 數量。作者報告：selective decoding 約 **0.35 Hz**、平均間隔約 **2.85 秒** 時，可以接近 uniform decoding **1 Hz** 的 CIDEr，因而減少約 **2.85 倍 decoder operations**。

這個結果的實際含義是「在相近品質下，需要更少次文字 decoder 呼叫」。論文的 controlled latency experiment 顯示單次生成延遲仍可相近；VL-JEPA 的優勢在於可以把 encoder/predictor 與 decoder 解耦，對 classification、retrieval 或串流語義監控只跑前者。要把 2.85 倍轉成端到端 latency、GPU energy 或 wall-clock throughput，還需要把 encoder refresh rate、clustering overhead、memory、I/O 與 decoder batch scheduling 一起測量。

### 5.5 消融實驗告訴我們什麼？

消融結果支持三個設計判斷。第一，去掉 caption pretraining 會讓 classification 平均下降 **21.7** 個百分點，retrieval 平均下降 **17.3** 個百分點，說明 SFT 不能取代前面的 vision-language alignment。第二，Y-Encoder learning-rate multiplier 約 0.05–0.10 最穩定；設為 0 或 1.0 都會退化。第三，InfoNCE 整體優於 cosine、L1 與 L2，因為它同時提供 alignment 與 anti-collapse 的 batch-level regularization。

Predictor 的 bidirectional attention 也很重要。若保留原本 causal attention，視覺 token 不能充分看到後面的 query token，VQA 下降約 1.9 個百分點。這是一個和使用者關注的 attention modulation 很接近的細節：注意力遮罩不只是工程設定，而是決定模型能否把 query 條件傳回視覺表示的結構性選擇。

## 六、相關研究脈絡

### 6.1 從 I-JEPA 到 VL-JEPA

I-JEPA 將單張影像切成 context block 與 target block，使用 context encoder 與 predictor 預測 target encoder 的 representation。它的關鍵主張是，在 representation space 預測影像的可預測結構，比重建每一個像素更能聚焦於高層語義；target encoder 的 exponential moving average 也被用來穩定學習 [4]。

VL-JEPA 沿用「predict representation、不要重建所有表面細節」的基本精神，但把 target 從影像 patch 改成文字 embedding，並加入文字 query 條件。這個改變讓 JEPA 從單模態表徵學習變成可處理 caption、retrieval、分類與 VQA 的多任務接口。差異也在於本文採用雙向 InfoNCE、聯合訓練 Predictor 與 Y-Encoder，而不是把 EMA target encoder 作為主要實作。

### 6.2 V-JEPA 2：視覺表徵、latent dynamics 與 zero-shot planning

V-JEPA 2 先用網路規模影片做 masked representation prediction，再以少量 interaction data 訓練 action-conditioned predictor。V-JEPA 2-AC 凍結視覺 encoder，預測動作條件下的未來影片 representation，進一步用於 robot planning 與 zero-shot control [5]。

VL-JEPA 使用 V-JEPA 2 ViT-L 作為 X-Encoder，並在 WorldPrediction-WM 上做狀態轉移候選判別。兩者的關係應該精確區分：V-JEPA 2-AC 是 latent dynamics model，VL-JEPA 是 conditional semantic target predictor；VL-JEPA 的 WorldPrediction-WM 實驗尚未證明 action-conditioned long-horizon rollout。這個差異正好指向下一步研究：讓 VL-JEPA 的 semantic embedding stream 與 V-JEPA 2-AC 的 state transition latent space 對齊，再用 energy 或 uncertainty 決定是否更新世界模型。

### 6.3 CLIP 與生成式 VLM 的位置

CLIP 以 image encoder 與 text encoder 將配對影像—文字拉近、未配對樣本推遠；推理時把類別名稱或描述轉成文字 embedding，便能建立 zero-shot classifier [3]。VL-JEPA 繼承了 shared embedding space 的 zero-shot 優勢，但不再只做 image/text 的獨立配對，而是以 $(S_V,X_Q)$ 預測條件化 target embedding，因而可在同一模型中兼顧分類、retrieval 與文字 readout。

相對地，生成式 VLM 直接以 next-token cross-entropy 學習文字輸出。VL-JEPA 不會自動擁有生成式 VLM 的所有能力，因為它的語義 embedding 可能丟失精確詞序、引用內容與長鏈 reasoning 所需的離散細節。這也是論文結論中承認仍需擴充 reasoning、tool use 與 agentic behavior 評估的原因。

## 七、對使用者偏好方向的研究啟發

### 7.1 Energy-based Transformer：把 semantic compatibility 變成可監控 energy

InfoNCE 的正樣本相似度可以被視為一種相對 compatibility energy，但 VL-JEPA 沒有把它推導為完整 EBM。下一步可以定義

$$
E_t=-\frac{\operatorname{sim}(\hat S_{Y,t},S_{Y,t})}{\tau}
+\lambda\,R_{\mathrm{uniformity}}(S_{Y,t}),
$$

再用 energy difference

$$
\Delta E_t=E_t-E_{t-1}
$$

作為串流語義是否改變的訊號。若 $|\Delta E_t|$ 很小，系統維持 latent monitoring 而不呼叫 decoder；若 compatibility energy 突然改變，才啟動 readout。這比單純以 embedding variance 做 clustering 更接近可解釋的 reliability gate。

更進一步，可以讓 Transformer block 的 attention logits 加入 state-dependent energy bias，或把 prediction-target mismatch 當作 inference-time preconditioner。這會把「語義預測」與 Energy-based Transformer 的「在能量地形上更新狀態」接起來，但需要新的穩定性分析，不能直接從 VL-JEPA 的 InfoNCE 得到 energy descent 保證。

### 7.2 JEPA：從單一 target embedding 到多時間尺度 predictive critic

本文的 target 是一個文字語義 embedding。對影片或機器人，可以把 target 拆成不同時間尺度：短期 motion、物件狀態、長期 task intent。令

$$
S_Y^{(k)}=f_Y^{(k)}(Y^{(k)}),
\qquad
\hat S_Y^{(k)}=g_\theta^{(k)}(S_V,X_Q),
$$

並以多尺度 loss

$$
\mathcal{L}=\sum_k w_k\mathcal{L}_{\mathrm{InfoNCE}}^{(k)}
$$

訓練。短期 embedding 可以觸發即時 action update，長期 embedding 則只在事件或目標改變時更新。這會使 VL-JEPA 的 selective decoding 從「何時產生文字」推進到「何時更新不同層級的世界語義」。

### 7.3 VAR-based：把 semantic target 接到 coarse-to-fine token hierarchy

VAR 的生成順序天然具有 coarse-to-fine scale。可以讓每一個尺度 $s$ 的 visual token group 預測一個 semantic target $S_Y^{(s)}$，而不是只在最後一次生成完整文本。模型可同時學習

$$
\hat S_Y^{(s)}=g_\theta(S_{\le s},X_Q),
$$

並以跨尺度 consistency penalty

$$
\mathcal{L}_{\mathrm{scale}}
=\sum_{s=1}^{L-1}
\left\|\operatorname{proj}(\hat S_Y^{(s)})-
\operatorname{proj}(\hat S_Y^{(s+1)})\right\|_2^2
$$

約束粗尺度先確立物件/場景語義，細尺度再補外觀與關係。這個方向可以把 VL-JEPA 的 semantic stream 與 VAR 的 scale-wise state 結合，研究在不完整 decoding 下能否先判斷 prompt 是否已被滿足，或在早期尺度就停止不必要的採樣。

### 7.4 Training-free 與 attention modulation：清楚劃定可延伸邊界

VL-JEPA 本身不是 training-free。它最接近 training-free 的地方，是 inference 時只依賴已學到的 embedding geometry 來選擇何時解碼。因此一個可行的後續實驗是凍結 X-Encoder、Predictor 與 Y-Encoder，只額外計算一個不更新參數的 gate：

$$
q_t=\alpha\,\|\hat S_{Y,t}-\hat S_{Y,t-1}\|_2
+\beta\,E_t
+\gamma\,\operatorname{Var}(\hat S_{Y,t-w:t}),
$$

當 $q_t>\delta$ 才解碼或調節 attention。這屬於 inference policy，不是重新訓練 backbone；但必須在 zero-shot transfer、不同影片長度、domain shift 與校準誤差上驗證，才可稱為 robust training-free control。

在 attention modulation 方面，論文已顯示取消 causal mask、讓 query 與 visual embeddings 雙向注意，對 VQA 有實質影響。下一步可測試 query-conditioned temperature、head-wise gate 或 KV scaling：當 embedding uncertainty 高時，提高與 query 對應的 cross-modal heads；當 semantic stream 穩定時，降低 decoder-triggering 頻率。這比單純 post-hoc threshold 更接近模型內部的注意力調制。

### 7.5 Zero-shot：已完成的範圍與尚未完成的問題

本文確實展示了 zero-shot classification 與 text-to-video retrieval，並以 Y-Encoder candidate matching 產生判別式結果。然而，它沒有展示 zero-shot open-ended captioning 的通用 reasoning 能力，也沒有以未見 action space 做完整 latent planning。下一步應加入新的類別詞彙、跨資料集影片、未見 prompt template，以及 calibration/error detection，並比較 CLIP、SigLIP2、V-JEPA 2 與 VL-JEPA 在同一資料與 compute budget 下的 transfer。

## 八、個人評價與限制

### 8.1 我認為最重要的貢獻

第一，VL-JEPA 把「文字生成」拆成兩個階段：先得到可用於決策的 semantic embedding，再按需求產生 surface text。這使同一個模型的 latent state 可以直接服務 classification、retrieval 與 discriminative VQA，而不是所有任務都經過昂貴的 token decoder。

第二，論文沒有只報一個大型 benchmark，而是做了相對有說服力的 controlled comparison。共同 visual encoder、資料、batch 與訓練條件下，latent prediction 在 15M samples 的 caption 與 classification 表現優於 token prediction，這為「embedding supervision 可能更 sample-efficient」提供了直接實驗證據。

第三，selective decoding 把 embedding space 轉成可部署的操作策略。2.85 倍不是抽象的 representation claim，而是對解碼頻率與 CIDEr 的 Pareto curve；即使它還不是完整端到端加速，研究問題已被定義得足夠清楚。

### 8.2 需要保留的限制

第一，訓練成本仍然很高。四週、24 個節點、每節點 8 張 H200 的 pretraining 不能被描述為 lightweight training。論文的效率優勢主要發生在 prediction objective 與 inference decoder usage，而非整個開發流程都便宜。

第二，zero-shot 表格存在資料量不對稱。VL-JEPA 的 3.3B samples 低於 PE-Core 的 86B，這既是其 sample efficiency 的正面訊號，也意味著不同資料混合、影像/影片比例與 text quality 可能影響比較。需要 matched compute、matched data quality 與多次 seed 才能把因果結論說得更強。

第三，selective decoding 的 clustering 看起來需要整段 embedding stream 才能做 temporal agglomeration。若目標是嚴格線上、因果的 streaming inference，還要測量新 frame 到來時的增量更新成本、segment merge/split 延遲與錯過事件的風險。2.85 倍 operation reduction 不等於 2.85 倍總系統加速。

第四，VQA 與 WorldPrediction-WM 多數是 candidate matching。這個介面很適合檢驗語義對齊，但不等同於自由生成、長鏈推理或可控制世界模型。論文自己也承認 reasoning、tool use 與 agentic behavior 尚未充分評估。

第五，版本 metadata 存在差異。arXiv v2 列出 10 位作者並包含 Yejin Bang；camera-ready PDF 列出 9 位作者並使用 Jade Yu，且正式 ICLR proceedings HTML metadata 對 Yu 的名字顯示也有不一致。本文以 camera-ready PDF 的「Published as a conference paper at ICLR 2026」與作者列為正式版本依據，同時保留 arXiv 版本差異。這不改變方法與實驗的核心判讀，但寫作時不能把不同版本作者表混為一談。

### 8.3 對研究方向的總評

我會把 VL-JEPA 評為一篇 **對 JEPA 與多模態推理介面很有啟發、但不應被過度包裝為通用 VLM 替代品** 的論文。它的真正價值在於提出一個乾淨的中間層：這個中間層足以支援語義決策，又可以在需要時接上文字 decoder。對使用者目前關注的方向而言，最值得延伸的是「將 latent prediction error、embedding disagreement 與 attention entropy 統一為可校準的 energy/reliability score」，再將它用於 training-free gating、VAR scale scheduling 或 zero-shot control。

## 九、可立即實驗的三個研究題目

| 題目 | 最小可行實驗 | 主要評估 |
|---|---|---|
| **Energy-Gated VL-JEPA** | 凍結 VL-JEPA，將 cosine compatibility energy 與 temporal embedding variance 組成 decoder gate | 同等 CIDEr 下的 decoder operations、端到端 latency、energy、missed-event rate |
| **Scale-wise JEPA for VAR** | 在 VAR 每個尺度加入 semantic target head，訓練 coarse-to-fine latent consistency | FID、GenEval/文本對齊、早停比例、不同尺度的 prompt satisfaction |
| **Causal Selective JEPA** | 以線上 EMA/滑動統計取代 offline agglomerative clustering，禁止回看未來 frames | streaming delay、memory、event recall、gate calibration、domain shift |

其中第一個題目最接近 Energy-based Transformer 與 training-free attention modulation 的交集。若能證明同一個 compatibility energy 同時可作為 decoder trigger、attention temperature 與 uncertainty proxy，就可能得到比單純 threshold 更通用的 inference-time control layer。

## 十、結論

VL-JEPA 的核心不是「把文字 decoder 拿掉」，而是把多模態模型的主要中間產物重新定義為 **可被多任務消費的語義 embedding**。訓練時，模型以 JEPA-style prediction 在 latent space 對齊視覺條件與文字 target；推理時，這個 embedding 可直接做分類、retrieval 或候選答案匹配，只有需要人類可讀輸出時才解碼。

本文最可靠的實驗結論有三個。第一，在作者控制的 training setting 下，embedding prediction 比 token prediction 展現更好的 sample efficiency。第二，VL-JEPA BASE 在 8 個 zero-shot video classification 與 8 個 retrieval benchmark 的平均結果高於作者選用的 generalist baselines。第三，embedding-guided selective decoding 在相近 CIDEr 下減少約 2.85 倍 decoder operations。

最需要避免的三個誤讀也有三個。第一，它不是 training-free；第二，2.85 倍不是自動等於端到端 2.85 倍加速；第三，WorldPrediction-WM 的 candidate matching 不是完整 action-conditioned world-model rollout。

對未來研究而言，VL-JEPA 提供了一個很好的接口：以 semantic prediction 產生可監控狀態，再用 energy、attention 或 scale-wise policy 決定何時更新與何時生成。這正是它與 Energy-based Transformer、JEPA world model、VAR、training-free inference、attention modulation 與 zero-shot transfer 之間最值得繼續挖掘的交集。

## References

[1]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/9144aded4e536bc5fb7bfc660a2d7a3d-Abstract-Conference.html "ICLR 2026 Proceedings: VL-JEPA"

[2]: https://arxiv.org/html/2512.10942v2 "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language, arXiv HTML v2"

[3]: https://arxiv.org/html/2103.00020 "Learning Transferable Visual Models From Natural Language Supervision, CLIP"

[4]: https://arxiv.org/html/2301.08243 "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture, I-JEPA"

[5]: https://arxiv.org/html/2506.09985 "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"

[6]: https://arxiv.org/abs/2512.10942 "VL-JEPA arXiv abstract and version metadata"

[7]: https://proceedings.iclr.cc/paper_files/paper/2026/file/9144aded4e536bc5fb7bfc660a2d7a3d-Paper-Conference.pdf "VL-JEPA camera-ready paper PDF, ICLR 2026"

---

**作者：** Manus AI  
**研究日期：** 2026-09-14
