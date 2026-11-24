# AI Daily

2026-11-24

## 論文資料

**Plan Big, Paint Small: Semantic-Aware Computation Allocation for Efficient Autoregressive Video Generation**；作者 Yixuan Luo、Yanjing Li。arXiv v1 發表於 2026-11-20，分類 cs.CV。官方紀錄未列會議或期刊接收，本文將它視為**尚未證實正式 venue 的預印本**。arXiv 頁面未列作者所屬機構，故不臆測研究單位背景。[1] [2]

## 核心貢獻

自回歸影片逐段生成，若每一段都使用大型模型，延遲很高；若全程用小型模型，語意與動態品質又可能下降。本文提出 BigSmall：讓 14B 模型負責影片開場，以及偵測到新內容時的「草圖」步驟；1.3B 模型則渲染其餘片段。兩個模型共享 latent space 與去噪排程，因此可以在同一段生成過程中交接。作者另設計近似、視窗化、延遲建立的跨模型 KV cache，避免大型規劃器每段都必須重新處理完整歷史。[1]

它是**推論時免訓練的模型分工策略**，不是新的單一生成模型，也不需要為分工再訓練模型；但仍須有相容的大小模型組合，且變化門檻是在保留提示詞上校準。因此，論文沒有宣稱「不需任何校準的嚴格零樣本」。[1]

## 技術方法

令 $B$ 為大型規劃器、$S$ 為小型渲染器，$F_c$ 表示第 $c$ 個影片片段由大型模型負責的去噪步數。第 $k$ 步的乾淨 latent 預測為：

$$
\hat{x}_{0,c}^{(k)} =
\begin{cases}
B(x_c^{(k)}, t_k \mid \mathcal{C}_B), & k \le F_c, \\
S(x_c^{(k)}, t_k \mid \mathcal{C}_S), & k > F_c.
\end{cases}
$$

其中 $x_c^{(k)}$ 是第 $c$ 段在去噪步 $k$ 的狀態，$t_k$ 是時間步，$\mathcal{C}_B$ 與 $\mathcal{C}_S$ 是各模型各自的 KV cache。首段設 $F_0=4$，由大型模型完整規劃；若後續片段需要引入新內容，設 $F_c=1$，讓大型模型先定下布局，再由小模型完成；其餘片段設 $F_c=0$，只用小模型渲染。[1]

是否需要「草圖」由前兩個已生成片段的特徵變化決定。將前一段各 token 與前前段附近 $\pm2$ 個位置的 token 配對，計算殘差 $r_c$，再定義：

$$
s_c = \operatorname{mean}\!\left(\operatorname{MaxPool}_{3\times3}(r_c)\right),
\qquad s_c \ge \tau \;\Rightarrow\; \text{呼叫大型模型作草圖}.
$$

門檻 $\tau$ 以保留提示詞校準，使目標跳過率為 70%；每兩段至多連續跳過兩次，提示詞切換後的前兩段則強制呼叫大型模型。這個規則用片段層級的視覺變化控制昂貴計算，不是每幀或每個去噪步都重跑大型模型。[1]

![論文 Figure 4 局部：同一個「小熊洗碗」提示詞在不同影片時間點的生成比較；BigSmall 列標示 10.0 FPS，其他方法為 4.6–6.5 FPS。這是論文質性示例，不能單獨代表整體評測。來源：[1]](../../../../asset/2026-11-24-plan-big-paint-small/fig4-bigsmall-teddy-bear.png)

## 實驗結果

標準 VBench 測試使用 946 個 extended prompts、每個提示詞五個 seed，輸出 832×480、81 幀影片；計時在單張 RTX PRO 6000 Blackwell 96-GB GPU 上以 BF16 和未編譯的 PyTorch SDPA 執行。BigSmall 的 DiT 時間是每支 8.11 秒、10.0 FPS；Krea-Realtime-14B 基線為 24.50 秒、3.3 FPS，換算為 **3.02× DiT 加速**。語意分數為 82.18 對 82.21，配對差 +0.01，95% CI 為 −0.22 至 +0.26；動態分數為 74.2 對 74.7，多物件分數為 92.3 對 92.5。[1]

這裡的計時**不含 VAE 解碼**；納入 VAE 後，論文報告的端到端加速為 2.34×。此外，語意差的區間跨過 0，只表示這組資料沒有明確顯示差異，不能單靠它宣稱已完成預先設界的統計等效性檢定。[1]

在 LoCoT2V-Bench 的長影片測試中，234 個段落提示詞平均生成約 47 秒影片。BigSmall 對 14B 基線達 3.14× 加速，平均分數高 1.78（配對 95% CI：+0.89 至 +2.67）。較快的 BigSmall-Fast 只在第一段使用 14B，達 3.77× VBench 加速，但在動態提示詞上的 RAFT 平均峰值光流比 14B 低 9.8%，呈現速度與動態保真之間的取捨。[1]

## 相關研究

Self Forcing 以模型自身生成的影片作為後續片段上下文，處理訓練與推論間的 exposure bias；BigSmall 使用同一類自回歸渲染器，但研究焦點不同：它是在既有生成流程中分配大小模型的計算。[1] [3]

SRDiffusion 也讓大型與小型模型合作，不過它依去噪階段分工，由大型模型處理高噪聲步驟、小型模型負責低噪聲細節；BigSmall 則沿影片時間軸分工，著重大模型何時需要再次規劃新內容。[1] [5] SDVG 採小模型先草擬、再由大模型驗證，BigSmall 則先以大模型建立語意，再讓小模型延續，形成另一種計算配置。[1] [4]

## 評價與可延伸想法

本文最有啟發性的觀察是：自回歸生成的計算預算不必平均分配到每個片段。若大模型的主要價值是建立場景與新事件，便可把昂貴步驟留給「內容改變」時刻，而把穩定延續交給小模型。論文同時提供長影片結果和配對信賴區間，讓效能取捨比單報 FPS 更容易判讀。[1]

限制也明確：核心結果建立在 Krea-Realtime-14B 與 Self-Forcing-1.3B 這組共用 latent space／排程的模型上；門檻需要提示詞校準；主要速度數字不含 VAE；測試計時使用特定 GPU 與軟體設定。不同模型家族能否互換、跨域門檻能否免校準，仍需要實驗回答。[1]

可供後續研究的方向（以下為延伸構想，非本文結果）：

1. **JEPA 事件觸發器：**以 JEPA 的未來表徵預測誤差取代局部殘差，區分鏡頭運動與真正的物件／場景變化；比較相同大模型呼叫率下的漏觸發率與影片品質。
2. **VAR 的尺度分工：**把「大型模型規劃新內容、小型模型延續」搬到 visual autoregressive next-scale 生成，測試大模型只產生粗尺度或新語意 token、小模型補細尺度是否保留構圖與細節。
3. **EBT 能量門控：**用能量或信心作為呼叫大型模型的觸發分數，檢驗它能否比固定閾值更穩健地控制算力；尤其應測低變化但語意關鍵的事件。
4. **免校準泛化與注意力調制：**在不更新權重下，把觸發器移至新領域，並於提示詞切換時重置或調節跨片段注意力／快取；以未參與校準的提示詞檢驗泛化，避免把提示詞校準後的推論策略稱為嚴格零樣本。

## References

[1]: https://arxiv.org/html/2611.28402v1 "Plan Big, Paint Small: Semantic-Aware Computation Allocation for Efficient Autoregressive Video Generation, arXiv v1 full text"
[2]: https://arxiv.org/abs/2611.28402 "arXiv record for Plan Big, Paint Small"
[3]: https://neurips.cc/virtual/2025/poster/116208 "Self Forcing, NeurIPS 2025 poster"
[4]: https://arxiv.org/abs/2604.17397 "Speculative Decoding for Autoregressive Video Generation"
[5]: https://arxiv.org/abs/2505.19151 "SRDiffusion: Accelerate Video Diffusion Inference via Sketching-Rendering Cooperation"
