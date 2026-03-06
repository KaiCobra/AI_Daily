# AI Daily: VAREdit - 指令引導圖像編輯的VAR新標竿
**論文標題**: Visual Autoregressive Modeling for Instruction-Guided Image Editing
**作者**: Qingyang Mao, Qi Cai, Yehao Li, Yingwei Pan, Mingyue Cheng, Ting Yao, Qi Liu, Tao Mei
**機構**: HiDream.ai Inc., University of Science and Technology of China
**發表時間**: 2026年2月6日 (ICLR 2026)
**arXiv編號**: [2508.15772](https://arxiv.org/abs/2508.15772)
**代碼**: [https://github.com/HiDream-ai/VAREdit](https://github.com/HiDream-ai/VAREdit)

---

## 核心貢獻與創新點

VAREdit 是首個專為**指令引導圖像編輯（Instruction-Guided Image Editing）**設計的**視覺自回歸（VAR）**框架。相較於主流的擴散模型（Diffusion Models）方法，VAREdit 通過其獨特的**次世代預測（Next-Scale Prediction）**範式，從根本上解決了擴散模型因全局去噪過程導致的「意外編輯」和「指令遵循度不足」的問題。VAREdit 不僅在編輯精準度和背景保留上超越了現有的 SOTA 方法，更在效率上實現了巨大突破，一張 512x512 的圖像編輯僅需 **1.2 秒**，比同等規模的 UltraEdit 快 **2.2 倍**。

其核心創新在於提出了 **Scale-Aligned Reference (SAR) 模塊**：

> 為了有效解決在僅使用最精細尺度（finest-scale）的源圖像特徵進行條件化時，產生的嚴重尺度不匹配問題（scale mismatch），我們引入了 SAR 模塊。該模塊將尺度匹配的條件化信息注入到第一個自註意力層中，從而彌合了高頻細節可能干擾粗糙目標特徵預測的鴻溝。

這一設計使得 VAREdit 能夠在保持高效率的同時，精準地捕捉和執行編輯指令，實現了前所未有的編輯質量。

![VAREdit 編輯範例](asset/VAREdit_demo_gallery.webp)
*圖1: VAREdit 在多種複雜編輯任務中的出色表現，包括物體增減、屬性修改和風格轉換，同時完美保留了背景細節。*

---

## 技術方法簡述

VAREdit 的整體架構建立在 **Infinity** 模型 [2] 的基礎之上，採用多尺度視覺權杖化器（multi-scale visual tokenizer）和一個基於 Transformer 的生成模型。其核心是將圖像編輯問題重構為一個**條件化的次世代預測問題**。

### VAREdit 的條件化生成過程

模型在生成目標圖像的每一個尺度（scale）的殘差圖 `R_k(tgt)` 時，會同時考慮三個條件：
1.  **先前已生成的目標殘差圖** `R_1:k-1(tgt)`
2.  **原始的源圖像** `I(src)`
3.  **文字編輯指令** `t`

$$ p(R_{1:K}^{(tgt)} | I^{(src)}, t) = \prod_{k=1}^{K} p(R_k^{(tgt)} | R_{1:k-1}^{(tgt)}, I^{(src)}, t) $$

### Scale-Aligned Reference (SAR) 模塊

為了在高效的「僅使用最精細尺度條件化」和效果好的「全尺度條件化」之間取得平衡，作者通過對自註意力機制的分析發現，模型對尺度感知條件的敏感性主要集中在**第一個自註意力層**。基於此，他們設計了 SAR 模塊：

1.  **動態生成參考特徵**: SAR 模塊通過對最精細尺度的源特徵圖 `F_K(src)` 進行降採樣，動態生成與目標尺度 `k` 相匹配的參考特徵 `F_k(ref)`。
2.  **精準注入**: 在預測目標尺度 `k` 的 token 時，僅在**第一個自註意力層**中，將查詢 `Q_k(tgt)` 與這個新生成的、尺度對齊的參考特徵 `F_k(ref)` 以及先前已生成的目標 token 歷史進行註意力計算。
3.  **後續層級**: 在所有後續的自註意力層中，模型僅關註最精細尺度的源特徵 `F_K(src)` 和目標的歷史殘差，以進行局部細節的精煉。

![VAREdit 架構圖](asset/VAREdit_architecture.webp)
*圖2: VAREdit 整體架構圖，展示了 SAR 模塊如何在第一個自註意力層中注入尺度對齊的參考信息。*

這種混合策略使得 VAREdit 能夠以極高的效率捕捉到關鍵的尺度依賴性，從而實現了精準且高效的編輯。

---

## 實驗結果與性能指標

VAREdit 在多個標準圖像編輯基準測試（EMU-Edit, PIE-Bench）上均取得了 SOTA 性能，尤其是在 GPT-4o 評分的 **GPT-Balance**（編輯成功率和背景保留度的調和平均值）指標上，VAREdit-8B 分別比之前的最佳開源模型高出 **64.9%** 和 **45.3%**。

| 方法 | 模型大小 | EMU-Edit (GPT-Bal.) | PIE-Bench (GPT-Bal.) | 推理時間 (512px) |
| :--- | :--- | :--- | :--- | :--- |
| InstructPix2Pix | 1.1B | 2.923 | 4.034 | 3.5s |
| UltraEdit | 7.7B | 4.541 | 5.580 | 2.6s |
| OmniGen | 3.8B | 4.666 | 3.498 | 16.5s |
| ICEdit | 17.0B | 4.785 | 4.933 | 8.4s |
| **VAREdit** | **2.2B** | **5.662** | **6.996** | **0.7s** |
| **VAREdit** | **8.4B** | **7.892** | **8.105** | **1.2s** |

**關鍵結論**:
- **卓越的編輯質量**: VAREdit 在指令遵循度和背景保留方面均顯著優於所有對比的開源方法。
- **極高的推理效率**: VAREdit-8B 的推理速度遠超同等規模的擴散模型，展現了 VAR 框架在效率上的巨大潛力。
- **優秀的擴展性**: 從 2B 到 8B 模型的性能提升表明，VAREdit 的性能可以通過擴大模型和數據規模進一步增強。

---

## 相關研究背景

VAREdit 的出現並非偶然，它建立在視覺自回歸模型（VAR）近年來飛速發展的基礎之上。其直接的基礎模型是 **Infinity** [2]，一個通過位元級（bitwise）自回歸建模實現高解析度圖像生成的強大模型。Infinity 引入的無限詞彙權杖化器和位元級自校正機制，為 VAREdit 提供了堅實的生成底座。

與此同時，VAREdit 也與另一篇重要的 VAR 編輯論文 **AREdit** [3] 形成了鮮明對比。AREdit 是一個**無需訓練（Training-Free）**的框架，它通過巧妙的隨機性緩存和權杖重組機制，在不重新訓練模型的情況下實現了高效編輯。然而，由於缺乏針對編輯任務的微調，AREdit 在處理複雜指令時的能力不如經過專門訓練的 VAREdit。

| 特性 | VAREdit | AREdit |
| :--- | :--- | :--- |
| **核心思想** | 微調（Tuning-based） | 免訓練（Training-Free） |
| **編輯方式** | 次世代預測 + SAR 模塊 | 隨機性緩存 + 自適應蒙版 |
| **優勢** | 編輯指令遵循度極高，效果精準 | 推理速度極快，無需額外訓練 |
| **適用場景** | 對編輯質量要求極高的專業場景 | 追求極致效率和快速部署的場景 |

VAREdit 的成功，標誌著 VAR 模型在圖像編輯領域從「免訓練」的探索階段，正式邁入了「精細化微調」的成熟階段，展現了與擴散模型分庭抗禮的巨大潛力。

---

## 個人評價與意義

VAREdit 無疑是2026年初圖像編輯領域最令人矚目的工作之一。它不僅在技術上提出了一個優雅且高效的解決方案來應對擴散模型的固有缺陷，更重要的是，它為 VAR 模型在精細化、可控生成任務中的應用開闢了全新的道路。

**核心意義**: 
1.  **範式轉移的潛力**: VAREdit 證明了 VAR 模型在指令遵循和避免意外編輯方面具有天然優勢，這可能會促使研究社區重新評估 VAR 與擴散模型在圖像編輯任務中的地位。
2.  **效率與質量的完美結合**: 在保持甚至超越 SOTA 編輯質量的同時，實現了數量級的效率提升，這對於推動生成式 AI 進入實時交互應用至關重要。
3.  **Attention 調節的新思路**: SAR 模塊的設計思想——即在模型的不同層級注入不同粒度的條件信息——為未來如何更高效地進行多模態信息融合提供了寶貴的啟示。

總而言之，VAREdit 不僅是一個性能卓越的圖像編輯工具，更是一篇充滿洞見的論文。它為下一代生成模型，特別是在追求**速度、控制和保真度**的終極平衡上，提供了一個強有力的競爭者和一個充滿希望的發展方向。

---

### 參考文獻
[1] Mao, Q., Cai, Q., Li, Y., Pan, Y., Cheng, M., Yao, T., Liu, Q., & Mei, T. (2026). *Visual Autoregressive Modeling for Instruction-Guided Image Editing*. In The Fourteenth International Conference on Learning Representations. ([https://arxiv.org/abs/2508.15772](https://arxiv.org/abs/2508.15772))
[2] Han, J., Liu, J., Jiang, Y., Yan, B., Zhang, Y., Yuan, Z., Peng, B., & Liu, X. (2025). *Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis*. arXiv preprint arXiv:2412.04431. ([https://arxiv.org/abs/2412.04431](https://arxiv.org/abs/2412.04431))
[3] Wang, Y., Guo, L., Li, Z., Huang, J., Wang, P., Wen, B., & Wang, J. (2025). *Training-Free Text-Guided Image Editing with Visual Autoregressive Model*. arXiv preprint arXiv:2503.23897. ([https://arxiv.org/abs/2503.23897](https://arxiv.org/abs/2503.23897))
