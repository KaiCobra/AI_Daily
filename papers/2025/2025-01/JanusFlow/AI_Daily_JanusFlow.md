# AI Daily: JanusFlow - 融合自回歸與 Rectified Flow 的統一多模態新典範

**日期:** 2026年01月05日

## 論文基本資訊

| 項目 | 內容 |
| --- | --- |
| **論文標題** | JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation [1] |
| **作者** | Yiyang Ma, Xingchao Liu, Xiaokang Chen, et al. |
| **研究單位** | DeepSeek-AI, Peking University, The University of Hong Kong, Tsinghua University |
| **發表會議** | CVPR 2025 |
| **arXiv** | [2411.07975](https://arxiv.org/abs/2411.07975) |
| **開源項目** | [github.com/deepseek-ai/Janus](https://github.com/deepseek-ai/Janus) [2] |

---

## 核心貢獻與創新點

JanusFlow 提出了一個極簡而強大的統一框架，在單一模型內實現了**圖像理解**與**圖像生成**兩大核心功能，突破了以往統一多模態模型在性能和架構複雜性上的瓶頸。其核心創新點可歸納如下：

1.  **架構創新：融合自回歸與 Rectified Flow**
    JanusFlow 巧妙地將自回歸語言模型（Autoregressive Models）與前沿的生成模型技術 **Rectified Flow** [3] 結合。最關鍵的發現是，Rectified Flow 的訓練可以無縫整合到大型語言模型（LLM）的框架中，無需對現有架構進行複雜的修改。這大大簡化了統一模型的設計複雜度。

2.  **性能優化：解耦編碼器與表示對齊**
    為了解決單一模型中「理解」與「生成」兩個任務間的潛在衝突，JanusFlow 採用了兩大關鍵策略：
    *   **解耦編碼器 (Decoupling Encoders)**：為理解任務和生成任務設計了獨立的視覺編碼器，避免了單一編碼器在兩種截然不同的任務需求下拉扯，從而導致性能下降的問題。
    *   **表示對齊 (Representation Alignment)**：在統一訓練過程中，策略性地對齊這兩個解耦編碼器的特徵表示，確保模型在執行不同任務時能夠共享和傳遞有效的語義信息。

3.  **卓越性能與效率**
    實驗結果表明，JanusFlow 在僅有 **1.3B** 的緊湊參數規模下，其性能不僅全面超越了現有的統一多模態模型，甚至在多項基準測試中達到或超過了許多為特定任務設計的專門模型（如 LLaVA-v1.5、Qwen-VL-Chat、SDXL 等）[1]。

---

## 技術方法簡述

JanusFlow 的架構分為「理解」和「生成」兩個協同工作的模塊，兩者共享一個核心的大型語言模型。

*   **理解模塊 (Understanding: Autoregressive)**：此模塊遵循標準的多模態語言模型設計，接收圖像和文本提示作為輸入。一個專用的**理解編碼器 (Understanding Encoder)** 將圖像轉換為特徵嵌入，與文本嵌入一同送入 LLM。LLM 以自回歸的方式預測並生成文本響應，完成對輸入內容的理解和問答。

*   **生成模塊 (Generation: Rectified Flow)**：此模塊負責根據文本提示生成圖像。一個獨立的**生成編碼器 (Generation Encoder)** 處理文本提示，其輸出作為條件引導 **Rectified Flow** 模型。Rectified Flow 是一個基於常微分方程（ODE）的生成過程，它從一個隨機噪聲分佈出發，通過學習一個速度場（Velocity Field），逐步將噪聲轉換為清晰的圖像。相比於傳統的擴散模型，Rectified Flow 提供了更直接、高效的生成路徑。

> 論文指出：「我們的關鍵發現表明，Rectified Flow 可以直接在大型語言模型框架內進行訓練，從而消除了複雜架構修改的需要。」[1]

這種設計使得 JanusFlow 能夠在一個統一的訓練循環中，同時優化其理解和生成的能力，而解耦的編碼器則保證了各自任務的專業性不受影響。

---

## 實驗結果與性能指標

JanusFlow 在多個標準多模態基準測試中取得了領先的成績，充分證明了其設計的有效性。

| 基準測試 (Benchmark) | JanusFlow-1.3B 分數 | 說明 |
| --- | --- | --- |
| **多模態理解** | | |
| MMBench | 74.9 | 綜合多模態能力評估 |
| SEED-Bench | 70.5 | 多模態理解與推理 |
| GQA | 60.3 | 視覺問答 |
| **文本到圖像生成** | | |
| GenEval | 9.51 | 生成質量與指令遵循評估 |
| DPG-Bench | 0.63 | 指令遵循評估 |
| MJHQ-30k (FID) | 80.09% | 圖像生成質量 (FID 分數) |

與其他模型的比較顯示，JanusFlow-1.3B 在多模態理解任務上超越了 LLaVA-v1.5 和 Qwen-VL-Chat 等專門的對話模型。在圖像生成方面，其性能可與 SDv1.5 和 SDXL 等成熟的擴散模型相媲美，同時在指令遵循方面表現更佳。

---

## 相關研究背景

統一多模態理解與生成是通往通用人工智能（AGI）的關鍵一步。近年來的研究主要分為兩條路徑：

1.  **分離式模型**：通常將一個預訓練的 LLM 與一個強大的圖像生成模型（如擴散模型）結合。這種方法雖然有效，但通常需要複雜的接口設計和多階段訓練，導致架構臃腫且效率低下。

2.  **統一式模型**：嘗試在單一模型內完成所有任務。早期的工作如 **TokenFlow** [4] 專注於設計統一的圖像 Tokenizer，而 **FlowAR** [5] 則探索了自回歸與 Flow Matching 的結合。然而，這些模型往往難以在理解和生成兩個任務之間取得理想的平衡。

JanusFlow 正是站在這些研究的基礎上，通過引入 Rectified Flow 並設計解耦的編碼器架構，成功地解決了先前統一模型中的核心挑戰，為該領域提供了一個更簡潔、高效且性能強大的新範式。

---

## 個人評價與意義

JanusFlow 的出現為多模態 AI 領域帶來了重要的啟示。它不僅僅是一個性能強大的新模型，更重要的是，它證明了**在統一框架內實現高性能的理解與生成是完全可行的**，並且無需過於複雜的架構設計。

**對研究者的啟發：**
*   **AR + Flow/Diffusion 的潛力**：JanusFlow 的成功激發了我們對自回歸模型與 Flow-based 或 Diffusion-based 模型更深度融合的想像。這種組合可能成為未來多模態基礎模型的標準架構。
*   **解耦與對齊的重要性**：在統一模型設計中，「解耦」特定任務的處理模塊以避免衝突，同時通過「對齊」來保持信息一致性，這一思想極具參考價值。
*   **效率與性能的平衡**：在追求模型規模的同時，JanusFlow 展示了通過巧妙的架構設計，在相對較小的參數規模下也能實現 SOTA 性能的可能性。

總體而言，JanusFlow 是對用戶要求中 **VAR-based** 和 **Flow Matching** 思想的一次完美實踐，它不僅在技術上具有創新性，其開源的代碼和模型也極大地推動了社區的發展，為開發下一代更通用、更高效的 AI 系統鋪平了道路。

---

## 參考文獻

[1] Ma, Y., Liu, X., Chen, X., et al. (2025). *JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). [https://arxiv.org/abs/2411.07975](https://arxiv.org/abs/2411.07975)

[2] DeepSeek-AI. (2024). *Janus: Unified Multimodal Understanding and Generation Models*. GitHub Repository. [https://github.com/deepseek-ai/Janus](https://github.com/deepseek-ai/Janus)

[3] Liu, X., et al. (2022). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. arXiv preprint arXiv:2209.03003.

[4] Qu, L., Zhang, H., Liu, Y., et al. (2025). *TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[5] Anonymous. (2024). *FlowAR: Scale-wise Autoregressive Image Generation Meets Flow Matching*. arXiv preprint arXiv:2412.15205.
