# AI Daily: DreamVAR - 當VAR遇上強化學習，實現高保真主體驅動圖像生成

> 論文標題：DreamVAR: Taming Reinforced Visual Autoregressive Model for High-Fidelity Subject-Driven Image Generation
> 
> 論文連結：[https://arxiv.org/abs/2601.22507](https://arxiv.org/abs/2601.22507)
> 
> 發表會議：ICASSP 2026
> 
> 作者：Xin Jiang, Jingwen Chen, Yehao Li, Yingwei Pan, Kezhou Chen, Zechao Li, Ting Yao, Tao Mei

---

## 核心貢獻

在主體驅動（subject-driven）圖像生成領域，擴散模型（Diffusion Models）如DreamBooth和IP-Adapter已取得顯著成功，但視覺自回歸（Visual Autoregressive, VAR）模型的潛力卻鮮少被挖掘。**DreamVAR是首個將VAR模型與強化學習（Reinforcement Learning）相結合，專門用於主體驅動圖像生成的框架**。它不僅在保持主體外觀一致性上超越了頂尖的擴散模型方法，更為VAR模型在條件生成任務中的應用開闢了新思路。

DreamVAR的核心創新點主要有二：

1.  **主體特徵預填充（Subject Feature Pre-filling）機制**：傳統VAR模型在進行條件生成時，通常將條件標記（conditional tokens）與目標圖像標記（target image tokens）在各個尺度上交錯排列，這導致了訓練與推理階段的不一致性（train-test discrepancy）。DreamVAR創新地提出「預填充」策略，即在自回歸生成開始前，一次性將所有尺度的參考主體特徵全部置於序列前端。這種設計分離了條件與目標，簡化了自回歸的依賴關係，有效緩解了不一致性問題，從而顯著提升了生成圖像的主體保真度。

2.  **多獎勵強化學習（Multi-Reward Reinforcement Learning）**：為了同時優化生成圖像的「主體一致性」和「語義對齊度」，DreamVAR引入了基於群體相對策略優化（Group Relative Policy Optimization, GRPO）的強化學習機制。它設計了兩個獨立的獎勵函數，分別評估生成圖像與參考主體在視覺特徵上的相似度（主體一致性），以及圖像與文本提示在語義上的匹配度（語義對齊），通過加權組合這兩個獎勵來指導模型進行端到端的優化。

![DreamVAR框架圖](asset/dreamvar_framework.webp)
*圖1：DreamVAR的整體框架。(a)部分展示了通過主體特徵預填充來緩解訓練-測試不一致性的流程。(b)部分展示了使用GRPO強化學習進行優化的流程。*

---

## 技術方法詳解

### 視覺自回歸模型（VAR）基礎

VAR模型將圖像生成過程重新定義為一個「下一尺度預測」（next-scale prediction）的自回歸過程。它首先使用視覺標記器（visual tokenizer）將圖像編碼為多個不同解析度的離散標記圖（token maps），記為 \(I = (I_1, I_2, ..., I_K)\)。生成過程則是根據條件 \(C\)（如文本提示）和之前已生成的較小尺度的標記圖，來預測下一個尺度的標記圖。其自回歸似然函數可以表示為：

$$ p(I_1, I_2, ..., I_K | C) = \prod_{k=1}^{K} p(I_k | I_1, ..., I_{k-1}, C) $$

### 主體特徵預填充（Subject Feature Pre-filling）

DreamVAR的關鍵在於如何高效地將參考主體 \(I^s\) 的信息融入生成過程。它首先使用視覺標記器提取 \(I^s\) 的多尺度特徵 \((I^s_1, ..., I^s_K)\)。與傳統方法在每個尺度 \(k\) 將 \(I^s_k\) 和 \(I_k\) 混合不同，DreamVAR將所有主體特徵 \((I^s_1, ..., I^s_K)\) 拼接成一個序列，並在生成過程開始前將其「預填充」到模型中。這樣，模型在預測任何目標圖像標記 \(I_k\) 之前，就已經「看過」了完整的參考主體信息，從而能更好地保持主體一致性。

### 基於GRPO的多獎勵強化學習

為了進一步提升生成質量，DreamVAR採用GRPO算法進行微調。這個過程的核心是設計合理的獎勵函數 \(R\)，它由兩部分加權組成：

$$ R = \alpha R^i + \gamma R^s $$

- **主體一致性獎勵 (\(R^i\))**: 使用DINO或CLIP的圖像編碼器，計算生成圖像和參考主體圖像在特徵空間的餘弦相似度，以獎勵模型生成與參考主體外觀更接近的圖像。
- **語義對齊獎勵 (\(R^s\))**: 使用CLIP的圖文對比能力，計算生成圖像與文本提示的匹配程度，確保生成內容符合文本描述。

GRPO的訓練目標函數如下，旨在最大化標準化後的獎勵，同時通過KL散度項 \(D_{KL}(\pi_\theta || \pi_{ref})\) 防止模型偏離預訓練權重太遠，保證生成的多樣性和穩定性。

$$ J_{GRPO}(\theta) = \mathbb{E}_{C, \{I_g\}_{g=1}^G \sim \pi_{\theta_{old}}}} [\sum_{g=1}^G \frac{R_g - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)} \sum_{l=1}^L (\min(r_l(\theta)A_l, \text{clip}(r_l(\theta), 1-\epsilon, 1+\epsilon)A_l)) - \beta D_{KL}(\pi_\theta || \pi_{ref})] $$

---

## 實驗結果

DreamVAR在標準的主體驅動生成評測集DreamBench上進行了廣泛的實驗，並與當前最先進的擴散模型方法進行了比較。

### 定量比較

實驗結果（如下表所示）表明，DreamVAR在主體一致性指標（DINO和CLIP-I）上全面超越了所有基線模型，包括DreamBooth、IP-Adapter和OmniControl。這證明了其在保持參考主體外觀方面的卓越能力。值得注意的是，DreamVAR僅用了約2B的參數，遠小於UNO等模型（12B），卻取得了更優的性能。

![定量比較結果](asset/dreamvar_table1.webp)
*表1：在DreamBench上的定量比較結果。DINO和CLIP-I越高，表示主體一致性越好；CLIP-T越高，表示文本對齊越好。DreamVAR在主體一致性上取得了最佳成績。*

### 定性比較

從下圖的生成樣本可以看出，DreamVAR能夠在遵循文本提示的同時，高度還原參考主體（如第一行的狗、第二行的毛絨玩具、第三行的運動鞋）的獨特外觀和細節。相比之下，其他方法要麼無法準確還原主體（如IP-Adapter），要麼生成的主體與文本描述不符（如OmniControl）。

![定性比較結果](asset/dreamvar_figure2.webp)
*圖2：與不同方法的定性比較。DreamVAR在保持主體外觀和遵循文本提示之間取得了最佳平衡。*

### 消融實驗

論文還通過消融實驗驗證了各個組件的有效性。結果表明：
- **獎勵函數**：同時使用主體一致性獎勵（\(R^i\)）和語義對齊獎勵（\(R^s\)）能取得最佳的綜合效果。僅使用\(R^i\)會犧牲文本對齊度。
- **預填充機制**：與其他條件注入策略（如僅使用最後一層尺度特徵、使用多尺度特徵、交錯式注入）相比，預填充機制在各項指標上均表現最優，證明了其在緩解訓練-推理不一致性問題上的有效性。

![獎勵函數消融實驗](asset/dreamvar_figure3.webp)
*圖3：強化學習獎勵的消融研究。左側為僅使用主體一致性獎勵的結果，右側為完整DreamVAR的結果，後者在背景和細節上更符合文本提示。*

---

## 相關研究背景

主體驅動的圖像生成旨在根據用戶提供的少數幾張參考圖像，生成該主體在不同場景、不同風格下的新圖像。這一領域的研究大致可分為三類：

1.  **模型微調（Fine-tuning）**: 以DreamBooth [1] 為代表，通過在少量參考圖像上微調整個或部分擴散模型來「學習」主體的概念。這類方法效果好，但需要為每個新主體重新訓練，成本較高。

2.  **適配器（Adapter）**: 以IP-Adapter [2] 為代表，通過訓練一個輕量的適配器模塊來將主體特徵注入到預訓練的擴散模型中。這類方法無需微調大模型，效率高，但有時在主體保真度上有所欠缺。

3.  **自回歸模型（Autoregressive Models）**: 在DreamVAR之前，很少有工作探索VAR模型在這一任務上的應用。相關的研究如`Fine-Tuning Visual Autoregressive Models for Subject-Driven Generation` [3] 雖然是首次嘗試，但指出了直接微調VAR模型的困難。DreamVAR則通過創新的預填充和強化學習機制，成功克服了這些挑戰。

同時，將強化學習（特別是PPO、GRPO等算法）應用於優化生成模型（包括AR和Diffusion）的對齊和質量，已成為一個新興的研究熱點 [4, 5]。DreamVAR正是這一趨勢在VAR模型和主體驅動生成任務上的成功實踐。

---

## 個人評價與意義

DreamVAR是一項非常具有啟發性的工作，它成功地將VAR模型的潛力從無條件生成擴展到了複雜的、細粒度的條件生成任務中。這篇論文最大的亮點在於**問題的精準定位和解決方案的巧妙設計**。

- **對問題的洞察**：作者敏銳地指出了VAR模型在條件生成中存在的「訓練-推理不一致性」這一核心痛點，這是阻礙其應用於此類任務的關鍵障礙。
- **解決方案的優雅**：「預填充」機制是一個非常簡潔而高效的解決方案，它沒有引入複雜的模塊，而是通過改變數據的組織方式，從根本上解決了問題。這體現了深刻的工程智慧。
- **強化學習的有效應用**：將GRPO引入，並設計了針對性的雙重獎勵，使得模型可以在「長得像」和「聽得懂」這兩個目標之間取得精妙的平衡，這是生成模型對齊的關鍵。

從更廣泛的意義上看，DreamVAR的成功表明，**VAR模型作為一個統一、高效的生成範式，其潛力遠未被充分發掘**。當前生成領域由擴散模型主導，但其多步採樣帶來的速度瓶頸始終存在。VAR模型憑藉其單次前向傳播的快速推理能力，在交互式應用和大規模部署上具有天然優勢。DreamVAR為如何將VAR模型應用於更廣泛、更複雜的生成任務提供了寶貴的範例和思路，可能會激發更多研究者重新審視並探索VAR架構的可能性。

---

### 參考文獻

[1] Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K. (2023). DreamBooth: Fine-tuning text-to-image diffusion models for subject-driven generation. In *CVPR*.

[2] Ye, H., Zhang, J., Liu, S., Han, X., & Wei, Y. (2023). IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv preprint arXiv:2308.06721*.

[3] Chung, S., Lee, J., Kim, H., & Lee, J. (2025). Fine-Tuning Visual Autoregressive Models for Subject-Driven Generation. In *ICCV*.

[4] Yuan, S., et al. (2025). AR-GRPO: Training Autoregressive Image Generation Models via Reinforcement Learning. *arXiv preprint arXiv:2508.06924*.

[5] Sun, S., et al. (2026). VAR RL Done Right: Tackling Asynchronous Policy in Visual Autoregressive Generation. *arXiv preprint arXiv:2601.02256*.
