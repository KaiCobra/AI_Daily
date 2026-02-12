# AI Daily: Visual Autoregressive Modelling for Monocular Depth Estimation

**日期：** 2025年12月31日

## 論文基本資訊

| 項目 | 內容 |
| --- | --- |
| **論文標題** | Visual Autoregressive Modelling for Monocular Depth Estimation [1] |
| **作者** | Amir El-Ghoussani, André Kaup, Nassir Navab, Gustavo Carneiro, Vasileios Belagiannis |
| **研究機構** | Friedrich-Alexander University Erlangen-Nuremberg, Germany; Technical University of Munich, Germany |
| **發表** | arXiv (2025年12月27日提交)，預計為CVPR 2025等頂會論文 |
| **代碼** | [https://github.com/amirelghoussani/VAR-Depth](https://github.com/amirelghoussani/VAR-Depth) |

---

## 論文核心貢獻與創新點

在單目深度估計領域，基於擴散模型（Diffusion Models）的方法近年來取得了顯著進展，但其高昂的計算成本和對大規模數據集的依賴限制了其應用。為應對此挑戰，來自德國FAU和TUM的研究團隊提出了一種基於**視覺自回歸（Visual Autoregressive, VAR）**先驗的新型單目深度估計方法，為生成式深度估計提供了新的解決路徑。

該研究的核心創新點在於，它首次將在圖像生成領域取得巨大成功的VAR模型（由Tian et al.在NeurIPS 2024上提出並獲最佳論文獎 [2]）成功應用於單目深度估計任務。與擴散模型依賴多步去噪的迭代過程不同，VAR模型通過從粗到細的「下一尺度預測」範式，在固定的自回歸階段中高效生成結果。

**主要貢獻總結如下：**

1.  **方法創新**：提出了一種新穎的生成式深度估計框架，將預訓練的大規模文本到圖像VAR模型與專為深度估計設計的**尺度條件上採樣機制（scale-wise conditional upsampling）**相結合。
2.  **效率優勢**：該方法在訓練效率上表現出色，僅需約**74K**的合成數據進行微調，即可在多個室內基準測試中達到最先進的性能，顯著低於擴散模型通常所需的數據量。
3.  **性能卓越**：在NYUv2、KITTI等標準數據集上，該方法在受約束的訓練條件下，性能優於現有的基於擴散模型的方法，證明了VAR先驗在幾何感知任務中的有效性。
4.  **理論補充**：為深度估計領域建立了一個補充性的生成模型家族，證明自回歸先驗可作為與擴散先驗並行發展的有效幾何感知生成工具。

---

## 技術方法簡述

本文方法的核心是利用一個預訓練的VAR模型作為先驗，並通過一個精巧的微調和採樣策略將其適應於深度估計任務。整體流程可分為三個關鍵部分：

1.  **背景：視覺自回歸建模 (VAR)**
    該方法基於Tian等人提出的VAR框架 [2]，該框架通過一個多尺度VQ-GAN將圖像編碼為一系列逐步降低分辨率的離散標記圖。然後，一個Transformer模型被訓練用於自回歸地預測更高分辨率的標記圖，其條件是所有較低分辨率的標記圖。這種「下一尺度預測」機制避免了傳統自回歸模型逐像素預測的低效問題。

2.  **方法：條件化微調與上採樣**
    為了將預訓練的VAR模型應用於深度估計，研究者設計了一個特定的微調協議。他們用一個小型MLP替換了原始的文本編碼器，以處理來自RGB圖像的條件編碼。關鍵的創新在於**條件上採樣機制**，該機制通過一個U-Net架構的網絡，在自回歸生成過程中有效地將幾何條件信息（來自輸入圖像）跨尺度傳播，解決了標準VAR模型在處理複雜條件時信號丟失的問題。

3.  **採樣：結合分類器自由引導**
    在推理階段，為了平衡先驗知識和條件信息，該方法採用了類似於擴散模型中的**分類器自由引導（Classifier-Free Guidance）**策略。通過在每個尺度上結合來自條件生成模型和無條件先驗模型的預測，系統能夠生成既忠實於輸入圖像幾何結構又具有良好先驗分佈的深度圖。整個推理過程在10個固定的自回歸階段內完成。

---

## 實驗結果與性能指標

該研究在多個主流的單目深度估計基準數據集上進行了廣泛評估，包括NYUv2、KITTI、ETH3D和DIODE。實驗結果表明，本文提出的VAR方法在多項指標上均取得了具有競爭力的結果。

**關鍵性能表現：**

- **室內場景**：在NYUv2數據集上，該方法在AbsRel（絕對相對誤差）等關鍵指標上達到了**0.094**，優於Marigold [3]、DDP等一系列基於擴散模型或其他生成模型的方法。
- **戶外場景**：在KITTI數據集上，該方法同樣展現了強勁的性能，證明了其對不同場景的適應能力。
- **數據效率**：與需要數百萬甚至數十億圖像對進行訓練的擴散模型相比，本文方法僅使用74K合成數據進行微調，顯示出極高的數據效率和更低的訓練成本。

下表總結了該方法與其他SOTA方法在NYUv2和KITTI數據集上的性能對比：

| 方法 | 先驗模型 | 訓練數據量 | NYUv2 (AbsRel ↓) | KITTI (AbsRel ↓) |
| :--- | :--- | :--- | :--- | :--- |
| DepthAnything v2 | DINOv2-L | 142M | 0.095 | 9.4 |
| Metric3D | DINOv2-L | 142M | 0.096 | 9.8 |
| Marigold v1.1 | SD-2.1B | 74K | 0.101 | 10.1 |
| **Ours (本文方法)** | **Swin-100M** | **74K** | **0.094** | **9.8** |

*數據來源於原論文表1，越低越好。*

---

## 相關研究背景

本文的研究建立在自回歸模型和生成式深度估計的堅實基礎之上。視覺自回歸建模正成為繼擴散模型之後，圖像生成領域的下一個研究熱點。

- **VAR的崛起**：Tian等人提出的VAR模型 [2] 因其高效的訓練和推理能力，以及卓越的生成質量，獲得了NeurIPS 2024的最佳論文獎，引發了學術界的廣泛關注。此後，迅速出現了如ControlVAR、M-VAR等改進工作。
- **生成式深度估計的演進**：在VAR之前，以Marigold [3] 為代表的擴散模型方法在零樣本深度估計方面取得了巨大成功，證明了大規模生成模型先驗在幾何理解任務中的潛力。此外，DepthFM等工作探索了流匹配（Flow Matching）在深度估計中的應用。
- **競爭與融合**：在2025年的頂級會議（如CVPR、IJCAI）上，已有多篇論文（如DepthART [4]、Scalable Autoregressive Monocular Depth Estimation [5]）幾乎同時探索將VAR應用於深度估計，顯示出該方向的巨大潛力和激烈的研究競爭。

本文正是在這一背景下，率先提出了一套完整且高效的VAR深度估計框架，並通過實驗證明了其相對於擴散模型的競爭優勢。

---

## 個人評價與意義

這篇論文不僅是一次成功的技術應用，更為生成式AI在3D視覺領域的發展提供了重要的啟示。

**評價：**

1.  **選題前沿且影響深遠**：論文精準地抓住了VAR模型這一新興熱點，並將其應用於深度估計這一基礎視覺任務，展示了極高的學術敏感度和創新性。其研究成果很可能引領一波基於自回歸模型進行3D視覺研究的新浪潮。
2.  **技術方案巧妙且務實**：面對將2D生成先驗應用於3D幾何任務的挑戰，論文提出的「條件上採樣」和「分類器自由引導」方案非常巧妙，在理論和實踐上都取得了良好平衡。特別是其高數據效率的特點，使其在實際應用中比動輒需要海量數據的擴散模型更具可行性。
3.  **實驗充分且具說服力**：論文在多個標準數據集上進行了詳盡的實驗，並與當前最先進的擴散模型方法進行了直接比較，用數據證明了其方法的有效性和優越性。

**意義：**

- **對學術研究而言**，它開闢了除擴散模型之外的另一條通往高質量生成式深度估計的道路，鼓勵研究者探索更多樣化的生成模型在3D視覺中的應用。
- **對工業應用而言**，該方法在訓練成本和推理速度上的潛在優勢，使其比擴散模型更有可能被部署到資源受限的邊緣設備或對實時性要求高的應用中，如自動駕駛、AR/VR等。

總體而言，這是一篇高質量的研究工作，兼具理論創新和實用價值，無疑是近期深度學習和計算機視覺領域最值得關注的論文之一。

---

### 參考文獻

[1] El-Ghoussani, A., et al. (2025). *Visual Autoregressive Modelling for Monocular Depth Estimation*. arXiv:2512.22653.
[2] Tian, K., et al. (2024). *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*. In *Advances in Neural Information Processing Systems*.
[3] Ke, B., et al. (2024). *Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation*. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.
[4] Patni, R., et al. (2025). *DepthART: Monocular Depth Estimation as Autoregressive Refinement Task*. In *Proceedings of the International Joint Conference on Artificial Intelligence*.
[5] Wang, J., et al. (2025). *Scalable Autoregressive Monocular Depth Estimation*. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.
