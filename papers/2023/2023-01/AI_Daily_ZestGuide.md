# AI Daily: ZestGuide - 零樣本空間佈局條件下的文本到圖像擴散模型

## 論文基本資訊

- **論文標題**: Zero-shot spatial layout conditioning for text-to-image diffusion models
- **作者**: Guillaume Couairon, Marlène Careil, Matthieu Cord, Stéphane Lathuilière, Jakob Verbeek
- **發表會議/期刊**: ICCV 2023
- **研究單位**: Meta AI, Sorbonne Université, Valeo.ai, LTCI, Télécom Paris, IP Paris
- **論文連結**: [https://arxiv.org/abs/2306.13754](https://arxiv.org/abs/2306.13754)
- **關鍵詞**: `Zero-Shot`, `Diffusion Models`, `Spatial Layout Control`, `Cross-Attention`, `Training-Free`

---

## 核心貢獻與創新點

**ZestGuide** 提出了一種創新的 **零樣本（Zero-shot）** 方法，旨在解決現有文本到圖像擴散模型在遵循精確空間佈局指令方面的限制。傳統模型難以單純通過文本提示來控制生成物體的具體位置和形狀，而 ZestGuide 通過結合分割掩碼（segmentation masks）和自由形式的文本描述，實現了對生成內容的精確空間控制，且 **無需任何額外訓練**。

其核心創新點包括：

1.  **零樣本引導**: ZestGuide 是一種即插即用的引導方法，可直接應用於預訓練的擴散模型，無需對模型進行任何微調或重新訓練，大大降低了應用門檻。
2.  **利用交叉注意力提取分割**: 該方法巧妙地利用了擴散模型 U-Net 中交叉注意力層（cross-attention layers）隱含的空間信息。通過提取這些注意力圖，ZestGuide 能夠在生成過程中推斷出物體的語義分割，從而避免了對外部預訓練分割模型的依賴。
3.  **基於梯度的引導損失**: ZestGuide 計算推斷出的分割圖與用戶提供的輸入分割掩碼之間的損失，並利用該損失的梯度來引導擴散模型的去噪過程，从而使生成的圖像在空間佈局上與輸入掩碼保持一致。
4.  **卓越的性能**: 實驗結果表明，與先前的 SOTA 方法（如 Paint with Words）相比，ZestGuide 在 COCO 數據集上的 **mIoU 指標提升了 5 到 10 個點**，同時保持了相當的圖像生成質量（FID 分數），證明了其在佈局控制精度上的顯著優勢。

---

## 技術方法簡述

ZestGuide 的核心在於一個引導循環，它在擴散模型的每個去噪步驟中，對齊生成的圖像與用戶提供的空間佈局。

![ZestGuide 方法示意圖](./asset/zestguide_figure2.png)
*圖 1: ZestGuide 方法示意圖，展示了如何通過分割掩碼和文本描述來引導圖像生成*

### 1. 擴散模型基礎

擴散模型通過一個去噪過程來生成圖像。給定一個帶噪聲的圖像 $\mathbf{x}_t$ 和一個文本提示 $y$，去噪網絡 $\epsilon_\theta(\mathbf{x}_t, t, \rho(y))$ 會預測用於產生 $\mathbf{x}_t$ 的噪聲。訓練目標是最小化預測噪聲與真實噪聲之間的差異：

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, t, \epsilon} \left[ \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t, \rho(y))\|^2 \right]$$

### 2. 從交叉注意力中提取零樣本分割

ZestGuide 的關鍵洞察是，U-Net 中的交叉注意力圖譜已經包含了豐富的空間信息，可以被用來進行零樣本分割。對於給定的文本標記 $T_k$，其在 U-Net 層 $\ell$ 的交叉注意力圖譜 $\mathbf{A}_\ell^k$ 計算如下：

$$\mathbf{A}_\ell^k = \text{Softmax}\left(\frac{\mathbf{Q}_\ell (\mathbf{K}_\ell^k)^T}{\sqrt{d}}\right)$$

其中 $\mathbf{Q}_\ell$ 是圖像特徵的查詢，$\mathbf{K}_\ell^k$ 是文本標記 $T_k$ 的鍵。通過在所有層和注意力頭上對特定文本標記的注意力圖譜進行平均，可以得到對應於該文本的分割圖 $\mathbf{S}_i$：

$$\mathbf{S}_i = \frac{1}{L} \sum_{\ell=1}^{L} \sum_{j=1}^{K} [[T_j \in \mathcal{T}_i]] \mathbf{A}_\ell^j$$

### 3. 空間自引導損失

為了將生成的內容與用戶提供的分割掩碼 $\mathbf{S}_i$ 對齊，ZestGuide 設計了一個名為 $\mathcal{L}_{\text{Zest}}$ 的損失函數，它由兩部分組成：

$$\mathcal{L}_{\text{Zest}} = \sum_{i=1}^{K} \left( \mathcal{L}_{\text{BCE}}(\bar{\mathbf{S}}_i, \mathbf{S}_i) + \mathcal{L}_{\text{BCE}}\left(\frac{\bar{\mathbf{S}}_i}{\|\bar{\mathbf{S}}_i\|_\infty}, \mathbf{S}_i\right) \right)$$

這個損失函數比較了從注意力圖中提取的分割 $\bar{\mathbf{S}}_i$ 和目標分割 $\mathbf{S}_i$。第二項對注意力圖進行了歸一化，以平衡不同物體分割的梯度，從而實現更穩定的引導。

在去噪的每一步，通過計算 $\mathcal{L}_{\text{Zest}}$ 關於帶噪聲圖像 $\mathbf{x}_t$ 的梯度 $\nabla_{\mathbf{x}_t} \mathcal{L}_{\text{Zest}}$，並將其應用於噪聲預測，從而引導生成過程。

---

## 實驗結果和性能指標

ZestGuide 在 COCO-Stuff 驗證集上進行了廣泛的實驗，並在三個不同的設置下（Eval-all, Eval-filtered, Eval-few）與多種基線方法進行了比較。

**性能亮點**：

- **佈局對齊精度 (mIoU)**: 在最能代表真實世界應用場景的 `Eval-few` 設置中，ZestGuide 的 mIoU 達到了 **46.9**，遠超之前的零樣本方法 Paint with Words (PwW) 的 23.8，甚至優於需要微調的 SpaText (23.8)。
- **圖像質量 (FID)**: ZestGuide 在保持高質量圖像生成方面表現出色，其 FID 分數（越低越好）與其他頂級方法相當，例如在 `Eval-few` 設置中 FID 為 **21.0**。
- **推理效率**: 作為一種零樣本方法，ZestGuide 的推理速度遠快於需要外部自分類器的方法。例如，生成一張圖像 ZestGuide 大約需要 15 秒，而 LDM w/ External Classifier 則需要 1 分鐘。

| Method | Zero-shot | Eval-few (mIoU ↑) | Eval-few (FID ↓) |
| :--- | :---: | :---: | :---: |
| LDM w/ PwW | ✓ | 29.6 | 25.8 |
| LDM w/ MultiDiffusion | ✓ | 19.6 | 21.1 |
| **LDM w/ ZestGuide (ours)** | ✓ | **46.9** | **21.0** |
| SD w/ SpaText (Finetuned) | ✗ | 23.8 | 16.2 |
| LDM w/ External Classifier | ✗ | 20.5 | 23.7 |

---

## 相關研究背景

ZestGuide 的研究建立在空間條件圖像生成和無需訓練的擴散模型適應這兩個活躍的研究領域之上。

- **空間條件生成**: 早期的工作主要依賴於 GAN，如 SPADE 和 OASIS，但這些方法通常需要大量的帶有像素級標註的數據集進行訓練。擴散模型的出現為這一領域帶來了新的可能性，例如 PITI 和 LayoutDiffusion，但它們仍然需要微調或在特定數據上訓練。
- **無需訓練的適應**: 近期研究發現，預訓練擴散模型的交叉注意力圖譜中包含了豐富的語義和空間信息。像 Prompt-to-Prompt、pix2pix-zero 和 eDiff-I 等方法已經探索了通過操縱注意力圖譜來實現圖像編輯。ZestGuide 正是沿著這一思路，將其應用於更具挑戰性的、從零開始的空間佈局生成任務。

ZestGuide 與這些方法的不同之處在於，它專注於利用這種內在的注意力信息來實現對圖像生成過程的精確、零樣本空間佈局控制，而無需依賴任何外部模型或對擴散模型本身進行修改。

---

## 個人評價和意義

ZestGuide 為文本到圖像生成領域帶來了一個優雅而強大的解決方案，完美地平衡了用戶對 **精確空間控制** 的需求和 **模型應用的便捷性**。它最大的亮點在於其 **“零樣本”** 的特性，使得非專業用戶也能夠像“指揮家”一樣，通過簡單的分割掩碼和文本描述，精確地安排生成圖像中各個元素的位置和內容。

從技術角度看，該研究深刻地揭示了預訓練擴散模型內部交叉注意力機制的潛力，證明了這些看似“黑盒”的模型內部已經學習到了豐富的結構化知識。ZestGuide 不僅僅是一個工具，更像是一把鑰匙，解鎖了模型內部隱藏的空間理解能力。

對於內容創作者和設計師而言，ZestGuide 大大降低了實現複雜創意構圖的技術門檻。未來，我們可以期待更多基於類似原理的工具出現，讓 AI 圖像生成不再僅僅是“隨機抽卡”，而是成為一種可控、可預期的創意實現過程。這項工作無疑為下一代可控內容生成技術的發展鋪平了道路。
