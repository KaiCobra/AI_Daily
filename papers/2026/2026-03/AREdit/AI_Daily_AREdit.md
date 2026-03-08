# AI Daily: AREdit - 首個基於VAR的免訓練文字引導圖像編輯框架

## 論文基本信息
- **論文標題**: Training-Free Text-Guided Image Editing with Visual Autoregressive Model
- **作者**: Yufei Wang, Lanqing Guo, Zhihao Li, Jiaxing Huang, Pichao Wang, Bihan Wen, Jian Wang
- **發表會議**: ICCV 2025
- **arXiv ID**: [2503.23897](https://arxiv.org/abs/2503.23897)
- **GitHub**: [wyf0912/AREdit](https://github.com/wyf0912/AREdit)

## 核心貢獻與創新點

文字引導的圖像編輯技術在生成式AI領域扮演著重要角色。過去的方法主要依賴擴散模型（Diffusion Models）或整流流（Rectified Flows），這些方法通常需要透過反轉技術（Inversion）從輸入圖像中提取結構化噪聲。然而，反轉過程中的不準確性容易導致誤差傳播，進而引發非預期的修改並降低圖像保真度。此外，文本提示與圖像特徵之間的糾纏，經常導致在只需局部編輯時發生全局性的改變。

為了解決這些挑戰，本研究提出了 **AREdit**，這是**第一個基於視覺自回歸模型（Visual AutoRegressive, VAR）的免訓練（Training-Free）文字引導圖像編輯框架**。該框架的核心創新在於完全消除了顯式反轉的需求，同時確保了精確且受控的修改。

AREdit 引入了一種創新的**隨機性快取機制（Randomness Caching）**，能夠儲存原始圖像的 token 索引和概率分佈，從而捕捉源提示與圖像之間的關係。基於此快取，研究團隊設計了**自適應細粒度遮罩策略（Adaptive Fine-grained Masking）**，動態識別並限制修改範圍於相關區域，有效防止了非預期的改變。最後，透過**Token重組方法（Token Re-assembling）**進一步精煉編輯過程，顯著提升了編輯的多樣性、保真度與控制力。

![AREdit 編輯效果展示](../../../../assets/AREdit_qualitative.webp)
*圖 1：AREdit 在多種編輯任務上的定性結果，展現了其在保持非編輯區域結構的同時，精確執行文字指令的能力。*

## 技術方法簡述

AREdit 建立在最新的 VAR 基礎模型 Infinity-2B 之上。VAR 模型將圖像生成重新定義為從粗到細（coarse-to-fine）的下一個尺度預測（next-scale prediction）。

### 1. 隨機性快取（Randomness Caching）
在推理階段，編碼器 $\mathcal{E}$ 對輸入圖像 $\mathbf{I}$ 進行前向傳播，預測 bit labels $\mathbf{R}_{queue}$ 和 transformer 輸入 $\mathbf{P}_{queue}$。對於每個尺度 $k$，計算概率分佈 $\mathbf{P}_k = \mathcal{T}(\mathbf{F}_{k-1}, \Phi(t_S))$。系統會快取這些 bit labels 和概率分佈，為後續的編輯提供必要的結構信息。

### 2. 自適應細粒度遮罩（Adaptive Fine-grained Masking）
為了在保真度與生成能力之間取得平衡，AREdit 引入了超參數 $\gamma$。對於 $k \le \gamma$ 的早期步驟，系統直接重用快取的 bit labels，以保留圖像的低頻特徵（如整體佈局）。對於 $k > \gamma$ 的步驟，則使用自適應遮罩來決定哪些 bit labels 需要被修改：

$$ \mathbf{M}_k = \mathbb{I}\left[ (\mathbf{P}_k[..., \mathbf{R}_k] - \mathbf{P}_k^{tgt}[..., \mathbf{R}_k]) > \tau \right] $$

其中 $\tau$ 是控制保真度與創意的超參數。目標分佈 $\mathbf{P}_k^{tgt}$ 是由 transformer 在目標提示 $t_T$ 的條件下預測得出的。這個公式的直觀意義是：只有當某個 bit label 在目標提示下的概率顯著低於在源提示下的概率時，才認為該區域需要被編輯。

![AREdit 框架圖](../../../../assets/AREdit_framework.webp)
*圖 2：AREdit 的整體框架。透過快取機制與自適應遮罩，實現了精確的局部編輯。*

### 3. Token 重組與注意力控制（Token Re-assembling & Attention Control）
獲得細粒度遮罩 $\mathbf{M}_k$ 後，系統將其用於引導 token 採樣：

$$ \mathbf{R}_k^{tgt} = \mathbf{M}^k \odot \mathbf{R}_k' + (1 - \mathbf{M}^k) \odot \mathbf{R}_k $$

其中 $\mathbf{R}_k'$ 是從目標分佈 $\mathbf{P}_k^{tgt}$ 中隨機採樣的 bit labels。此外，AREdit 也兼容現有的注意力控制方法，透過對齊函數映射目標提示與源提示的 token 索引，進一步實現細粒度的語義控制。

## 實驗結果與性能指標

研究團隊在 PIE-Bench 基準測試上對 AREdit 進行了全面的評估。PIE-Bench 包含 700 張圖像與 10 種不同的編輯類型。

### 定量比較
與現有基於 Diffusion 和 Rectified Flow 的 SOTA 方法相比，AREdit 在多項指標上展現了卓越的性能：

| Method | Base Model | Structure Distance↓ | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP Sim (Whole/Edited)↑ |
|--------|-----------|---------------------|-------|-------|--------|--------------------------|
| Prompt2Prompt | diffusion | 0.0694 | 17.87 | 0.7114 | 0.2088 | 25.01 / 22.44 |
| MasaCtrl | diffusion | 0.0284 | 22.17 | 0.7967 | 0.1066 | 23.96 / 21.16 |
| PnP-DirInv | diffusion | 0.0243 | 22.46 | 0.7968 | 0.1061 | 25.41 / 22.62 |
| LEDits++ | diffusion | 0.0431 | 19.64 | 0.7767 | 0.1334 | 26.42 / 23.37 |
| RF-Inversion | flow | 0.0406 | 20.82 | 0.7192 | 0.0793 | 25.20 / 22.11 |
| **AREdit (Ours)** | **VAR** | **0.0305** | **24.19** | **0.8370** | **0.0870** | **25.42 / 22.77** |

*表 1：AREdit 在背景保留（PSNR, SSIM）和編輯質量（CLIP Similarity）上均達到了頂尖水準。*

### 計算效率
AREdit 最大的優勢之一在於其驚人的推理速度。在單張 Nvidia A100 GPU 上處理 1K 解析度的圖像：

| Method | Backbone | Resolution | Speed |
|--------|---------|-----------|-------|
| LEDits++ | SDXL | 1K | 19s (12s) |
| RFInversion | Flux | 1K | 27s (13.5s) |
| **AREdit** | **Infinity** | **1K** | **2.5s (1.2s)** |

*表 2：括號內為不包含快取或反轉的後續運行時間。AREdit 的速度比現有方法快了約 9 倍！*

![AREdit 視覺比較](../../../../assets/AREdit_comparison.webp)
*圖 3：與其他方法的視覺比較。AREdit 在保留非編輯區域（如背景和人物身份）方面表現出色。*

## 相關研究背景

本研究建立在幾個重要的前沿技術之上：

1. **Visual AutoRegressive Modeling (VAR)**：由 Tian 等人於 NeurIPS 2024 提出，將圖像生成重新定義為 next-scale prediction，首次讓自回歸模型在圖像生成質量上超越了 Diffusion Transformer (DiT)。
2. **Infinity**：由 Han 等人於 CVPR 2025 提出的 Bitwise VAR 模型，能夠極速生成高解析度、真實感的圖像。AREdit 正是採用了 Infinity-2B 作為其強大的基礎模型。
3. **Training-Free Image Editing**：過去的方法如 Prompt-to-Prompt (P2P)、Plug-and-Play (PnP) 和 RF-Inversion 等，主要依賴擴散模型或整流流的反轉技術。AREdit 則開闢了一條基於 VAR 架構的全新路徑。

## 個人評價和意義

AREdit 的出現具有里程碑式的意義。它不僅是**第一個基於 VAR 架構的免訓練圖像編輯框架**，更重要的是，它巧妙地利用了 VAR 模型的特性（離散 token 和概率分佈），徹底繞過了 Diffusion 模型中繁瑣且容易出錯的 Inversion 過程。

這種架構上的轉換帶來了兩個巨大的紅利：
1. **極致的速度**：1K 解析度圖像編輯僅需 1.2 秒，這使得實時、交互式的圖像編輯應用成為可能。
2. **精確的局部控制**：透過比較源提示和目標提示下的 token 概率分佈差異，AREdit 能夠非常自然且精準地定位需要修改的區域，這在解決「牽一髮而動全身」的編輯痛點上表現得極為優雅。

對於關注 VAR-based、training-free 和 zero-shot 領域的研究者來說，AREdit 展示了自回歸模型在下游任務（如下游編輯、修復等）中的巨大潛力。它證明了 VAR 不僅在生成速度上具有優勢，在細粒度控制和特徵解耦方面同樣大有可為。這無疑將激發更多基於 VAR 架構的創新應用。
