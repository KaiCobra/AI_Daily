# AI Daily: RAE - 重新定義擴散模型， Yann LeCun 親自下場的降維打擊

> **論文標題**: Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders
> **發表單位**: New York University, Meta AI (FAIR)
> **作者**: Shengbang Tong, Boyang Zheng, Ziteng Wang, Bingda Tang, Nanye Ma, Ellis Brown, Jihan Yang, Rob Fergus, **Yann LeCun**, Saining Xie
> **發表時間**: 2026年1月22日
> **論文連結**: [https://arxiv.org/abs/2601.16208](https://arxiv.org/abs/2601.16208)
> **專案頁面**: [https://rae-dit.github.io/](https://rae-dit.github.io/)

## 總結

由圖靈獎得主 Yann LeCun 參與的最新研究，為大規模文本到圖像生成領域帶來了突破。這篇論文介紹了一種名為 **表示自動編碼器 (Representation Autoencoders, RAE)** 的新方法，旨在取代目前主流的變分自動編碼器 (Variational Autoencoders, VAE)，成為擴散模型的新基石。研究表明，RAE 不僅在性能上全面超越 VAE，還具備更快的收斂速度和更強的穩定性，尤其是在大規模模型訓練中，成功解決了 VAE 長期存在的過擬合問題。這項工作不僅簡化了擴散模型的設計，更為未來構建統一的視覺理解與生成模型鋪平了道路。

## 核心貢獻

這篇論文的核心貢獻可以總結為以下幾點：

1.  **RAE 在大規模 T2I 生成中的優越性**：首次將 RAE 擴展到大規模、自由形式的文本到圖像生成任務，並證明其在所有模型規模（從 5 億到 98 億參數）上均顯著優於當前最先進的 VAE（如 FLUX）。

2.  **解決 VAE 的災難性過擬合問題**：實驗發現，基於 VAE 的模型在微調超過 64 個 epoch 後會出現災難性的過擬合，而 RAE 模型即使在 256 個 epoch 的訓練後依然保持穩定，並持續提升性能。

3.  **簡化擴散模型框架**：研究發現，隨著模型規模的擴大，一些複雜的設計（如寬擴散頭和噪聲增強解碼）變得不再必要，從而簡化了模型架構。唯一保持關鍵作用的是**維度相關的噪聲調度 (dimension-dependent noise scheduling)**。

4.  **更快的收斂速度和更高的生成質量**：與 VAE 相比，RAE 在訓練過程中收斂速度更快（在 GenEval 上快 4.0 倍，在 DPG-Bench 上快 4.6 倍），並能生成更高質量的圖像。

![RAE vs VAE pre-training](asset/RAE_Scaling/fig1-01.png)
*圖 1：RAE 在預訓練階段的收斂速度明顯快於 VAE。*

## 技術方法詳解

### RAE 的核心思想

傳統的 VAE 試圖將圖像壓縮到一個低維的潛在空間，這不可避免地會丟失細節和語義信息。而 RAE 的核心思想是**在高維的語義空間中進行擴散**。它不學習一個壓縮的表示，而是利用一個**預訓練且凍結的強大視覺編碼器**（如 SigLIP-2）將圖像轉換為高維的語義表示，然後訓練一個輕量級的解碼器來從這些表示中重建圖像。

### 框架組成

RAE 框架主要由兩部分組成：

1.  **凍結的表示編碼器 (Frozen Representation Encoder)**：論文使用了 Google 的 SigLIP-2 模型。這個編碼器已經在海量數據上進行了預訓練，能夠提取豐富的語義特徵。

2.  **輕量級的解碼器 (Lightweight Decoder)**：這個解碼器的任務是將編碼器生成的高維語義表示還原為像素級的圖像。由於編碼器是固定的，訓練過程只集中在優化解碼器上。

### 訓練流程

![RAE 訓練流程](asset/RAE_Scaling/fig3-04.png)
*圖 2：RAE 的訓練流程圖。左側為解碼器訓練，右側為統一模型訓練。*

訓練分為兩個階段：

1.  **解碼器訓練**：使用 RAE 編碼器產生的表示（黃色標記）作為輸入，訓練 RAE 解碼器重建原始圖像。

2.  **統一模型訓練**：將文本 Token 和圖像的潛在表示輸入到一個自回歸模型中，然後由擴散 Transformer 進行處理，最終生成圖像。

### 數學公式解析

RAE 的訓練目標是最小化重建損失。對於給定的輸入圖像 $x$，編碼器 $E$ 將其映射為潛在表示 $z = E(x)$。解碼器 $D$ 則試圖從 $z$ 重建圖像 $\hat{x} = D(z)$。損失函數通常包含 LPIPS 損失、對抗性損失和 Gram 損失：

$$ L(x, \hat{x}) = \lambda_1 L_{\text{LPIPS}}(x, \hat{x}) + \lambda_2 L_{\text{Adv}}(x, \hat{x}) + \lambda_3 L_{\text{Gram}}(x, \hat{x}) $$

在擴散過程中，噪聲調度至關重要。論文強調了**維度相關的噪聲調度**的必要性。對於一個給定的時間步 $t \in [0, 1]$ 和參考維度 $n_{\text{ref}}$，調整後的時間步 $t_{\text{shift}}$ 計算如下：

$$ t_{\text{shift}} = \frac{t_{\text{ref}}}{1 + (\alpha - 1) t_{\text{ref}}} \quad \text{其中} \quad \alpha = \sqrt{\frac{m}{n_{\text{ref}}}} $$

這裡 $m$ 是實際的潛在維度。這個公式確保了噪聲水平能根據潛在空間的維度進行自適應調整，這對於在高維空間中進行有效的擴散至關重要。

## 實驗結果

論文進行了大量實驗來驗證 RAE 的有效性。

### 重建質量比較

下表比較了 RAE 和 VAE 在不同數據集上的重建性能（以 FID 分數衡量，越低越好）。

![RAE vs VAE 重建質量](asset/RAE_Scaling/tables-03.png)
*表 1 & 2：左側為不同訓練數據對 RAE 重建質量的影響；右側為 RAE 與 VAE 的重建性能比較。*

從右側的 **Table 2** 可以看出，無論是在 ImageNet、YFCC 還是文本數據集上，RAE 的 FID 分數都顯著低於 SDXL 和 FLUX 所使用的 VAE，證明了其卓越的重建能力。

### 擴展性和穩定性

實驗結果表明，RAE 具有出色的擴展性。隨著模型參數從 5 億增加到 98 億，其性能穩定提升。更重要的是，RAE 解決了 VAE 在長時間微調過程中普遍存在的過擬合問題，顯示出更強的訓練穩定性。

## 相關研究背景

這項研究建立在近年來圖像生成領域多個關鍵進展的基礎之上：

- **Diffusion Transformers (DiT)**：由 Peebles 等人在 2022 年提出，用 Transformer 替代 U-Net 作為擴散模型的骨幹網絡，顯著提升了模型的可擴展性。
- **Representation Learning**：以 CLIP、DINOv2 和 SigLIP 為代表的自監督學習模型，能夠學習到強大的視覺表示，為 RAE 提供了堅實的基礎。
- **Autoencoder 架構**：從 VAE、VQ-VAE 到本次提出的 RAE，自動編碼器一直是生成模型中用於數據壓縮和表示學習的核心組件。

## 個人評價與意義

這篇論文無疑是繼 DiT 之後，擴散模型領域的又一里程碑式的工作。Yann LeCun 的親自下場，也預示著這項技術的重要性。

RAE 的提出，可以說是一次「降維打擊」。它巧妙地繞開了 VAE 在表示能力和訓練穩定性上的瓶頸，通過「站在巨人（強大的預訓練編碼器）的肩膀上」，實現了更簡單、更高效、更強大的生成模型。這種「借力」的思想，也體現了當前 AI 領域「基礎模型 + 微調」這一主流範式的精髓。

更令人興奮的是，RAE 為**統一的視覺模型**開闢了道路。由於視覺理解（編碼）和生成（解碼）可以在同一個共享的語義空間中進行，未來我們或許能看到一個模型同時完成圖像識別、檢測、分割和生成等多種任務，實現真正的通用視覺智能。這項研究不僅僅是對現有技術的改進，更可能從根本上改變我們構建和思考生成模型的方式。

## 參考文獻

1.  Tong, S., et al. (2026). *Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders*. arXiv:2601.16208.
2.  Peebles, W., & Xie, S. (2023). *Scalable Diffusion Models with Transformers*. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
3.  Zheng, B., et al. (2025). *Diffusion Transformers with Representation Autoencoders*. arXiv:2510.11690.
4.  Tschannen, M., et al. (2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*. arXiv:2502.14786.
