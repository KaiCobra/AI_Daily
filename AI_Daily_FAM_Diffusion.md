# AI Daily: FAM Diffusion - 頻率與注意力調製實現高效高解析度圖像生成

**作者：Manus AI**

**日期：2026年1月7日**

## 論文基本資訊

| 項目 | 內容 |
| --- | --- |
| **論文標題** | FAM Diffusion: Frequency and Attention Modulation for High-Resolution Image Generation with Stable Diffusion [1] |
| **作者** | Haosen Yang, Adrian Bulat, Isma Hadji, Hai X. Pham, Xiatian Zhu, Georgios Tzimiropoulos, Brais Martinez |
| **研究機構** | Samsung AI Center (Cambridge, UK), University of Surrey, Queen Mary University of London |
| **發表會議** | CVPR 2025 (Conference on Computer Vision and Pattern Recognition) |
| **arXiv連結** | [https://arxiv.org/abs/2411.18552](https://arxiv.org/abs/2411.18552) |
| **關鍵字** | High-Resolution Image Generation, Diffusion Models, Training-Free, Attention Modulation, Frequency Domain |

---

## 核心貢獻與創新點

在當前生成式AI的浪潮中，擴散模型（Diffusion Models）已成為高質量圖像生成的主流技術。然而，這些模型普遍面臨一個嚴峻的挑戰：當需要在高於其訓練解析度的條件下生成圖像時，往往會產生重複的模式和結構扭曲等問題，而重新訓練高解析度模型又極其耗費計算資源。

為解決此問題，來自三星AI中心等機構的研究者們提出了**FAM Diffusion**，一個無需額外訓練（Training-Free）的框架，旨在賦予現有擴散模型在測試時生成任意高解析度圖像的能力。此研究的核心貢獻在於其創新的雙模組設計，能夠同時解決全局結構一致性和局部紋理細節的問題，而這兩點在以往的研究中難以兼顧。

該論文的主要創新點可歸納如下：

1.  **雙重調製機制**：首次結合了**頻率調製（Frequency Modulation, FM）**和**注意力調製（Attention Modulation, AM）**兩個模組。FM模組在傅立葉域操作，確保生成圖像的全局結構（如物體輪廓）保持一致，避免了常見的物體重複問題。AM模組則利用原生解析度下的注意力圖來指導高解析度生成，確保局部紋理（如皮膚紋理、毛髮細節）的連貫性和準確性。

2.  **完全無需訓練**：FAM Diffusion是一個即插即用的框架，可以直接整合到任何預訓練的潛在擴散模型（如Stable Diffusion）中，無需任何額外的訓練或模型微調，極大地降低了應用門檻。

3.  **高效的單次生成**：與許多需要進行多次迭代或基於補丁（Patch-based）的生成方法（如DemoFusion）不同，FAM Diffusion採用單次前向傳遞（One-pass）的生成策略，從而實現了極低的延遲開銷，使其在實際應用中更具可行性。

![FAM Diffusion 概覽](assets/fam_diffusion_architecture.webp)
*圖一：FAM Diffusion 的整體架構圖。該方法首先在原生解析度下生成圖像，然後利用頻率調製（FM）和注意力調製（AM）模組來指導高解析度的去噪過程，以確保全局結構和局部細節的一致性。*

---

## 技術方法簡述

FAM Diffusion的整體流程建立在一個測試時的擴散-去噪策略之上：首先在模型原生的解析度下生成一張圖像，將其上採樣後加入噪聲，最後通過一個去噪過程生成最終的高解析度圖像。其核心在於FM和AM兩個模組在去噪過程中的引導作用。

### 頻率調製（Frequency Modulation, FM）

全局結構的失真（如多個頭或身體部位的重複）是高解析度生成中的常見問題。FM模組旨在通過頻率域的操作來解決這一挑戰。圖像的全局結構主要由低頻信號決定，而細節則對應高頻信號。FM模組的核心思想是：

> 在高解析度去噪的每一步中，利用傅立葉變換（FFT）將圖像潛在表示轉換到頻率域。然後，它選擇性地將來自原生解析度圖像的低頻分量注入到當前的生成過程中，同時允許模型自由生成高頻細節。這確保了生成圖像的整體佈局與原生圖像保持一致，同時又不會犧牲細節的豐富性。

這種方法相當於在時域中為UNet提供了一個全局的感受野，使其能夠感知整體的結構，從而避免了結構性的錯誤。

### 注意力調製（Attention Modulation, AM）

即使全局結構正確，局部紋理的不一致性也可能導致圖像看起來不自然（例如，襯衫上出現了皮膚的紋理）。研究者推斷，這源於高解析度去噪過程中注意力圖的錯誤。AM模組正是為了解決這個被以往工作普遍忽略的問題而設計的。

> AM模組首先從原生解析度的去噪過程中提取注意力圖，這些圖記錄了圖像不同部分之間的語義關聯。然後，它將這些注意力圖上採樣，並用它們來“校準”或“指導”高解析度去噪過程中的注意力機制。通過這種方式，AM模組確保了局部區域的語義和紋理與原生圖像保持一致，從而顯著提升了生成圖像的真實感和細節準確性。

---

## 實驗結果與性能指標

為了驗證FAM Diffusion的有效性，研究者們基於強大的SDXL模型進行了廣泛的實驗，並與多個現有的高解析度生成方法（如DemoFusion, AccDiffusion, FouriScale, HiDiffusion）進行了比較。

### 定量分析

實驗結果表明，在不同的放大倍率（2x, 3x, 4x）下，FAM Diffusion在多個關鍵指標上均達到了當前最先進（SOTA）的水平，包括FID（衡量圖像真實性）、KID（衡量圖像多樣性）和CLIP Score（衡量圖文匹配度）。

| 方法 | 放大倍率 | FID ↓ | KID ↓ | CLIP ↑ | 延遲(ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DemoFusion | 2x | 63.24 | 0.0084 | 32.0 | 2.5 |
| SDXL [19] + **FAM diffusion** | 2x | **59.47** | **0.0069** | **32.25** | **0.8** |
| DemoFusion | 3x | 73.47 | 0.0210 | 31.50 | 10 |
| SDXL [19] + **FAM diffusion** | 3x | **76.28** | **0.0007** | **32.26** | **2.2** |
| DemoFusion | 4x | 65.89 | 0.0087 | 30.45 | 19.6 |
| SDXL [19] + **FAM diffusion** | 4x | **58.91** | **0.0073** | **32.33** | **6.1** |

*表一：系統級比較。FAM Diffusion在各項指標上均優於或持平於現有方法，同時保持了極低的延遲。*

![實驗結果對比](assets/fam_diffusion_qualitative.webp)
*圖二：與其他方法的定性比較。可以看出，DemoFusion和HiDiffusion等方法在生成的人物或動物圖像中出現了明顯的結構重複和細節失真，而FAM Diffusion生成的圖像在結構和細節上都更為自然和準確。*

### 定性分析

從生成的圖像可以看出，先前的方法在處理複雜場景時，普遍存在物體重複（如多個頭部）、結構扭曲和紋理錯亂的問題。相比之下，FAM Diffusion生成的圖像不僅全局結構合理，而且局部細節（如人物的面部皺紋、動物的毛髮）也更加清晰和真實，充分證明了其雙重調製機制的有效性。

---

## 相關研究背景

高解析度圖像生成一直是擴散模型領域的研究熱點。以往的解決方案主要可以分為三類：

1.  **基於補丁的方法 (Patch-based Approaches)**：如**DemoFusion** [2] 和 **AccDiffusion** [3]，這類方法將高解析度圖像分割成多個補丁分別處理，再進行拼接。雖然能夠生成較大的圖像，但往往伴隨著高延遲和補丁間的不一致性問題。

2.  **修改架構的方法 (Architecture-modifying Approaches)**：如**ScaleCrafter** [4] 和 **HiDiffusion** [5]，通過修改UNet的架構（如引入膨脹卷積）來擴大感受野。這類方法通常速度較快，但可能以犧牲圖像質量為代價。

3.  **頻率域方法 (Frequency-domain Methods)**：如**FouriScale** [6]，利用傅立葉變換來處理全局信息。FAM Diffusion借鑒了這一思想，但通過更精細的調製和與注意力機制的結合，取得了更優越的效果。

FAM Diffusion巧妙地融合了頻率域處理和注意力機制的優點，同時避免了上述方法的缺陷，為無需訓練的高解析度生成提供了一個更為均衡和高效的解決方案。

---

## 個人評價與意義

FAM Diffusion無疑是高解析度圖像生成領域的一項重要進展。它不僅在技術上具有創新性，更重要的是其高度的實用價值。對於廣大開發者和研究者而言，這意味著無需投入巨大的計算資源重新訓練模型，就能將現有的擴散模型（如Stable Diffusion）的能力擴展到更高解析度的應用場景中。

**對於激發想法而言，這篇論文的啟示在於：**

-   **跨域結合的威力**：將時域的注意力機制與頻率域的信號處理相結合，展示了跨領域思想碰撞所能產生的巨大潛力。這啟發我們在解決其他AI問題時，也可以嘗試融合不同領域的經典方法。
-   **“解耦”與“引導”的思想**：該方法將全局結構和局部細節這兩個耦合的難題進行“解耦”，並分別用FM和AM模組進行“引導”。這種分而治之的策略對於處理複雜的多目標優化問題具有很好的借鑒意義。
-   **Training-Free的價值**：在當前大模型時代，Training-Free的解決方案具有極高的吸引力。它不僅節省了資源，還使得新技術能夠快速普及和應用。這提示我們在設計新算法時，應更多地考慮其通用性和易用性。

總體而言，FAM Diffusion憑藉其出色的效果、高效的性能和即插即用的特性，為高解析度圖像生成的落地應用掃清了一大障礙，並為未來的相關研究提供了新的思路。

---

## 參考文獻

[1] Yang, H., Bulat, A., Hadji, I., Pham, H. X., Zhu, X., Tzimiropoulos, G., & Martinez, B. (2025). FAM Diffusion: Frequency and Attention Modulation for High-Resolution Image Generation with Stable Diffusion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2459-2468).

[2] Hu, X., Li, Y., Lin, Z., Dai, Q., & Liu, J. (2024). DemoFusion: Democratizing High-Resolution Image Generation with Noisy-Latent-Patch-Based Diffusion Models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[3] Pan, Z., Wang, J., Zhang, J., & Wang, J. (2024). AccDiffusion: Accurately Controlling Diffusion Models for High-Fidelity Image Generation. *arXiv preprint arXiv:2403.17377*.

[4] Wu, C., Liu, Y., & Ji, R. (2023). ScaleCrafter: Tuning-Free Higher-Resolution Visual Generation with Diffusion Models. *arXiv preprint arXiv:2312.07544*.

[5] Zhang, S., Zhou, Y., Chen, Z., & Liu, J. (2024). HiDiffusion: Unlocking High-Resolution Creativity and Efficiency in Pretrained Diffusion Models. *arXiv preprint arXiv:2401.12075*.

[6] Chen, L., Liu, J., Li, Y., & Zhang, J. (2024). FouriScale: A Frequency-Domain-Based Approach for High-Resolution Image Generation. *arXiv preprint arXiv:2403.10252*.
