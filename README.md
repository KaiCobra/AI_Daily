# AI Daily

## 今日閱讀

**[Just-in-Time (JiT) — 2026-03-17：免訓練空間加速 Diffusion Transformers，FLUX.1-dev 實現 7x 加速 (CVPR 2026)](papers/2026/2026-03/JiT/AI_Daily_JiT.md)**

本文提出 **JiT（Just-in-Time）**（CVPR 2026），一個**免訓練（Training-Free）**的 Diffusion Transformers 空間加速框架。其核心洞見在於：圖像生成是「先全局後細節」的過程，對所有空間區域均等計算是嚴重的效率浪費。JiT 設計了**空間近似生成 ODE（SAG-ODE）**，通過動態選擇的稀疏 anchor token 子集驅動完整 latent state 演化（$\frac{d\mathbf{y}}{dt} = \mathbf{\Pi}_k \mathbf{u}_\theta(\mathbf{S}_k^\top \mathbf{y}, t)$），並以**確定性微流（DMF）**確保新 token 激活時的無縫過渡，配合**重要性引導的 Token 激活（ITA）**動態分配計算資源到高頻細節區域。在 FLUX.1-dev 上，JiT 以 **7x** 加速（3.67 秒）實現幾乎無損的生成質量，ImageReward 達 0.9746，T2I-Comp 達 0.4961，全面超越 RALU、Bottleneck Sampling、TeaCache 等方法。

---

**[StyleGallery — 2026-03-15：免訓練語義感知個性化風格遷移，支持任意數量參考圖](papers/2026/2026-03/StyleGallery/AI_Daily_StyleGallery.md)**

本文提出 **StyleGallery**（CVPR 2026），一個**免訓練（Training-Free）**且**語義感知（Semantic-aware）**的圖像風格遷移框架。其核心創新在於：利用擴散模型內部的 UNet 中間特徵進行 K-means 語義聚類，通過三維相似度（統計特徵、DINOv2 語義相似度、幾何位置）實現精確的區域匹配，並以 **Sparse Attention**（語義遮罩稀疏化 Q/K/V）結合能量函數引導採樣優化，在無需額外訓練或語義分割遮罩的情況下，實現細粒度的區域級風格遷移。StyleGallery 支持任意數量的風格參考圖，在 ArtFID 指標上以 **24.536** 大幅超越次優方法 StyTR-2（25.804），為個性化風格定制提供了全新的 Zero-shot 解決方案。

---

**[Delta-K — 2026-03-14：免訓練 Cross-Attention Key 空間增強，解決多實例概念遺漏問題](papers/2026/2026-03/Delta-K/AI_Daily_Delta-K.md)**

本文提出 **Delta-K**，一個與主幹網路無關、隨插即用的**免訓練（Training-Free）**推理框架，專門解決文本到圖像生成中的**多實例概念遺漏（Concept Omission）**問題。論文核心洞見在於：概念遺漏並非單純的注意力激活不足，而是發生在擴散採樣**早期語義規劃階段**的 Key 空間語義匹配失敗。Delta-K 透過 VLM 識別缺失概念，計算差異鍵向量 $\Delta K = K_{input}(P) - K_{input}(P_{mask})$，並在早期去噪步驟中動態注入，配合 Adam 線上優化調度強度 $\alpha_t$，在 T2I-CompBench 上將 SDXL 的 Complex 分數提升 +0.0302，Spatial 提升 +0.0355，同時保持推理效率。

---

> 每天一篇論文，跟上 AI 最新潮流

AI 領域發展太快，每天都有新論文發表，卻沒時間一一追蹤？**AI Daily** 為你精選每天最值得關注的 AI 論文，用深入淺出的方式解析最前沿的研究成果。

## 我們在做什麼？

每天從海量論文中，挑選最具價值的研究進行深度解析：

- **精選優質論文** — 只挑最新、最有影響力的研究
- **深入淺出解析** — 複雜理論也能輕鬆理解
- **每日更新** — 保持對 AI 領域的敏銳度
- **涵蓋熱門領域** — 深度學習、圖像生成、計算機視覺等

### 關注焦點

- **圖像生成技術** — Diffusion Models、Autoregressive Models、Flow Matching
- **深度學習前沿** — 最新架構與訓練方法
- **計算機視覺** — 圖像理解、深度估計、多模態學習

### 論文來源

來自全球頂尖研究機構的最新成果：Stanford、MIT、清華、北大、Meta FAIR、NVIDIA、Google、Adobe 等。優先收錄 CVPR、ICCV、NeurIPS、ICML、AAAI、ICLR 等頂級會議論文。

---

## 目錄結構

```
papers/
├── YYYY/
│   └── YYYY-MM/
│       └── PaperName/
│           ├── AI_Daily_PaperName.md   # 論文閱讀筆記
│           └── asset/                   # 論文圖片（從原始 PDF 提取）
skills/
└── pdf-image-extractor/                 # PDF 圖片提取工具
```

---

## 論文索引（按日期）

### 2023

| 日期 | 論文 | 主題 | arXiv |
|------|------|------|-------|
| 2023-01 | [ZestGuide](papers/2023/2023-01/ZestGuide/AI_Daily_ZestGuide.md) | Zero-shot 空間佈局條件生成 | [2306.13754](https://arxiv.org/abs/2306.13754) |

### 2024

| 日期 | 論文 | 主題 | arXiv |
|------|------|------|-------|
| 2024-11 | [SoFlow](papers/2024/2024-11/SoFlow/AI_Daily_SoFlow.md) | Second-order Flow Matching | [2411.05520](https://arxiv.org/abs/2411.05520) |
| 2024-12 | [FlowAR](papers/2024/2024-12/FlowAR/AI_Daily_FlowAR.md) | Flow Matching + Autoregressive | [2412.15205](https://arxiv.org/abs/2412.15205) |

### 2025

| 日期 | 論文 | 主題 | arXiv |
|------|------|------|-------|
| 2025-01 | [DiverseVAR](papers/2025/2025-01/DiverseVAR/AI_Daily_DiverseVAR.md) | VAR 多樣性生成 | [2501.09751](https://arxiv.org/abs/2501.09751) |
| 2025-01 | [EditAR](papers/2025/2025-01/EditAR/AI_Daily_EditAR.md) | 自回歸圖像編輯 | [2501.04699](https://arxiv.org/abs/2501.04699) |
| 2025-01 | [JanusFlow](papers/2025/2025-01/JanusFlow/AI_Daily_JanusFlow.md) | 統一多模態理解與生成 | [2411.07975](https://arxiv.org/abs/2411.07975) |
| 2025-01 | [Multimodal AR Vision Encoders](papers/2025/2025-01/Multimodal_AR_Vision_Encoders/AI_Daily_Multimodal_AR_Vision_Encoders.md) | 多模態自回歸視覺編碼器 | [2501.01030](https://arxiv.org/abs/2501.01030) |
| 2025-01 | [RadAR](papers/2025/2025-01/RadAR/AI_Daily_RadAR.md) | 放射學自回歸模型 | [2501.04382](https://arxiv.org/abs/2501.04382) |
| 2025-01 | [VAR Depth Estimation](papers/2025/2025-01/VAR_Depth_Estimation/AI_Daily_VAR_Depth_Estimation.md) | VAR 深度估計 | [2512.22653](https://arxiv.org/abs/2512.22653) |
| 2025-03 | [AREdit](papers/2025/2025-03/AREdit/AI_Daily_AREdit.md) | 無需訓練的 VAR 圖像編輯 | [2501.13101](https://arxiv.org/abs/2501.13101) |

### 2026-01

| 日期 | 論文 | 主題 | arXiv |
|------|------|------|-------|
| 2026-01 | [AR-Omni](papers/2026/2026-01/AR-Omni/AI_Daily_AR-Omni.md) | 純自回歸 Any-to-Any 多模態生成 | [2601.06849](https://arxiv.org/abs/2601.06849) |
| 2026-01 | [Alterbute](papers/2026/2026-01/Alterbute/AI_Daily_Alterbute.md) | 編輯物體內在屬性 | [2601.10714](https://arxiv.org/abs/2601.10714) |
| 2026-01 | [Decentralized AR](papers/2026/2026-01/Decentralized_AR/AI_Daily_Decentralized_AR.md) | 去中心化自回歸生成 | [2601.03915](https://arxiv.org/abs/2601.03915) |
| 2026-01 | [DreamVAR](papers/2026/2026-01/DreamVAR/AI_Daily_DreamVAR.md) | VAR + 強化學習主體驅動生成 | [2601.01122](https://arxiv.org/abs/2601.01122) |
| 2026-01 | [FAM Diffusion](papers/2026/2026-01/FAM_Diffusion/AI_Daily_FAM_Diffusion.md) | 頻率與注意力調製高效生成 | [2411.18552](https://arxiv.org/abs/2411.18552) |
| 2026-01 | [FlexVAR](papers/2026/2026-01/FlexVAR/AI_Daily_FlexVAR.md) | 靈活高效視覺自回歸生成 | [2601.06850](https://arxiv.org/abs/2601.06850) |
| 2026-01 | [LooseRoPE](papers/2026/2026-01/LooseRoPE/AI_Daily_LooseRoPE.md) | 內容感知注意力操控 | [2601.07256](https://arxiv.org/abs/2601.07256) |
| 2026-01 | [MMD-Guidance](papers/2026/2026-01/MMD-Guidance/AI_Daily_MMD-Guidance.md) | 無需訓練的分佈自適應引導 | [2601.08379](https://arxiv.org/abs/2601.08379) |
| 2026-01 | [Mirai](papers/2026/2026-01/Mirai/AI_Daily_Mirai.md) | 自回歸視覺生成預見未來 | [2601.06838](https://arxiv.org/abs/2601.06838) |
| 2026-01 | [PackCache](papers/2026/2026-01/PackCache/AI_Daily_PackCache.md) | 無需訓練加速自回歸影片生成 | [2601.07777](https://arxiv.org/abs/2601.07777) |
| 2026-01 | [RAE](papers/2026/2026-01/RAE/AI_Daily_RAE.md) | 擴散 + 自回歸挑戰 VAE | [2601.16208](https://arxiv.org/abs/2601.16208) |
| 2026-01 | [ResTok](papers/2026/2026-01/ResTok/AI_Daily_ResTok.md) | 高效高保真視覺自回歸生成 | [2601.03955](https://arxiv.org/abs/2601.03955) |
| 2026-01 | [SSG](papers/2026/2026-01/SSG/AI_Daily_SSG.md) | 尺度空間引導解放 VAR 潛力 | [2602.05534](https://arxiv.org/abs/2602.05534) |
| 2026-01 | [Sissi](papers/2026/2026-01/Sissi/AI_Daily_Sissi.md) | 零樣本風格引導圖像生成 | [2601.06605](https://arxiv.org/abs/2601.06605) |
| 2026-01 | [SoftCFG](papers/2026/2026-01/SoftCFG/AI_Daily_SoftCFG.md) | 不確定性引導穩定 VAR | [2510.00996](https://arxiv.org/abs/2510.00996) |
| 2026-01 | [TP-Blend](papers/2026/2026-01/TP-Blend/AI_Daily_TP-Blend.md) | 雙提示注意力配對物體風格融合 | [2601.08011](https://arxiv.org/abs/2601.08011) |
| 2026-01 | [ToProVAR](papers/2026/2026-01/ToProVAR/AI_Daily_ToProVAR.md) | 注意力熵高效優化 VAR | [2506.08908](https://arxiv.org/abs/2506.08908) |
| 2026-01 | [Unraveling MMDiT](papers/2026/2026-01/Unraveling_MMDiT/AI_Daily_Unraveling_MMDiT.md) | 無需訓練分析增強 MMDiT | [2601.02211](https://arxiv.org/abs/2601.02211) |
| 2026-01 | [Untwisting RoPE](papers/2026/2026-01/Untwisting_RoPE/AI_Daily_Untwisting_RoPE.md) | DiT 共享注意力頻率控制 | [2602.05013](https://arxiv.org/abs/2602.05013) |
| 2026-01 | [VAR-LIDE](papers/2026/2026-01/VAR-LIDE/AI_Daily_VAR-LIDE.md) | VAR + VLM 零參考圖像修復 | [2511.18591](https://arxiv.org/abs/2511.18591) |
| 2026-01 | [VAR-Scaling](papers/2026/2026-01/VAR-Scaling/AI_Daily_VAR-Scaling.md) | VAR 推理時縮放策略 | [2601.07293](https://arxiv.org/abs/2601.07293) |
| 2026-01 | [VAR RL Done Right](papers/2026/2026-01/VAR_RL_Done_Right/AI_Daily_VAR_RL_Done_Right.md) | VAR 異步策略衝突解決 | [2601.02256](https://arxiv.org/abs/2601.02256) |
| 2026-01 | [VideoAR](papers/2026/2026-01/VideoAR/AI_Daily_VideoAR.md) | 自回歸影片生成 | [2601.05966](https://arxiv.org/abs/2601.05966) |
| 2026-01 | [iFSQ](papers/2026/2026-01/iFSQ/AI_Daily_iFSQ.md) | 統一 AR 與 Diffusion 高效生成 | [2601.17124](https://arxiv.org/abs/2601.17124) |

### 2026-02

| 日期 | 論文 | 主題 | arXiv |
|------|------|------|-------|
| 2026-02 | [ARPG](papers/2026/2026-02/ARPG/AI_Daily_ARPG.md) | 隨機並行解碼自回歸生成 | [2503.10568](https://arxiv.org/abs/2503.10568) |
| 2026-02 | [BAR](papers/2026/2026-02/BAR/AI_Daily_BAR.md) | Masked Bit Modeling 自回歸生成 | [2602.09024](https://arxiv.org/abs/2602.09024) |
| 2026-02 | [BitDance](papers/2026/2026-02/BitDance/AI_Daily_BitDance.md) | 二元 Token 自回歸生成 | [2602.14041](https://arxiv.org/abs/2602.14041) |
| 2026-02 | [DCAG](papers/2026/2026-02/DCAG/AI_Daily_DCAG.md) | 雙通道注意力引導圖像編輯 | [2602.18022](https://arxiv.org/abs/2602.18022) |
| 2026-02 | [DIAMOND](papers/2026/2026-02/DIAMOND/AI_Daily_DIAMOND.md) | 直接推論修正 Flow Matching | [2602.00883](https://arxiv.org/abs/2602.00883) |
| 2026-02 | [EchoGen](papers/2026/2026-02/EchoGen/AI_Daily_EchoGen.md) | VAR 主體驅動零樣本合成 | [2509.26127](https://arxiv.org/abs/2509.26127) |
| 2026-02 | [FusionEdit](papers/2026/2026-02/FusionEdit/AI_Daily_FusionEdit.md) | 語義融合注意力調製圖像編輯 | [2602.08725](https://arxiv.org/abs/2602.08725) |
| 2026-02 | [HSI-VAR](papers/2026/2026-02/HSI-VAR/AI_Daily_HSI-VAR.md) | VAR 高光譜影像修復 | [2602.00749](https://arxiv.org/abs/2602.00749) |
| 2026-02 | [LapFlow](papers/2026/2026-02/LapFlow/AI_Daily_LapFlow.md) | 拉普拉斯金字塔 + 流匹配 | [2602.19461](https://arxiv.org/abs/2602.19461) |
| 2026-02 | [Light Forcing](papers/2026/2026-02/Light_Forcing/AI_Daily_Light_Forcing.md) | 稀疏注意力加速自回歸影片擴散 | [2602.04789](https://arxiv.org/abs/2602.04789) |
| 2026-02 | [Look-Ahead Look-Back Flows](papers/2026/2026-02/Look_Ahead_Look_Back_Flows/AI_Daily_Look_Ahead_Look_Back_Flows.md) | 無需訓練軌跡平滑化生成 | [2602.09449](https://arxiv.org/abs/2602.09449) |
| 2026-02 | [NOVA](papers/2026/2026-02/NOVA/AI_Daily_NOVA.md) | 熵引導 VAR 自適應加速 | [2602.01345](https://arxiv.org/abs/2602.01345) |
| 2026-02 | [PTQ4ARVG](papers/2026/2026-02/PTQ4ARVG/AI_Daily_PTQ4ARVG.md) | 自回歸視覺生成量化框架 | [2601.21238](https://arxiv.org/abs/2601.21238) |
| 2026-02 | [RFC](papers/2026/2026-02/RFC/AI_Daily_RFC.md) | 輸出入關係加速 DiT 特徵快取 | [2602.19506](https://arxiv.org/abs/2602.19506) |
| 2026-02 | [Semantic Bottleneck](papers/2026/2026-02/Semantic_Bottleneck/AI_Daily_Semantic_Bottleneck.md) | DiT 條件嵌入語義瓶頸 (ICLR 2026) | [2602.21596](https://arxiv.org/abs/2602.21596) |
| 2026-02 | [SparVAR](papers/2026/2026-02/SparVAR/AI_Daily_SparVAR.md) | VAR 稀疏性高效加速 | [2602.04361](https://arxiv.org/abs/2602.04361) |
| 2026-02 | [VAREdit](papers/2026/2026-02/VAREdit/AI_Daily_VAREdit.md) | 指令引導 VAR 圖像編輯 | [2508.15772](https://arxiv.org/abs/2508.15772) |

### 2026-03

| 日期 | 論文 | 主題 | arXiv |
|------|------|------|-------|
| 2026-03 | [AlignVAR](papers/2026/2026-03/AlignVAR/AI_Daily_AlignVAR.md) | VAR 全域一致性超解析度 | [2603.00589](https://arxiv.org/abs/2603.00589) |
| 2026-03 | [MVAR](papers/2026/2026-03/MVAR/AI_Daily_MVAR.md) | 馬可夫假設線性複雜度 VAR | [2505.12742](https://arxiv.org/abs/2505.12742) |
| 2026-03 | [PixelRush](papers/2026/2026-03/PixelRush/AI_Daily_PixelRush.md) | 20秒 4K 單步擴散模型 | [2602.12769](https://arxiv.org/abs/2602.12769) |
| 2026-03 | [SCALAR](papers/2026/2026-03/SCALAR/AI_Daily_SCALAR.md) | VAR 尺度感知可控生成 | [2507.19946](https://arxiv.org/abs/2507.19946) |
| 2026-03 | [StepVAR](papers/2026/2026-03/StepVAR/AI_Daily_StepVAR.md) | 結構紋理引導 VAR 剪枝 | [2603.01757](https://arxiv.org/abs/2603.01757) |
| 2026-03 | [AREdit](papers/2026/2026-03/AREdit/AI_Daily_AREdit.md) | 首個 VAR-based 免訓練文字引導圖像編輯 (ICCV 2025) | [2503.23897](https://arxiv.org/abs/2503.23897) |
| 2026-03-09 | [Self-Flow](papers/2026/2026-03/SelfFlow/AI_Daily_SelfFlow.md) | 自監督 Flow Matching，Dual-Timestep 信息不對稱 | [2603.06507](https://arxiv.org/abs/2603.06507) |
| 2026-03-11 | [LayerBind](papers/2026/2026-03/LayerBind/AI_Daily_LayerBind.md) | 免訓練 DiT 區域佈局與遮擋控制 (CVPR 2026) | [2603.05769](https://arxiv.org/abs/2603.05769) |
| 2026-03-13 | [Scale-wise AR Style-Aligned](papers/2026/2026-03/2026-03-13-Scale-wise-Autoregressive-Style-Aligned.md) | 首個 VAR-based 免訓練風格對齊圖像生成，推理速度快 6× 以上 | [2504.06144](https://arxiv.org/abs/2504.06144) |
| 2026-03-12 | [KV-Lock & RAISE](papers/2026/2026-03/2026-03-12-KV-Lock-and-RAISE.md) | 免訓練 KV 注意力控制影片編輯 & 需求自適應演化 T2I 對齊 (CVPR 2026) | [2603.09657](https://arxiv.org/abs/2603.09657) / [2603.00483](https://arxiv.org/abs/2603.00483) |
| 2026-03 | [ATM (ISLock)](papers/2026-03-07-ATM-ISLock.md) | 首個 AR 模型免訓練圖像編輯 (ICCV 2025) | [2504.10434](https://arxiv.org/abs/2504.10434) |
| 2026-03 | [Rethinking Global Text Conditioning](papers/2026/2026-03/Rethinking_Global_Text_Conditioning/AI_Daily_Rethinking_Global_Text_Conditioning.md) | DiT 全域文本條件機制 | [2602.09268](https://arxiv.org/abs/2602.09268) |
| 2026-03-14 | [Delta-K](papers/2026/2026-03/Delta-K/AI_Daily_Delta-K.md) | 免訓練 Cross-Attention Key 空間增強，解決多實例概念遺漏 | [2603.10210](https://arxiv.org/abs/2603.10210) |
| 2026-03-15 | [StyleGallery](papers/2026/2026-03/StyleGallery/AI_Daily_StyleGallery.md) | 免訓練語義感知個性化風格遷移，支持任意數量參考圖 (CVPR 2026) | [2603.10354](https://arxiv.org/abs/2603.10354) |
| 2026-03-17 | [JiT](papers/2026/2026-03/JiT/AI_Daily_JiT.md) | 免訓練空間加速 DiT，FLUX.1-dev 實現 7x 加速 (CVPR 2026) | [2603.10744](https://arxiv.org/abs/2603.10744) |

### 比較分析

| 日期 | 報告 | 主題 |
|------|------|------|
| 2026-03 | [Modulation Guidance 比較](papers/2026/2026-03/Modulation_Guidance_Comparison/AI_Daily_Modulation_Guidance_Comparison.md) | Modulation Guidance 系列論文深度比較 |
| 2026-03 | [雙維度控制場景分析](papers/2026/2026-03/Dual_Dimensional_Control_Scenarios/AI_Daily_Dual_Dimensional_Control_Scenarios.md) | 雙維度協同控制應用場景案例分析 |

---

## 論文索引（按主題）

### Visual Autoregressive Models (VAR)

**架構與改進：** [FlexVAR](papers/2026/2026-01/FlexVAR/AI_Daily_FlexVAR.md) | [DiverseVAR](papers/2025/2025-01/DiverseVAR/AI_Daily_DiverseVAR.md) | [ResTok](papers/2026/2026-01/ResTok/AI_Daily_ResTok.md) | [RAE](papers/2026/2026-01/RAE/AI_Daily_RAE.md) | [iFSQ](papers/2026/2026-01/iFSQ/AI_Daily_iFSQ.md) | [BAR](papers/2026/2026-02/BAR/AI_Daily_BAR.md) | [BitDance](papers/2026/2026-02/BitDance/AI_Daily_BitDance.md) | [MVAR](papers/2026/2026-03/MVAR/AI_Daily_MVAR.md)

**加速與效率：** [SparVAR](papers/2026/2026-02/SparVAR/AI_Daily_SparVAR.md) | [NOVA](papers/2026/2026-02/NOVA/AI_Daily_NOVA.md) | [PackCache](papers/2026/2026-01/PackCache/AI_Daily_PackCache.md) | [ToProVAR](papers/2026/2026-01/ToProVAR/AI_Daily_ToProVAR.md) | [StepVAR](papers/2026/2026-03/StepVAR/AI_Daily_StepVAR.md) | [PTQ4ARVG](papers/2026/2026-02/PTQ4ARVG/AI_Daily_PTQ4ARVG.md) | [VAR-Scaling](papers/2026/2026-01/VAR-Scaling/AI_Daily_VAR-Scaling.md)

**引導與控制：** [SSG](papers/2026/2026-01/SSG/AI_Daily_SSG.md) | [SoftCFG](papers/2026/2026-01/SoftCFG/AI_Daily_SoftCFG.md) | [SCALAR](papers/2026/2026-03/SCALAR/AI_Daily_SCALAR.md) | [VAR RL Done Right](papers/2026/2026-01/VAR_RL_Done_Right/AI_Daily_VAR_RL_Done_Right.md)

**應用（編輯/修復/超解析度）：** [AREdit (ICCV 2025)](papers/2026/2026-03/AREdit/AI_Daily_AREdit.md) | [AREdit (2025-03)](papers/2025/2025-03/AREdit/AI_Daily_AREdit.md) | [EditAR](papers/2025/2025-01/EditAR/AI_Daily_EditAR.md) | [VAREdit](papers/2026/2026-02/VAREdit/AI_Daily_VAREdit.md) | [VAR-LIDE](papers/2026/2026-01/VAR-LIDE/AI_Daily_VAR-LIDE.md) | [AlignVAR](papers/2026/2026-03/AlignVAR/AI_Daily_AlignVAR.md) | [HSI-VAR](papers/2026/2026-02/HSI-VAR/AI_Daily_HSI-VAR.md) | [VAR Depth Estimation](papers/2025/2025-01/VAR_Depth_Estimation/AI_Daily_VAR_Depth_Estimation.md) | [ATM (ISLock)](papers/2026-03-07-ATM-ISLock.md)

**風格與主體驅動：** [Scale-wise AR Style-Aligned](papers/2026/2026-03/2026-03-13-Scale-wise-Autoregressive-Style-Aligned.md) | [DreamVAR](papers/2026/2026-01/DreamVAR/AI_Daily_DreamVAR.md) | [EchoGen](papers/2026/2026-02/EchoGen/AI_Daily_EchoGen.md) | [Sissi](papers/2026/2026-01/Sissi/AI_Daily_Sissi.md)

### Autoregressive Generation (General)

[FlowAR](papers/2024/2024-12/FlowAR/AI_Daily_FlowAR.md) | [Mirai](papers/2026/2026-01/Mirai/AI_Daily_Mirai.md) | [AR-Omni](papers/2026/2026-01/AR-Omni/AI_Daily_AR-Omni.md) | [Decentralized AR](papers/2026/2026-01/Decentralized_AR/AI_Daily_Decentralized_AR.md) | [ARPG](papers/2026/2026-02/ARPG/AI_Daily_ARPG.md) | [VideoAR](papers/2026/2026-01/VideoAR/AI_Daily_VideoAR.md) | [Multimodal AR Vision Encoders](papers/2025/2025-01/Multimodal_AR_Vision_Encoders/AI_Daily_Multimodal_AR_Vision_Encoders.md) | [RadAR](papers/2025/2025-01/RadAR/AI_Daily_RadAR.md)

### Diffusion Models & Flow Matching

**自監督表示學習：** [Self-Flow](papers/2026/2026-03/SelfFlow/AI_Daily_SelfFlow.md)

**架構與機制分析：** [Unraveling MMDiT](papers/2026/2026-01/Unraveling_MMDiT/AI_Daily_Unraveling_MMDiT.md) | [Semantic Bottleneck](papers/2026/2026-02/Semantic_Bottleneck/AI_Daily_Semantic_Bottleneck.md) | [Rethinking Global Text Conditioning](papers/2026/2026-03/Rethinking_Global_Text_Conditioning/AI_Daily_Rethinking_Global_Text_Conditioning.md) | [Untwisting RoPE](papers/2026/2026-01/Untwisting_RoPE/AI_Daily_Untwisting_RoPE.md)

**引導與加速：** [MMD-Guidance](papers/2026/2026-01/MMD-Guidance/AI_Daily_MMD-Guidance.md) | [Light Forcing](papers/2026/2026-02/Light_Forcing/AI_Daily_Light_Forcing.md) | [RFC](papers/2026/2026-02/RFC/AI_Daily_RFC.md) | [DIAMOND](papers/2026/2026-02/DIAMOND/AI_Daily_DIAMOND.md) | [Look-Ahead Look-Back Flows](papers/2026/2026-02/Look_Ahead_Look_Back_Flows/AI_Daily_Look_Ahead_Look_Back_Flows.md) | [SoFlow](papers/2024/2024-11/SoFlow/AI_Daily_SoFlow.md) | [LapFlow](papers/2026/2026-02/LapFlow/AI_Daily_LapFlow.md)

**高效生成：** [FAM Diffusion](papers/2026/2026-01/FAM_Diffusion/AI_Daily_FAM_Diffusion.md) | [PixelRush](papers/2026/2026-03/PixelRush/AI_Daily_PixelRush.md) | [JiT (CVPR 2026)](papers/2026/2026-03/JiT/AI_Daily_JiT.md)

### Image Editing (Training-Free)

[AREdit (ICCV 2025)](papers/2026/2026-03/AREdit/AI_Daily_AREdit.md) | [ATM (ISLock)](papers/2026-03-07-ATM-ISLock.md) | [DCAG](papers/2026/2026-02/DCAG/AI_Daily_DCAG.md) | [FusionEdit](papers/2026/2026-02/FusionEdit/AI_Daily_FusionEdit.md) | [Alterbute](papers/2026/2026-01/Alterbute/AI_Daily_Alterbute.md) | [LooseRoPE](papers/2026/2026-01/LooseRoPE/AI_Daily_LooseRoPE.md) | [TP-Blend](papers/2026/2026-01/TP-Blend/AI_Daily_TP-Blend.md) | [ZestGuide](papers/2023/2023-01/ZestGuide/AI_Daily_ZestGuide.md) | [LayerBind](papers/2026/2026-03/LayerBind/AI_Daily_LayerBind.md) | [Delta-K](papers/2026/2026-03/Delta-K/AI_Daily_Delta-K.md)

### Style Transfer (Training-Free)

[StyleGallery (CVPR 2026)](papers/2026/2026-03/StyleGallery/AI_Daily_StyleGallery.md) | [Scale-wise AR Style-Aligned](papers/2026/2026-03/2026-03-13-Scale-wise-Autoregressive-Style-Aligned.md) | [Sissi](papers/2026/2026-01/Sissi/AI_Daily_Sissi.md) | [TP-Blend](papers/2026/2026-01/TP-Blend/AI_Daily_TP-Blend.md)

### Unified / Multi-modal

[JanusFlow](papers/2025/2025-01/JanusFlow/AI_Daily_JanusFlow.md) | [AR-Omni](papers/2026/2026-01/AR-Omni/AI_Daily_AR-Omni.md)

---

## 工具

- **[PDF Image Extractor](skills/pdf-image-extractor/)** — 從論文 PDF 中提取高品質圖片的工具

---

## 保持聯繫

喜歡我們的內容嗎？歡迎關注、分享、交流！如果你有想看的論文主題或建議，也歡迎隨時提出。

> **Contact Me**: k20010928@gmail.com

---

*每天進步一點點，與 AI 一起成長。*

*Last Updated: 2026-03-17*

### 2026-03-17
* [Just-in-Time (JiT)](papers/2026/2026-03/JiT/AI_Daily_JiT.md) - 免訓練空間加速 Diffusion Transformers，FLUX.1-dev 實現 7x 加速 (CVPR 2026)。

### 2026-03-16
* [Reflective Flow Sampling Enhancement (RF-Sampling)](papers/2026/2026-03/RF-Sampling/README.md) - 專為 Flow Matching 模型（如 FLUX）設計的 Training-free 推論增強框架，首次在 FLUX 上實現 Test-time Scaling。
