# Awesome LLM Pre-training

**中文版** | [English Version](https://github.com/RUCAIBox/awesome-llm-pretraining/blob/main/README.md)

预训练是研发大语言模型的第一个训练阶段，也是最为重要的一个阶段。随着开源社区在模型架构、训练策略、开源数据集、数据方法等方面的完善，我们持续关注可用于大模型预训练的资源，以回馈开源社区中的大语言模型的开发者。

相比于完整的综述，我们覆盖的范围将局限于预训练相关的常用资源和前沿尝试，以快速上手大语言模型预训练。同时我们欢迎开源社区提交更新，以共同促进大模型的发展。

> 相关项目链接：[[LLMSurvey](https://github.com/RUCAIBox/LLMSurvey)] [[YuLan-Chat](https://github.com/RUC-GSAI/YuLan-Chat)] | [[YuLan-Mini](https://github.com/RUC-GSAI/YuLan-Mini)]

## 目录

- [技术报告](#一技术报告)
- [训练策略](#二训练策略)
- [开源数据集](#三开源数据集)
- [数据方法](#四数据方法)

## 一、技术报告

技术报告的背后往往都是成百上千的算力资源作为支撑，因此很推荐阅读一些开源技术报告。

### 1.1 Dense模型

1. **The Llama 3 Herd of Models**. [[paper](https://arxiv.org/abs/2407.21783)]
2. **Qwen2.5 Technical Report**. [[paper](https://arxiv.org/abs/2412.15115)]
3. **Gemma 3 Technical Report**. [[paper](https://arxiv.org/abs/2503.19786)]
4. **Nemotron-4 340B Technical Report**. [[paper](https://arxiv.org/abs/2406.11704)]
5. **Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs**. [[paper](https://arxiv.org/abs/2504.07866)]
6. **Baichuan 2: Open Large-scale Language Models**. [[paper](https://arxiv.org/abs/2309.10305)]

### 1.2 MoE模型

1. **DeepSeek-V3 Technical Report**. [[paper](https://arxiv.org/abs/2412.19437)]
2. **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2401.06066)]
3. **Mixtral of Experts**. [[paper](https://arxiv.org/abs/2401.04088)]
4. **Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2406.06563)]
5. **Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs**. [[paper](https://arxiv.org/abs/2503.05139)]
6. **OLMoE: Open Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2409.02060)]
7. **Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent**. [[paper](https://arxiv.org/abs/2411.02265)]

### 1.3 带开源数据集的模型

1. **YuLan-Mini: An Open Data-efficient Language Model**. [[code](https://github.com/RUC-GSAI/YuLan-Mini?tab=readme-ov-file)] [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[paper](https://arxiv.org/abs/2412.17743)]
2. **MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series**. [[paper](https://arxiv.org/abs/2405.19327)]
3. **LLM360: Towards Fully Transparent Open-Source LLMs**. [[paper](https://arxiv.org/abs/2312.06550)]
4. **Nemotron-4 15B Technical Report**. [[paper](https://arxiv.org/abs/2402.16819)]

### 1.4 训练/数据策略

1. **Phi-4 Technical Report**. [[paper](https://arxiv.org/abs/2412.08905)]
2. **OLMo: Accelerating the Science of Language Models**. [[paper](https://arxiv.org/abs/2402.00838)]
3. **2 OLMo 2 Furious**. [[paper](https://arxiv.org/abs/2501.00656)]
4. **Yi: Open Foundation Models by 01.AI**. [[paper](https://arxiv.org/abs/2403.04652)]
5. **MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies**. [[paper](https://arxiv.org/abs/2404.06395)]

### 1.5 混合/线性模型

1. **Falcon Mamba: The First Competitive Attention-free 7B Language Model**. [[paper](https://arxiv.org/abs/2410.05355)]
2. **MiniMax-01: Scaling Foundation Models with Lightning Attention**. [[paper](https://arxiv.org/abs/2501.08313)]
3. **Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models**. [[paper](https://arxiv.org/abs/2504.03624)]


[⬆️ 返回目录](#目录)

<details>
<summary>全部技术报告</summary>

### LLaMA 系列

1. **LLaMA: Open and Efficient Foundation Language Models**. [[paper](https://arxiv.org/abs/2302.13971)]
2. **Llama 2: Open Foundation and Fine-Tuned Chat Models**. [[paper](https://arxiv.org/abs/2307.09288)]
3. **The Llama 3 Herd of Models**. [[paper](https://arxiv.org/abs/2407.21783)]

### Qwen 系列

1. **Qwen Technical Report**. [[paper](https://arxiv.org/abs/2309.16609)]
2. **Qwen2 Technical Report**. [[paper](https://arxiv.org/abs/2407.10671)]
3. **Qwen2.5 Technical Report**. [[paper](https://arxiv.org/abs/2412.15115)]

### DeepSeek 系列

1. **DeepSeek LLM: Scaling Open-Source Language Models with Longtermism**. [[paper](https://arxiv.org/abs/2401.02954)]
2. **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2401.06066)]
3. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**. [[paper](https://arxiv.org/abs/2402.03300)]
4. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**. [[paper](https://arxiv.org/abs/2405.04434)]
5. **DeepSeek-V3 Technical Report**. [[paper](https://arxiv.org/abs/2412.19437)]

### Gemma 系列

1. **Gemma: Open Models Based on Gemini Research and Technology**. [[paper](https://arxiv.org/abs/2403.08295)]
2. **Gemma 2: Improving Open Language Models at a Practical Size**. [[paper](https://arxiv.org/abs/2408.00118)]
3. **Gemma 3 Technical Report**. [[paper](https://arxiv.org/abs/2503.19786)]

### Gemini 系列

1. **Gemini: A Family of Highly Capable Multimodal Models**. [[paper](https://arxiv.org/abs/2312.11805)]
2. **Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context**. [[paper](https://arxiv.org/abs/2403.05530v5)]

### Mistral 系列

1. **Mistral 7B**. [[paper](https://arxiv.org/abs/2310.06825)]
2. **Mixtral of Experts**. [[paper](https://arxiv.org/abs/2401.04088)]

### Phi 系列

1. **Textbooks Are All You Need**. [[paper](https://arxiv.org/abs/2306.11644)]
2. **Textbooks Are All You Need II: phi-1.5 technical report**. [[paper](https://arxiv.org/abs/2309.05463)]
3. **Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone**. [[paper](https://arxiv.org/abs/2404.14219)]
4. **Phi-4 Technical Report**. [[paper](https://arxiv.org/abs/2412.08905)]

### GLM 系列

1. **GLM: General Language Model Pretraining with Autoregressive Blank Infilling**. [[paper](https://arxiv.org/abs/2103.10360)]
2. **GLM-130B: An Open Bilingual Pre-trained Model**. [[paper](https://arxiv.org/abs/2210.02414)]
3. **ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools**. [[paper](https://arxiv.org/abs/2406.12793)]

### Baichuan 系列

1. **Baichuan 2: Open Large-scale Language Models**. [[paper](https://arxiv.org/abs/2309.10305)]
2. **Baichuan-M1: Pushing the Medical Capability of Large Language Models**. [[paper](https://arxiv.org/abs/2502.12671)]

### Falcon 系列

1. **The Falcon Series of Open Language Models**. [[paper](https://arxiv.org/abs/2311.16867)]
2. **Falcon2-11B Technical Report**. [[paper](https://arxiv.org/abs/2407.14885)]
3. **Falcon Mamba: The First Competitive Attention-free 7B Language Model**. [[paper](https://arxiv.org/abs/2410.05355)]

### InternLM 系列

1. **InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities**. [[paper](https://github.com/InternLM/InternLM-techreport/blob/main/InternLM.pdf)]
2. **InternLM2 Technical Report**. [[paper](https://arxiv.org/abs/2403.17297)]

### MiniCPM

1. **MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies**. [[paper](https://arxiv.org/abs/2404.06395)]

### Yi 系列

1. **Yi: Open Foundation Models by 01.AI**. [[paper](https://arxiv.org/abs/2403.04652)]
2. **Yi-Lightning Technical Report**. [[paper](https://arxiv.org/abs/2412.01253)]

### Minimax 系列

1. **MiniMax-01: Scaling Foundation Models with Lightning Attention**. [[paper](https://arxiv.org/abs/2501.08313)]

### Reka 系列

1. **Reka Core, Flash, and Edge: A Series of Powerful Multimodal Language Models**. [[paper](https://arxiv.org/abs/2404.12387v1)]

### Skywork 系列

1. **Skywork: A More Open Bilingual Foundation Model**. [[paper](https://arxiv.org/abs/2310.19341)]
2. **Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2406.06563)]

### Hunyuan 系列

1. **Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent**. [[paper](https://arxiv.org/abs/2411.02265)]

### Nemotron 系列

1. **Nemotron-4 15B Technical Report**. [[paper](https://arxiv.org/abs/2402.16819)]
2. **Nemotron-4 340B Technical Report**. [[paper](https://arxiv.org/abs/2406.11704)]
3. **Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models**. [[paper](https://arxiv.org/abs/2504.03624)]

### Ling 系列

1. **Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs**. [[paper](https://arxiv.org/abs/2503.05139)]

### OLMo 系列

1. **OLMo: Accelerating the Science of Language Models**. [[paper](https://arxiv.org/abs/2402.00838)]
2. **2 OLMo 2 Furious**. [[paper](https://arxiv.org/abs/2501.00656)]
3. **OLMoE: Open Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2409.02060)]

### YuLan 系列

1. **YuLan: An Open-source Large Language Model**. [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[code](https://github.com/RUC-GSAI/YuLan-Chat)] [[paper](https://arxiv.org/abs/2406.19853)]
2. **YuLan-Mini: An Open Data-efficient Language Model**. [[code](https://github.com/RUC-GSAI/YuLan-Mini?tab=readme-ov-file)] [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[paper](https://arxiv.org/abs/2412.17743)]

### MAP-Neo 系列

1. **MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series**. [[paper](https://arxiv.org/abs/2405.19327)]

### LLM360 项目

1. **LLM360: Towards Fully Transparent Open-Source LLMs**. [[paper](https://arxiv.org/abs/2312.06550)]

[⬆️ 返回目录](#目录)
</details>

## 二、训练策略

我们从训练框架、训练策略、可解释性、模型架构改进、学习率退火等方面讨论了训练策略。

### 2.1 训练框架

最常使用的训练框架为Megatron-LM，提供了一个良好的开箱即用的高效基准。结合其他库可以达到更好的训练速度。

1. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**. [[code](https://github.com/NVIDIA/Megatron-LM)] [[paper](https://arxiv.org/abs/1909.08053)]
   > 最常用的预训练框架，上手门槛高但更加稳定
2. **Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts**. [[paper](https://arxiv.org/abs/2502.19811)]
   > MoE计算通信重叠
3. **DeepEP: an efficient expert-parallel communication library**. [[code](https://github.com/deepseek-ai/DeepEP)]
   > 专家并行加速
4. **DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling**. [[code](https://github.com/deepseek-ai/DeepGEMM)]
   > 利用Hopper的异步特性加速FP8矩阵乘法
5. **Liger Kernel: Efficient Triton Kernels for LLM Training**. [[code](https://github.com/linkedin/Liger-Kernel)] [[paper](https://arxiv.org/abs/2410.10989)]
   > Triton加速算子库

### 2.2 训练策略

关于超参数Scaling Law、并行策略、初始化策略、优化器选择、FP8训练等。

1. **Predictable Scale: Part I -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining**. [[paper](http://arxiv.org/abs/2503.04715)] [[homepage](https://step-law.github.io/)]
   > 关于超参数的 Scaling Law
2. **The Ultra-Scale Playbook: Training LLMs on GPU Clusters**. [[demo](https://huggingface.co/spaces/nanotron/ultrascale-playbook)]
   > 可视化并行策略显存占用
3. **A Spectral Condition for Feature Learning**. [[paper](https://arxiv.org/abs/2310.17813)]
   > MuP的进阶版本
4. **Muon is Scalable for LLM Training**. [[code](https://github.com/MoonshotAI/Moonlight)] [[paper](https://arxiv.org/abs/2502.16982)]
   > 高效优化器
5. **COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training**. [[paper](https://arxiv.org/abs/2410.19313)] [[code](https://github.com/NVlabs/COAT)]
   > 优化器状态和激活值也为FP8的训练
6. **Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2309.13206)]
   > 关于MoE的Scaling Law

### 2.3 可解释性

我们不完全列举了一些对于预训练有启发的可解释性工作。

1. **On the Biology of a Large Language Model**. [[blog](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)]
2. **Physics of Language Models**. [[homepage](https://physics.allen-zhu.com/)]
3. **In-context Learning and Induction Heads**. [[blog](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)] [[paper](https://arxiv.org/abs/2209.11895)]
4. **Rethinking Reflection in Pre-Training**. [[paper](https://arxiv.org/abs/2504.04022)]

### 2.4 模型架构改进

我们不完全列举了一些近期针对模型架构的改进。


1. **Gated Delta Networks: Improving Mamba2 with Delta Rule**. [[paper](https://arxiv.org/abs/2412.06464)]
2. **RWKV-7 "Goose" with Expressive Dynamic State Evolution**. [[paper](https://arxiv.org/abs/2503.14456)]
3. **Mixture of Hidden-Dimensions Transformer**. [[paper](https://arxiv.org/abs/2412.05644)]
4. **Titans: Learning to Memorize at Test Time**. [[paper](https://arxiv.org/abs/2501.00663)]
5. **Ultra-Sparse Memory Network**. [[paper](https://arxiv.org/abs/2411.12364)]
6. **Large Language Diffusion Models**. [[paper](https://arxiv.org/abs/2502.09992)]
7. **Better & Faster Large Language Models via Multi-token Prediction**. [[paper](https://arxiv.org/abs/2402.10738)]
8. **Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing**. [[paper](https://arxiv.org/abs/2402.18158)]
9. **Stick-breaking Attention**. [[code](https://github.com/shawntan/stickbreaking-attention)] [[paper](https://arxiv.org/abs/2410.17980)]
10. **Forgetting Transformer: Softmax Attention with a Forget Gate**. [[paper](https://arxiv.org/abs/2310.17045)]
11. **Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention**. [[unofficial code](https://github.com/fla-org/native-sparse-attention)] [[paper](https://arxiv.org/abs/2502.11089)]
12. **MoBA: Mixture of Block Attention for Long-Context LLMs**. [[code](https://github.com/MoonshotAI/MoBA)] [[paper](https://arxiv.org/abs/2502.13189)]
13. **KV Shifting Attention Enhances Language Modeling**. [[paper](https://arxiv.org/abs/2404.05922)]
14. **Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models**. [[paper](https://arxiv.org/abs/2402.19481)]
15. **Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts**. [[paper](https://arxiv.org/abs/2402.18140)]
16. **ReLU2 Wins: Discovering Efficient Activation Functions for Sparse LLMs**. [[paper](https://arxiv.org/abs/2402.10517)]
17. **μnit Scaling: Simple and Scalable FP8 LLM Training**. [[paper](https://arxiv.org/abs/2402.14809)]

### 2.5 学习率退火

学习率退火往往和数据质量筛选相结合。

1. **MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies**. [[paper](http://arxiv.org/abs/2404.06395)]
2. **Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations**. [[paper](https://arxiv.org/abs/2405.18392)]
3. **Scaling Law with Learning Rate Annealing**. [[paper](https://arxiv.org/abs/2408.11029)]

[⬆️ 返回目录](#目录)

## 三、开源数据集

我们主要从网页、数学、代码、通用四个方面讨论现有开源数据集。

### 3.1 网页

网页数据将构成预训练中的核心语料。

1. **DCLM**. [[paper](https://arxiv.org/abs/2406.11794)] [[resource](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)]
   > 开源网页数据集，经过Fasttext等筛选后得到的3.8T数据集
2. **FineWeb-Edu** [[paper](https://arxiv.org/abs/2406.17557)] [[resource](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)]
   > 教育质量打分语料，从FineWeb中筛选打分，对于知识密集型题目有一定效果
3. **Nemotron-CC-HQ**. [[paper](https://arxiv.org/abs/2412.02595)] [[resource](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html)]
   > 英伟达的CC语料
4. **Chinese-FineWeb-Edu**. [[resource](https://huggingface.co/collections/opencsg/high-quality-chinese-training-datasets-66cfed105f502ece8f29643e)]
   > OpenCSG开源的中文教育质量打分语料，从Map-CC、SkyPile、WuDao、Wanjuan等筛选打分
5. **FineWeb2: A sparkling update with 1000s of languages** [[resource](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)]
   > 多语言数据集


### 3.2 数学

数学预训练语料可以显著提升基模的数学能力以及后训练的上限。

1. **MegaMath: Pushing the Limits of Open Math Corpora**. [[resource](https://huggingface.co/datasets/meta-math/MegaMath)] [[paper](https://arxiv.org/abs/2504.02807)]
   > 开源最大的高质量数学CC语料
2. **JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models**. [[resource](https://github.com/RUCAIBox/JiuZhang3.0)] [[paper](https://arxiv.org/abs/2405.14365)]
   > 合成数学指令数据
3. **mlfoundations-dev/stackoverflow_math**. [[resource](https://huggingface.co/datasets/mlfoundations-dev/stackoverflow_math)]
   > 数学相关提问
4. **DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning**. [[resource](https://github.com/zwhe99/DeepMath)] [[paper](https://arxiv.org/abs/2504.11456)]
   > 高难度数学数据集
5. **YuLan-Mini: An Open Data-efficient Language Model**. [[code](https://github.com/RUC-GSAI/YuLan-Mini?tab=readme-ov-file)] [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[paper](https://arxiv.org/abs/2412.17743)]
   > 收集开源Lean定理证明数据集

### 3.3 代码

代码数据不仅可以增强基模生成代码的能力，还可以增强数学、逻辑等方面。

1. **OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models**. [[resource](https://huggingface.co/collections/OpenCoder-LLM/opencoder-datasets-672e6db6a0fed24bd69ef1c2)] [[paper](https://arxiv.org/abs/2411.04905)]
   > 从 The-Stack-V2 中清洗
2. **SmolLM-corpus**. [[resource](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)]
   > Python教育质量打分
3. **The-Stack-V2**. [[resource](https://huggingface.co/datasets/bigcode/the-stack-v2)]
   > 最大规模未清洗的代码数据
4. **YuLan-Mini: An Open Data-efficient Language Model**. [[code](https://github.com/RUC-GSAI/YuLan-Mini?tab=readme-ov-file)] [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[paper](https://arxiv.org/abs/2412.17743)]
   > 以教育质量清洗Jupyter-Notebook和Python数据
5. **HuggingFaceTB/issues-kaggle-notebooks**. [[resource](https://huggingface.co/datasets/HuggingFaceTB/issues-kaggle-notebooks)]
   > GitHub Issues和Kaggle Notebooks数据
6. **mlfoundations-dev/stackoverflow**. [[resource](https://huggingface.co/datasets/mlfoundations-dev/stackoverflow)]
   > 编程问答论坛
7. **Magicoder: Empowering Code Generation with OSS-Instruct**. [[resource](https://github.com/ise-uiuc/magicoder)] [[paper](https://arxiv.org/abs/2312.02120)]
   > 利用开源代码生成合成指令数据训练

### 3.4 通用（书籍、百科、指令、长上下文等）

通用数据往往是较为稀缺的长尾数据，对于后训练模型的可用性起到至关重要的作用。

1. **YuLan: An Open-source Large Language Model**. [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[code](https://github.com/RUC-GSAI/YuLan-Chat)] [[paper](https://arxiv.org/abs/2406.19853)]
   > 长尾知识增强和多种通用数据源清洗
2. **MinerU: An Open-Source Solution for Precise Document Content Extraction**. [[code](https://github.com/opendatalab/MinerU)] [[paper](https://arxiv.org/abs/2409.18839)]
   > PDF转Markdown，兼容性较强
3. **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**. [[homepage](https://pile.eleuther.ai/)] [[paper](https://arxiv.org/abs/2101.00027)]
   > arXiv、对话、DM Math等
4. **Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research**. [[resource](https://huggingface.co/datasets/allenai/dolma)] [[paper](https://arxiv.org/abs/2402.00159)]
   > 百科、书籍、论文、Reddit等
5. **WanJuan: A Comprehensive Multimodal Dataset for Advancing English and Chinese Large Models**. [[resource](https://github.com/opendatalab/WanJuan)] [[paper](https://arxiv.org/abs/2308.10755)]
   > 法律、考试、新闻、专利、百科等
6. **MAmmoTH2: Scaling Instructions from the Web**. [[resource](https://arxiv.org/abs/2401.12246)] [[paper](https://github.com/togethercomputer/MAmmoTH)]
   > 针对网页的问答
7. **togethercomputer/Long-Data-Collections**. [[resource](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections)]
   > 从RedPajama、Pile、P3等数据集过滤的书籍、论文、网页和指令
8. **Longattn: Selecting long-context training data via token-level attention**. [[resource](https://github.com/Lyun0912-wu/LongAttn)] [[paper](https://arxiv.org/abs/2502.16860)]
   > 长程依赖的问答

[⬆️ 返回目录](#目录)

## 四、数据方法

数据集往往配合高质量的数据方法。我们从分词器、数据配比和课程、数据合成等方面详细阐述。

### 4.1 分词器

分词是模型重要又常被忽视的一块，会显著影响模型在数学、知识等方面能力。

1. **SuperBPE: Space Travel for Language Models**. [[code](https://github.com/PythonNut/superbpe)] [[paper](http://arxiv.org/abs/2503.13423)]
   > 多单词的分词器训练方式
2. **Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies**. [[code](https://github.com/sail-sg/scaling-with-vocab)] [[demo](https://huggingface.co/spaces/sail/scaling-with-vocab-demo)] [[paper](https://arxiv.org/abs/2407.13623)]
   > 预测词表大小
3. **Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs**. [[code](https://github.com/aadityasingh/tokenizationcounts)] [[paper](https://arxiv.org/abs/2402.14903)]
   > 数字的分词方式比较

### 4.2 数据配比和课程

多阶段预训练往往能使得模型充分学习高质量、少量的数据。在继续预训练（CPT）阶段引入更多的数学、代码、CoT甚至长思维链数据，将构成下一代预训练模型的核心能力。

1. **Nemotron-4 15B Technical Report**. [[paper](https://arxiv.org/abs/2402.16819)]
   > 分为 8T 预训练 + 更少数据规模的 CPT
2. **YuLan-Mini: An Open Data-efficient Language Model**. [[code](https://github.com/RUC-GSAI/YuLan-Mini?tab=readme-ov-file)] [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[paper](https://arxiv.org/abs/2412.17743)]
   > 使用教育分数进行课程数据
3. **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining**. [[code](https://github.com/mlfoundations/doremi)] [[paper](https://arxiv.org/abs/2305.10429)]
   > 预训练数据混合比例优化
4. **Efficient Online Data Mixing For Language Model Pre-Training**. [[paper](https://arxiv.org/abs/2312.02406)]
   > 在线数据混合
5. **Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance**. [[paper](https://arxiv.org/abs/2305.05461)]
   > 数据混合定律
6. **Data Mixture Inference: What do BPE Tokenizers Reveal about their Training Data?**. [[paper]()]
   > 通过 BPE 分词器的合并规则，破解GPT等商业模型的数据比例
7. **CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training**. [[homepage](https://research.nvidia.com/labs/lpr/climb/)] [[paper](https://arxiv.org/abs/2504.13161)]
   > 基于聚类的迭代数据混合自举框架
8. **Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens**. [[demo](https://huggingface.co/spaces/liujch1998/infini-gram)] [[homepage](https://infini-gram.io/)] [[paper](https://arxiv.org/abs/2401.17377)]
   > 为大规模预训练数据集构建索引，以检查数据质量

### 4.3 数据合成

除了前文提到的数学和代码的合成数据，我们总结了部分通用的合成数据方法和资源。除此之外，在预训练后期使用更多的长思维数据，也逐渐成为值得探索的方向。

1. **Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems**. [[resource](https://huggingface.co/datasets/RUC-AIBOX/long_form_thought_data_5k)] [[code](https://github.com/RUCAIBox/Slow_Thinking_with_LLMs)] [[paper](https://arxiv.org/abs/2412.09413)]
   > 基于长思维链合成数据的模仿学习
2. **Knowledge-Instruct: Effective Continual Pre-training from Limited Data using Instructions**. [[code](https://github.com/meniData1/knowledge-instruct)] [[paper](https://arxiv.org/abs/2504.05571)]
   > 生成信息密集型的合成指令数据，从有限的语料库中学习知识
3. **LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs**. [[resource](https://huggingface.co/datasets/THUDM/LongWriter-6k)] [[code](https://github.com/THUDM/LongWriter)] [[paper](https://arxiv.org/abs/2408.07055)]
   > 构造长文本 Creative Writing
4. **Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use**. [[paper](https://arxiv.org/abs/2504.04736)]
   > 多步骤推理数据合成，将复杂任务分解为子轨迹，结合强化学习优化数据生成
5. **WildChat: 1M ChatGPT Interaction Logs in the Wild**. [[resource](https://huggingface.co/datasets/allenai/WildChat-1M)] [[paper](https://arxiv.org/abs/2405.01470)]
   > 用户真实对话的开源数据集
6. **Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing**. [[resource](https://huggingface.co/Magpie-Align)] [[code](https://github.com/magpie-align/magpie)] [[paper](https://arxiv.org/abs/2406.08464)]
   > 对齐数据合成

[⬆️ 返回目录](#目录)

## 贡献

如果您对于项目内容有建议，欢迎提交 [Issue](https://github.com/RUCAIBox/awesome-llm-pretraining/issues/new) 和 PR，以共同促进大模型的发展。

