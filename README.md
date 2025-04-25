# Awesome LLM Pre-training

[中文版](https://github.com/RUCAIBox/awesome-llm-pretraining/blob/main/README_ZH.md) | **English Version**

Pre-training is the first and most crucial training stage in the development of large language models. As the open-source community continues to improve in areas such as model architecture, training strategies, open-source datasets, and data methods, we are committed to continuously tracking resources available for large model pre-training to give back to developers in the open-source large language model community.

Compared to comprehensive reviews, our scope is limited to commonly used resources and cutting-edge attempts related to pre-training, aiming to help users quickly get started with large language model pre-training. We also welcome contributions and updates from the open-source community to jointly promote the development of large models.

> Related project links: [[LLMSurvey](https://github.com/RUCAIBox/LLMSurvey)] [[YuLan-Chat](https://github.com/RUC-GSAI/YuLan-Chat)] | [[YuLan-Mini](https://github.com/RUC-GSAI/YuLan-Mini)]

## Table of Contents

- [Technical Reports](#i-technical-reports)
- [Training Strategies](#ii-training-strategies)
- [Open-source Datasets](#iii-open-source-datasets)
- [Data Methods](#iv-data-methods)

## I. Technical Reports

Technical reports often rely on hundreds or thousands of computing resources. Therefore, it is highly recommended to read some open-source technical reports.

### 1.1 Dense Models

1. **The Llama 3 Herd of Models**. [[paper](https://arxiv.org/abs/2407.21783)]
2. **Qwen2.5 Technical Report**. [[paper](https://arxiv.org/abs/2412.15115)]
3. **Gemma 3 Technical Report**. [[paper](https://arxiv.org/abs/2503.19786)]
4. **Nemotron-4 340B Technical Report**. [[paper](https://arxiv.org/abs/2406.11704)]
5. **Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs**. [[paper](https://arxiv.org/abs/2504.07866)]
6. **Baichuan 2: Open Large-scale Language Models**. [[paper](https://arxiv.org/abs/2309.10305)]

### 1.2 MoE Models

1. **DeepSeek-V3 Technical Report**. [[paper](https://arxiv.org/abs/2412.19437)]
2. **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2401.06066)]
3. **Mixtral of Experts**. [[paper](https://arxiv.org/abs/2401.04088)]
4. **Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2406.06563)]
5. **Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs**. [[paper](https://arxiv.org/abs/2503.05139)]
6. **OLMoE: Open Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2409.02060)]
7. **Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent**. [[paper](https://arxiv.org/abs/2411.02265)]

### 1.3 Models with Open-source Datasets

1. **YuLan-Mini: An Open Data-efficient Language Model**. [[code](https://github.com/RUC-GSAI/YuLan-Mini?tab=readme-ov-file)] [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[paper](https://arxiv.org/abs/2412.17743)]
2. **MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series**. [[paper](https://arxiv.org/abs/2405.19327)]
3. **LLM360: Towards Fully Transparent Open-Source LLMs**. [[paper](https://arxiv.org/abs/2312.06550)]
4. **Nemotron-4 15B Technical Report**. [[paper](https://arxiv.org/abs/2402.16819)]

### 1.4 Training/Data Strategies

1. **Phi-4 Technical Report**. [[paper](https://arxiv.org/abs/2412.08905)]
2. **OLMo: Accelerating the Science of Language Models**. [[paper](https://arxiv.org/abs/2402.00838)]
3. **2 OLMo 2 Furious**. [[paper](https://arxiv.org/abs/2501.00656)]
4. **Yi: Open Foundation Models by 01.AI**. [[paper](https://arxiv.org/abs/2403.04652)]
5. **MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies**. [[paper](https://arxiv.org/abs/2404.06395)]

### 1.5 Hybrid/Linear Models

1. **Falcon Mamba: The First Competitive Attention-free 7B Language Model**. [[paper](https://arxiv.org/abs/2410.05355)]
2. **MiniMax-01: Scaling Foundation Models with Lightning Attention**. [[paper](https://arxiv.org/abs/2501.08313)]
3. **Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models**. [[paper](https://arxiv.org/abs/2504.03624)]

[⬆️ Back to Table of Contents](#table-of-contents)

<details>
<summary>All Technical Reports</summary>

### LLaMA Series

1. **LLaMA: Open and Efficient Foundation Language Models**. [[paper](https://arxiv.org/abs/2302.13971)]
2. **Llama 2: Open Foundation and Fine-Tuned Chat Models**. [[paper](https://arxiv.org/abs/2307.09288)]
3. **The Llama 3 Herd of Models**. [[paper](https://arxiv.org/abs/2407.21783)]

### Qwen Series

1. **Qwen Technical Report**. [[paper](https://arxiv.org/abs/2309.16609)]
2. **Qwen2 Technical Report**. [[paper](https://arxiv.org/abs/2407.10671)]
3. **Qwen2.5 Technical Report**. [[paper](https://arxiv.org/abs/2412.15115)]

### DeepSeek Series

1. **DeepSeek LLM: Scaling Open-Source Language Models with Longtermism**. [[paper](https://arxiv.org/abs/2401.02954)]
2. **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2401.06066)]
3. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**. [[paper](https://arxiv.org/abs/2402.03300)]
4. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**. [[paper](https://arxiv.org/abs/2405.04434)]
5. **DeepSeek-V3 Technical Report**. [[paper](https://arxiv.org/abs/2412.19437)]

### Gemma Series

1. **Gemma: Open Models Based on Gemini Research and Technology**. [[paper](https://arxiv.org/abs/2403.08295)]
2. **Gemma 2: Improving Open Language Models at a Practical Size**. [[paper](https://arxiv.org/abs/2408.00118)]
3. **Gemma 3 Technical Report**. [[paper](https://arxiv.org/abs/2503.19786)]

### Gemini Series

1. **Gemini: A Family of Highly Capable Multimodal Models**. [[paper](https://arxiv.org/abs/2312.11805)]
2. **Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context**. [[paper](https://arxiv.org/abs/2403.05530v5)]

### Mistral Series

1. **Mistral 7B**. [[paper](https://arxiv.org/abs/2310.06825)]
2. **Mixtral of Experts**. [[paper](https://arxiv.org/abs/2401.04088)]

### Phi Series

1. **Textbooks Are All You Need**. [[paper](https://arxiv.org/abs/2306.11644)]
2. **Textbooks Are All You Need II: phi-1.5 technical report**. [[paper](https://arxiv.org/abs/2309.05463)]
3. **Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone**. [[paper](https://arxiv.org/abs/2404.14219)]
4. **Phi-4 Technical Report**. [[paper](https://arxiv.org/abs/2412.08905)]

### GLM Series

1. **GLM: General Language Model Pretraining with Autoregressive Blank Infilling**. [[paper](https://arxiv.org/abs/2103.10360)]
2. **GLM-130B: An Open Bilingual Pre-trained Model**. [[paper](https://arxiv.org/abs/2210.02414)]
3. **ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools**. [[paper](https://arxiv.org/abs/2406.12793)]

### Baichuan Series

1. **Baichuan 2: Open Large-scale Language Models**. [[paper](https://arxiv.org/abs/2309.10305)]
2. **Baichuan-M1: Pushing the Medical Capability of Large Language Models**. [[paper](https://arxiv.org/abs/2502.12671)]

### Falcon Series

1. **The Falcon Series of Open Language Models**. [[paper](https://arxiv.org/abs/2311.16867)]
2. **Falcon2-11B Technical Report**. [[paper](https://arxiv.org/abs/2407.14885)]
3. **Falcon Mamba: The First Competitive Attention-free 7B Language Model**. [[paper](https://arxiv.org/abs/2410.05355)]

### InternLM Series

1. **InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities**. [[paper](https://github.com/InternLM/InternLM-techreport/blob/main/InternLM.pdf)]
2. **InternLM2 Technical Report**. [[paper](https://arxiv.org/abs/2403.17297)]

### MiniCPM

1. **MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies**. [[paper](https://arxiv.org/abs/2404.06395)]

### Yi Series

1. **Yi: Open Foundation Models by 01.AI**. [[paper](https://arxiv.org/abs/2403.04652)]
2. **Yi-Lightning Technical Report**. [[paper](https://arxiv.org/abs/2412.01253)]

### Minimax Series

1. **MiniMax-01: Scaling Foundation Models with Lightning Attention**. [[paper](https://arxiv.org/abs/2501.08313)]

### Reka Series

1. **Reka Core, Flash, and Edge: A Series of Powerful Multimodal Language Models**. [[paper](https://arxiv.org/abs/2404.12387v1)]

### Skywork Series

1. **Skywork: A More Open Bilingual Foundation Model**. [[paper](https://arxiv.org/abs/2310.19341)]
2. **Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2406.06563)]

### Hunyuan Series

1. **Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent**. [[paper](https://arxiv.org/abs/2411.02265)]

### Nemotron Series

1. **Nemotron-4 15B Technical Report**. [[paper](https://arxiv.org/abs/2402.16819)]
2. **Nemotron-4 340B Technical Report**. [[paper](https://arxiv.org/abs/2406.11704)]
3. **Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models**. [[paper](https://arxiv.org/abs/2504.03624)]

### Ling Series

1. **Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs**. [[paper](https://arxiv.org/abs/2503.05139)]

### OLMo Series

1. **OLMo: Accelerating the Science of Language Models**. [[paper](https://arxiv.org/abs/2402.00838)]
2. **2 OLMo 2 Furious**. [[paper](https://arxiv.org/abs/2501.00656)]
3. **OLMoE: Open Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2409.02060)]

### YuLan Series

1. **YuLan: An Open-source Large Language Model**. [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[code](https://github.com/RUC-GSAI/YuLan-Chat)] [[paper](https://arxiv.org/abs/2406.19853)]
2. **YuLan-Mini: An Open Data-efficient Language Model**. [[code](https://github.com/RUC-GSAI/YuLan-Mini?tab=readme-ov-file)] [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[paper](https://arxiv.org/abs/2412.17743)]

### MAP-Neo Series

1. **MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series**. [[paper](https://arxiv.org/abs/2405.19327)]

### LLM360 Project

1. **LLM360: Towards Fully Transparent Open-Source LLMs**. [[paper](https://arxiv.org/abs/2312.06550)]

[⬆️ Back to Table of Contents](#table-of-contents)
</details>

## II. Training Strategies

We discuss training strategies from aspects such as training frameworks, training strategies, interpretability, model architecture improvements, and learning rate annealing.

### 2.1 Training Frameworks

The most commonly used training framework is Megatron-LM, which provides a good out-of-the-box and efficient benchmark. Combining it with other libraries can achieve better training speed.

1. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**. [[code](https://github.com/NVIDIA/Megatron-LM)] [[paper](https://arxiv.org/abs/1909.08053)]
   > The most commonly used pre-training framework, with a high entry threshold but more stability.
2. **Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts**. [[paper](https://arxiv.org/abs/2502.19811)]
   > Computation-communication overlapping for MoE.
3. **DeepEP: an efficient expert-parallel communication library**. [[code](https://github.com/deepseek-ai/DeepEP)]
   > Expert parallel acceleration.
4. **DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling**. [[code](https://github.com/deepseek-ai/DeepGEMM)]
   > Accelerating FP8 matrix multiplication using the asynchronous features of Hopper.
5. **Liger Kernel: Efficient Triton Kernels for LLM Training**. [[code](https://github.com/linkedin/Liger-Kernel)] [[paper](https://arxiv.org/abs/2410.10989)]
   > Triton acceleration operator library.

<details>
<summary>All Training Frameworks</summary>

1. **Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters**. [[code](https://github.com/deepspeedai/DeepSpeed)] [[paper](https://dl.acm.org/doi/10.1145/3394486.3406703)]
   > Zero-redundancy data parallelism
2. **TorchTitan: One-stop PyTorch native solution for production ready LLM pretraining**. [[code](https://github.com/pytorch/torchtitan)] [[paper](https://openreview.net/forum?id=SFN6Wm7YBI)]
   > Torch-native parallelism based on DTensor
3. **Flash Linear Attention** [[code](https://github.com/fla-org/flash-linear-attention/tree/main)]
   > Efficient Triton-based implementations for state-of-the-art linear attention models

</details>

### 2.2 Training Strategies

Regarding hyperparameter Scaling Law, parallel strategies, initialization strategies, optimizer selection, FP8 training, etc.

1. **Predictable Scale: Part I -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining**. [[paper](http://arxiv.org/abs/2503.04715)] [[homepage](https://step-law.github.io/)]
   > About the Scaling Law of hyperparameters.
2. **The Ultra-Scale Playbook: Training LLMs on GPU Clusters**. [[demo](https://huggingface.co/spaces/nanotron/ultrascale-playbook)]
   > Visualizing the memory usage of parallel strategies.
3. **A Spectral Condition for Feature Learning**. [[paper](https://arxiv.org/abs/2310.17813)]
   > An advanced version of MuP.
4. **Muon is Scalable for LLM Training**. [[code](https://github.com/MoonshotAI/Moonlight)] [[paper](https://arxiv.org/abs/2502.16982)]
   > An efficient optimizer.
5. **COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training**. [[paper](https://arxiv.org/abs/2410.19313)] [[code](https://github.com/NVlabs/COAT)]
   > Training with optimizer states and activation values also in FP8.
6. **Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2309.13206)]
   > About the Scaling Law of MoE.

### 2.3 Interpretability

We list some interpretability works that are inspiring for pre-training.

1. **On the Biology of a Large Language Model**. [[blog](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)]
2. **Physics of Language Models**. [[homepage](https://physics.allen-zhu.com/)]
3. **In-context Learning and Induction Heads**. [[blog](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)] [[paper](https://arxiv.org/abs/2209.11895)]
4. **Rethinking Reflection in Pre-Training**. [[paper](https://arxiv.org/abs/2504.04022)]

### 2.4 Model Architecture Improvements

We list some recent improvements to model architectures.

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

### 2.5 Learning Rate Annealing

Learning rate annealing is often combined with data quality screening.

1. **MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies**. [[paper](http://arxiv.org/abs/2404.06395)]
2. **Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations**. [[paper](https://arxiv.org/abs/2405.18392)]
3. **Scaling Law with Learning Rate Annealing**. [[paper](https://arxiv.org/abs/2408.11029)]

[⬆️ Back to Table of Contents](#table-of-contents)

## III. Open-source Datasets

We discuss existing open-source datasets mainly from four aspects: web pages, mathematics, code, and general-purpose.

### 3.1 Web Pages

Web page data will form the core corpus in pre-training.

1. **DCLM**. [[paper](https://arxiv.org/abs/2406.11794)] [[resource](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)]
   > An open-source web page dataset, a 3.8T dataset obtained after screening by Fasttext, etc.
2. **FineWeb-Edu** [[paper](https://arxiv.org/abs/2406.17557)] [[resource](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)]
   > A corpus for educational quality scoring, screened and scored from FineWeb, which has certain effects on knowledge-intensive questions.
3. **Nemotron-CC-HQ**. [[paper](https://arxiv.org/abs/2412.02595)] [[resource](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html)]
   > NVIDIA's CC corpus.
4. **Chinese-FineWeb-Edu**. [[resource](https://huggingface.co/collections/opencsg/high-quality-chinese-training-datasets-66cfed105f502ece8f29643e)]
   > An open-source Chinese educational quality scoring corpus by OpenCSG, screened and scored from Map-CC, SkyPile, WuDao, Wanjuan, etc.
5. **FineWeb2: A sparkling update with 1000s of languages** [[resource](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)]
   > A multilingual dataset.

### 3.2 Mathematics

Mathematical pre-training corpora can significantly improve the mathematical ability of the base model and the upper limit of post-training.

1. **MegaMath: Pushing the Limits of Open Math Corpora**. [[resource](https://huggingface.co/datasets/meta-math/MegaMath)] [[paper](https://arxiv.org/abs/2504.02807)]
   > The largest open-source high-quality mathematical CC corpus.
2. **JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models**. [[resource](https://github.com/RUCAIBox/JiuZhang3.0)] [[paper](https://arxiv.org/abs/2405.14365)]
   > Synthetic mathematical instruction data.
3. **mlfoundations-dev/stackoverflow_math**. [[resource](https://huggingface.co/datasets/mlfoundations-dev/stackoverflow_math)]
   > Math-related questions.
4. **DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning**. [[resource](https://github.com/zwhe99/DeepMath)] [[paper](https://arxiv.org/abs/2504.11456)]
   > A high-difficulty mathematical dataset.
5. **YuLan-Mini: An Open Data-efficient Language Model**. [[code](https://github.com/RUC-GSAI/YuLan-Mini?tab=readme-ov-file)] [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[paper](https://arxiv.org/abs/2412.17743)]
   > Collecting open-source Lean theorem proving datasets.

### 3.3 Code

Code data can not only enhance the code generation ability of the base model but also improve its mathematical and logical abilities.

1. **OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models**. [[resource](https://huggingface.co/collections/OpenCoder-LLM/opencoder-datasets-672e6db6a0fed24bd69ef1c2)] [[paper](https://arxiv.org/abs/2411.04905)]
   > Cleaned from The-Stack-V2.
2. **SmolLM-corpus**. [[resource](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)]
   > Python educational quality scoring.
3. **The-Stack-V2**. [[resource](https://huggingface.co/datasets/bigcode/the-stack-v2)]
   > The largest-scale uncleaned code data.
4. **YuLan-Mini: An Open Data-efficient Language Model**. [[code](https://github.com/RUC-GSAI/YuLan-Mini?tab=readme-ov-file)] [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[paper](https://arxiv.org/abs/2412.17743)]
   > Cleaning Jupyter-Notebook and Python data with educational quality.
5. **HuggingFaceTB/issues-kaggle-notebooks**. [[resource](https://huggingface.co/datasets/HuggingFaceTB/issues-kaggle-notebooks)]
   > GitHub Issues and Kaggle Notebooks data.
6. **mlfoundations-dev/stackoverflow**. [[resource](https://huggingface.co/datasets/mlfoundations-dev/stackoverflow)]
   > Programming Q&A forum.
7. **Magicoder: Empowering Code Generation with OSS-Instruct**. [[resource](https://github.com/ise-uiuc/magicoder)] [[paper](https://arxiv.org/abs/2312.02120)]
   > Training with synthetic instruction data generated from open-source code.

### 3.4 General-purpose (Books, Encyclopedias, Instructions, Long Contexts, etc.)

General-purpose data is often scarce long-tail data, which plays a crucial role in the usability of post-training models.

1. **YuLan: An Open-source Large Language Model**. [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[code](https://github.com/RUC-GSAI/YuLan-Chat)] [[paper](https://arxiv.org/abs/2406.19853)]
   > Long-tail knowledge enhancement and cleaning of various general-purpose data sources.
2. **MinerU: An Open-Source Solution for Precise Document Content Extraction**. [[code](https://github.com/opendatalab/MinerU)] [[paper](https://arxiv.org/abs/2409.18839)]
   > Converting PDF to Markdown with strong compatibility.
3. **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**. [[homepage](https://pile.eleuther.ai/)] [[paper](https://arxiv.org/abs/2101.00027)]
   > arXiv, conversations, DM Math, etc.
4. **Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research**. [[resource](https://huggingface.co/datasets/allenai/dolma)] [[paper](https://arxiv.org/abs/2402.00159)]
   > Encyclopedias, books, papers, Reddit, etc.
5. **WanJuan: A Comprehensive Multimodal Dataset for Advancing English and Chinese Large Models**. [[resource](https://github.com/opendatalab/WanJuan)] [[paper](https://arxiv.org/abs/2308.10755)]
   > Law, exams, news, patents, encyclopedias, etc.
6. **MAmmoTH2: Scaling Instructions from the Web**. [[resource](https://arxiv.org/abs/2401.12246)] [[paper](https://github.com/togethercomputer/MAmmoTH)]
   > Q&A for web pages.
7. **togethercomputer/Long-Data-Collections**. [[resource](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections)]
   > Filtered books, papers, web pages, and instructions from datasets such as RedPajama, Pile, and P3.
8. **Longattn: Selecting long-context training data via token-level attention**. [[resource](https://github.com/Lyun0912-wu/LongAttn)] [[paper](https://arxiv.org/abs/2502.16860)]
   > Q&A for long-range dependencies.

[⬆️ Back to Table of Contents](#table-of-contents)

## IV. Data Methods

Datasets are often paired with high-quality data methods. We elaborate on this from aspects such as tokenizers, data配比 and courses, and data synthesis.

### 4.1 Tokenizers

Tokenization is an important but often overlooked part of the model, which can significantly affect the model's ability in mathematics, knowledge, etc.

1. **SuperBPE: Space Travel for Language Models**. [[code](https://github.com/PythonNut/superbpe)] [[paper](http://arxiv.org/abs/2503.13423)]
   > A multi-word tokenizer training method.
2. **Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies**. [[code](https://github.com/sail-sg/scaling-with-vocab)] [[demo](https://huggingface.co/spaces/sail/scaling-with-vocab-demo)] [[paper](https://arxiv.org/abs/2407.13623)]
   > Predicting the vocabulary size.
3. **Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs**. [[code](https://github.com/aadityasingh/tokenizationcounts)] [[paper](https://arxiv.org/abs/2402.14903)]
   > Comparing the tokenization methods of numbers.

### 4.2 Data Mixing and Curriculum

Multi-stage pre-training often enables the model to fully learn high-quality and small-scale data. Introducing more mathematical, code, CoT, and even long-thinking chain data in the continued pre-training (CPT) stage will form the core capabilities of the next generation of pre-trained models.

1. **Nemotron-4 15B Technical Report**. [[paper](https://arxiv.org/abs/2402.16819)]
   > Divided into 8T pre-training + CPT with a smaller data scale.
2. **YuLan-Mini: An Open Data-efficient Language Model**. [[code](https://github.com/RUC-GSAI/YuLan-Mini?tab=readme-ov-file)] [[resource](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)] [[paper](https://arxiv.org/abs/2412.17743)]
   > Using educational scores for course data.
3. **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining**. [[code](https://github.com/mlfoundations/doremi)] [[paper](https://arxiv.org/abs/2305.10429)]
   > Optimizing the pre-training data mixing ratio.
4. **Efficient Online Data Mixing For Language Model Pre-Training**. [[paper](https://arxiv.org/abs/2312.02406)]
   > Online data mixing.
5. **Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance**. [[paper](https://arxiv.org/abs/2305.05461)]
   > Data mixing laws.
6. **Data Mixture Inference: What do BPE Tokenizers Reveal about their Training Data?**. [[paper]()]
   > Cracking the data ratio of commercial models such as GPT through the merging rules of BPE tokenizers.
7. **CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training**. [[homepage](https://research.nvidia.com/labs/lpr/climb/)] [[paper](https://arxiv.org/abs/2504.13161)]
   > A clustering-based iterative data mixing bootstrapping framework.
8. **Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens**. [[demo](https://huggingface.co/spaces/liujch1998/infini-gram)] [[homepage](https://infini-gram.io/)] [[paper](https://arxiv.org/abs/2401.17377)]
   > Building an index for large-scale pre-training datasets to check data quality.

### 4.3 Data Synthesis

In addition to the synthetic data for mathematics and code mentioned above, we summarize some general synthetic data methods and resources. Moreover, using more long-thinking data in the later stage of pre-training is also becoming a direction worthy of exploration.

1. **Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems**. [[resource](https://huggingface.co/datasets/RUC-AIBOX/long_form_thought_data_5k)] [[code](https://github.com/RUCAIBox/Slow_Thinking_with_LLMs)] [[paper](https://arxiv.org/abs/2412.09413)]
   > Imitation learning based on long-thinking chain synthetic data.
2. **Knowledge-Instruct: Effective Continual Pre-training from Limited Data using Instructions**. [[code](https://github.com/meniData1/knowledge-instruct)] [[paper](https://arxiv.org/abs/2504.05571)]
   > Generating synthetic instruction data rich in information to learn knowledge from a limited corpus.
3. **LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs**. [[resource](https://huggingface.co/datasets/THUDM/LongWriter-6k)] [[code](https://github.com/THUDM/LongWriter)] [[paper](https://arxiv.org/abs/2408.07055)]
   > Constructing long-text Creative Writing.
4. **Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use**. [[paper](https://arxiv.org/abs/2504.04736)]
   > Multi-step reasoning data synthesis, decomposing complex tasks into sub-trajectories and optimizing data generation with reinforcement learning.
5. **WildChat: 1M ChatGPT Interaction Logs in the Wild**. [[resource](https://huggingface.co/datasets/allenai/WildChat-1M)] [[paper](https://arxiv.org/abs/2405.01470)]
   > An open-source dataset of real user conversations.
6. **Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing**. [[resource](https://huggingface.co/Magpie-Align)] [[code](https://github.com/magpie-align/magpie)] [[paper](https://arxiv.org/abs/2406.08464)]
   > Alignment data synthesis.

[⬆️ Back to Table of Contents](#table-of-contents)

## Contribution

If you have suggestions for the project content, please submit [Issues](https://github.com/RUCAIBox/awesome-llm-pretraining/issues/new) and PRs to jointly promote the development of large language models.
