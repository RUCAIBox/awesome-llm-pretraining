# Awesome LLM Pre-training

Pre-training is the first and most crucial training stage in the development of large language models. As the open-source community continues to improve in areas such as model architecture, training strategies, open-source datasets, and data methods, we consistently monitor the resources available for large model pre-training to give back to the developers of large language models in the open-source community.

Compared to a comprehensive review, our coverage will be limited to common resources and cutting-edge attempts related to pre-training, enabling users to quickly get started with large language model pre-training. Meanwhile, we welcome updates from the open-source community to jointly promote the development of large models.

## Table of Contents

- [Technical Reports](#technical-reports)
- [Training Strategies](#training-strategies)
- [Open-Source Datasets](#open-source-datasets)
- [Data Methods](#data-methods)

## Technical Reports

Behind technical reports often lie hundreds or thousands of computing resources. Therefore, it is highly recommended to read some open-source technical reports.

### Dense Models

1. **The Llama 3 Herd of Models**. [[paper](https://arxiv.org/abs/2407.21783)]
2. **Qwen2.5 Technical Report**. [[paper](https://arxiv.org/abs/2412.15115)]
3. **Gemma 3 Technical Report**. [[paper](https://arxiv.org/abs/2503.19786)]
4. **Nemotron-4 340B Technical Report**. [[paper](https://arxiv.org/abs/2406.11704)]
5. **Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs**. [[paper](https://arxiv.org/abs/2504.07866)]
6. **Baichuan 2: Open Large-scale Language Models**. [[paper](https://arxiv.org/abs/2309.10305)]

### MoE Models

1. **DeepSeek-V3 Technical Report**. [[paper](https://arxiv.org/abs/2412.19437)]
2. **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2401.06066)]
3. **Mixtral of Experts**. [[paper](https://arxiv.org/abs/2401.04088)]
4. **Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2406.06563)]
5. **Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs**. [[paper](https://arxiv.org/abs/2503.05139)]
6. **OLMoE: Open Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2409.02060)]
7. **Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent**. [[paper](https://arxiv.org/abs/2411.02265)]

### Models with Open-Source Datasets

1. **YuLan-Mini: An Open Data-efficient Language Model**. [[paper](https://arxiv.org/abs/2412.17743)]
2. **MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series**. [[paper](https://arxiv.org/abs/2405.19327)]
3. **LLM360: Towards Fully Transparent Open-Source LLMs**. [[paper](https://arxiv.org/abs/2312.06550)]
4. **Nemotron-4 15B Technical Report**. [[paper](https://arxiv.org/abs/2402.16819)]

### Training/Data Strategies

1. **Phi-4 Technical Report**. [[paper](https://arxiv.org/abs/2412.08905)]
2. **OLMo: Accelerating the Science of Language Models**. [[paper](https://arxiv.org/abs/2402.00838)]
3. **2 OLMo 2 Furious**. [[paper](https://arxiv.org/abs/2501.00656)]
4. **Yi: Open Foundation Models by 01.AI**. [[paper](https://arxiv.org/abs/2403.04652)]
5. **MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies**. [[paper](https://arxiv.org/abs/2404.06395)]

### Hybrid/Linear Models

1. **Falcon Mamba: The First Competitive Attention-free 7B Language Model**. [[paper](https://arxiv.org/abs/2410.05355)]
2. **MiniMax-01: Scaling Foundation Models with Lightning Attention**. [[paper](https://arxiv.org/abs/2501.08313)]
3. **Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models**. [[paper](https://arxiv.org/abs/2504.03624)]

<details>
<summary>All Technical Reports</summary>

## LLaMA Series

1. **LLaMA: Open and Efficient Foundation Language Models**. [[paper](https://arxiv.org/abs/2302.13971)]
2. **Llama 2: Open Foundation and Fine-Tuned Chat Models**. [[paper](https://arxiv.org/abs/2307.09288)]
3. **The Llama 3 Herd of Models**. [[paper](https://arxiv.org/abs/2407.21783)]

---

## Qwen Series

1. **Qwen Technical Report**. [[paper](https://arxiv.org/abs/2309.16609)]
2. **Qwen2 Technical Report**. [[paper](https://arxiv.org/abs/2407.10671)]
3. **Qwen2.5 Technical Report**. [[paper](https://arxiv.org/abs/2412.15115)]

---

## DeepSeek Series

1. **DeepSeek LLM: Scaling Open-Source Language Models with Longtermism**. [[paper](https://arxiv.org/abs/2401.02954)]
2. **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2401.06066)]
3. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**. [[paper](https://arxiv.org/abs/2402.03300)]
4. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**. [[paper](https://arxiv.org/abs/2405.04434)]
5. **DeepSeek-V3 Technical Report**. [[paper](https://arxiv.org/abs/2412.19437)]

---

## Gemma Series

1. **Gemma: Open Models Based on Gemini Research and Technology**. [[paper](https://arxiv.org/abs/2403.08295)]
2. **Gemma 2: Improving Open Language Models at a Practical Size**. [[paper](https://arxiv.org/abs/2408.00118)]
3. **Gemma 3 Technical Report**. [[paper](https://arxiv.org/abs/2503.19786)]

---

## Gemini Series

1. **Gemini: A Family of Highly Capable Multimodal Models**. [[paper](https://arxiv.org/abs/2312.11805)]
2. **Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context**. [[paper](https://arxiv.org/abs/2403.05530v5)]

---

## Mistral Series

1. **Mistral 7B**. [[paper](https://arxiv.org/abs/2310.06825)]
2. **Mixtral of Experts**. [[paper](https://arxiv.org/abs/2401.04088)]

---

## Phi Series

1. **Textbooks Are All You Need**. [[paper](https://arxiv.org/abs/2306.11644)]
2. **Textbooks Are All You Need II: phi-1.5 technical report**. [[paper](https://arxiv.org/abs/2309.05463)]
3. **Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone**. [[paper](https://arxiv.org/abs/2404.14219)]
4. **Phi-4 Technical Report**. [[paper](https://arxiv.org/abs/2412.08905)]

---

## GLM Series

1. **GLM: General Language Model Pretraining with Autoregressive Blank Infilling**. [[paper](https://arxiv.org/abs/2103.10360)]
2. **GLM-130B: An Open Bilingual Pre-trained Model**. [[paper](https://arxiv.org/abs/2210.02414)]
3. **ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools**. [[paper](https://arxiv.org/abs/2406.12793)]

---

## Baichuan Series

1. **Baichuan 2: Open Large-scale Language Models**. [[paper](https://arxiv.org/abs/2309.10305)]
2. **Baichuan-M1: Pushing the Medical Capability of Large Language Models**. [[paper](https://arxiv.org/abs/2502.12671)]

---

## Falcon Series

1. **The Falcon Series of Open Language Models**. [[paper](https://arxiv.org/abs/2311.16867)]
2. **Falcon2-11B Technical Report**. [[paper](https://arxiv.org/abs/2407.14885)]
3. **Falcon Mamba: The First Competitive Attention-free 7B Language Model**. [[paper](https://arxiv.org/abs/2410.05355)]

---

## InternLM Series

1. **InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities**. [[paper](https://github.com/InternLM/InternLM-techreport/blob/main/InternLM.pdf)]
2. **InternLM2 Technical Report**. [[paper](https://arxiv.org/abs/2403.17297)]

---

## MiniCPM

1. **MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies**. [[paper](https://arxiv.org/abs/2404.06395)]

---

## Yi Series

1. **Yi: Open Foundation Models by 01.AI**. [[paper](https://arxiv.org/abs/2403.04652)]
2. **Yi-Lightning Technical Report**. [[paper](https://arxiv.org/abs/2412.01253)]

---

## Minimax Series

1. **MiniMax-01: Scaling Foundation Models with Lightning Attention**. [[paper](https://arxiv.org/abs/2501.08313)]

---

## Reka Series

1. **Reka Core, Flash, and Edge: A Series of Powerful Multimodal Language Models**. [[paper](https://arxiv.org/abs/2404.12387v1)]

---

## Skywork Series

1. **Skywork: A More Open Bilingual Foundation Model**. [[paper](https://arxiv.org/abs/2310.19341)]
2. **Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2406.06563)]

---

## Hunyuan Series

1. **Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent**. [[paper](https://arxiv.org/abs/2411.02265)]

---

## Nemotron Series

1. **Nemotron-4 15B Technical Report**. [[paper](https://arxiv.org/abs/2402.16819)]
2. **Nemotron-4 340B Technical Report**. [[paper](https://arxiv.org/abs/2406.11704)]
3. **Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models**. [[paper](https://arxiv.org/abs/2504.03624)]

---

## Ling Series

1. **Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs**. [[paper](https://arxiv.org/abs/2503.05139)]

---

## OLMo Series

1. **OLMo: Accelerating the Science of Language Models**. [[paper](https://arxiv.org/abs/2402.00838)]
2. **2 OLMo 2 Furious**. [[paper](https://arxiv.org/abs/2501.00656)]
3. **OLMoE: Open Mixture-of-Experts Language Models**. [[paper](https://arxiv.org/abs/2409.02060)]

---

## Yulan Series

1. **YuLan: An Open-source Large Language Model**. [[paper](https://arxiv.org/abs/2406.19853)]
2. **YuLan-Mini: An Open Data-efficient Language Model**. [[paper](https://arxiv.org/abs/2412.17743)]

---

## MAP-Neo Series

1. **MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series**. [[paper](https://arxiv.org/abs/2405.19327)]

---

## LLM360 Project

1. **LLM360: Towards Fully Transparent Open-Source LLMs**. [[paper](https://arxiv.org/abs/2312.06550)]

</details>

## Training Strategies

### Training Frameworks

The most commonly used training framework is Megatron-LM, which provides a good out-of-the-box and efficient benchmark.

1. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**
   > The most commonly used pre-training framework
2. **Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters**
   > Zero-redundancy data parallelism
3. **Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts**. 
   > MoE computation-communication overlapping
4. **DeepEP: an efficient expert-parallel communication library**
   > Expert parallel acceleration
5. **DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling**
   > Accelerating FP8 matrix multiplication using the asynchronous features of Hopper
6. **Liger Kernel: Efficient Triton Kernels for LLM Training**
   > Triton acceleration operator library

### Training Strategies

1. **Predictable Scale: Part I -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining**
   > Regarding the Scaling Law of hyperparameters
2. **The Ultra-Scale Playbook: Training LLMs on GPU Clusters**
   > Visualizing the memory usage of parallel strategies
3. **A Spectral Condition for Feature Learning**
   > An advanced version of MuP
4. **Muon is Scalable for LLM Training**
   > An efficient optimizer
5. **COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training**


