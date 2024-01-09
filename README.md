# 701_FinalProject

Recent advancements in the domain of large language models have led to a focus on integrating human feedback during the fine-tuning process. This strategy aims to align the behaviors of these models with human preferences such that they have higher performance on benchmark tests or soft human evaluation in specific tasks. However, a standardized method for collecting and integrating human feedback into language models remains an open challenge. This study seeks to conduct a review of current theories and practices concerning the incorporation of human preferences into language models, particularly on methods that optimize the frame work of Reinforcement Learning with Human Feedback (RLHF). We implement some of the experiments and aim to provide insights towards the performance and reproducibility of these methods.

First, clone the LlaMA-Factory with the following command:

```
git clone https://github.com/hiyouga/LLaMA-Factory.git
conda create -n llama_factory python=3.10
conda activate llama_factory
cd LLaMA-Factory
pip install -r requirements.txt
```
