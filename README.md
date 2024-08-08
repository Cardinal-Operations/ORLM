# ORLM: Training Large Language Models for Optimization Modeling

<p align="center" width="100%">
<a ><img src="./imgs/orlm_method.png" alt="OR-Instruct" style="width: 80%; min-width: 300px; display: block; margin: auto;"></a>
</p> 

This project explores training open-source LLMs for optimization modeling. We identify four critical requirements for the training dataset of OR LLMs, design and implement OR-Instruct, a semi-automated process for creating synthetic data tailored to specific requirements. We also introduce the IndustryOR benchmark, the first industrial benchmark for testing LLMs on solving real-world OR problems. We apply the data from OR-Instruct to various open-source LLMs of 7b size (termed as ORLMs), resulting in a significantly improved capability for optimization modeling. [Read our paper here](https://arxiv.org/abs/2405.17743).

## News
- 🔥 We're excited to offer an [Interactive DEMO](https://huggingface.co/spaces/tangzhy/ORLM) of the ORLM-LLaMA-3-8B model, thanks to donations of NVIDIA-A100 from [Huggingface ZeroGPU](https://huggingface.co/zero-gpu-explorers).
- 🔥 We released a sample of the OR-Instruct Data for training LLMs! [View it here](https://huggingface.co/datasets/CardinalOperations/OR-Instruct-Data-3K).
- 🔥 Our [**ORLM-LLaMA-3-8B**](https://huggingface.co/CardinalOperations/ORLM-LLaMA-3-8B) model achieves SOTA on the [NL4OPT](https://huggingface.co/datasets/CardinalOperations/NL4OPT), [MAMO](https://huggingface.co/datasets/CardinalOperations/MAMO), and [IndustryOR](https://huggingface.co/datasets/CardinalOperations/IndustryOR) benchmarks!
- 🔥 We launched the IndustryOR benchmark, the first industrial benchmark, consists of 100 real-world OR problems! [Check it out](https://huggingface.co/datasets/CardinalOperations/IndustryOR).


| Model             | Checkpoint | License | NL4OPT | MAMO EasyLP | MAMO ComplexLP | IndustryOR | Micro Avg | Macro Avg |
|-------------------|------------|---------|--------|-------------|----------------|------------|-----------|-----------|
| ORLM-LLaMA-3-8B   | 🤗 <a href="https://huggingface.co/CardinalOperations/ORLM-LLaMA-3-8B" target="_blank">HF Link</a> | <a href="https://llama.meta.com/llama3/license/" target="_blank">llama3</a> | 85.7% | 82.3% | 37.4% | 38.0% | 71.4% | 60.8% |

                                                                                               

## Performances

Below is the comparison of performance on the NL4OPT, MAMO, and IndustryOR benchmarks. Values marked with a <sup>*</sup> are directly copied from original papers, with blanks where data were not reported. The highest results are highlighted in bold.

| **Method**                                     | **NL4OPT**              | **MAMO EasyLP**       | **MAMO ComplexLP**  | **IndustryOR**    | **Micro Avg**   | **Macro Avg**   |
|------------------------------------------------|-------------------------|-----------------------|----------------------|-------------------|-----------------|-----------------|
| *Methods based on PLMs*                        |                         |                       |                      |                   |                 |                 |
| `tag-BART`                                     | 47.9%<sup>*</sup>               | -                     | -                    | -                 | -               | -               |
| *Methods based on GPT-3.5*                     |                         |                       |                      |                   |                 |                 |
| `Standard`                                     | 42.4%<sup>*</sup>                | -                     | -                    | -                 | -               | -               |
| `Reflexion`                                    | 50.7%<sup>*</sup>                | -                     | -                    | -                 | -               | -               |
| `Chain-of-Experts`                             | 58.9%<sup>*</sup>                | -                     | -                    | -                 | -               | -               |
| *Methods based on GPT-4*                       |                         |                       |                      |                   |                 |                 |
| `Standard`                                     | 47.3%<sup>*</sup>                | 66.5%<sup>*</sup>              | 14.6%<sup>*</sup>             | 28.0%             | 50.2%           | 39.1%           |
| `Reflexion`                                    | 53.0%<sup>*</sup>                | -                     | -                    | -                 | -               | -               |
| `Chain-of-Experts`                             | 64.2%<sup>*</sup>                | -                     | -                    | -                 | -               | -               |
| `OptiMUS`                                      | 78.8%<sup>*</sup>                | -                     | -                    | -                 | -               | -               |
| *ORLMs based on open-source LLMs*              |                         |                       |                      |                   |                 |                 |
| `ORLM-Mistral-7B`                              | 84.4%                   | 81.4%                 | 32.0%                | 27.0%             | 68.8%           | 56.2%           |
| `ORLM-Deepseek-Math-7B-Base`                   | **86.5%**               | 82.2%                 | **37.9%**            | 33.0%             | 71.2%           | 59.9%           |
| `ORLM-LLaMA-3-8B`                              | 85.7%                   | **82.3%**             | 37.4%                | **38.0%**         | **71.4%**       | **60.8%**       |

## Setup

To get started, clone ORLM and install the required packages:

```bash
git clone https://github.com/Cardinal-Operations/ORLM.git
cd ORLM
pip install -r requirements.txt
```

## Inference

Prompting Template:
```text
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}

# Response:
```

Please replace the `{Question}` with any natural language OR question.

To run a sample inference, use this command:

```bash
cd ORLM
python scripts/inference.py --model_name_or_path <path_to_local_orlm_directory> --tensor_parallel_size <num_gpus>
```

## Evaluation

First, we prompt the ORLMs to generate a complete solution that includes both a mathematical model and a program (refer to `eval/generate.py`). We then extract this program and run it to obtain the predicted optimal value using parallel processing (see `eval/execute.py`, currently supporting only the COPT solver). We evaluate the accuracy by comparing the execution results with the ground truth optimal value. Note that variations in results may occur due to differences in computing resources, as executions are performed in parallel. Additionally, for hard examples like IndustryOR, where the number of variables may increase significantly, consider applying for [COPT web licenses](https://copt.shanshu.ai/license/home). Otherwise, execution may directly fail.

Here's how to evaluate the ORLM models on various benchmarks:

```bash
# (Optional) If you have trouble accessing the Hugging Face website, you can set an alternative endpoint:
# export HF_ENDPOINT=https://hf-mirror.com
cd ORLM
sh scripts/eval.NL4OPT.sh <path_to_local_orlm_directory>
sh scripts/eval.MAMO_EasyLP.sh <path_to_local_orlm_directory>
sh scripts/eval.MAMO_ComplexLP.sh <path_to_local_orlm_directory>
sh scripts/eval.IndustryOR.sh <path_to_local_orlm_directory>
```

We also provide detailed completions and execution results in the `results` directory for the ORLM-LLaMA-3-8B model on the above benchmarks.

## Citation
Please cite the paper if you refer to our model, code, data or paper.

```
@article{tang2024orlm,
  title={ORLM: Training Large Language Models for Optimization Modeling},
  author={Tang, Zhengyang and Huang, Chenyu and Zheng, Xin and Hu, Shixi and Wang, Zizhuo and Ge, Dongdong and Wang, Benyou},
  journal={arXiv preprint arXiv:2405.17743},
  year={2024}
}
```
