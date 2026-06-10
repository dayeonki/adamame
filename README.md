<div align="center">

 # AdaMame: A Training Recipe for <br> Adaptive Multilingual Reasoning


<p align="center">
 <img width="922" height="296" alt="Screenshot 2026-06-07 at 2 45 17 PM" src="https://github.com/user-attachments/assets/8c12a2fb-d040-4323-aa54-f2a53fd3433c" />
</p>

<a href=https://dayeonki.github.io/>Dayeon Ki</a><sup>1</sup>, <a href=https://www.cs.jhu.edu/~kevinduh/>Kevin Duh</a><sup>2</sup>, <a href=https://www.cs.umd.edu/~marine/>Marine Carpuat<a><sup>1</sup> <br>
<sup>1</sup>University of Maryland, <sup>2</sup>Johns Hopkins University
<br>

This repository contains the code and dataset for our paper <br> **AdaMame: A Training Recipe for Adaptive Multilingual Reasoning**.

<p>
  <a href="" target="_blank" style="text-decoration:none">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat&logo=arxiv" alt="arXiv">
  </a>
</p>

</div>

---
## 👾 TL;DR
We introduce AdaMame, a two-stage SFT+RL training recipe that resolves language collapse by adaptively aligning reasoning language to the query language.

## 📰 News
- **`2026-06-09`** We release our code, dataset, and models!


## ✏️ Content
- [🗺️ Overview](#overview)
- [🚀 Quick Start](#quick_start)
  - [Environment Setup](#environment-setup)
  - [Model Download](#model-download)
  - [Data Preparation](#data-preparation)
  - [Stage 1: SFT](#stage-1-sft)
  - [Stage 2: RL](#stage-2-rl)
  - [Model Evaluation](#model-evaluation)
- [🤲 Citation](#citation)
- [📧 Contact](#contact)

---


<a id="overview"></a>
## 🗺️ Overview

<p align="center">
<img width="381" height="313" alt="Screenshot 2026-06-07 at 2 48 25 PM" src="https://github.com/user-attachments/assets/bfcbc617-f993-4a85-9538-3cb2b474de6a" />
</p>

While Large Reasoning Models (LRMs) show strong performance in English, they often fail to reason in the language of the query, a phenomenon known as *language collapse*. 
Existing RL-based fixes typically add a binary language fidelity reward to the accuracy objective, yet still incur trade-off in accuracy, mid-trace code-switching, and excessive token usage. 

We propose **AdaMame**, a two-stage training recipe for multilingual mathematical reasoning that addresses these limitations by adaptively aligning the reasoning language to the query language without compromising accuracy. 
The first SFT stage fine-tunes on naturally occurring reasoning traces across five languages to establish multilingual reasoning capability. In the subsequent RL stage, we introduce AdaMame-GRPO, an adaptation of Group Relative Policy Optimization (GRPO) in which a query-conditioned alignment factor grows progressively during training, guiding the model to first explore diverse reasoning languages before exploiting reasoning in the query language. 
We show that **AdaMame-GRPO** achieves Pareto-optimal performance across reasoning accuracy, language fidelity, and token efficiency.


### Results

<div align="center">
<img width="903" height="362" alt="Screenshot 2026-06-07 at 2 49 00 PM" src="https://github.com/user-attachments/assets/1e1a3b9e-c025-4df2-830e-186865480494" />
</div>


<a id="quick_start"></a>
## 🚀 Quick Start

Our pipeline has four stages, each in its own top-level directory:
- `0_data`: Build the SFT and RL training data.
- `1_sft`: **Stage 1 (SFT)** — multilingual reasoning warm-up with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
- `2_rl`: **Stage 2 (RL)** — AdaMame-GRPO training built on [verl](https://github.com/volcengine/verl).
- `3_eval`: Evaluation and RL data selection utilities.

We train and evaluate on **five languages**: French (`fr`), Japanese (`ja`), Korean (`ko`), Portuguese (`pt`), and Thai (`th`).

### Environment Setup

The two training stages use separate environments. Conda specifications are provided in `2_rl/environment`:
- **SFT (LLaMA-Factory)**: `conda env create -f 2_rl/environment/llama_factory_env.yaml`
- **RL (verl)**: `conda env create -f 2_rl/environment/verl_env.yaml`

### Model Download

You can download the models from HuggingFace:
- **SFT** versions
  - Distill-Qwen-1.5B: https://huggingface.co/zoeyki/sft_distill1.5b
  - Qwen-4B: https://huggingface.co/zoeyki/sft_qwen4b
- **GRPO** versions
  - Distill-Qwen-1.5B: https://huggingface.co/zoeyki/grpo_distill1.5b
  - Qwen-4B: https://huggingface.co/zoeyki/grpo_qwen4b
- **AdaMame-GRPO** versions
  - Distill-Qwen-1.5B: https://huggingface.co/zoeyki/adamame_grpo_distill1.5b
  - Qwen-4B: https://huggingface.co/zoeyki/adamame_grpo_qwen4b

### Data Preparation
#### (1) SFT Data Preparation

- All code is located in `0_data/sft_data`.

We start from naturally occurring multilingual reasoning traces and clean them into the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) instruction format. Run the numbered scripts in order:

```bash
python -u 0_data/sft_data/1_split_data.py          # language-split + math-verify filtering
python -u 0_data/sft_data/2_remove_heading_prefix.py  # strip leading "## " headings
python -u 0_data/sft_data/3_count_trace_length.py     # report reasoning-trace token lengths
python -u 0_data/sft_data/4_ensure_final_check.py     # keep traces with a valid final \boxed{} answer
python -u 0_data/sft_data/5_add_instructions.py       # prepend per-language step-by-step instruction
python -u 0_data/sft_data/6_make_llamafactory_format.py  # convert to {instruction, input, output}
python -u 0_data/sft_data/7_random_shuffle.py         # merge languages and shuffle
```

- `1_split_data.py` uses [`lingua`](https://github.com/pemistahl/lingua-py) for language identification and [`math_verify`](https://github.com/huggingface/Math-Verify) to validate answers.
- `5_add_instructions.py` prepends the localized *"Reason step by step and put your final answer in `\boxed{}`"* instruction for each language.
- The final per-language files are written to `0_data/sft_data/llamafactory_format/` (e.g., `ko_gpt5nano_full.json`) and registered through `1_sft/data/dataset_info.json` as the `sft_data` dataset.

#### (2) RL Data Preparation

- All code is located in `0_data/rl_data`.

We build query-conditioned RL prompts from math questions, generating localized reasoning targets per language.

```bash
python -u 0_data/rl_data/make_reasoning.py \
  --input $PATH_TO_INPUT_JSONL \
  --output $PATH_TO_OUTPUT_JSONL \
  --model $MODEL \
  --prompt-lang $LANG \
  --max-output-tokens $MAX_OUTPUT_TOKENS \
  --reasoning-effort $REASONING_EFFORT
```

Arguments for the code are:
- `--input`: Path to the input jsonl file (e.g., `SFT/trit/fr.jsonl`)
- `--output`: Path to the output jsonl file
- `--model`: Generation model (default `gpt-5-nano`; set your `OPENAI_API_KEY`)
- `--prompt-lang`: ISO code of the target language (`fr`, `ja`, `ko`, `pt`, `th`)
- `--max-output-tokens`: Maximum number of output tokens (default 50000)
- `--reasoning-effort`: Reasoning effort, one of `low`, `medium`, `high` (default `medium`)

The RL training/validation splits are sampled and filtered from these generations using the utilities in [Model Evaluation](#model-evaluation) (`sample_for_rl.py`, `filter_for_rl.py`, `convert_to_parquet.py`). The final parquet files used in the paper are provided under `2_rl/data/train` and `2_rl/data/val` (sourced from DAPO-Math-17k).

### Stage 1. SFT

The first stage establishes multilingual reasoning capability by fine-tuning the base model on the cleaned traces with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Each launcher trains a LoRA adapter and then merges it into a standalone model.

For **Distill-Deepseek-1.5B**,
```bash
cd 1_sft
sbatch bash_scripts/sft_distill1.5b.sh
```

For **Qwen3-4B**,
```bash
cd 1_sft
sbatch bash_scripts/sft_qwen4b.sh
```

Each launcher runs two steps via `llamafactory-cli`:
1. **Train**: `llamafactory-cli train stage1_scripts/sft_{model}.yaml`
2. **Merge**: `llamafactory-cli export stage1_scripts/merge_{model}.yaml`

Key settings (see `1_sft/stage1_scripts/*.yaml`):
- `model_name_or_path`: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (`template: deepseekr1`) or `Qwen/Qwen3-4B` (`template: qwen3_nothink`)
- `finetuning_type`: `lora` with `lora_target: all`
- `dataset`: `sft_data` (defined in `1_sft/data/dataset_info.json`)
- `cutoff_len`: 8192, `num_train_epochs`: 4.0, `learning_rate`: 2.0e-4, `lr_scheduler_type`: cosine
- The merged model is exported to `1_sft/model/{model}/sft`, which becomes the `BASE_MODEL` for Stage 2.

### Stage 2. RL

The second stage runs **AdaMame-GRPO**, our adaptive variant of GRPO implemented on top of [verl](https://github.com/volcengine/verl). A query-conditioned alignment factor grows over the course of training, so the model first **explores** diverse reasoning languages and gradually **exploits** reasoning in the query language. The core advantage estimator is `compute_adamame_outcome_advantage` in `2_rl/verl/trainer/ppo/core_algos.py`, selected via `algorithm.adv_estimator=adamame_align`.

For **Distill-Deepseek-1.5B**,
```bash
cd 2_rl
ALIGN_BONUS=2.0 sbatch bash_scripts/adamame_grpo_distill1.5b.sh
```

For **Qwen3-4B**,
```bash
cd 2_rl
ALIGN_BONUS=2.0 sbatch bash_scripts/adamame_grpo_qwen4b.sh
```

Each launcher (1) calls the matching trainer script in `2_rl/stage2_scripts/trainer/`, then (2) merges the trained LoRA checkpoint into a full model with `scripts/merge_lora_checkpoint.py`. The main RL knobs are:
- `ALIGN_BONUS`: **B** denotes the maximum query-language alignment bonus (default `2.0`). Override at launch, e.g. `ALIGN_BONUS=2.0 bash ...`.
- `algorithm.adv_estimator`: `adamame_align` for AdaMame-GRPO, or `grpo` for the baseline (see `bash_scripts/grpo_*.sh`).
- `actor_rollout_ref.rollout.n`: 8 rollouts per prompt, `temperature`: 0.8
- `model.lora_rank`: 32, `model.lora_alpha`: 64, `actor.optim.lr`: 1e-5, `kl_loss_coef`: 0.001
- `data.train_files` / `data.val_files`: parquet files under `2_rl/data` (DAPO-Math-17k)
- The diversity weight `a(t)` decays from its maximum at step 0 to 1.0 at the final step, while the alignment weight `q(t)` grows from `0.1·B` to `B`, yielding the explore-then-exploit schedule described in the paper.

To merge a checkpoint manually:
```bash
python 2_rl/scripts/merge_lora_checkpoint.py \
  --actor_dir $CHECKPOINT_DIR/global_step_${STEP}/actor \
  --base_model $BASE_MODEL \
  --output_dir $OUTPUT_DIR
```

### Model Evaluation

All evaluation and RL-selection utilities live in `3_eval`. We evaluate on **MGSM-v2** (`mgsmv2`) and **MSVAMP** (`msvamp`).

#### (1) Compute metrics

Score the generated traces on (1) final-answer accuracy, (2) language fidelity, (3) code-switching, and (4) token usage:
```bash
python -u 3_eval/metrics.py
```
Set `DATASET_TYPE` (`mgsmv2` or `msvamp`) and `MODEL_NAMES` at the top of the file. Fidelity is measured at the line and word level (`check_line_level`, `check_word_level`), and mid-trace code-switching is detected with `analyze_code_switching`.

#### (2) Build RL data from evaluation candidates

To turn candidate generations into RL training data:
```bash
# Sample non-overlapping train/val questions per language
python -u 3_eval/sample_for_rl.py --n 2000 --m 0 --seed 42

# Keep prompts with mixed correctness (1 <= num_correct < 8)
python -u 3_eval/filter_for_rl.py

# Convert jsonl splits to parquet for verl
python -u 3_eval/convert_to_parquet.py --data-source $DATA_SOURCE
```

Arguments of note:
- `sample_for_rl.py`: `--n` as train samples per language, `--m` as val samples per language (non-overlapping), `--seed` as random seed
- `filter_for_rl.py`: retains prompts where the model is partially correct across its 8 candidates, providing learning signal for GRPO
- `convert_to_parquet.py`: `--data-source` as value written to each row's `data_source` field


---

<a id="citation"></a>
## 🤲 Citation
If you find our work useful in your research, please consider citing our work:
```
TBD
```

<a id="contact"></a>
## 📧 Contact
For questions, issues, or collaborations, please reach out to [dayeonki@umd.edu](mailto:dayeonki@umd.edu).
