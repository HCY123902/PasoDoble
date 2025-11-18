# Better LLM reasonign via Dual-Play

This is the repostory for the paper ["Better LLM reasonign via Dual-Play"](https://arxiv.org/abs/2511.11881). Our project page is at https://hcy123902.github.io/PasoDoble

## Setup

```
conda create -n pasodoble python=3.10.16
conda activate pasodoble

git clone https://github.com/PasoDoble-Cornell/PasoDoble.git
cd PasoDoble
pip install -r requirements.txt

# Install flash-attention separately
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# (Optional) If your current binutils version is lower than 2.38, upgrade with
conda install -c conda-forge binutils=2.40

mkdir history_record
```

## Supervised Finetuning

To create SFT datasets, run

```
export OPENAI_API_KEY="{your_api_key}"
bash generate_sft.sh &> history_record/generate_sft_qwen3_14b.out &
```

This creates a SFT dataset JSON object for the Proposer and another dataset JSON object prefixed with "solver_" for the Solver. You can then use libraries such as LLaMAFactory to separately supervise finetune a Proposer and a Solver from the same model of your choice.

## Training

We tried 2 training settings that are both functional. The first setting uses 4 A100 80G GPUs (2 for vLLM and 2 for trainers) and it works for models under 1B, and the second uses 8 A100 GPUs (2 for vLLM and 6 for trainers) and it works for 1B and larger models. The other hyperparameters are kept the same except for the number of parallel processes for the trainer.

Set `SOLVER_MODEL_NAME` and `PROPOSER_MODEL_NAME` to the corresponding Solver and Proposer checkpoints after SFT.

Online:
```
bash train_online.sh &> history_record/train_online_4_cards.out &
```
or 
```
bash train_online.sh &> history_record/train_online_8_cards.out &
```

Offline:
```
bash train_offline.sh &> history_record/train_offline_4_cards.out &
```
or 
```
bash train_offline.sh &> history_record/train_offline_8_cards.out &
```

Note: If you want to train with different number of GPUs, make sure the following is true:
* `CUDA_VISIBLE_DEVICES={corresponding_indices}` when revoking `train_*.py` in `train_*.sh`
* `PROPOSER_NUM_GENERATIONS` in `train_*.sh` is divisible by `{num_cards}`
* `num_processes={num_cards}` in `configs/accelerate_config.yaml`
* `train_batch_size={num_cards}` in `configs/proposer_deepspeed_config.json` and `configs/solver_deepspeed_config.json`.

## Trained checkpoints

| **Model** | **Training** | **Download** |
| :------------: | :------------: | :------------: |
| PasoDoble Qwen2.5-0.5B | online | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen2.5-0.5b-solver-online-new)   |
| PasoDoble Qwen2.5-0.5B | offline | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen2.5-0.5b-solver-offline)   |
| PasoDoble Qwen2.5-1.5B  | online | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen2.5-1.5b-solver-online)   |
| PasoDoble Qwen2.5-1.5B  | offline | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen2.5-1.5b-solver-offline)   |
| PasoDoble Qwen2.5-3B  | online | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen2.5-3b-solver-online)   |
| PasoDoble Qwen2.5-3B  | offline | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen2.5-3b-solver-offline)   |
| PasoDoble Qwen3-0.6B  | online | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen3-0.6b-solver-online)   |
| PasoDoble Qwen3-0.6B  | offline | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen3-0.6b-solver-offline)   |
| PasoDoble Qwen3-1.7B  | online | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen3-1.7b-solver-online)   |
| PasoDoble Qwen3-1.7B  | offline | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen3-1.7b-solver-offline)   |
| PasoDoble Qwen3-4B  | online | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen3-4b-solver-online)   |
| PasoDoble Qwen3-4B  | offline | [🤗 HuggingFace](https://huggingface.co/PasoDoble-Cornell/Qwen3-4b-solver-offline)   |

## Citation

```
@article{zhang2025pasodoble,
  title={Better LLM Reasoning via Dual-Play},
  author={Zhengxin Zhang and Chengyu Huang and Aochong Oliver Li and Claire Cardie},
  eprint={2511.11881},
  archivePrefix={arXiv},
  year={2025},
  url={https://arxiv.org/abs/2511.11881}
}
```
