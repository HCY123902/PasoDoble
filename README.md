# PasoDoble

This is the repostory for the paper "Better LLM reasonign via Dual-Play".

## Setup

```
uv venv pasodoble --python 3.10.16
source pasodoble/bin/activate

git clone https://github.com/PasoDoble-Cornell/PasoDoble.git
cd PasoDoble
uv pip install -r requirements.txt

# Install flash-attention separately
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Supervised Finetuning

To create SFT datasets, run

```
export OPENAI_API_KEY="{your_api_key}"
bash generate_sft.sh &> generate_sft_qwen3_14b.out &
```

This creates a SFT dataset JSON object for the Proposer and another dataset JSON object prefixed with "solver_" for the Solver. You can then use libraries such as LLaMAFactory to separately supervise finetune a Proposer and a Solver from the same model of your choice.

## Training

We tried 2 training settings that are both functional. For models smaller than 3B, we use 4 A100 80G GPUs (2 for vLLM and 2 for trainers). For 3B and 4B models, we use 8 H100 GPUs (2 for vLLM and 6 for trainers). The latter uses more parallel processes for the trainers, but other hyperparameters are kept the same.

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
bash train_online.sh &> history_record/train_offline_4_cards.out &
```
or 
```
bash train_online.sh &> history_record/train_offline_8_cards.out &
```


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
  journal={Conference/Journal Name},
  year={2025},
  url={https://hcy123902.github.io/PasoDoble}
}
```