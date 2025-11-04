# PasoDoble

This is the repostory for the paper "Better LLM reasonign via Dual-Play".

## Setup

```
uv venv pasodoble --python 3.10.16
source pasodoble

git clone https://github.com/HCY123902/PasoDoble.git
cd PasoDoble
uv pip install -r requirements.txt
```

## Supervised Finetuning

To create SFT datasets, run

```
export OPENAI_API_KEY="{your_api_key}"
bash generate_sft.sh &> generate_sft_qwen3_14b.out &
```

This creates a SFT dataset JSON object for the Proposer and another dataset JSON object prefixed with "solver_" for the Solver. You can then use libraries such as LLaMAFactory to separately supervise finetune a Proposer and a Solver from the same model of your choice.

## Training

We tried 2 training settings that are both functional. The first setting uses 4 A100 80G GPUs (2 for vLLM and 2 for trainers) and the second uses 8 A6000 GPUs (2 for vLLM and 6 for trainers). The latter uses more parallel processes for the trainers, but other hyperparameters are kept the same.

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
