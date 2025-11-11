import argparse

from datasets import load_dataset
from trl import GRPOConfig
from pasodoble_online import PasoDoble
from reward_utils import accuracy, rand_accuracy
from knowledge_base import Knowledgebase
import os
import torch
from prompt import PROPOSER_PROMPT_WITH_KNOWLEDGE, PROPOSER_PROMPT_WITHOUT_KNOWLEDGE, SOLVER_SYSTEM_PROMPT
# from accelerate import Accelerator
# Exact match for math problems

def main():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--model_name", type=str, required=True,
                        # help="Pretrained model name or path")
    parser.add_argument(f"--proposer_model_name", type=str, default="Qwen/Qwen3-1.7B",
                        help="Pretrained proposer model name or path")
    parser.add_argument(f"--solver_model_name", type=str, default="Qwen/Qwen3-1.7B",
                        help="Pretrained solver model name or path")
    parser.add_argument("--proposer_deepspeed_path", type=str, required=True,
                        help="Path to deepspeed config file")
    parser.add_argument("--solver_deepspeed_path", type=str, required=True,
                        help="Path to deepspeed config file")
    parser.add_argument("--wandb_run_name", type=str, required=True,
                        help="Wandb run name")
    
    
    parser.add_argument("--output_dir_proposer", type=str, default="./output_proposer",
                        help="Output directory for proposer training results")
    parser.add_argument("--output_dir_solver", type=str, default="./output_solver",
                        help="Output directory for solver training results")
    parser.add_argument("--use_knowledge", type=int, default=1,
                        help="Use knowledge or not")
    parser.add_argument("--use_ds", type=int, default=0,
                        help="Use deepspeed or not")
    # parser.add_argument("--num_epochs", type=int, default=5,
    #                     help="Number of training epochs")
    # parser.add_argument("--batch_size", type=int, default=1,
    #                     help="Per device batch size")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of steps")
    parser.add_argument("--max_proposer_prompt_length", type=int, default=256,
                        help="Maximum proposer prompt sequence length")
    parser.add_argument("--max_solver_prompt_length", type=int, default=256,
                        help="Maximum solver prompt sequence length")
    parser.add_argument("--max_proposer_completion_length", type=int, default=800,
                        help="Maximum proposer completion sequence length")
    parser.add_argument("--max_solver_completion_length", type=int, default=800,
                        help="Maximum solver completion sequence length")
    parser.add_argument("--save_steps", type=int, default=10,
                        help="Number of steps between model checkpoints")
    # parser.add_argument("--restore", type=bool, default=False,
    #                     help="Restore from checkpoint or not")
    
    parser.add_argument("--proposer_temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--solver_temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--epsilon_high", type=float, default=None,
                        help="Epsilon high")
    parser.add_argument("--passing_rate_lower_threshold", type=float, default=0.0,
                        help="Passing rate lower threshold")
    parser.add_argument("--passing_rate_upper_threshold", type=float, default=1.0,
                        help="Passing rate upper threshold")
    parser.add_argument("--proposer_num_generations", type=int, default=1,
                        help="Number of generations for proposer")
    parser.add_argument("--solver_num_generations", type=int, default=1,
                        help="Number of generations for solver")
    #parser.add_argument("--max_sub_batch_size", type=int, default=8,
    #                     help="Maximum sub batch size")
    parser.add_argument("--use_vllm", type=bool, default=True,
                        help="Use vllm or not")
    parser.add_argument("--vllm_mode", type=str, default="server")
    parser.add_argument("--proposer_vllm_server_host", type=str, default="", help="The proposer server IP")
    parser.add_argument("--proposer_vllm_server_port", type=str, default="", help="The proposer server port")
    parser.add_argument("--proposer_vllm_client_port", type=str, default="51216", help="The proposer client port")
    parser.add_argument("--solver_vllm_server_host", type=str, default="", help="The solver server IP")
    parser.add_argument("--solver_vllm_server_port", type=str, default="", help="The solver server port")
    parser.add_argument("--solver_vllm_client_port", type=str, default="51217", help="The solver client port")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3,
                        help="GPU memory utilization")
    parser.add_argument("--loss_type", type=str, default="grpo",
                        help="Loss type")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="Beta for DSR")
    parser.add_argument("--solver_reward_func", type=str, default="accuracy",
                        help="Solver reward function")
    
    args = parser.parse_args()
    
    if args.use_knowledge == 1:
        knowledge_base = Knowledgebase("YouAreSpecialToMe/filtered_MegaMath")
        args.use_knowledge = True
    else:
        knowledge_base = None
        args.use_knowledge = False

    if args.use_ds == 1:
        args.use_ds = True
    else:
        args.use_ds = False
    
    if args.use_vllm == 1:
        args.use_vllm = True
    else:
        args.use_vllm = False
    
    
    proposer_prompt = PROPOSER_PROMPT_WITH_KNOWLEDGE if args.use_knowledge == 1 else PROPOSER_PROMPT_WITHOUT_KNOWLEDGE
    solver_prompt = SOLVER_SYSTEM_PROMPT
    
    proposer_grpoconfig = GRPOConfig(
        logging_steps=10,
        bf16=True,
        beta=args.beta,
        output_dir=args.output_dir_proposer,
        loss_type=args.loss_type,
        lr_scheduler_type="cosine",
        top_entropy_quantile=1.0,
        gradient_checkpointing=True,
        per_device_train_batch_size=args.proposer_num_generations,
        generation_batch_size=args.proposer_num_generations * args.solver_num_generations,
        save_steps=args.save_steps,
        # loss_type="grpo",
        # warmup_ratio=0.0,
        # learning_rate=1e-06,
        # deepspeed=args.deepspeed_path,
        gradient_accumulation_steps=1,
        num_generations=args.proposer_num_generations,
        report_to='wandb',
        max_prompt_length=args.max_proposer_prompt_length,
        max_completion_length=args.max_proposer_completion_length,
        fp16_full_eval=True,
        bf16_full_eval=False,
        epsilon_high=args.epsilon_high,
        run_name="Proposer-"+os.path.basename(args.output_dir_proposer),
        temperature=args.proposer_temperature,
        log_completions=True,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_host=args.proposer_vllm_server_host,
        vllm_server_port=args.proposer_vllm_server_port,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    
    solver_grpoconfig = GRPOConfig(
        logging_steps=10,
        bf16=True,
        beta=args.beta,
        output_dir=args.output_dir_solver,
        loss_type=args.loss_type,
        lr_scheduler_type="cosine",
        top_entropy_quantile=1.0,
        gradient_checkpointing=True,
        save_steps=args.save_steps,
        # warmup_ratio=0.0,
        # learning_rate=1e-06,
        # deepspeed=args.deepspeed_path,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=args.solver_num_generations,
        generation_batch_size=args.solver_num_generations * args.proposer_num_generations,
        num_generations=args.solver_num_generations,
        report_to='wandb',
        max_prompt_length=args.max_solver_prompt_length,
        max_completion_length=args.max_solver_completion_length,
        temperature=args.solver_temperature,
        epsilon_high=args.epsilon_high,
        fp16_full_eval=True,
        bf16_full_eval=False,
        run_name="Solver-"+os.path.basename(args.output_dir_solver),
        log_completions=True,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_host=args.solver_vllm_server_host,
        vllm_server_port=args.solver_vllm_server_port,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
        
    paso_doble = PasoDoble(
        proposer_model=args.proposer_model_name,
        solver_model=args.solver_model_name,
        proposer_grpoconfig=proposer_grpoconfig,
        solver_grpoconfig=solver_grpoconfig,
        #  ccelerator=accelerator,
        proposer_prompt=proposer_prompt,
        solver_prompt=solver_prompt,
        solver_reward_func=accuracy if (args.solver_reward_func == "accuracy") else (rand_accuracy if (args.solver_reward_func == "rand_accuracy") else None),
        passing_rate_lower_threshold=args.passing_rate_lower_threshold,
        passing_rate_upper_threshold=args.passing_rate_upper_threshold,
        knowledge_base=knowledge_base,
        max_steps=args.max_steps,
        # save_steps=args.save_steps,
        use_knowledge=args.use_knowledge,
        use_ds=args.use_ds,
        proposer_ds_config=args.proposer_deepspeed_path,
        solver_ds_config=args.solver_deepspeed_path,
        other_args=args
    )
    
    paso_doble.train()

if __name__=="__main__":
    main()
    
