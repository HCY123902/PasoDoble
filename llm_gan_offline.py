from proposergrpo_trainer import ProposerGRPOTrainer
from solvergrpo_trainer import SolverGRPOTrainer
from knowledge_base import Knowledgebase
import re
import regex
from typing import Tuple

from tqdm import tqdm
import torch
from accelerate.utils import DeepSpeedPlugin, gather_object
import torch.distributed as dist
from jaccorbsimilarity import DiversityRewardManager
from question_buffer import QuestionBuffer

from transformers import is_wandb_available

if is_wandb_available():
    import wandb

from prompt import *
from reward_utils import *

class LlmGan:
    def __init__(
            self, 
            proposer_model, 
            solver_model, 
            proposer_grpoconfig, 
            solver_grpoconfig, 
            knowledge_base, 
            proposer_prompt, 
            solver_prompt, 
            use_knowledge=True, 
            use_ds=False, 
            proposer_ds_config=None, 
            solver_ds_config=None, 
            solver_reward_func=None, 
            passing_rate_lower_threshold=0.0, 
            passing_rate_upper_threshold=1.0, 
            max_steps=1000, 
            proposer_update_step=30, 
            solver_update_step=30,
            other_args=None, 
        ):
        self.proposer_model = proposer_model
        self.solver_model = solver_model
        self.proposer_config = proposer_grpoconfig
        self.solver_config = solver_grpoconfig
        self.diversity_reward_manager = DiversityRewardManager()
        self.proposer_update_step = 10
        self.solver_update_step = 5
        if use_ds:
            proposer_deepspeed_config = proposer_ds_config
            solver_deepspeed_config = solver_ds_config

            proposer_ds_plugin = DeepSpeedPlugin(hf_ds_config=proposer_deepspeed_config)
            solver_ds_plugin = DeepSpeedPlugin(hf_ds_config=solver_deepspeed_config)
            deepspeed_plugins = {'proposer': proposer_ds_plugin, 'solver': solver_ds_plugin}
            self.proposer_config.deepspeed_plugin = deepspeed_plugins
            self.solver_config.deepspeed_plugin = None
        
        
        self.max_steps = max_steps
        self.knowledge_base = knowledge_base
        self.proposer_prompt = proposer_prompt
        self.solver_prompt = solver_prompt
        self.passing_rate_lower_threshold = passing_rate_lower_threshold
        self.passing_rate_upper_threshold = passing_rate_upper_threshold
        
        self.use_knowledge = use_knowledge
        self.proposer_trainer = ProposerGRPOTrainer(proposer_model,solver_reward_func, args=proposer_grpoconfig, diversity_reward_manager=self.diversity_reward_manager, passing_rate_lower_threshold=self.passing_rate_lower_threshold, passing_rate_upper_threshold=self.passing_rate_upper_threshold, vllm_client_port=other_args.proposer_vllm_client_port)
        self.solver_trainer = SolverGRPOTrainer(solver_model, solver_reward_func, args=solver_grpoconfig, passing_rate_lower_threshold=self.passing_rate_lower_threshold, passing_rate_upper_threshold=self.passing_rate_upper_threshold, vllm_client_port=other_args.solver_vllm_client_port)
        
        
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        if self.rank == 0:
            wandb.init(name=other_args.wandb_run_name)

        self.replay_buffer = QuestionBuffer(removal_threshold=3, passing_rate_upper_threshold=self.passing_rate_upper_threshold, passing_rate_lower_threshold=self.passing_rate_lower_threshold, rank=self.rank, world_size=self.world_size)
        
    def validate_passing_rate(self, passing_rate):
        return ((passing_rate > self.passing_rate_lower_threshold) & (passing_rate < self.passing_rate_upper_threshold)).any().item()

    def validate_advantages(self, advantages):
        return (advantages != 0.0).any().item()
    
    def get_prompt(self, role, tokenizer, question):
        # temporary remove chat template
        if role == "proposer":
            sys_prompt = self.proposer_prompt
            if self.use_knowledge:
                user_prompt = PROPOSER_USER_PROMPT_WITH_KNOWLEDGE.replace("{knowledge}", question)
            else:
                user_prompt = PROPOSER_USER_PROMPT_WITHOUT_KNOWLEDGE
        else:
            sys_prompt = self.solver_prompt
            user_prompt = SOLVER_USER_PROMPT.replace("{question}", question)
        new_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tokenize=False,
            add_generation_prompt=True,
        ) # Qwen3 tokenizer has an argument "enable_thinking", which is turned on by default
        if not new_prompt.endswith("<think>\n") and not new_prompt.endswith("<think>"):
            new_prompt = new_prompt + "<think>\n"

        return new_prompt
    
    
    def _sync_proposer_data(self):
        """在所有rank间同步proposer数据"""
        if not self.is_distributed:
            knowledge_data = None
            if self.use_knowledge:
                knowledge_data = self.knowledge_base.sample(1)[0]
                # knowledge_data = self.knowledge_base.get_knowledge_test(1)[0]
                proposer_data = [{'prompt': self.get_prompt(role="proposer", tokenizer=self.proposer_trainer.processing_class, question=knowledge_data)}]
            else:
                proposer_data = [{'prompt': self.get_prompt(role="proposer", tokenizer=self.proposer_trainer.processing_class, question="")}]
            return proposer_data, knowledge_data
        
        knowledge_data = None
        
        if self.rank == 0:
            if self.use_knowledge:
                knowledge_data = self.knowledge_base.sample(1)[0]
                # knowledge_data = self.knowledge_base.get_knowledge_test(1)[0]
        

        knowledge_data_list = [knowledge_data]
        dist.broadcast_object_list(knowledge_data_list, src=0)
        knowledge_data = knowledge_data_list[0]
        
        if self.use_knowledge:
            assert knowledge_data is not None, f"Rank {self.rank} did not receive knowledge data"
            proposer_data = [{'prompt': self.get_prompt(role="proposer", tokenizer=self.proposer_trainer.processing_class, question=knowledge_data)}]
        else:
            proposer_data = [{'prompt': self.get_prompt(role="proposer", tokenizer=self.proposer_trainer.processing_class, question="")}]
        
        return proposer_data, knowledge_data

    

    def proposer_reward_hacking_check(self, question_answer_pairs, passing_rate):
        passing_rate_list = passing_rate.tolist()
        new_passing_rate = []
        for q, p in zip(question_answer_pairs, passing_rate_list):
            
            if q['question'] is None or q['answer'] is None:
                new_passing_rate.append(0.0)
            elif str(q['question']) in str(q['answer']) or extract_answer_proposer(q['question']) is not None:
                new_passing_rate.append(0.0)
            elif str(q['answer']) in str(q['question']):
                new_passing_rate.append(0.0)
            else:
                new_passing_rate.append(p)

        return torch.tensor(new_passing_rate, dtype=passing_rate.dtype, device=passing_rate.device)
    
    def train(self):
        self.solver_trainer.set_train_mode()
        self.proposer_trainer.set_train_mode()
        pbar = tqdm(range(self.max_steps), desc="Training Iterations", disable=self.rank != 0)

        
        for step in pbar:
            proposer_update_step = 0
            solver_update_step = 0
            
            # self._sync_random_seed(step)
            
            # print(f"proposer_update_step: {proposer_update_step}, solver_update_step: {solver_update_step}")
            
            while proposer_update_step < self.proposer_update_step:
                batch_data = {'proposer_data': {}, 'solver_data': {}}
                proposer_data, knowledge_data = self._sync_proposer_data()
                
                # 生成completions
                proposer_completions, batch_data['proposer_data'] = self.proposer_trainer._generate_completions(proposer_data)
                
                question_answer_pairs = []
                for i, completion in enumerate(proposer_completions):
                    question = extract_question(completion)
                    answer = extract_answer_proposer(completion)
                    question_answer_pairs.append({'question': question, 'answer': answer})
                    # print(f"question: {question}, answer: {answer}")
                    
                solver_data = [{
                    'prompt': self.get_prompt(role="solver", tokenizer=self.solver_trainer.processing_class, question=q['question']),
                    'answer': q['answer']
                } for q in question_answer_pairs]   

                batch_data['solver_data'], log_metrics = self.solver_trainer._generate_and_score_completions(solver_data)
                
                all_question_answer_pairs = gather_object(question_answer_pairs)
                
                passing_rate = batch_data['solver_data']['passing_rate']
                batch_data['solver_data']['passing_rate'] = self.proposer_reward_hacking_check(all_question_answer_pairs, batch_data['solver_data']['passing_rate'])
                new_passing_rate = batch_data['solver_data']['passing_rate']
                
                
                batch_data['proposer_data']['question'] = [q['question'] for q in all_question_answer_pairs]
                batch_data['proposer_data']['advantages'], all_advantages = self.proposer_trainer._score_completions(batch_data['proposer_data'], new_passing_rate, knowledge_data=knowledge_data)
                
                skip = "no"

                if not self.validate_passing_rate(new_passing_rate):
                    skip = "pr"
                
                if not self.validate_advantages(all_advantages):
                    skip = "adv"
                
                if skip in ["adv", "pr", "div"]:
                    del batch_data
                    self.proposer_trainer.clear_prompt_completion_textual_logs()
                    self.solver_trainer.clear_textual_logs()
                    if self.rank==0:
                        if skip == "adv":
                            print("The current batch is skipped for proposer and solver since all_advantages is {}".format(all_advantages))
                        elif skip == "pr":
                            print("The current batch is skipped for solver since passing_rate is {}".format(new_passing_rate))
                        for q, p, np in zip(all_question_answer_pairs, passing_rate.tolist(), new_passing_rate.tolist()):
                            print("Question: {}\nAnswer: {}\nPassing rate: {}\nNew passing rate: {}".format(q["question"], q["answer"], p, np))
                        print("\n\n")
                    continue
                
                if self.rank == 0:
                    self.replay_buffer.add_question_list(all_question_answer_pairs, passing_rate.tolist())
                
                self.proposer_trainer.print_completions = (proposer_update_step % 10 == 0)
                self.proposer_trainer.train_batch(batch_data['proposer_data'])
                
                proposer_update_step += 1

                del batch_data
                torch.cuda.empty_cache()
            
            if self.rank == 0:
                print("Starting to train solver with {} questions".format(self.replay_buffer.size()))

            while solver_update_step < self.solver_update_step:
                batch_data = {'proposer_data': {}, 'solver_data': {}}

                break_solver = [False]
                if self.rank == 0 and self.replay_buffer.is_empty():
                    print("The buffer is empty. Stop the current solver iteration.")
                    break_solver = [True]

                dist.broadcast_object_list(break_solver, src=0)

                if break_solver[0] == True:
                    break
            
                if self.rank == 0:
                    question_answer_pairs = self.replay_buffer.get_multiple_questions(self.proposer_trainer.num_generations)
                else:
                    question_answer_pairs = [None] * self.proposer_trainer.num_generations
                
                dist.broadcast_object_list(question_answer_pairs, src=0)
                
                tp_qa_size = self.proposer_trainer.num_generations // self.solver_trainer.vllm_tensor_parallel_size    
                tp_qa_slice = slice(self.rank * tp_qa_size, (self.rank+1) * tp_qa_size)
                question_answer_pairs = question_answer_pairs[tp_qa_slice]
                
                solver_data = [{
                    'prompt': self.get_prompt(role="solver", tokenizer=self.solver_trainer.processing_class, question=q[0]),
                    'answer': q[1]
                } for q in question_answer_pairs]
            

                batch_data['solver_data'], log_metrics = self.solver_trainer._generate_and_score_completions(solver_data)

                # Experimental Eviction Logic
                # if self.rank == 0:
                #     self.replay_buffer.update_after_training_batch(batch_data['solver_data']["passing_rate"].tolist())

                passing_rate = batch_data(['solver_data']['passing_rate'])
                if not self.validate_passing_rate(passing_rate):
                    skip = "pr"

                if skip in ["adv", "pr", "div"]:
                    del batch_data
                    self.proposer_trainer.clear_prompt_completion_textual_logs()
                    self.solver_trainer.clear_textual_logs()
                    if self.rank==0:
                        if skip == "adv":
                            print("The current batch is skipped for proposer and solver since all_advantages is {}".format(all_advantages))
                        elif skip == "pr":
                            print("The current batch is skipped for solver since passing_rate is {}".format(new_passing_rate))
                        for q, p, np in zip(all_question_answer_pairs, passing_rate.tolist(), new_passing_rate.tolist()):
                            print("Question: {}\nAnswer: {}\nPassing rate: {}\nNew passing rate: {}".format(q["question"], q["answer"], p, np))
                        print("\n\n")
                    continue

                self.solver_trainer.print_completions = (solver_update_step % 10 == 0)
                self.solver_trainer.train_batch(batch_data['solver_data'],log_metrics, filter_batch_data=True)
                solver_update_step += 1
                
                del batch_data
                torch.cuda.empty_cache()
            