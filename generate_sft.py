import os

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


from tqdm import tqdm

import re

import random
import json

from transformers import AutoTokenizer

from argparse import ArgumentParser

from reward_utils import accuracy, extract_answer_solver, extract_answer_proposer, extract_question

from time import strftime, localtime

from openai import OpenAI

from knowledge_base import Knowledgebase
import copy

from prompt import *

client = OpenAI()

WITHOUT_KNOWLEDGE_MSGS = [
    {"role": "system", "content": PROPOSER_PROMPT_WITHOUT_KNOWLEDGE}, 
    {"role": "user", "content": PROPOSER_USER_PROMPT_WITHOUT_KNOWLEDGE}
]

WITH_KNOWLEDGE_MSGS = [
    {"role": "system", "content": PROPOSER_PROMPT_WITH_KNOWLEDGE}, 
    {"role": "user", "content": PROPOSER_USER_PROMPT_WITH_KNOWLEDGE}
]

SOLVER_MSGS = [
    {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
    {"role": "user", "content": SOLVER_USER_PROMPT}
]

def add_think_token(output_text):
    if not output_text.startswith("<think>\n") and not output_text.startswith("<think>"):
        output_text = "<think>\n" + output_text
    return output_text

def get_solver_msgs(question):
    prompt_messages = copy.deepcopy(SOLVER_MSGS)
    prompt_messages[1]["content"] = question
    return prompt_messages

def get_proposer_w_know_msgs(knowledge):
    prompt_messages = copy.deepcopy(WITH_KNOWLEDGE_MSGS)
    prompt_messages[1]["content"] = prompt_messages[1]["content"].replace("{knowledge}", knowledge)
    return prompt_messages

def get_solver_response_from_gpt(question):
    prompt_messages = get_solver_msgs(question)
    completion = client.chat.completions.create(
        model="gpt-5-mini",
        messages=prompt_messages
    )
    
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens

    print("Prompt tokens: {}; Response tokens: {}; Cost: {}".format(prompt_tokens, completion_tokens, (prompt_tokens/1000000)*0.25 + (completion_tokens/1000000)*2.00))

    return completion.choices[0].message.content

def apply_prompt_chat_template(example, role, tokenizer, is_qwen3, use_knowledge, knowledge=None, question=None):
    if role == "proposer":
        if use_knowledge:
            prompt_messages = get_proposer_w_know_msgs(knowledge)
        else:
            prompt_messages = WITHOUT_KNOWLEDGE_MSGS
    elif role == "solver":
        prompt_messages = get_solver_msgs(question)
    
    new_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    if is_qwen3:
        if not new_prompt.endswith("<think>\n") and not new_prompt.endswith("<think>"):
            new_prompt = new_prompt + "<think>\n"

    example["text_prompt"] = new_prompt

    return example





if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--peft_dir", type=str)
    parser.add_argument("--other_suffix", type=str, default="")
    parser.add_argument("--use_knowledge", type=int, default=1, help="Use knowledge or not")
    parser.add_argument("--generate_solver_sft", action="store_true")
    parser.add_argument("--train_set_size", type=int, default=2800)

    args = parser.parse_args()

    use_peft = args.peft_dir is not None and len(args.peft_dir) > 0
    if use_peft:
        peft_dir = args.peft_dir

        generator = peft_dir.split("/")[-1]
    else:
        generator = args.model_name_or_path.split("/")[-1]


    if args.use_knowledge:
        knowledge_base = Knowledgebase("YouAreSpecialToMe/filtered_MegaMath")

    max_model_len = 6144
    max_new_tokens = 6144
    temperature = 0.8
    num_sampling_sequences = 6
    use_prompt_with_partial_resp = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path, 
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
    )

    # sampling_seed = args.seed

    sampling_params = SamplingParams(
        n=num_sampling_sequences,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
        # seed=sampling_seed,
    )

    model_n_p_casefold = args.model_name_or_path.casefold()

    is_qwen3 = "qwen3" in model_n_p_casefold

    batch_size = 4
    save_interval = 5
    temp_suffix = "" if temperature == 0.8 else "_temp_{:.0e}".format(temperature)
    num_seq_suffix = "" if num_sampling_sequences == 5 else "_num_seq_{}".format(num_sampling_sequences)


    
    res = []
    res_path = os.path.join(args.output_dir, "src_{}{}{}{}.json".format(generator, num_seq_suffix, temp_suffix, args.other_suffix))
    
    solver_res = []
    solver_res_path = res_path.replace("src_", "solver_src_")

    print("Result path is {}".format(res_path))

    if os.path.exists(res_path):
        with open(res_path, "r") as src_json:
            res = json.load(src_json)
    
    if os.path.exists(solver_res_path):
        with open(solver_res_path, "r") as solver_res_path:
            solver_res = json.load(solver_res_path)

    valid_count = sum([s["is_proposer_answer_correct"] for s in res])
    solver_valid_count = sum([ss["is_solver_answer_correct"] for ss in solver_res])
    step_count = 0

    print("=========Start sampling responses from Proposer=========")

    while valid_count < args.train_set_size:
        
        prompts = []
        knowledges = []

        for _ in range(batch_size):
            knowledge = None
            if args.use_knowledge:
                knowledge = knowledge_base.sample(1)[0]
            knowledges.append(knowledge)
            prompts.append(apply_prompt_chat_template({}, "proposer", tokenizer, is_qwen3=is_qwen3, use_knowledge=args.use_knowledge, knowledge=knowledge)["text_prompt"])
        

        

        print("Generation starts")

        # import pdb; pdb.set_trace()

        if use_peft:
            outputs = llm.generate(prompts, sampling_params, lora_request=LoRARequest("adapter", 1, peft_dir), use_tqdm=False)
        else:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
       
        print("Generation completed")

        for p, o, k in zip(prompts, outputs, knowledges):
            output_texts = [output.text for output in o.outputs]
            output_logprobs = [output.cumulative_logprob for output in o.outputs]

            assert len(output_texts) == num_sampling_sequences, "{}; {}".format(len(output_texts), num_sampling_sequences)

            for output_text in output_texts:
                question = extract_question(output_text)
                gold = str(extract_answer_proposer(output_text))

                if question == "None" or question is None:
                    question = ""

                if gold == "None" or gold is None:
                    gold = ""

                if is_qwen3:
                    new_output_text = add_think_token(output_text)

                msgs = WITHOUT_KNOWLEDGE_MSGS
                if args.use_knowledge:
                    msgs = get_proposer_w_know_msgs(k)

                new_example = {
                    "problem": question,
                    "response": new_output_text,
                    "solution": gold,
                    "solver_gpt_response": "",
                    "solver_gpt_pred": "",
                    "level": "",
                    "is_proposer_answer_correct": 0,
                    "is_valid": 0,
                    "knowledge": k if k is not None else "",
                    "conversations": msgs + [{"role": "assistant", "content": new_output_text}]
                }

                if not question:
                    print("No valid question from proposer. Skipping", flush=True)
                    # res.append(new_example)
                    continue

                if not gold:
                    print("No valid answer from proposer. Skipping", flush=True)
                    # res.append(new_example)
                    continue

                try:
                    resp = get_solver_response_from_gpt(question)
                except Exception as e:
                    print("Encounter exception when calling GPT:\n\n{}\n\nSkipping".format(e), flush=True)
                    # res.append(new_example)
                    continue
                hits = accuracy([resp], [gold])

                pred = str(extract_answer_solver(resp))

                if pred == "None" or pred is None:
                    pred = ""

                assert len(hits) == 1

                new_example["solver_gpt_response"] = resp
                new_example["solver_gpt_pred"] = pred
                new_example["is_valid"] = 1
                valid_count = valid_count + 1

                if not hits[0]:
                    print("GPT answer {} does not match that of proposer {}. Skipping".format(extract_answer_solver(resp), gold), flush=True)
                    # res.append(new_example)
                    continue
                
                new_example["is_proposer_answer_correct"] = 1

                res.append(new_example)
        

        step_count = step_count + 1
        
        if step_count % save_interval == 0 and step_count > 0:
            with open(res_path, "w") as res_json:
                json.dump(res, res_json, indent=4)
    


    corr_count = sum([s["is_proposer_answer_correct"] for s in res])
    valid_count = sum([s["is_valid"] for s in res])

    print("Accuracy of the proposer answer: {}\n{} percent of the proposer responses are valid.".format(corr_count/valid_count, valid_count/len(res)))

    with open(res_path, "w") as res_json:
        json.dump(res, res_json, indent=4)


    if not args.generate_solver_sft:
        exit()

    step_count = 0

    print("=========Start sampling responses from Solver=========")

    for proposer_example in res:
        question = proposer_example["problem"]
        solver_prompts = [apply_prompt_chat_template({}, "solver", tokenizer, is_qwen3=is_qwen3, use_knowledge=False, knowledge=None, question=question)["text_prompt"]]

        

        if use_peft:
            solver_outputs = llm.generate(solver_prompts, sampling_params, lora_request=LoRARequest("adapter", 1, peft_dir), use_tqdm=False)
        else:
            solver_outputs = llm.generate(solver_prompts, sampling_params, use_tqdm=True)
        
        assert len(solver_outputs) == 1
        solver_output_texts = [solver_output.text for solver_output in solver_outputs[0].outputs]

        for solver_output_text in solver_output_texts:
            solver_gold = str(extract_answer_solver(solver_output_text))

            if solver_gold == "None" or solver_gold is None:
                solver_gold = ""
            
            if is_qwen3:
                new_solver_output_text = add_think_token(solver_output_text)

            solver_msgs = get_solver_msgs(question)

            new_solver_example = {
                "problem": question,
                "response": new_solver_output_text,
                "solution": proposer_example["solution"],
                "solver_gpt_response": proposer_example["solver_gpt_response"],
                "solver_gpt_pred": proposer_example["solver_gpt_pred"],
                "level": "",
                "is_solver_answer_correct": 0,
                "is_valid": 0,
                "knowledge": proposer_example["knowledge"],
                "conversations": solver_msgs + [{"role": "assistant", "content": new_solver_output_text}]
            }

            if not solver_gold:
                print("No valid answer from solver. Skipping.", flush=True)
                # solver_res.append(new_solver_example
                continue

            new_solver_example["is_valid"] = 1
            solver_valid_count = solver_valid_count + 1

            solver_hits = accuracy([proposer_example["solver_gpt_response"]], [solver_gold])

            if not solver_hits[0]:
                print("GPT answer {} does not match that of solver {}. Skipping".format(extract_answer_solver(new_solver_example["solver_gpt_response"]), solver_gold), flush=True)
                # solver_res.append(new_solver_example)
                continue

            new_solver_example["is_solver_answer_correct"] = 1

            solver_res.append(new_solver_example)

        step_count = step_count + 1
        
        if step_count % (batch_size * save_interval) == 0 and step_count > 0:
            with open(solver_res_path, "w") as solver_res_json:
                json.dump(solver_res, solver_res_json, indent=4)


solver_corr_count = sum([s["is_solver_answer_correct"] for s in solver_res])
solver_valid_count = sum([s["is_valid"] for s in solver_res])

print("Accuracy of the solver answer: {}\n{} percent of the solver responses are valid.".format(solver_corr_count/solver_valid_count, solver_valid_count/len(solver_res)))

with open(solver_res_path, "w") as solver_res_json:
    json.dump(solver_res, solver_res_json, indent=4)