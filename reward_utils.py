import re
import regex

try:
    from inference.math_eval import math_verify_score
except Exception as e:
    from math_eval import math_verify_score

import random

def extract_answer_solver(completion):
    """Extract solution from the model output using <answer> tags, \\boxed{} content, or final answers"""
    pattern = regex.compile(r"""
\\boxed \s* \{ (?P<content> (?&braced) ) \}
(?(DEFINE)
  (?<braced>
      (?:
          \\[{}]            
        | \{ (?&braced) \}  
        | [^{}]             
      )*
  )
)
""", regex.VERBOSE | regex.DOTALL)

    boxed_matches = [m.group("content") for m in pattern.finditer(completion)]

    if boxed_matches:
        # Return the last \box{} content from the answer tag
        return boxed_matches[-1].strip()
    else:
        return None
    
def extract_answer_proposer(completion):
    """Extract solution from the model output using <answer> tags, \\boxed{} content, or final answers"""

    pattern = regex.compile(r"""
\\boxed \s* \{ (?P<content> (?&braced) ) \}
(?(DEFINE)
  (?<braced>
      (?:
          \\[{}]            
        | \{ (?&braced) \}  
        | [^{}]             
      )*
  )
)
""", regex.VERBOSE | regex.DOTALL)

    boxed_matches = [m.group("content") for m in pattern.finditer(completion)]

    if boxed_matches:
        # Return the last \box{} content from the answer tag
        return boxed_matches[-1].strip()
    else:        
        return None



def extract_number(text):
    """Extract the last numerical value from text"""
    if text is None:
        return None

    # Clean the text
    text = text.replace(',', '').replace('$', '')

    # Find all numbers in the text
    matches = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    return float(matches[-1]) if matches else None



def extract_question(completion: str) -> str | None:
    """
    查找最后一次出现的 </think>，并在其之后搜索第一个 <problem>...</problem> 匹配。
    返回匹配内容（不包括标签），如果未找到标签则返回 None。
    """
    marker = "</think>"
    pos = completion.rfind(marker)
    if pos == -1:
        return ""

    rest = completion[pos + len(marker):]
    match = re.search(r"<problem>(.*?)</problem>", rest, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""


def accuracy(completions, ground_truth, **kwargs) -> list[float]:
    try:
        hits = []
        for comp, gold in zip(completions, ground_truth):
            pred = extract_answer_solver(comp)
            if pred is None or gold is None:
                hits.append(0.0)
                continue

            hit = math_verify_score(pred, gold)
            if hit:
                hits.append(1.0)
            else:
                hits.append(0.0)
        return hits
    except Exception as e:
        print(f"Math parsing failed: {e}")
        return [0.0] * len(completions)

def rand_accuracy(completions, ground_truth, **kwargs) -> list[float]:
    hits = []
    for comp, gold in zip(completions, ground_truth):
        pred = extract_answer_solver(comp)
        if pred is None or gold is None:
            hits.append(0.0)
            continue
        hits.append(random.choice([1.0, 0.0]))
    return hits