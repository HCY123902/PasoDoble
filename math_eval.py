# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
import argparse
import pandas as pd
import os
import re

from functools import lru_cache

_FRAC_RE   = re.compile(r'\\frac\{([^{}]+)\}\{([^{}]+)\}')
_TEXT_RE   = re.compile(r'\\text\{([^{}]*)\}')
_CHOICE_RE = re.compile(r'^\(([A-Z])\)$')
_SPACES_RE = re.compile(r'\s+')

def _unwrap_text(s: str) -> str:
    return _TEXT_RE.sub(r'\1', s)

def _frac_to_div(s: str) -> str:
    # \dfrac -> \frac
    if '\\dfrac' in s:
        s = s.replace('\\dfrac', '\\frac')
    # \frac{a}{b} -> (a)/(b)
    return _FRAC_RE.sub(r'(\1)/(\2)', s)

def _strip_noise(s: str) -> str:
    if ',\\!' in s:
        s = s.replace(',\\!', '')
    if '\\!' in s:
        s = s.replace('\\!', '')
    s = _SPACES_RE.sub(' ', s).strip()
    return s

def _post_math500_variants(gt: str):
    """返回 ground truth 的一组等价变体（按你的 MATH-500 hack）"""
    variants = {gt}
    # \text{...} -> ...
    if gt.startswith('\\text{') and gt.endswith('}'):
        variants.add(_unwrap_text(gt))
    # (A) -> A
    m = _CHOICE_RE.match(gt)
    if m:
        variants.add(m.group(1))
    if '\\in' in gt:
        right = gt.split('\\in', 1)[-1].strip()
        variants.add(right)
    return variants

def _split_set_form(s: str):
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        parts = [p for p in parts if p]
        return frozenset(parts)
    return None

@lru_cache(maxsize=8192)
def _last_boxed_clean(s: str):
    x = last_boxed_only_string(s)
    return None if x is None else remove_boxed(x)

@lru_cache(maxsize=8192)
def _first_boxed_clean(s: str):
    x = first_boxed_only_string(s)
    return None if x is None else remove_boxed(x)

def _normalize_once(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = _unwrap_text(s)
    s = _frac_to_div(s)
    s = _strip_noise(s)
    return s

def _quick_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    # 逗号集合等价
    sa, sb = _split_set_form(a), _split_set_form(b)
    return (sa is not None and sb is not None and sa == sb)

def _try_is_equiv(a: str, b: str) -> bool:
    try:
        return is_equiv(a, b)
    except Exception:
        return False

def _try_math_verify(a: str, b: str) -> bool:
    try:
        from math_verify import verify, parse
    except Exception:
        return False

    try:
        pa = parse(a)
        pb = parse(b)
        return verify(pb, pa, float_rounding=6, numeric_precision=15)
    except Exception:
        return False

def _candidate_solutions(solution_str: str, response_str: str | None):

    cands = []
    seen = set()

    for getter in (_last_boxed_clean, _first_boxed_clean):
        v = getter(solution_str) if solution_str is not None else None
        if v and v not in seen:
            seen.add(v); cands.append(v)

    if solution_str and solution_str not in seen:
        seen.add(solution_str); cands.append(solution_str)

    if response_str and not math_if_boxed(solution_str):
        for getter in (_last_boxed_clean, _first_boxed_clean):
            v = getter(response_str)
            if v and v not in seen:
                seen.add(v); cands.append(v)
        if response_str not in seen:
            seen.add(response_str); cands.append(response_str)

    return cands

def math_verify_score(solution_str: str, ground_truth: str, response_str: str = None) -> float:
    gt_variants_raw = _post_math500_variants(ground_truth)
    gt_norm = { _normalize_once(g) for g in gt_variants_raw }

    cand_raw = _candidate_solutions(solution_str, response_str)
    cand_norm = [ _normalize_once(c) for c in cand_raw ]

    for a in cand_norm:
        for b in gt_norm:
            if _quick_equal(a, b):
                return 1.0

    for a in cand_norm:
        for b in gt_norm:
            if _try_is_equiv(a, b):
                return 1.0

    for a in cand_norm:
        for b in gt_norm:
            if _try_math_verify(a, b):
                return 1.0

    return 0.0

def math_if_boxed(s) -> bool:
    try:
        return _last_boxed_clean(s) is not None
    except Exception:
        return False

def compute_score(solution_str, ground_truth) -> float:
    try:
        last_answer  = _last_boxed_clean(solution_str)
        first_answer = _first_boxed_clean(solution_str)
        gt_last      = _last_boxed_clean(ground_truth)

        if gt_last is not None:
            gt_last = _normalize_once(gt_last)

        for ans in (last_answer, first_answer):
            if not ans:
                continue
            ans = _normalize_once(ans)
            if gt_last and _quick_equal(ans, gt_last):
                return 1.0
            if gt_last and _try_is_equiv(ans, gt_last):
                return 1.0
    except Exception:
        pass

    return 0.0


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def first_boxed_only_string(string):
    idx = string.find("\\boxed")   # CHANGED from rfind → find
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[1].split("$")[0]
    if idx < 0:
        idx = string.find("\\fbox")  # CHANGED from rfind → find
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

if __name__ == "__main__":
    """
    conda activate zero
    python reward_score/math500.py --file_path ./results/math500amc23/benchmark
    python reward_score/math500.py --input_dir ./results/olympiadbench/benchmark
    python reward_score/math500.py --input_dir ./results/allmath/benchmark
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=False)
    parser.add_argument("--input_dir", type=str, required=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.input_dir if args.input_dir else args.file_path

    pred_col = "pred"
    gt_col = "ground_truth"
    if "benchmark" in source:
        response_col = "response"
    elif "distract" in source:
        response_col = "post_distraction_response"
    elif "teacher" in source:
        response_col = "student_response"
    
    if_strict_answer = False  # Add this variable definition
    if_boxed = True
    
    if args.file_path:
        df = pd.read_pickle(args.file_path)
        if "gt" in df.columns:
            df = df.rename(columns={"gt": gt_col})
            df.to_pickle(args.file_path)
            
        df["model_is_correct"] = df.apply(lambda x: math_verify_score(x[pred_col], x[gt_col], x[response_col]), axis=1)
        df.to_pickle(args.file_path)

    else:
        for fname in os.listdir(args.input_dir):
            if fname.endswith(".pickle"):
                print("Evaluating {}".format(fname))
                df = pd.read_pickle(os.path.join(args.input_dir, fname))
                if "gt" in df.columns:
                    df = df.rename(columns={"gt": gt_col})
                    df.to_pickle(os.path.join(args.input_dir, fname))
                
                if ("model_is_correct" in df.columns 
                    or "original_correct" in df.columns 
                    or "distractor_correct" in df.columns) and not args.overwrite:
                    print("Skipping {} because it already has model_is_correct column".format(fname))
                    continue
                
                if "distract" in source:
                    df["original_correct"] = df.apply(lambda x: math_verify_score(x["pred"], x["solution"], x["post_distraction_response"]), axis=1)
                    df["distractor_correct"] = df.apply(lambda x: math_verify_score(x["pred"], x["distractor_solution"], x["post_distraction_response"]), axis=1)
                else:
                    df["model_is_correct"] = df.apply(lambda x: math_verify_score(x[pred_col], x[gt_col], x[response_col]), axis=1)
                
                is_correct_col = "model_is_correct" if "model_is_correct" in df.columns else "original_correct"
                if if_strict_answer:
                    df.loc[df[response_col] == df[pred_col], is_correct_col] = 0.0

                if if_boxed:
                    df.loc[~df["if_boxed"], is_correct_col] = 0.0

                df.to_pickle(os.path.join(args.input_dir, fname))