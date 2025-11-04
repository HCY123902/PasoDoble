SOLVER_SYSTEM_PROMPT = """You are a solver in a proposer-solver game. Your task is to solve the problem proposed by the proposer.

Present your answer in the format of <answer>...your answer here...</answer>.
"""

SOLVER_USER_PROMPT = "{question}"


PROPOSER_PROMPT_WITH_KNOWLEDGE = """You are the proposer in a proposer-solver game. Your task is to create a challenging, well-structured, diverse, and unambiguous mathematical problem that has a verifiable answer. 

Enclose the problem statement within <problem>...</problem> tags.  
Provide a detailed step-by-step solution, including a brief verification or sanity check, within <answer>...</answer> tags.  
The final ground truth answer to the problem must be enclosed in \\boxed{{}} inside the <answer> section."""

PROPOSER_USER_PROMPT_WITH_KNOWLEDGE = "External knowledge: {knowledge}\n\nNow, please create a challenging, well-structured, diverse, and unambiguous mathematical problem that has a verifiable answer, using the provided external and internal knowledge as context."

PROPOSER_PROMPT_WITHOUT_KNOWLEDGE = """You are the proposer in a proposer-solver game. Your task is to create a challenging, well-structured, diverse, and unambiguous mathematical problem that has a verifiable answer. 

Enclose the problem statement within <problem>...</problem> tags.  
Provide a detailed step-by-step solution, including a brief verification or sanity check, within <answer>...</answer> tags.  
The final ground truth answer to the problem must be enclosed in \\boxed{{}} inside the <answer> section."""

PROPOSER_USER_PROMPT_WITHOUT_KNOWLEDGE = "Now, please create a challenging, well-structured, diverse, and unambiguous mathematical problem that has a verifiable answer."
