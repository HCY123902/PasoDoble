import random
import numpy as np
import torch
import torch.nn.functional as F

def entropy_from_logits(logits, chunk_size: int = 1) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits* without
    materialising the full soft-max in memory.
    The batch dimension is processed in chunks of size `chunk_size` so that
    only a subset of rows is expanded to probabilities at any one time.
    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all
            leading dimensions are preserved.
        chunk_size (`int`, *optional*, defaults to `1`):
            Number of rows to process per iteration.
    Returns:
        `torch.Tensor`:
            Entropy values with shape `logits.shape[:-1]`.
    """
    per_token_entropies = []
    for logits_chunk in logits.split(chunk_size, dim=0):
        logps = F.log_softmax(logits_chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        per_token_entropies.extend(chunk_entropy)

    per_token_entropies = torch.stack(per_token_entropies)
    return per_token_entropies

def print_completions_from_role(role, step, prompts, completions, rewards, advantages, num_to_print=6):
    count = 0

    for i in range(len(prompts)):
        print("==========================================\nRole: {}\n\nStep: {}\n\nPrompt: {}\n\nCompletion: {}\n\nReward: {}; Advantages: {}\n==========================================\n".format(
            role,
            step,
            prompts[i],
            completions[i],
            "; ".join(["{}: {}".format(key, rewards[key][i]) for key in rewards]),
            advantages[i]
        ))
        count = count + 1
        if count >= num_to_print:
            break

def set_seed_1(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_knowledge_ids(tokenizer, knowledge_data):
    if knowledge_data is None:
        return None
    return tokenizer.encode(knowledge_data, add_special_tokens=False)