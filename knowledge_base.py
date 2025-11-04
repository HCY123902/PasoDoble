from datasets import load_dataset
import random

class Knowledgebase:
    def __init__(self, knowledge_base_path=None, config_name=None, split="train"):
        random.seed(42)
        self.knowledge_base_path = knowledge_base_path
        
        if knowledge_base_path is not None:
            self.knowledge_base = load_dataset(knowledge_base_path, split="train")
        else:
            self.knowledge_base = None
            
        self.knowledge_test = [
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. The answer is 72.",
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? Weng earns $12/60 = $0.20 per minute. Weng earned $0.20*50 = $10 yesterday. The answer is $10.",
            "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm? Randy has 60/2 = 30 coconut trees. Randy has 30-5 = 25 coconut trees. Randy has 60+25 = 85 trees in all on his farm. The answer is 85."
        ]

    def get_knowledge_test(self, num_samples=1):
        available_samples = len(self.knowledge_test)
        if num_samples > available_samples:
            print(f"Warning: Requested {num_samples} samples, but only {available_samples} available. Returning all available samples.")
            return self.knowledge_test
        return random.sample(self.knowledge_test, num_samples)

    def sample(self, num_samples=1):
        dataset_size = len(self.knowledge_base)
        if num_samples > dataset_size:
            print(f"Warning: Requested {num_samples} samples, but dataset only has {dataset_size} items. Returning all texts.")
            return self.knowledge_base["text"]

        indices = random.sample(range(dataset_size), num_samples)
        subset = self.knowledge_base.select(indices)
        return subset["text"]
    
    def get_knowledge_base(self):
        """返回完整的知识库"""
        return self.knowledge_base
    
    def get_knowledge_base_size(self):
        """返回知识库的大小"""
        return len(self.knowledge_base)
    
    def get_sample_by_index(self, index):
        """根据索引获取特定样例"""
        if 0 <= index < len(self.knowledge_base):
            return self.knowledge_base[index]
        else:
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.knowledge_base)}")
