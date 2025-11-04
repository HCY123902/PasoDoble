from collections import deque

class DiversityRewardManager:
    def __init__(self, maxlen=100, similarity_threshold=0.8, length_tolerance=5, length_tolerance_mode='absolute'):
        """
        Args:
            maxlen: Maximum number of historical questions in the buffer
            similarity_threshold: Jaccard Similarity Threshold
            length_tolerance: degree of tolerance to treat completions to be similar according to their lengths, not currently used
            length_tolerance_mode: 'relative' or 'absolute'
        """
        self.history_deque = deque(maxlen=maxlen)
        self.history_sets = deque(maxlen=maxlen)
        self.history_lengths = deque(maxlen=maxlen)
        
        self.buffer = []
        
        self.similarity_threshold = similarity_threshold
        self.length_tolerance = length_tolerance
        self._length_tolerance_mode = length_tolerance_mode
        self.maxlen = maxlen
    
    def add_to_history(self, new_completion_ids, new_set, new_length):
        """
        Args:
            new_completion_ids: New completion_ids list
            new_set: New completion_ids set
            new_length: New completion_ids length
        """
        if len(self.history_deque) >= self.maxlen:
            self.history_deque.popleft()
            if self.history_sets:
                self.history_sets.popleft()
            if self.history_lengths:
                self.history_lengths.popleft()
        
        self.history_deque.append(new_completion_ids)
        self.history_sets.append(new_set)
        self.history_lengths.append(new_length)
    
    def add(self, new_completion_ids=None):
        """
        Args:
            new_completion_ids: Optional，New completion_ids list
        """
        if new_completion_ids is not None:
            new_set = set(new_completion_ids)
            new_length = len(new_completion_ids)
            self.buffer.append((new_completion_ids, new_set, new_length))
        
        for completion_ids, completion_set, length in self.buffer:
            self.add_to_history(completion_ids, completion_set, length)
        
        self.clear_buffer()
    
    def diversity_reward(self, new_completion_ids, knowledge_data_ids):
        """        
        Args:
            new_completion_ids: New completion_ids list
            knowledge_data_ids: Knowledge_data_ids list
        Returns:
            float: Diversity reward (0-100)
        """
        new_set = set(new_completion_ids)
        new_length = len(new_completion_ids)
        similar_count = 0
        
        for i, historical_set in enumerate(self.history_sets):
            similarity = self._calculate_jaccard_similarity(new_set, historical_set)
            
            if similarity > self.similarity_threshold:
                similar_count += 1
                # continue
            
            # historical_length = self.history_lengths[i]
            # if self._is_length_similar(new_length, historical_length):
            #     similar_count += 1
        
        self.buffer.append((new_completion_ids, new_set, new_length))
        
        total_records = len(self.history_sets)
        if total_records == 0:
            return 1.0 
        
        diversity_reward = (total_records - similar_count) / total_records
        return max(0.0, diversity_reward)
    
    def clear_buffer(self):
        self.buffer.clear()
    
    def get_buffer_count(self):
        return len(self.buffer)
    
    def get_buffer_items(self):
        return [(completion_ids, length) for completion_ids, _, length in self.buffer]
    
    def _calculate_jaccard_similarity(self, set1, set2):
        if not set1 and not set2:
            return 1.0
        
        max_intersection = min(len(set1), len(set2))
        min_union = max(len(set1), len(set2))
        max_possible_similarity = max_intersection / min_union if min_union > 0 else 0.0
        
        if max_possible_similarity <= self.similarity_threshold:
            return 0.0 
        
        intersection_size = len(set1 & set2)
        union_size = len(set1) + len(set2) - intersection_size
        similarity = intersection_size / union_size if union_size > 0 else 0.0
        
        return similarity
    
    def _is_length_similar(self, new_length, historical_length):
        """
        Check if 2 lengths are within the degree of tolerance to be treated as similar
        
        Args:
            new_length: New length
            historical_length: Historic length
            
        Returns:
            bool: Similar or not
        """
        if self._length_tolerance_mode == 'absolute':
            return abs(new_length - historical_length) <= self.length_tolerance
        else:  # relative mode
            if historical_length == 0:
                return new_length == 0
            relative_diff = abs(new_length - historical_length) / historical_length
            return relative_diff <= self.length_tolerance
    
    def set_length_tolerance_absolute(self, absolute_tolerance):
        """ 
        Args:
            absolute_tolerance: Degree of absolute tolerance (number of tokens)
        """
        self.length_tolerance = absolute_tolerance
        self._length_tolerance_mode = 'absolute'
    
    def set_length_tolerance_relative(self, relative_tolerance):
        """
        Args:
            relative_tolerance: Degree of absolute tolerance (percentage)
        """
        self.length_tolerance = relative_tolerance
        self._length_tolerance_mode = 'relative'
    
    def get_history_count(self):
        return len(self.history_deque)
    
    def get_detailed_similarities(self, new_completion_ids):
        """
        Get detailed similarity statistics in between new and historic completion_ids
        
        Returns:
            list: Contains detailed statistics in between new and historic completion_ids
        """
        new_set = set(new_completion_ids)
        new_length = len(new_completion_ids)
        
        # 只与历史记录比较
        similarities = []
        for i, historical_set in enumerate(self.history_sets):
            historical_length = self.history_lengths[i]
            
            # 计算Jaccard相似度
            similarity = self._calculate_jaccard_similarity(new_set, historical_set)
            
            # 检查长度相似性
            length_similar = self._is_length_similar(new_length, historical_length)
            
            # 是否同时满足两个条件
            both_similar = similarity > self.similarity_threshold and length_similar
            
            similarities.append({
                'index': i,
                'historical_ids': list(self.history_deque)[i],
                'historical_length': historical_length,
                'jaccard_similarity': similarity,
                'length_similar': length_similar,
                'both_conditions_met': both_similar,
                'length_diff': abs(new_length - historical_length)
            })
        
        return similarities
    
    def get_statistics(self):
        history_lengths = list(self.history_lengths) if self.history_lengths else []
        buffer_lengths = [length for _, _, length in self.buffer] if self.buffer else []
        all_lengths = history_lengths + buffer_lengths
        
        return {
            'total_records': len(self.history_deque),
            'buffer_records': len(self.buffer),
            'total_including_buffer': len(self.history_deque) + len(self.buffer),
            'length_stats': {
                'history': {
                    'min': min(history_lengths) if history_lengths else None,
                    'max': max(history_lengths) if history_lengths else None,
                    'avg': sum(history_lengths) / len(history_lengths) if history_lengths else None,
                    'values': history_lengths
                },
                'buffer': {
                    'min': min(buffer_lengths) if buffer_lengths else None,
                    'max': max(buffer_lengths) if buffer_lengths else None,
                    'avg': sum(buffer_lengths) / len(buffer_lengths) if buffer_lengths else None,
                    'values': buffer_lengths
                },
                'all': {
                    'min': min(all_lengths) if all_lengths else None,
                    'max': max(all_lengths) if all_lengths else None,
                    'avg': sum(all_lengths) / len(all_lengths) if all_lengths else None,
                    'values': all_lengths
                }
            },
            'similarity_threshold': self.similarity_threshold,
            'length_tolerance': self.length_tolerance,
            'tolerance_mode': self._length_tolerance_mode
        }

