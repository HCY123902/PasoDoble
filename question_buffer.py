from typing import List, Tuple, Optional

class QuestionBuffer:
    """Simplified question buffer for managing question queues during training"""
    
    def __init__(self, removal_threshold: int = 3, passing_rate_upper_threshold: float = 1.0, passing_rate_lower_threshold: float = 0.0, rank: int = 0, world_size: int = 2):
        """
        Args:
            removal_threshold: Number of times without update before removing question
        """
        self.buffer: List[Tuple[str, str, float, int]] = []  # (question, answer, passing_rate, times_without_update)
        self.rank = rank
        self.world_size = world_size
        self.current_index = 0  # Current training position
        self.removal_threshold = removal_threshold
        self.passing_rate_upper_threshold = passing_rate_upper_threshold
        self.passing_rate_lower_threshold = passing_rate_lower_threshold
        # Batch operation related
        self.last_batch_size = 0  # Number of questions in last batch get
        self.last_batch_idxes = []  # Start index of last batch get
    
    def add_question(self, question: str, answer: str, passing_rate: float = 0.0) -> None:
        """Add question to buffer
        
        Args:
            question: Question content
            answer: Answer content
            passing_rate: Initial passing rate
        """
        if passing_rate < self.passing_rate_upper_threshold and passing_rate > self.passing_rate_lower_threshold:
            self.buffer.append((question, answer, passing_rate, 0))
    
    def add_question_list(self, qa_pairs: List[dict], passing_rates: List[float]) -> None:
        """Add multiple questions from list
        
        Args:
            qa_pairs: List of dictionaries with 'question' and 'answer' keys
            passing_rates: List of initial passing rates
        """
        for qa_pair, passing_rate in zip(qa_pairs, passing_rates):
            self.add_question(qa_pair['question'], qa_pair['answer'], passing_rate)
    
    def get_next_question(self) -> Optional[Tuple[str, str]]:
        """Get next training question in order
        
        Returns:
            (question, answer) tuple, or None if buffer is empty
        """
        if not self.buffer:
            return None
        
        # Circular fetch
        if self.current_index >= len(self.buffer):
            self.current_index = 0
        
        question = self.buffer[self.current_index][0]
        answer = self.buffer[self.current_index][1]
        self.current_index += 1
        return (question, answer)
    
    def get_safe_index(self, idx):
        new_idx = idx
        if new_idx >= len(self.buffer):
            new_idx = 0
        return new_idx

    def get_multiple_questions(self, count: int = 1) -> List[Tuple[str, str]]:
        """Batch get multiple training questions
        
        Args:
            count: Number of questions to get
            
        Returns:
            List of (question, answer) tuples, empty list if buffer is empty
        """
        if not self.buffer or count <= 0:
            return []
        
        qa_pairs = []
        
        # self.last_batch_size = min(count, len(self.buffer))
        self.last_batch_size = count
        
        for i in range(self.last_batch_size):
            # Circular fetch
            self.current_index = self.get_safe_index(self.current_index)
            self.last_batch_idxes.append(self.current_index)
            
            question = self.buffer[self.current_index][0]
            answer = self.buffer[self.current_index][1]
            qa_pairs.append((question, answer))
            self.current_index += 1

        self.current_index = self.get_safe_index(self.current_index)
        
        return qa_pairs
    
    def update_after_training(self, new_passing_rate: float) -> bool:
        """Update current question state after training
        
        Args:
            new_passing_rate: New passing rate
            
        Returns:
            Whether update was successful
        """
        if self.is_empty():
            return False
        
        # Rollback to previous question position for update
        update_index = self.last_batch_idxes[-1]

        if new_passing_rate == 1.0:
            print("New passing rate is {}. Removing the question {}.".format(new_passing_rate, self.buffer[update_index][0][:30]))
            self._remove_question_at_index(update_index)
            return True
        
        # Get current question
        question, answer, rate, count = self.buffer[update_index]
        
        # Update passing rate
        if new_passing_rate - rate > 0.01:
            new_rate = new_passing_rate
            new_count = 0
        elif new_passing_rate == 0.0:
            new_count = self.removal_threshold - 1
        else:
            new_rate = rate
            new_count = count + 1
        
        self.buffer[update_index] = (question, answer, new_rate, new_count)
        
        # Check if removal is needed
        if new_count >= self.removal_threshold:
            self._remove_question_at_index(update_index)
        
        return True
    
    def update_after_training_batch(self, passing_rates: List[float]) -> bool:
        """Batch update question states after training
        
        Args:
            passing_rates: List of new passing rates, order corresponds to last get_multiple_questions
            
        Returns:
            Whether update was successful
        """
        if not passing_rates or self.last_batch_size == 0:
            return False
        
        if len(passing_rates) != self.last_batch_size:
            print(f"Warning: Number of passing rates ({len(passing_rates)}) doesn't match last batch size ({self.last_batch_size})")
            return False
        
        indices_to_remove = []
        
        for i, actual_index in enumerate(self.last_batch_idxes):
            passing_rate = passing_rates[i]
            
            if passing_rate == 1.0:
                print("New passing rate is {}. Removing the question: {}...".format(passing_rate, self.buffer[actual_index][0][:30]))
                indices_to_remove.append(actual_index)
                continue
            
            # Get current question
            question, answer, rate, count = self.buffer[actual_index]
            
            # Update passing rate
            if passing_rate - rate > 0.01:
                new_rate = passing_rate
                new_count = 0
            else:
                new_rate = rate
                new_count = count + 1
            
            # Update data in buffer
            self.buffer[actual_index] = (question, answer, new_rate, new_count)
            
            # Check if removal is needed
            if new_count >= self.removal_threshold:
                indices_to_remove.append(actual_index)
        
        for index in sorted(indices_to_remove, reverse=True):
            self._remove_question_at_index(index)
        
        # Adjust current_index
        if self.buffer:
            self.current_index = self.get_safe_index(self.current_index)
        
        # Reset batch operation state
        self.last_batch_size = 0
        self.last_batch_idxes = []
        
        return True
    
    def _remove_question_at_index(self, index: int) -> bool:
        """Remove question at specified index
        
        Args:
            index: Index of question to remove
            
        Returns:
            Whether removal was successful
        """
        if index < 0 or index >= len(self.buffer):
            return False
        
        # Remove question
        removed = self.buffer.pop(index)
        print(f"Removed question: {removed[0][:30]}... (passing rate: {removed[2]:.1f}, not updated: {removed[3]} times)")
        
        # Adjust current index
        if self.current_index > index:
            self.current_index -= 1
        elif self.current_index >= len(self.buffer) and self.buffer:
            self.current_index = self.get_safe_index(self.current_index)
        
        return True
    
    def remove_current_question(self) -> bool:
        """Remove question at current position
            
        Returns:
            Whether removal was successful
        """
        if self.is_empty() or self.current_index >= len(self.buffer):
            return False
        
        return self._remove_question_at_index(self.current_index)
    
    def get_all_questions(self) -> List[Tuple[str, str, float, int]]:
        """Get all question states
        
        Returns:
            List of (question, answer, passing_rate, times_without_update)
        """
        return self.buffer.copy()
    
    def size(self) -> int:
        """Return number of questions in buffer"""
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.buffer) == 0
    
    def reset_position(self) -> None:
        """Reset training position to beginning"""
        self.current_index = 0
        self.last_batch_size = 0
        self.last_batch_idxes = []
    
    def __str__(self) -> str:
        if self.is_empty():
            return "QuestionBuffer(empty)"
        
        result = f"QuestionBuffer({len(self.buffer)} questions, current position: {self.current_index}):\n"
        for i, (question, answer, rate, count) in enumerate(self.buffer):
            marker = " -> " if i == self.current_index else "    "
            result += f"{marker}[{i}] {question[:30]}... (rate: {rate:.1f}, not updated: {count})\n"
        
        return result.rstrip()


# Usage example
if __name__ == "__main__":
    import random
    
    # Create buffer, remove after 3 times without update
    buffer = QuestionBuffer(removal_threshold=3)
    
    buffer.add_question("What is Python?", "A high-level programming language", 0.8)
    buffer.add_question("How to use list comprehensions?", "Use [expr for item in iterable]", 0.6)
    buffer.add_question("What is recursion?", "A function calling itself", 0.4)
    buffer.add_question("How to handle exceptions?", "Use try-except blocks", 0.7)
    buffer.add_question("What is OOP?", "Object-oriented programming paradigm", 0.5)
    buffer.add_question("How to use decorators?", "Use @ symbol before function", 0.3)
    
    print("=== Initial State ===")
    print(buffer)
    
    print("\n=== Batch Training Test ===")
    
    # Test batch get and update
    for round_num in range(5):
        print(f"\n--- Round {round_num + 1} Batch Training ---")
        
        # Batch get 3 questions
        batch_size = 3
        questions = buffer.get_multiple_questions(batch_size)
        if not questions:
            print("No more questions!")
            break
        
        print(f"Batch retrieved questions ({len(questions)}):")
        for i, (q, a) in enumerate(questions):
            print(f"  {i+1}. Q: {q[:30]}... A: {a[:20]}...")
        
        # Simulate batch training results
        passing_rates = []
        for q_tuple in questions:
            old_rate = 0.0
            for question, answer, rate, count in buffer.get_all_questions():
                if question == q_tuple[0]:
                    old_rate = rate
                    break
            
            # Simulate training effect: chance to improve passing rate
            if random.random() < 0.7:  # 70% chance to improve
                new_rate = min(1.0, old_rate + random.uniform(0.1, 0.3))
            else:
                new_rate = old_rate
            
            passing_rates.append(new_rate)
            print(f"  {q_tuple[0][:20]}... -> passing rate: {old_rate:.2f} → {new_rate:.2f}")
        
        # Batch update
        success = buffer.update_after_training_batch(passing_rates)
        print(f"Batch update result: {'Success' if success else 'Failed'}")
        
        # Show current state
        print("Current state:")
        if buffer.is_empty():
            print("  All questions completed!")
        else:
            for q, a, rate, count in buffer.get_all_questions():
                print(f"  Q: {q[:20]}... -> rate: {rate:.2f}, not updated: {count} times")
    
    print(f"\n=== Training Completed ===")
    print(f"Remaining questions: {buffer.size()}")
    if not buffer.is_empty():
        print("Final state:")
        print(buffer)