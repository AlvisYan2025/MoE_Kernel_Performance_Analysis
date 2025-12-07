import json
import numpy as np
from collections import Counter
import math
from transformers import AutoTokenizer

class ImbalancePredictor:
    """
    Predicts routing imbalance level for MoE models based on input characteristics.
    
    Usage:
        predictor = ImbalancePredictor()
        score = predictor.predict(texts)  # texts is list of strings
    """
    
    def __init__(self, tokenizer_name="mistralai/Mixtral-8x7B-v0.1"):
        """Initialize with tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def predict(self, texts):
        """
        Compute predicted imbalance score for a list of text samples.
        
        Args:
            texts: List of strings OR list of dicts with 'text' field
        
        Returns:
            float: Imbalance score in [0, 1], where higher = more imbalanced
        """
        # Handle both raw strings and dicts with 'text' field
        if isinstance(texts[0], dict):
            texts = [sample['text'] for sample in texts]
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        if len(all_tokens) == 0:
            return 0.0
        
        # Compute component metrics
        metrics = self._compute_metrics(all_tokens)
        
        # Aggregate into single score
        score = self._aggregate_score(metrics)
        
        return score
    
    def predict_with_breakdown(self, texts):
        """
        Compute predicted imbalance score with detailed metric breakdown.
        
        Args:
            texts: List of strings OR list of dicts with 'text' field
        
        Returns:
            dict: {
                'imbalance_score': float,
                'metrics': dict of individual metrics
            }
        """
        # Handle both raw strings and dicts with 'text' field
        if isinstance(texts[0], dict):
            texts = [sample['text'] for sample in texts]
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        if len(all_tokens) == 0:
            return {'imbalance_score': 0.0, 'metrics': {}}
        
        # Compute component metrics
        metrics = self._compute_metrics(all_tokens)
        
        # Aggregate into single score
        score = self._aggregate_score(metrics)
        
        return {
            'imbalance_score': score,
            'metrics': metrics
        }
    
    def _compute_metrics(self, all_tokens):
        """Compute all component metrics"""
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        counter = Counter(all_tokens)
        
        return {
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'token_repetition_rate': 1.0 - (unique_tokens / total_tokens),
            'top10_concentration': self._top_k_concentration(all_tokens, counter, k=10),
            'vocabulary_diversity': unique_tokens / total_tokens,
            'unigram_entropy': self._unigram_entropy(counter, total_tokens),
            'gini_coefficient': self._gini_coefficient(counter),
        }
    
    def _aggregate_score(self, metrics):
        """
        Combine metrics into single imbalance score.
        
        Weights are based on expected correlation with routing imbalance:
        - High repetition → high imbalance
        - High concentration → high imbalance
        - Low diversity → high imbalance
        - Low entropy → high imbalance
        - High Gini → high imbalance
        """
        score = 0.0
        
        # Token repetition (higher = more imbalance)
        score += metrics['token_repetition_rate'] * 0.30
        
        # Top-10 concentration (higher = more imbalance)
        score += metrics['top10_concentration'] * 0.25
        
        # Inverse of vocabulary diversity (lower diversity = more imbalance)
        score += (1 - metrics['vocabulary_diversity']) * 0.20
        
        # Inverse of normalized entropy (lower entropy = more imbalance)
        max_entropy = math.log2(max(metrics['unique_tokens'], 2))
        normalized_entropy = metrics['unigram_entropy'] / max_entropy
        score += (1 - normalized_entropy) * 0.15
        
        # Gini coefficient (higher = more imbalance)
        score += metrics['gini_coefficient'] * 0.10
        
        return min(score, 1.0)  # Clamp to [0, 1]
    
    def _top_k_concentration(self, tokens, counter, k=10):
        """Proportion of tokens covered by top-k most frequent"""
        top_k_count = sum([count for _, count in counter.most_common(k)])
        return top_k_count / len(tokens)
    
    def _unigram_entropy(self, counter, total):
        """Shannon entropy of token distribution"""
        entropy = 0.0
        for count in counter.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy
    
    def _gini_coefficient(self, counter):
        """Gini coefficient of token frequency distribution"""
        frequencies = sorted(counter.values())
        n = len(frequencies)
        
        if n == 0:
            return 0.0
        
        cumsum = np.cumsum(frequencies)
        gini = (2 * np.sum((np.arange(1, n + 1) * frequencies))) / (n * np.sum(frequencies)) - (n + 1) / n
        
        return gini


# ============= Convenience Functions =============

def load_dataset(filepath):
    """
    Load a dataset from JSONL file.
    
    Args:
        filepath: Path to .jsonl file
    
    Returns:
        List of dicts with 'text' field
    """
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def predict_imbalance(texts, tokenizer_name="mistralai/Mixtral-8x7B-v0.1"):
    """
    Quick function to predict imbalance score.
    
    Args:
        texts: List of strings or list of dicts with 'text' field
        tokenizer_name: HuggingFace model name for tokenizer
    
    Returns:
        float: Imbalance score in [0, 1]
    """
    predictor = ImbalancePredictor(tokenizer_name=tokenizer_name)
    return predictor.predict(texts)


# ============= Example Usage =============

if __name__ == "__main__":
    # Example 1: Direct usage with text list
    predictor = ImbalancePredictor()
    
    texts = [
        "the the the the the",
        "Python programming Python programming",
        "hello world hello world"
    ]
    
    score = predictor.predict(texts)
    print(f"Imbalance score: {score:.3f}")
    
    # Example 2: With breakdown
    result = predictor.predict_with_breakdown(texts)
    print(f"\nDetailed breakdown:")
    print(f"  Imbalance score: {result['imbalance_score']:.3f}")
    print(f"  Token repetition: {result['metrics']['token_repetition_rate']:.3f}")
    print(f"  Top-10 concentration: {result['metrics']['top10_concentration']:.3f}")
    print(f"  Vocabulary diversity: {result['metrics']['vocabulary_diversity']:.3f}")
    
    # Example 3: Load from file
    # samples = load_dataset("datasets/high_repetition.jsonl")
    # score = predictor.predict(samples)
    # print(f"Dataset imbalance score: {score:.3f}")