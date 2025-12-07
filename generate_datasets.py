import json
import random
from pathlib import Path

# ============= Domain-Specific Content =============
DOMAINS = {
    "programming": [
        "Write a Python function to sort a list",
        "Explain object-oriented programming concepts",
        "Debug this JavaScript async/await code",
        "Create a REST API endpoint in Flask",
        "Implement binary search in C++",
    ],
    "science": [
        "Explain the process of photosynthesis",
        "What is quantum entanglement?",
        "Describe DNA replication mechanism",
        "How do black holes form?",
        "Explain Newton's laws of motion",
    ],
    "cooking": [
        "Recipe for homemade pasta sauce",
        "How to properly season a cast iron skillet",
        "Techniques for making sourdough bread",
        "Explain the Maillard reaction in cooking",
        "Best methods for caramelizing onions",
    ],
    "history": [
        "Causes of the French Revolution",
        "Explain the fall of the Roman Empire",
        "What led to World War I?",
        "Describe the Renaissance period",
        "Impact of the Industrial Revolution",
    ],
    "math": [
        "Solve this quadratic equation: x^2 + 5x + 6 = 0",
        "Explain the concept of derivatives",
        "What is the Pythagorean theorem?",
        "Calculate the area under a curve using integrals",
        "Prove that the sum of angles in a triangle is 180 degrees",
    ],
}

# ============= Generation Functions =============

def generate_high_repetition(n_samples=20, seq_length=50):
    """
    Expected: VERY HIGH imbalance
    Strategy: Repeat same token/phrase many times
    """
    datasets = []
    
    templates = [
        "the " * seq_length,
        "hello world " * (seq_length // 2),
        "Python programming " * (seq_length // 2),
        "1 2 3 4 5 " * (seq_length // 5),
    ]
    
    for i in range(n_samples):
        text = templates[i % len(templates)]
        datasets.append({
            "id": f"high_rep_{i}",
            "text": text.strip(),
            "category": "high_repetition"
        })
    
    return datasets

def generate_single_domain(domain, n_samples=20):
    """
    Expected: HIGH imbalance
    Strategy: All prompts from same domain
    """
    datasets = []
    prompts = DOMAINS[domain]
    
    for i in range(n_samples):
        datasets.append({
            "id": f"single_{domain}_{i}",
            "text": prompts[i % len(prompts)],
            "category": f"single_domain_{domain}"
        })
    
    return datasets

def generate_mixed_domains(n_samples=20, n_domains_per_sample=3):
    """
    Expected: MEDIUM imbalance
    Strategy: Mix multiple domains in each prompt
    """
    datasets = []
    all_domains = list(DOMAINS.keys())
    
    for i in range(n_samples):
        selected_domains = random.sample(all_domains, n_domains_per_sample)
        mixed_text = " ".join([
            random.choice(DOMAINS[d]) 
            for d in selected_domains
        ])
        
        datasets.append({
            "id": f"mixed_{i}",
            "text": mixed_text,
            "category": "mixed_domains"
        })
    
    return datasets

def generate_diverse_topics(n_samples=20):
    """
    Expected: LOW imbalance
    Strategy: Each prompt from different domain, max diversity
    """
    datasets = []
    all_domains = list(DOMAINS.keys())
    all_prompts = [p for prompts in DOMAINS.values() for p in prompts]
    
    # Sample without replacement for max diversity
    sampled = random.sample(all_prompts, min(n_samples, len(all_prompts)))
    
    for i, text in enumerate(sampled):
        datasets.append({
            "id": f"diverse_{i}",
            "text": text,
            "category": "diverse_topics"
        })
    
    return datasets

def generate_repeated_structure(n_samples=20):
    """
    Expected: HIGH imbalance
    Strategy: Same syntactic structure, different content
    """
    datasets = []
    
    template = "Explain the concept of {} in simple terms"
    concepts = [
        "machine learning", "blockchain", "photosynthesis",
        "democracy", "recursion", "entropy", "capitalism",
        "evolution", "quantum mechanics", "relativity"
    ]
    
    for i in range(n_samples):
        text = template.format(concepts[i % len(concepts)])
        datasets.append({
            "id": f"repeated_struct_{i}",
            "text": text,
            "category": "repeated_structure"
        })
    
    return datasets

def generate_varying_length(n_samples=20):
    """
    Expected: MEDIUM imbalance
    Strategy: Mix short and long prompts
    """
    datasets = []
    
    for i in range(n_samples):
        if i % 3 == 0:
            # Short
            text = "Hello"
        elif i % 3 == 1:
            # Medium
            text = "Write a short story about a robot learning to feel emotions"
        else:
            # Long
            text = " ".join(random.sample([p for prompts in DOMAINS.values() for p in prompts], 5))
        
        datasets.append({
            "id": f"varying_len_{i}",
            "text": text,
            "category": "varying_length"
        })
    
    return datasets

# ============= Main Generation =============

def generate_all_datasets(output_dir="datasets"):
    """Generate all dataset categories and save to output_dir"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating datasets in directory: {output_path.absolute()}\n")
    
    all_datasets = {
        "high_repetition": generate_high_repetition(n_samples=30),
        "single_domain_programming": generate_single_domain("programming", n_samples=30),
        "single_domain_science": generate_single_domain("science", n_samples=30),
        "mixed_domains": generate_mixed_domains(n_samples=30),
        "diverse_topics": generate_diverse_topics(n_samples=30),
        "repeated_structure": generate_repeated_structure(n_samples=30),
        "varying_length": generate_varying_length(n_samples=30),
    }
    
    # Save each category as separate JSONL file
    for category, data in all_datasets.items():
        filepath = output_path / f"{category}.jsonl"
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"✓ Saved {len(data):3d} samples → {filepath}")
    
    # Save combined dataset with all categories
    combined = []
    for category, data in all_datasets.items():
        combined.extend(data)
    
    combined_path = output_path / "all_datasets.jsonl"
    with open(combined_path, 'w') as f:
        for item in combined:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✓ Combined: {len(combined)} total samples → {combined_path}")
    print(f"✓ All datasets saved to: {output_path.absolute()}/")
    
    return all_datasets

if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    
    # Generate and save to datasets/ directory
    datasets = generate_all_datasets(output_dir="datasets")
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Generation Summary:")
    print("="*60)
    for category, data in datasets.items():
        print(f"  {category:35s}: {len(data):3d} samples")
    print("="*60)