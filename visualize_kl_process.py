#!/usr/bin/env python3
"""
Visualize KL Divergence Computation Process
Demonstrates step-by-step how KL divergence is computed in the analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def demonstrate_kl_computation():
    """Demonstrate KL divergence computation with simple example."""
    
    print("=" * 70)
    print("KL DIVERGENCE COMPUTATION DEMONSTRATION")
    print("=" * 70)
    
    # Step 1: Sample texts (simplified)
    print("\n" + "=" * 70)
    print("STEP 1: Sample Texts")
    print("=" * 70)
    
    surah_1_words = ["الله", "الرحمن", "الرحيم", "الحمد", "رب", "العالمين"]
    surah_2_words = ["الله", "الله", "الرحمن", "الكتاب", "يؤمنون", "الصلاة", "رب", "رب"]
    
    print(f"Surah 1 words: {surah_1_words}")
    print(f"Surah 2 words: {surah_2_words}")
    
    # Step 2: Word frequencies
    print("\n" + "=" * 70)
    print("STEP 2: Word Frequency Counting")
    print("=" * 70)
    
    freq_1 = Counter(surah_1_words)
    freq_2 = Counter(surah_2_words)
    
    print(f"\nSurah 1 frequencies:")
    for word, count in freq_1.most_common():
        print(f"  {word}: {count}")
    
    print(f"\nSurah 2 frequencies:")
    for word, count in freq_2.most_common():
        print(f"  {word}: {count}")
    
    # Step 3: Unified vocabulary
    print("\n" + "=" * 70)
    print("STEP 3: Create Unified Vocabulary")
    print("=" * 70)
    
    vocabulary = sorted(set(surah_1_words + surah_2_words))
    print(f"Unified vocabulary: {vocabulary}")
    print(f"Vocabulary size: {len(vocabulary)}")
    
    # Step 4: Frequency vectors
    print("\n" + "=" * 70)
    print("STEP 4: Create Frequency Vectors")
    print("=" * 70)
    
    vec_1 = np.array([freq_1.get(word, 0) for word in vocabulary])
    vec_2 = np.array([freq_2.get(word, 0) for word in vocabulary])
    
    print(f"\nVocabulary: {vocabulary}")
    print(f"Surah 1 vector: {vec_1}")
    print(f"Surah 2 vector: {vec_2}")
    
    # Step 5: Normalize to probabilities
    print("\n" + "=" * 70)
    print("STEP 5: Normalize to Probability Distributions")
    print("=" * 70)
    
    epsilon = 1e-10
    prob_1 = (vec_1 + epsilon) / np.sum(vec_1 + epsilon)
    prob_2 = (vec_2 + epsilon) / np.sum(vec_2 + epsilon)
    
    print(f"\nSurah 1 probabilities:")
    for word, prob in zip(vocabulary, prob_1):
        print(f"  P({word}) = {prob:.4f}")
    
    print(f"\nSurah 2 probabilities:")
    for word, prob in zip(vocabulary, prob_2):
        print(f"  P({word}) = {prob:.4f}")
    
    # Step 6: Compute KL divergence
    print("\n" + "=" * 70)
    print("STEP 6: Compute KL Divergence")
    print("=" * 70)
    
    # D_KL(P2 || P1) - How different is Surah 2 from Surah 1?
    kl_2_given_1 = np.sum(prob_2 * np.log(prob_2 / prob_1))
    
    # D_KL(P1 || P2) - How different is Surah 1 from Surah 2?
    kl_1_given_2 = np.sum(prob_1 * np.log(prob_1 / prob_2))
    
    print(f"\nD_KL(Surah 2 || Surah 1) = {kl_2_given_1:.4f}")
    print(f"  Interpretation: How much Surah 2 diverges from Surah 1")
    
    print(f"\nD_KL(Surah 1 || Surah 2) = {kl_1_given_2:.4f}")
    print(f"  Interpretation: How much Surah 1 diverges from Surah 2")
    
    print(f"\nAsymmetry: |{kl_2_given_1:.4f} - {kl_1_given_2:.4f}| = {abs(kl_2_given_1 - kl_1_given_2):.4f}")
    print(f"  This confirms: D_KL(A||B) ≠ D_KL(B||A)")
    
    # Step 7: Visualize
    print("\n" + "=" * 70)
    print("STEP 7: Visualize Probability Distributions")
    print("=" * 70)
    
    create_visualization(vocabulary, prob_1, prob_2, kl_2_given_1, kl_1_given_2)
    
    print("\nVisualization saved as 'kl_divergence_demo.png'")

def create_visualization(vocabulary, prob_1, prob_2, kl_forward, kl_reverse):
    """Create visualization of KL divergence computation."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Probability distributions
    ax1 = axes[0, 0]
    x = np.arange(len(vocabulary))
    width = 0.35
    
    ax1.bar(x - width/2, prob_1, width, label='Surah 1', alpha=0.8, color='blue')
    ax1.bar(x + width/2, prob_2, width, label='Surah 2', alpha=0.8, color='red')
    ax1.set_xlabel('Words')
    ax1.set_ylabel('Probability')
    ax1.set_title('Word Probability Distributions')
    ax1.set_xticks(x)
    ax1.set_xticklabels(vocabulary, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: KL divergence components
    ax2 = axes[0, 1]
    kl_components = prob_2 * np.log(prob_2 / prob_1)
    colors = ['red' if x > 0 else 'blue' for x in kl_components]
    
    ax2.bar(x, kl_components, color=colors, alpha=0.7)
    ax2.set_xlabel('Words')
    ax2.set_ylabel('KL Divergence Component')
    ax2.set_title(f'KL Divergence Components (Total: {kl_forward:.4f})')
    ax2.set_xticks(x)
    ax2.set_xticklabels(vocabulary, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Asymmetry demonstration
    ax3 = axes[1, 0]
    directions = ['Surah 2 → Surah 1', 'Surah 1 → Surah 2']
    kl_values = [kl_forward, kl_reverse]
    colors_asym = ['#e74c3c', '#3498db']
    
    bars = ax3.bar(directions, kl_values, color=colors_asym, alpha=0.8)
    ax3.set_ylabel('KL Divergence')
    ax3.set_title('Asymmetric Relationships')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, kl_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Explanation text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    explanation = f"""
    KL DIVERGENCE SUMMARY
    {'='*40}
    
    Formula: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))
    
    Forward Direction:
      D_KL(S2 || S1) = {kl_forward:.4f}
      "How different is S2 from S1?"
    
    Reverse Direction:
      D_KL(S1 || S2) = {kl_reverse:.4f}
      "How different is S1 from S2?"
    
    Asymmetry:
      |Forward - Reverse| = {abs(kl_forward - kl_reverse):.4f}
      
    What It Measures:
      ✓ Word frequency differences
      ✓ Vocabulary overlap
      ✓ Thematic content
      
    What It Doesn't Measure:
      ✗ Sentence structure
      ✗ Word order
      ✗ Grammar patterns
      ✗ Semantic similarity
    """
    
    ax4.text(0.1, 0.5, explanation, fontfamily='monospace',
            fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('kl_divergence_demo.png', dpi=300, bbox_inches='tight')
    print("  ✓ Visualization created")

def demonstrate_what_kl_misses():
    """Demonstrate what KL divergence doesn't capture."""
    
    print("\n" + "=" * 70)
    print("WHAT KL DIVERGENCE DOESN'T CAPTURE")
    print("=" * 70)
    
    # Example 1: Word order
    print("\n1. WORD ORDER (Not Captured)")
    sentence_a = "الله خلق السماوات والأرض"
    sentence_b = "خلق الله والأرض السماوات"
    
    print(f"  Sentence A: {sentence_a}")
    print(f"  Sentence B: {sentence_b}")
    print(f"  Same words? YES")
    print(f"  Same meaning? NO (word order matters)")
    print(f"  KL divergence: ~0.0 (treats as identical)")
    
    # Example 2: Synonyms
    print("\n2. SYNONYMS (Not Captured)")
    text_a = "الله الرحمن الغفور"
    text_b = "الله الرحيم العفو"
    
    print(f"  Text A: {text_a} (merciful, forgiving)")
    print(f"  Text B: {text_b} (compassionate, pardoning)")
    print(f"  Similar meaning? YES")
    print(f"  KL divergence: High (different words)")
    
    # Example 3: Sentence structure
    print("\n3. SENTENCE STRUCTURE (Not Captured)")
    pattern_a = "[Subject] [Verb] [Object]"
    pattern_b = "[Subject] [Verb] [Object]"
    
    print(f"  Both texts use pattern: {pattern_a}")
    print(f"  Same structure? YES")
    print(f"  KL divergence: Depends only on word choice")

if __name__ == "__main__":
    demonstrate_kl_computation()
    demonstrate_what_kl_misses()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
    KL Divergence in this analysis:
    
    ✓ MEASURES:
      - Word frequency distributions
      - Vocabulary overlap between surahs
      - Overall thematic content (via word choice)
      - Asymmetric relationships
    
    ✗ DOES NOT MEASURE:
      - Sentence structure or patterns
      - Word order or grammar
      - Semantic meaning (synonyms, context)
      - Similar sentence structures
      - Rhetorical patterns
    
    This is a VOCABULARY-BASED analysis, not a syntactic
    or semantic analysis. It's powerful for understanding
    thematic relationships but doesn't capture structural
    similarities.
    
    For sentence-level analysis, you would need:
      - N-gram models
      - Sentence embeddings (BERT, transformers)
      - Syntactic parsing
      - Semantic similarity measures
    """)
