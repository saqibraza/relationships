#!/usr/bin/env python3
"""
Sample Surah Pairs Analysis
============================

Detailed analysis of specific surah pairs showing:
- Individual method scores
- Unified scores
- Bidirectional relationships (A→B and B→A)
- Interpretation and insights
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple

# Sample pairs to analyze
SAMPLE_PAIRS = [
    (113, 114),  # Al-Falaq and An-Nas (protection surahs)
    (93, 94),    # Ad-Duha and Ash-Sharh (consolation surahs)
    (69, 101),   # Al-Haqqah and Al-Qari'ah (eschatological surahs)
    (2, 3),      # Al-Baqarah and Āl 'Imrān (long Medinan surahs)
    (2, 65),     # Al-Baqarah and At-Talaq (legislative surahs)
    (24, 33),    # An-Nur and Al-Ahzab (social guidance surahs)
]

SURAH_NAMES = {
    2: "Al-Baqarah", 3: "Āl 'Imrān", 24: "An-Nur", 33: "Al-Ahzab",
    65: "At-Talaq", 69: "Al-Haqqah", 93: "Ad-Duha", 94: "Ash-Sharh",
    101: "Al-Qari'ah", 113: "Al-Falaq", 114: "An-Nas"
}


class SamplePairsAnalyzer:
    """Analyzes specific surah pairs across all methods."""
    
    def __init__(self):
        self.matrices = {}
        self.load_all_matrices()
    
    def load_all_matrices(self):
        """Load all analysis matrices."""
        print("Loading all matrices...")
        
        # Load from unified results
        base_dirs = ['unified_results', 'normalized_results', 'advanced_results']
        
        # Try unified results first
        if os.path.exists('unified_results/unified_similarity_matrix.csv'):
            self.matrices['unified'] = pd.read_csv(
                'unified_results/unified_similarity_matrix.csv', 
                index_col=0
            )
            self.matrices['kl_divergence'] = pd.read_csv(
                'unified_results/kl_divergence_matrix.csv',
                index_col=0
            )
            self.matrices['bigram'] = pd.read_csv(
                'unified_results/bigram_matrix.csv',
                index_col=0
            )
            self.matrices['trigram'] = pd.read_csv(
                'unified_results/trigram_matrix.csv',
                index_col=0
            )
            self.matrices['embeddings'] = pd.read_csv(
                'unified_results/embeddings_matrix.csv',
                index_col=0
            )
            self.matrices['arabert'] = pd.read_csv(
                'unified_results/arabert_matrix.csv',
                index_col=0
            )
            print("✓ Loaded all 6 matrices from unified_results/")
        else:
            print("✗ Unified results not found. Please run unified_analysis.py first.")
            return False
        
        return True
    
    def get_pair_scores(self, surah_a: int, surah_b: int) -> Dict:
        """Get all scores for a surah pair (bidirectional)."""
        surah_a_label = f"Surah {surah_a}"
        surah_b_label = f"Surah {surah_b}"
        
        scores = {
            'surah_a': surah_a,
            'surah_b': surah_b,
            'name_a': SURAH_NAMES.get(surah_a, f"Surah {surah_a}"),
            'name_b': SURAH_NAMES.get(surah_b, f"Surah {surah_b}"),
            'methods': {}
        }
        
        for method, matrix in self.matrices.items():
            # A → B (forward)
            forward = matrix.loc[surah_a_label, surah_b_label]
            # B → A (reverse)
            reverse = matrix.loc[surah_b_label, surah_a_label]
            # Asymmetry
            asymmetry = forward - reverse
            
            scores['methods'][method] = {
                'forward': round(float(forward), 2),
                'reverse': round(float(reverse), 2),
                'asymmetry': round(float(asymmetry), 2),
                'average': round((float(forward) + float(reverse)) / 2, 2)
            }
        
        return scores
    
    def analyze_all_pairs(self) -> List[Dict]:
        """Analyze all sample pairs."""
        results = []
        
        for surah_a, surah_b in SAMPLE_PAIRS:
            scores = self.get_pair_scores(surah_a, surah_b)
            results.append(scores)
        
        return results
    
    def generate_markdown_report(self, results: List[Dict], output_file: str = "SAMPLE_PAIRS_ANALYSIS.md"):
        """Generate comprehensive markdown report."""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Detailed Analysis of Sample Surah Pairs\n\n")
            f.write("This document provides a comprehensive analysis of specific surah pairs ")
            f.write("across all 6 analysis methods (5 individual + 1 unified).\n\n")
            
            f.write("## Analysis Methods\n\n")
            f.write("| Method | Weight in Unified | What It Measures |\n")
            f.write("|--------|------------------|------------------|\n")
            f.write("| **KL Divergence** | 30% | Word frequency distributions |\n")
            f.write("| **Bigrams** | 10% | 2-word phrase patterns |\n")
            f.write("| **Trigrams** | 10% | 3-word phrase patterns |\n")
            f.write("| **Sentence Embeddings** | 35% | Deep semantic meaning |\n")
            f.write("| **AraBERT** | 15% | Arabic-specific contextual embeddings |\n")
            f.write("| **Unified** | 100% | Weighted combination of all methods |\n\n")
            
            f.write("## Reading the Results\n\n")
            f.write("For each pair, we show:\n")
            f.write("- **Forward (A→B)**: Similarity from Surah A to Surah B\n")
            f.write("- **Reverse (B→A)**: Similarity from Surah B to Surah A\n")
            f.write("- **Asymmetry**: Difference between forward and reverse (A→B - B→A)\n")
            f.write("  - Positive: A encompasses more of B's themes\n")
            f.write("  - Negative: B encompasses more of A's themes\n")
            f.write("  - Near zero: Symmetric relationship\n")
            f.write("- **Average**: Mean of bidirectional scores\n\n")
            
            f.write("---\n\n")
            
            # Analyze each pair
            for idx, pair_data in enumerate(results, 1):
                surah_a = pair_data['surah_a']
                surah_b = pair_data['surah_b']
                name_a = pair_data['name_a']
                name_b = pair_data['name_b']
                
                f.write(f"## Pair {idx}: Surah {surah_a} ({name_a}) ↔ Surah {surah_b} ({name_b})\n\n")
                
                # Summary table
                f.write("### Summary Table\n\n")
                f.write("| Method | A→B | B→A | Asymmetry | Average | Interpretation |\n")
                f.write("|--------|-----|-----|-----------|---------|----------------|\n")
                
                methods_order = ['unified', 'kl_divergence', 'bigram', 'trigram', 'embeddings', 'arabert']
                method_names = {
                    'unified': 'UNIFIED',
                    'kl_divergence': 'KL Divergence',
                    'bigram': 'Bigrams',
                    'trigram': 'Trigrams',
                    'embeddings': 'Embeddings',
                    'arabert': 'AraBERT'
                }
                
                for method in methods_order:
                    if method in pair_data['methods']:
                        scores = pair_data['methods'][method]
                        forward = scores['forward']
                        reverse = scores['reverse']
                        asymmetry = scores['asymmetry']
                        average = scores['average']
                        
                        # Interpretation
                        if abs(asymmetry) < 2:
                            interp = "Symmetric"
                        elif asymmetry > 0:
                            interp = f"{name_a} → {name_b}"
                        else:
                            interp = f"{name_b} → {name_a}"
                        
                        # Bold unified row
                        if method == 'unified':
                            f.write(f"| **{method_names[method]}** | **{forward}%** | **{reverse}%** | ")
                            f.write(f"**{asymmetry:+.2f}%** | **{average}%** | **{interp}** |\n")
                        else:
                            f.write(f"| {method_names[method]} | {forward}% | {reverse}% | ")
                            f.write(f"{asymmetry:+.2f}% | {average}% | {interp} |\n")
                
                # Detailed analysis
                f.write("\n### Detailed Analysis\n\n")
                
                unified_scores = pair_data['methods']['unified']
                unified_avg = unified_scores['average']
                
                # Overall similarity level
                if unified_avg >= 55:
                    level = "Very High"
                    desc = "These surahs are closely related across multiple dimensions"
                elif unified_avg >= 45:
                    level = "High"
                    desc = "Strong thematic and linguistic connections"
                elif unified_avg >= 35:
                    level = "Moderate"
                    desc = "Some shared themes and vocabulary"
                elif unified_avg >= 25:
                    level = "Low"
                    desc = "Limited connections, mostly distinct"
                else:
                    level = "Very Low"
                    desc = "Highly distinct with minimal overlap"
                
                f.write(f"**Unified Similarity Level**: {level} ({unified_avg}%)\n\n")
                f.write(f"{desc}.\n\n")
                
                # Method-specific insights
                f.write("#### Method-Specific Insights\n\n")
                
                kl_avg = pair_data['methods']['kl_divergence']['average']
                bigram_avg = pair_data['methods']['bigram']['average']
                trigram_avg = pair_data['methods']['trigram']['average']
                embed_avg = pair_data['methods']['embeddings']['average']
                arabert_avg = pair_data['methods']['arabert']['average']
                
                f.write(f"1. **Vocabulary (KL Divergence: {kl_avg}%)**\n")
                if kl_avg >= 30:
                    f.write(f"   - High vocabulary overlap. Surahs share many words.\n")
                elif kl_avg >= 15:
                    f.write(f"   - Moderate vocabulary sharing. Some common terms.\n")
                else:
                    f.write(f"   - Low vocabulary overlap. Distinct word choices.\n")
                
                f.write(f"\n2. **Phrases (Bigrams: {bigram_avg}%, Trigrams: {trigram_avg}%)**\n")
                if bigram_avg >= 5:
                    f.write(f"   - Significant phrase repetition. Common expressions shared.\n")
                elif bigram_avg >= 1:
                    f.write(f"   - Some phrase overlap. Occasional common patterns.\n")
                else:
                    f.write(f"   - Minimal phrase overlap. Unique linguistic styles.\n")
                
                f.write(f"\n3. **Semantics (Embeddings: {embed_avg}%)**\n")
                if embed_avg >= 85:
                    f.write(f"   - Very high semantic similarity. Core concepts strongly aligned.\n")
                elif embed_avg >= 75:
                    f.write(f"   - High semantic similarity. Related meanings and themes.\n")
                elif embed_avg >= 60:
                    f.write(f"   - Moderate semantic overlap. Some conceptual connections.\n")
                else:
                    f.write(f"   - Low semantic similarity. Different core messages.\n")
                
                f.write(f"\n4. **Arabic Context (AraBERT: {arabert_avg}%)**\n")
                if arabert_avg >= 85:
                    f.write(f"   - Strong Arabic linguistic patterns. Similar morphological and contextual usage.\n")
                elif arabert_avg >= 75:
                    f.write(f"   - Good Arabic context alignment. Related linguistic structures.\n")
                else:
                    f.write(f"   - Different Arabic linguistic contexts. Varied morphological patterns.\n")
                
                # Asymmetry analysis
                unified_asym = unified_scores['asymmetry']
                f.write(f"\n#### Asymmetry Analysis\n\n")
                if abs(unified_asym) < 2:
                    f.write(f"**Relationship Type**: Symmetric (asymmetry = {unified_asym:+.2f}%)\n\n")
                    f.write(f"The relationship is roughly equal in both directions. ")
                    f.write(f"Both surahs have similar levels of thematic overlap with each other.\n")
                elif unified_asym > 0:
                    f.write(f"**Relationship Type**: Directional (asymmetry = {unified_asym:+.2f}%)\n\n")
                    f.write(f"Surah {surah_a} ({name_a}) shows higher similarity toward Surah {surah_b} ({name_b}) ")
                    f.write(f"than vice versa. This suggests that Surah {surah_a} may encompass or reference ")
                    f.write(f"more of Surah {surah_b}'s themes.\n")
                else:
                    f.write(f"**Relationship Type**: Directional (asymmetry = {unified_asym:+.2f}%)\n\n")
                    f.write(f"Surah {surah_b} ({name_b}) shows higher similarity toward Surah {surah_a} ({name_a}) ")
                    f.write(f"than vice versa. This suggests that Surah {surah_b} may encompass or reference ")
                    f.write(f"more of Surah {surah_a}'s themes.\n")
                
                # Comparative insight
                f.write(f"\n#### Key Insight\n\n")
                
                # Compare vocabulary vs semantics
                vocab_sem_diff = embed_avg - kl_avg
                if vocab_sem_diff > 50:
                    f.write(f"**Discovery**: High semantic similarity ({embed_avg}%) despite ")
                    f.write(f"low vocabulary overlap ({kl_avg}%). ")
                    f.write(f"These surahs convey similar concepts using different words - ")
                    f.write(f"evidence of linguistic diversity with thematic unity.\n")
                elif vocab_sem_diff < -10:
                    f.write(f"**Discovery**: High vocabulary overlap ({kl_avg}%) but ")
                    f.write(f"lower semantic similarity ({embed_avg}%). ")
                    f.write(f"These surahs share many words but may use them in different semantic contexts.\n")
                else:
                    f.write(f"**Discovery**: Vocabulary ({kl_avg}%) and semantic similarity ({embed_avg}%) ")
                    f.write(f"are well-aligned. The surahs have proportional word and meaning overlap.\n")
                
                f.write("\n---\n\n")
            
            # Summary comparison
            f.write("## Summary Comparison of All Pairs\n\n")
            f.write("### Unified Similarity Rankings\n\n")
            
            # Sort by unified average
            sorted_results = sorted(
                results,
                key=lambda x: x['methods']['unified']['average'],
                reverse=True
            )
            
            f.write("| Rank | Pair | Unified Similarity | Category |\n")
            f.write("|------|------|--------------------|----------|\n")
            
            for rank, pair_data in enumerate(sorted_results, 1):
                surah_a = pair_data['surah_a']
                surah_b = pair_data['surah_b']
                name_a = pair_data['name_a']
                name_b = pair_data['name_b']
                unified_avg = pair_data['methods']['unified']['average']
                
                if unified_avg >= 55:
                    category = "Very High"
                elif unified_avg >= 45:
                    category = "High"
                elif unified_avg >= 35:
                    category = "Moderate"
                else:
                    category = "Low"
                
                f.write(f"| {rank} | {surah_a}×{surah_b} ({name_a} ↔ {name_b}) | ")
                f.write(f"{unified_avg}% | {category} |\n")
            
            # Method comparison
            f.write("\n### Method Comparison Across Pairs\n\n")
            f.write("Average scores for each method across all sample pairs:\n\n")
            
            f.write("| Method | Average Score | Std Dev |\n")
            f.write("|--------|--------------|----------|\n")
            
            for method in ['unified', 'kl_divergence', 'bigram', 'trigram', 'embeddings', 'arabert']:
                scores = [r['methods'][method]['average'] for r in results]
                avg = np.mean(scores)
                std = np.std(scores)
                method_name = {
                    'unified': 'Unified',
                    'kl_divergence': 'KL Divergence',
                    'bigram': 'Bigrams',
                    'trigram': 'Trigrams',
                    'embeddings': 'Embeddings',
                    'arabert': 'AraBERT'
                }[method]
                f.write(f"| {method_name} | {avg:.2f}% | {std:.2f}% |\n")
            
            f.write("\n---\n\n")
            f.write("**Analysis Date**: October 15, 2025  \n")
            f.write("**Methods**: 6 (KL Divergence, Bigrams, Trigrams, Embeddings, AraBERT, Unified)  \n")
            f.write("**Sample Pairs**: 6  \n")
            f.write("**Score Range**: 0-100%  \n")
        
        print(f"✓ Generated comprehensive report: {output_file}")
    
    def generate_csv_report(self, results: List[Dict], output_file: str = "sample_pairs_scores.csv"):
        """Generate CSV with all scores."""
        
        rows = []
        for pair_data in results:
            surah_a = pair_data['surah_a']
            surah_b = pair_data['surah_b']
            name_a = pair_data['name_a']
            name_b = pair_data['name_b']
            
            for method, scores in pair_data['methods'].items():
                rows.append({
                    'Surah_A': surah_a,
                    'Name_A': name_a,
                    'Surah_B': surah_b,
                    'Name_B': name_b,
                    'Method': method,
                    'Forward_A_to_B': scores['forward'],
                    'Reverse_B_to_A': scores['reverse'],
                    'Asymmetry': scores['asymmetry'],
                    'Average': scores['average']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"✓ Generated CSV report: {output_file}")


def format_matrix_to_2_decimals(csv_file: str):
    """Format a CSV matrix to 2 decimal places."""
    df = pd.read_csv(csv_file, index_col=0)
    df = df.round(2)
    df.to_csv(csv_file, float_format='%.2f')
    print(f"✓ Formatted {csv_file} to 2 decimal places")


def format_all_matrices():
    """Format all CSV matrices to 2 decimal places."""
    print("\n" + "="*70)
    print("FORMATTING ALL MATRICES TO 2 DECIMAL PLACES")
    print("="*70 + "\n")
    
    csv_files = [
        'unified_results/unified_similarity_matrix.csv',
        'unified_results/kl_divergence_matrix.csv',
        'unified_results/bigram_matrix.csv',
        'unified_results/trigram_matrix.csv',
        'unified_results/embeddings_matrix.csv',
        'unified_results/arabert_matrix.csv',
        'normalized_results/similarity_matrix_normalized.csv',
        'advanced_results/2gram_matrix.csv',
        'advanced_results/3gram_matrix.csv',
        'advanced_results/multilingual_embedding_matrix.csv',
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            format_matrix_to_2_decimals(csv_file)
        else:
            print(f"⚠ File not found: {csv_file}")
    
    print("\n✓ All matrices formatted successfully")


def main():
    """Main execution."""
    print("="*70)
    print("SAMPLE SURAH PAIRS ANALYSIS")
    print("="*70)
    print()
    print("Sample pairs to analyze:")
    for idx, (a, b) in enumerate(SAMPLE_PAIRS, 1):
        name_a = SURAH_NAMES.get(a, f"Surah {a}")
        name_b = SURAH_NAMES.get(b, f"Surah {b}")
        print(f"  {idx}. Surah {a} ({name_a}) ↔ Surah {b} ({name_b})")
    print()
    
    # Format all matrices first
    format_all_matrices()
    
    # Analyze sample pairs
    print("\n" + "="*70)
    print("ANALYZING SAMPLE PAIRS")
    print("="*70 + "\n")
    
    analyzer = SamplePairsAnalyzer()
    if not analyzer.matrices:
        print("✗ Could not load matrices. Please run unified_analysis.py first.")
        return
    
    results = analyzer.analyze_all_pairs()
    
    print("\n" + "="*70)
    print("GENERATING REPORTS")
    print("="*70 + "\n")
    
    analyzer.generate_markdown_report(results)
    analyzer.generate_csv_report(results)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - SAMPLE_PAIRS_ANALYSIS.md (comprehensive markdown report)")
    print("  - sample_pairs_scores.csv (detailed scores)")
    print("\nAll CSV matrices now formatted to 2 decimal places for better readability.")
    print("="*70)


if __name__ == "__main__":
    main()

