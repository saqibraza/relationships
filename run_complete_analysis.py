#!/usr/bin/env python3
"""
Complete Analysis Runner
========================

Runs both types of unified analysis:
1. Unified-All: Combines all 5 methods (KL, bigrams, trigrams, embeddings, AraBERT)
2. Unified-Semantic: Combines only semantic methods (embeddings + AraBERT)

Also generates sample pairs analysis with both unified types.
"""

import numpy as np
import pandas as pd
import os
import sys
from unified_analysis import UnifiedQuranAnalyzer
from sample_pairs_analysis import SamplePairsAnalyzer, SAMPLE_PAIRS, SURAH_NAMES, format_matrix_to_2_decimals
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_unified_all():
    """Run unified analysis with all 5 methods."""
    logger.info("\n" + "="*70)
    logger.info("UNIFIED ANALYSIS - ALL METHODS")
    logger.info("="*70)
    
    analyzer = UnifiedQuranAnalyzer(unified_type='all')
    analyzer.load_data()
    analyzer.compute_all_matrices()
    analyzer.compute_unified_matrix()
    analyzer.visualize_unified_matrix(save_path="unified_all_similarity_matrix.png")
    analyzer.compare_methods()
    
    # Save with specific naming
    os.makedirs("unified_results", exist_ok=True)
    
    # Save unified-all matrix
    np.save("unified_results/unified_all_matrix.npy", analyzer.unified_matrix)
    df = pd.DataFrame(analyzer.unified_matrix,
                     index=analyzer.surah_names,
                     columns=analyzer.surah_names).round(2)
    df.to_csv("unified_results/unified_all_matrix.csv", float_format='%.2f')
    
    # Save analysis results
    analysis = analyzer.analyze_unified_relationships()
    with open("unified_results/unified_all_results.txt", 'w', encoding='utf-8') as f:
        f.write("UNIFIED ANALYSIS - ALL METHODS (KL + N-grams + Embeddings + AraBERT)\n")
        f.write("="*70 + "\n\n")
        f.write("Weights:\n")
        for method, weight in analyzer.weights.items():
            if weight > 0:
                f.write(f"  • {method.replace('_', ' ').title()}: {weight*100:.1f}%\n")
        f.write(f"\nStatistics:\n")
        stats = analysis['statistics']
        f.write(f"  Mean: {stats['mean_similarity']:.2f}%\n")
        f.write(f"  Range: {stats['min_similarity']:.2f}% - {stats['max_similarity']:.2f}%\n")
    
    logger.info("✓ Unified-All analysis complete")
    return analyzer


def run_unified_semantic():
    """Run unified analysis with semantic methods only."""
    logger.info("\n" + "="*70)
    logger.info("UNIFIED ANALYSIS - SEMANTIC ONLY")
    logger.info("="*70)
    
    analyzer = UnifiedQuranAnalyzer(unified_type='semantic')
    analyzer.load_data()
    analyzer.compute_all_matrices()
    analyzer.compute_unified_matrix()
    analyzer.visualize_unified_matrix(save_path="unified_semantic_similarity_matrix.png")
    
    # Save with specific naming
    os.makedirs("unified_results", exist_ok=True)
    
    # Save unified-semantic matrix
    np.save("unified_results/unified_semantic_matrix.npy", analyzer.unified_matrix)
    df = pd.DataFrame(analyzer.unified_matrix,
                     index=analyzer.surah_names,
                     columns=analyzer.surah_names).round(2)
    df.to_csv("unified_results/unified_semantic_matrix.csv", float_format='%.2f')
    
    # Save analysis results
    analysis = analyzer.analyze_unified_relationships()
    with open("unified_results/unified_semantic_results.txt", 'w', encoding='utf-8') as f:
        f.write("UNIFIED ANALYSIS - SEMANTIC ONLY (Embeddings + AraBERT)\n")
        f.write("="*70 + "\n\n")
        f.write("Weights:\n")
        for method, weight in analyzer.weights.items():
            if weight > 0:
                f.write(f"  • {method.replace('_', ' ').title()}: {weight*100:.1f}%\n")
        f.write(f"\nStatistics:\n")
        stats = analysis['statistics']
        f.write(f"  Mean: {stats['mean_similarity']:.2f}%\n")
        f.write(f"  Range: {stats['min_similarity']:.2f}% - {stats['max_similarity']:.2f}%\n")
    
    logger.info("✓ Unified-Semantic analysis complete")
    return analyzer


def run_sample_pairs_analysis():
    """Run detailed sample pairs analysis with both unified types."""
    logger.info("\n" + "="*70)
    logger.info("SAMPLE PAIRS ANALYSIS")
    logger.info("="*70)
    
    # Load both unified matrices
    unified_all = pd.read_csv('unified_results/unified_all_matrix.csv', index_col=0)
    unified_semantic = pd.read_csv('unified_results/unified_semantic_matrix.csv', index_col=0)
    
    # Load individual method matrices
    kl = pd.read_csv('unified_results/kl_divergence_matrix.csv', index_col=0)
    bigram = pd.read_csv('unified_results/bigram_matrix.csv', index_col=0)
    trigram = pd.read_csv('unified_results/trigram_matrix.csv', index_col=0)
    embeddings = pd.read_csv('unified_results/embeddings_matrix.csv', index_col=0)
    arabert = pd.read_csv('unified_results/arabert_matrix.csv', index_col=0)
    
    # Generate detailed report
    with open("SAMPLE_PAIRS_ANALYSIS.md", 'w', encoding='utf-8') as f:
        f.write("# Detailed Analysis of Sample Surah Pairs\n\n")
        f.write("This document provides comprehensive analysis of specific surah pairs ")
        f.write("across all methods, including **two types of unified scores**:\n\n")
        f.write("1. **Unified-All**: Combines all 5 methods (KL divergence, bigrams, trigrams, embeddings, AraBERT)\n")
        f.write("2. **Unified-Semantic**: Combines only semantic methods (embeddings 70% + AraBERT 30%)\n\n")
        
        f.write("## Analysis Methods\n\n")
        f.write("| Method | Weight in Unified-All | Weight in Unified-Semantic | What It Measures |\n")
        f.write("|--------|---------------------|---------------------------|------------------|\n")
        f.write("| **KL Divergence** | 30% | 0% | Word frequency distributions |\n")
        f.write("| **Bigrams** | 10% | 0% | 2-word phrase patterns |\n")
        f.write("| **Trigrams** | 10% | 0% | 3-word phrase patterns |\n")
        f.write("| **Sentence Embeddings** | 35% | 70% | Deep semantic meaning |\n")
        f.write("| **AraBERT** | 15% | 30% | Arabic-specific contextual embeddings |\n\n")
        
        f.write("## Reading the Results\n\n")
        f.write("For each pair, we show **bidirectional scores** (A→B and B→A separately):\n")
        f.write("- **A→B**: Similarity from Surah A to Surah B\n")
        f.write("- **B→A**: Similarity from Surah B to Surah A\n")
        f.write("- **Asymmetry**: Difference (A→B minus B→A)\n")
        f.write("- **Average**: Mean of both directions\n\n")
        
        f.write("---\n\n")
        
        # Analyze each pair
        for idx, (surah_a, surah_b) in enumerate(SAMPLE_PAIRS, 1):
            name_a = SURAH_NAMES.get(surah_a, f"Surah {surah_a}")
            name_b = SURAH_NAMES.get(surah_b, f"Surah {surah_b}")
            
            f.write(f"## Pair {idx}: Surah {surah_a} ({name_a}) ↔ Surah {surah_b} ({name_b})\n\n")
            
            # Get scores
            sa_label = f"Surah {surah_a}"
            sb_label = f"Surah {surah_b}"
            
            def get_scores(matrix, label_a, label_b):
                forward = float(matrix.loc[label_a, label_b])
                reverse = float(matrix.loc[label_b, label_a])
                return forward, reverse, forward - reverse, (forward + reverse) / 2
            
            # All methods
            scores = {}
            scores['unified_all'] = get_scores(unified_all, sa_label, sb_label)
            scores['unified_semantic'] = get_scores(unified_semantic, sa_label, sb_label)
            scores['kl'] = get_scores(kl, sa_label, sb_label)
            scores['bigram'] = get_scores(bigram, sa_label, sb_label)
            scores['trigram'] = get_scores(trigram, sa_label, sb_label)
            scores['embeddings'] = get_scores(embeddings, sa_label, sb_label)
            scores['arabert'] = get_scores(arabert, sa_label, sb_label)
            
            # Summary table
            f.write("### Summary Table\n\n")
            f.write("| Method | A→B | B→A | Asymmetry | Average | Interpretation |\n")
            f.write("|--------|-----|-----|-----------|---------|----------------|\n")
            
            methods_display = [
                ('unified_all', 'UNIFIED-ALL', True),
                ('unified_semantic', 'UNIFIED-SEMANTIC', True),
                ('kl', 'KL Divergence', False),
                ('bigram', 'Bigrams', False),
                ('trigram', 'Trigrams', False),
                ('embeddings', 'Embeddings', False),
                ('arabert', 'AraBERT', False)
            ]
            
            for method_key, method_name, is_bold in methods_display:
                fwd, rev, asym, avg = scores[method_key]
                
                if abs(asym) < 2:
                    interp = "Symmetric"
                elif asym > 0:
                    interp = f"{name_a} → {name_b}"
                else:
                    interp = f"{name_b} → {name_a}"
                
                if is_bold:
                    f.write(f"| **{method_name}** | **{fwd:.2f}%** | **{rev:.2f}%** | ")
                    f.write(f"**{asym:+.2f}%** | **{avg:.2f}%** | **{interp}** |\n")
                else:
                    f.write(f"| {method_name} | {fwd:.2f}% | {rev:.2f}% | ")
                    f.write(f"{asym:+.2f}% | {avg:.2f}% | {interp} |\n")
            
            # Detailed analysis
            f.write("\n### Detailed Analysis\n\n")
            
            _, _, _, unified_all_avg = scores['unified_all']
            _, _, _, unified_sem_avg = scores['unified_semantic']
            
            f.write(f"**Unified-All Similarity**: {unified_all_avg:.2f}%  \n")
            f.write(f"**Unified-Semantic Similarity**: {unified_sem_avg:.2f}%\n\n")
            
            # Key insight
            diff = unified_sem_avg - unified_all_avg
            if diff > 5:
                f.write(f"**Key Insight**: Semantic similarity ({unified_sem_avg:.2f}%) is ")
                f.write(f"**{diff:.2f}% higher** than overall similarity ({unified_all_avg:.2f}%). ")
                f.write(f"This indicates strong semantic connection despite lower vocabulary/phrase overlap.\n")
            elif diff < -5:
                f.write(f"**Key Insight**: Overall similarity ({unified_all_avg:.2f}%) is higher ")
                f.write(f"than semantic alone ({unified_sem_avg:.2f}%), indicating strong ")
                f.write(f"vocabulary and phrase patterns boost the overall score.\n")
            else:
                f.write(f"**Key Insight**: Semantic and overall similarities are well-aligned ")
                f.write(f"(diff: {diff:.2f}%), indicating balanced similarity across all dimensions.\n")
            
            # Bidirectional analysis
            fwd_all, rev_all, asym_all, _ = scores['unified_all']
            if abs(asym_all) > 3:
                f.write(f"\n**Asymmetry Note**: ")
                if asym_all > 0:
                    f.write(f"Surah {surah_a} shows {abs(asym_all):.2f}% higher similarity toward ")
                    f.write(f"Surah {surah_b} ({fwd_all:.2f}%) than vice versa ({rev_all:.2f}%). ")
                    f.write(f"This suggests Surah {surah_a} may encompass more of Surah {surah_b}'s themes.\n")
                else:
                    f.write(f"Surah {surah_b} shows {abs(asym_all):.2f}% higher similarity toward ")
                    f.write(f"Surah {surah_a} ({rev_all:.2f}%) than vice versa ({fwd_all:.2f}%). ")
                    f.write(f"This suggests Surah {surah_b} may encompass more of Surah {surah_a}'s themes.\n")
            
            f.write("\n---\n\n")
        
        # Summary comparison
        f.write("## Summary Comparison\n\n")
        f.write("### Rankings by Unified-All Score\n\n")
        f.write("| Rank | Pair | Unified-All | Unified-Semantic | Difference |\n")
        f.write("|------|------|-------------|------------------|------------|\n")
        
        # Calculate rankings
        pair_scores = []
        for surah_a, surah_b in SAMPLE_PAIRS:
            sa_label = f"Surah {surah_a}"
            sb_label = f"Surah {surah_b}"
            name_a = SURAH_NAMES.get(surah_a, f"Surah {surah_a}")
            name_b = SURAH_NAMES.get(surah_b, f"Surah {surah_b}")
            
            _, _, _, all_avg = get_scores(unified_all, sa_label, sb_label)
            _, _, _, sem_avg = get_scores(unified_semantic, sa_label, sb_label)
            
            pair_scores.append({
                'pair': f"{surah_a}×{surah_b} ({name_a} ↔ {name_b})",
                'all': all_avg,
                'semantic': sem_avg,
                'diff': sem_avg - all_avg
            })
        
        pair_scores.sort(key=lambda x: x['all'], reverse=True)
        
        for rank, ps in enumerate(pair_scores, 1):
            f.write(f"| {rank} | {ps['pair']} | {ps['all']:.2f}% | {ps['semantic']:.2f}% | ")
            f.write(f"{ps['diff']:+.2f}% |\n")
        
        f.write("\n---\n\n")
        f.write("**Analysis Date**: October 16, 2025  \n")
        f.write("**Sample Pairs**: 6  \n")
        f.write("**Methods**: 7 (5 individual + 2 unified types)  \n")
        f.write("**Score Format**: Bidirectional (A→B and B→A shown separately)  \n")
    
    logger.info("✓ Sample pairs analysis complete")
    
    # Also create CSV
    csv_rows = []
    for surah_a, surah_b in SAMPLE_PAIRS:
        sa_label = f"Surah {surah_a}"
        sb_label = f"Surah {surah_b}"
        name_a = SURAH_NAMES.get(surah_a, f"Surah {surah_a}")
        name_b = SURAH_NAMES.get(surah_b, f"Surah {surah_b}")
        
        for method_name, matrix in [
            ('unified_all', unified_all),
            ('unified_semantic', unified_semantic),
            ('kl_divergence', kl),
            ('bigram', bigram),
            ('trigram', trigram),
            ('embeddings', embeddings),
            ('arabert', arabert)
        ]:
            fwd, rev, asym, avg = get_scores(matrix, sa_label, sb_label)
            csv_rows.append({
                'Surah_A': surah_a,
                'Name_A': name_a,
                'Surah_B': surah_b,
                'Name_B': name_b,
                'Method': method_name,
                'Forward_A_to_B': round(fwd, 2),
                'Reverse_B_to_A': round(rev, 2),
                'Asymmetry': round(asym, 2),
                'Average': round(avg, 2)
            })
    
    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv('sample_pairs_scores.csv', index=False)
    logger.info("✓ Sample pairs CSV saved")


def format_all_matrices():
    """Ensure all CSVs have 2 decimal places."""
    logger.info("\n" + "="*70)
    logger.info("FORMATTING ALL MATRICES")
    logger.info("="*70)
    
    csv_files = [
        'unified_results/unified_all_matrix.csv',
        'unified_results/unified_semantic_matrix.csv',
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
    
    logger.info("✓ All matrices formatted")


def main():
    """Run complete analysis."""
    print("\n" + "="*70)
    print("COMPLETE QURAN SEMANTIC ANALYSIS")
    print("="*70)
    print("\nThis will run:")
    print("  1. Unified-All (all 5 methods)")
    print("  2. Unified-Semantic (embeddings + AraBERT only)")
    print("  3. Sample pairs detailed analysis")
    print("  4. Format all matrices to 2 decimal places")
    print("\n" + "="*70 + "\n")
    
    # Run both unified analyses
    try:
        run_unified_all()
    except Exception as e:
        logger.error(f"Error in unified-all: {e}")
    
    try:
        run_unified_semantic()
    except Exception as e:
        logger.error(f"Error in unified-semantic: {e}")
    
    # Format matrices
    format_all_matrices()
    
    # Run sample pairs
    try:
        run_sample_pairs_analysis()
    except Exception as e:
        logger.error(f"Error in sample pairs: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  Unified-All:")
    print("    - unified_results/unified_all_matrix.csv")
    print("    - unified_results/unified_all_results.txt")
    print("    - unified_all_similarity_matrix.png")
    print("\n  Unified-Semantic:")
    print("    - unified_results/unified_semantic_matrix.csv")
    print("    - unified_results/unified_semantic_results.txt")
    print("    - unified_semantic_similarity_matrix.png")
    print("\n  Sample Pairs:")
    print("    - SAMPLE_PAIRS_ANALYSIS.md (with bidirectional scores)")
    print("    - sample_pairs_scores.csv")
    print("\n  All matrices formatted to 2 decimal places")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

