#!/usr/bin/env python3
"""
Complete Quran Semantic Analysis - Main Entry Point
====================================================

This is the MAIN script to run the complete analysis.

Usage:
    python scripts/run_complete_analysis.py

Generates:
    - Two unified matrices (All methods & Semantic only)
    - Individual method matrices
    - Sample pairs analysis with thematic explanations
    - All results in results/ directory
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unified_analysis import UnifiedQuranAnalyzer
from enhanced_pairs_analysis import generate_enhanced_analysis, format_matrix_to_2_decimals
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run complete analysis pipeline."""
    print("\n" + "="*70)
    print("QURAN SEMANTIC ANALYSIS - Complete Pipeline")
    print("="*70)
    print("\nThis will generate:")
    print("  1. Unified-All matrix (all 5 methods)")
    print("  2. Unified-Semantic matrix (embeddings + AraBERT)")
    print("  3. Individual method matrices")
    print("  4. Sample pairs analysis (10 pairs)")
    print("  5. Visualizations")
    print("\n" + "="*70 + "\n")
    
    # Run unified-all
    logger.info("Step 1/4: Computing Unified-All matrix...")
    try:
        analyzer_all = UnifiedQuranAnalyzer(unified_type='all')
        analyzer_all.load_data()
        analyzer_all.compute_all_matrices()
        analyzer_all.compute_unified_matrix()
        analyzer_all.visualize_unified_matrix(save_path="results/visualizations/unified_all_heatmap.png")
        
        # Save results
        analyzer_all.save_results("unified_results")
        logger.info("✓ Unified-All complete")
    except Exception as e:
        logger.error(f"Error in Unified-All: {e}")
    
    # Run unified-semantic
    logger.info("\nStep 2/4: Computing Unified-Semantic matrix...")
    try:
        analyzer_sem = UnifiedQuranAnalyzer(unified_type='semantic')
        analyzer_sem.load_data()
        analyzer_sem.compute_all_matrices()
        analyzer_sem.compute_unified_matrix()
        analyzer_sem.visualize_unified_matrix(save_path="results/visualizations/unified_semantic_heatmap.png")
        
        # Save results
        analyzer_sem.save_results("unified_results")
        logger.info("✓ Unified-Semantic complete")
    except Exception as e:
        logger.error(f"Error in Unified-Semantic: {e}")
    
    # Format all matrices
    logger.info("\nStep 3/4: Formatting all matrices to 2 decimals...")
    csv_files = [
        'unified_results/unified_all_matrix.csv',
        'unified_results/unified_semantic_matrix.csv',
        'unified_results/kl_divergence_matrix.csv',
        'unified_results/bigram_matrix.csv',
        'unified_results/trigram_matrix.csv',
        'unified_results/embeddings_matrix.csv',
        'unified_results/arabert_matrix.csv',
    ]
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            format_matrix_to_2_decimals(csv_file)
    logger.info("✓ All matrices formatted")
    
    # Sample pairs analysis
    logger.info("\nStep 4/4: Analyzing sample pairs...")
    try:
        generate_enhanced_analysis()
        logger.info("✓ Sample pairs analysis complete")
    except Exception as e:
        logger.error(f"Error in sample pairs: {e}")
    
    # Move results to organized locations
    logger.info("\nOrganizing results...")
    import shutil
    
    # Move matrices
    os.makedirs("results/matrices", exist_ok=True)
    for matrix in ['unified_all_matrix.csv', 'unified_semantic_matrix.csv', 
                   'kl_divergence_matrix.csv', 'bigram_matrix.csv', 
                   'trigram_matrix.csv', 'embeddings_matrix.csv', 'arabert_matrix.csv']:
        src = f"unified_results/{matrix}"
        dst = f"results/matrices/{matrix}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    
    # Move sample pairs
    os.makedirs("results/sample_pairs", exist_ok=True)
    if os.path.exists("ENHANCED_PAIRS_ANALYSIS.md"):
        shutil.copy("ENHANCED_PAIRS_ANALYSIS.md", "results/sample_pairs/")
    if os.path.exists("enhanced_pairs_scores.csv"):
        shutil.copy("enhanced_pairs_scores.csv", "results/sample_pairs/")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nResults location:")
    print("  📊 Matrices:        results/matrices/")
    print("  📈 Visualizations:  results/visualizations/")
    print("  📝 Sample Pairs:    results/sample_pairs/")
    print("\nKey files:")
    print("  - results/matrices/unified_all_matrix.csv")
    print("  - results/matrices/unified_semantic_matrix.csv")
    print("  - results/sample_pairs/ENHANCED_PAIRS_ANALYSIS.md")
    print("  - results/visualizations/unified_all_heatmap.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
