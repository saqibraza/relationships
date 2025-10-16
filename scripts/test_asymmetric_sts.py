#!/usr/bin/env python3
"""
Test Script for Asymmetric STS Implementation
==============================================

Quick test of the Asym-STS method on a few sample pairs
to verify the implementation before running full analysis.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.analysis.asymmetric_sts import AsymmetricSTSAnalyzer
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║        ASYMMETRIC STS - QUICK TEST (AraBERT)                   ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝\n")
    
    # Initialize analyzer with AraBERT
    logger.info("Initializing Asym-STS analyzer with AraBERT...")
    analyzer = AsymmetricSTSAnalyzer(model_type='arabert')
    
    # Load data
    logger.info("Loading Quran data...")
    analyzer.load_data()
    
    # Load model
    logger.info("Loading sentence transformer model...")
    analyzer.load_model()
    
    logger.info("\n" + "="*70)
    logger.info("TESTING ON SAMPLE PAIRS")
    logger.info("="*70 + "\n")
    
    # Test pairs
    test_pairs = [
        (113, 114, "Al-Falaq ↔ An-Nas", "Similar short protection surahs"),
        (2, 65, "Al-Baqarah ↔ At-Talaq", "Long comprehensive vs short focused"),
        (10, 11, "Yunus ↔ Hud", "Similar prophetic narratives"),
        (93, 94, "Ad-Duha ↔ Ash-Sharh", "Complementary consolation surahs"),
    ]
    
    for surah_a, surah_b, names, description in test_pairs:
        logger.info(f"📖 {names}")
        logger.info(f"   Description: {description}")
        
        # Analyze pair
        result = analyzer.analyze_pair(surah_a, surah_b)
        
        # Display results
        logger.info(f"\n   Sentence counts: {result['n_sentences_a']} vs {result['n_sentences_b']}")
        logger.info(f"   {surah_a}→{surah_b}: {result['a_to_b']:.2f}%")
        logger.info(f"   {surah_b}→{surah_a}: {result['b_to_a']:.2f}%")
        logger.info(f"   Asymmetry: {result['asymmetry']:.2f}%")
        logger.info(f"   Average: {result['average']:.2f}%")
        logger.info(f"\n   💡 {result['interpretation']}")
        logger.info("\n" + "-"*70 + "\n")
    
    # Validation tests
    logger.info("="*70)
    logger.info("VALIDATION TESTS")
    logger.info("="*70 + "\n")
    
    # Test 1: Self-similarity
    logger.info("Test 1: Self-similarity (should be ~100%)")
    result = analyzer.analyze_pair(1, 1)
    logger.info(f"   Surah 1→1: {result['a_to_b']:.2f}%")
    if result['a_to_b'] > 99.5:
        logger.info("   ✅ PASS: Self-similarity is high")
    else:
        logger.info("   ❌ FAIL: Self-similarity should be ~100%")
    
    # Test 2: Asymmetry for long vs short
    logger.info("\nTest 2: Asymmetry for long vs short (expect 65→2 > 2→65)")
    result = analyzer.analyze_pair(2, 65)
    logger.info(f"   2→65: {result['a_to_b']:.2f}%")
    logger.info(f"   65→2: {result['b_to_a']:.2f}%")
    if result['b_to_a'] > result['a_to_b']:
        logger.info(f"   ✅ PASS: 65→2 is {result['b_to_a'] - result['a_to_b']:.1f}% higher")
    else:
        logger.info("   ❌ FAIL: Expected 65→2 > 2→65")
    
    # Test 3: Symmetry for similar surahs
    logger.info("\nTest 3: Symmetry for similar surahs (113↔114 should be symmetric)")
    result = analyzer.analyze_pair(113, 114)
    logger.info(f"   113→114: {result['a_to_b']:.2f}%")
    logger.info(f"   114→113: {result['b_to_a']:.2f}%")
    logger.info(f"   Asymmetry: {result['asymmetry']:.2f}%")
    if result['asymmetry'] < 10:
        logger.info(f"   ✅ PASS: Asymmetry is low ({result['asymmetry']:.2f}%)")
    else:
        logger.info(f"   ⚠️  WARNING: Asymmetry higher than expected ({result['asymmetry']:.2f}%)")
    
    logger.info("\n" + "="*70)
    logger.info("✓ QUICK TEST COMPLETE!")
    logger.info("="*70)
    logger.info("\nTo run full analysis:")
    logger.info("  python src/analysis/asymmetric_sts.py")
    logger.info("\nOr add to unified analysis:")
    logger.info("  python scripts/run_complete_analysis.py --include-asym-sts")


if __name__ == '__main__':
    main()

