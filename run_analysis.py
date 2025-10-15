#!/usr/bin/env python3
"""
Simple script to run the Quran semantic analysis.
This script provides an easy way to run the analysis with different configurations.
"""

import argparse
import sys
import os
from quran_analysis import QuranAnalyzer
from utils import export_analysis_results
from config import ANALYSIS_MODES

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Run Quran Semantic Analysis')
    parser.add_argument('--mode', choices=list(ANALYSIS_MODES.keys()), 
                       default='standard', help='Analysis mode')
    parser.add_argument('--topics', type=int, default=None, 
                       help='Number of topics (overrides mode setting)')
    parser.add_argument('--passes', type=int, default=None, 
                       help='Number of LDA passes (overrides mode setting)')
    parser.add_argument('--output', default='results', 
                       help='Output directory')
    parser.add_argument('--jqurantree-jar', default=None,
                       help='Path to JQuranTree JAR file')
    parser.add_argument('--demo', action='store_true',
                       help='Run with sample data for demonstration')
    parser.add_argument('--advanced', action='store_true',
                       help='Run advanced analysis with additional features')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Quran Semantic Analysis")
    print("=" * 60)
    
    # Get configuration
    config = ANALYSIS_MODES[args.mode].copy()
    if args.topics:
        config['num_topics'] = args.topics
    if args.passes:
        config['lda_passes'] = args.passes
    
    print(f"Analysis mode: {args.mode}")
    print(f"Number of topics: {config['num_topics']}")
    print(f"LDA passes: {config['lda_passes']}")
    print(f"Output directory: {args.output}")
    
    try:
        # Initialize analyzer
        analyzer = QuranAnalyzer(jqurantree_jar_path=args.jqurantree_jar)
        
        if args.demo:
            print("\nRunning with sample data for demonstration...")
            # Use sample data
            analyzer.surahs = analyzer._create_sample_data()
            analyzer.surah_names = [f"Surah {i}" for i in range(1, 115)]
        else:
            print("\nExtracting Quran text...")
            analyzer.extract_quran_text()
        
        print("Preprocessing Arabic text...")
        analyzer.preprocess_all_surahs()
        
        print("Training LDA topic model...")
        analyzer.train_topic_model(
            num_topics=config['num_topics'],
            passes=config['lda_passes']
        )
        
        print("Computing asymmetric relationship matrix...")
        matrix = analyzer.compute_asymmetric_matrix()
        
        print("Creating visualizations...")
        fig = analyzer.visualize_matrix(save_path=os.path.join(args.output, "relationship_matrix.png"))
        
        print("Analyzing relationships...")
        analysis = analyzer.analyze_relationships()
        
        print("Saving results...")
        analyzer.save_results(args.output)
        
        if args.advanced:
            print("Running advanced analysis...")
            export_analysis_results(matrix, analyzer.surah_names, args.output)
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Analyzed {len(analyzer.surahs)} surahs")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Mean KL divergence: {matrix.mean():.4f}")
        print(f"Max KL divergence: {matrix.max():.4f}")
        
        print(f"\nTop 5 Relationships:")
        for i, rel in enumerate(analysis['top_relationships'][:5]):
            print(f"{i+1}. {rel['source']} → {rel['target']}: {rel['kl_divergence']:.4f}")
        
        print(f"\nResults saved to '{args.output}/' directory")
        print("Visualization saved as 'relationship_matrix.png'")
        
        if args.advanced:
            print("Advanced analysis results saved to 'advanced_results/' directory")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("Please check the installation and try again.")
        print("Run 'python test_installation.py' to diagnose issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()
