#!/usr/bin/env python3
"""
Unified Quran Analysis - Combined Multi-Method Relationship Matrix
==================================================================

This script combines all analysis methods into a single unified relationship matrix:
1. KL Divergence (statistical word frequency)
2. Bigram similarity (2-word phrases)
3. Trigram similarity (3-word phrases)
4. Sentence Embeddings (multilingual transformers)
5. AraBERT (Arabic-specific contextual embeddings)
6. Asymmetric STS (verse-level semantic coverage with AraBERT)

The unified matrix is a weighted combination of all methods, normalized to 0-100%.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.analysis.normalized_analysis import NormalizedQuranAnalyzer
from src.analysis.advanced_analysis import AdvancedQuranAnalyzer
from src.analysis.asymmetric_sts import AsymmetricSTSAnalyzer
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedQuranAnalyzer:
    """
    Combines all analysis methods into a unified relationship matrix.
    """
    
    def __init__(self, weights=None, unified_type='all'):
        """
        Initialize with optional custom weights for each method.
        
        Args:
            weights: Dict with keys: 'kl_divergence', 'bigram', 'trigram', 
                    'embeddings', 'arabert', 'asymmetric_sts'. Values should sum to 1.0.
                    If None, uses weights based on unified_type.
            unified_type: 'all' (all 6 methods) or 'semantic' (embeddings+arabert+asym_sts only)
        """
        self.normalized_analyzer = NormalizedQuranAnalyzer()
        self.advanced_analyzer = AdvancedQuranAnalyzer()
        self.asymmetric_sts_analyzer = AsymmetricSTSAnalyzer(model_type='arabert')
        self.unified_type = unified_type
        
        # Default weights based on type
        if weights is None:
            if unified_type == 'semantic':
                # Semantic only: embeddings + arabert + asymmetric_sts
                self.weights = {
                    'kl_divergence': 0.0,
                    'bigram': 0.0,
                    'trigram': 0.0,
                    'embeddings': 0.45,      # 45% - multilingual semantics
                    'arabert': 0.25,         # 25% - Arabic-specific contextual
                    'asymmetric_sts': 0.30   # 30% - verse-level coverage (AraBERT)
                }
            else:  # 'all'
                # All methods (6 total)
                self.weights = {
                    'kl_divergence': 0.25,   # 25% - statistical foundation
                    'bigram': 0.08,          # 8% - phrase patterns
                    'trigram': 0.07,         # 7% - longer phrases
                    'embeddings': 0.25,      # 25% - deep semantics
                    'arabert': 0.15,         # 15% - Arabic-specific
                    'asymmetric_sts': 0.20   # 20% - verse-level (AraBERT)
                }
        else:
            self.weights = weights
            # Normalize weights to sum to 1.0
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        logger.info(f"Initialized with unified_type='{unified_type}', weights: {self.weights}")
        
        self.matrices = {}
        self.unified_matrix = None
        self.unified_semantic_matrix = None
        self.surah_names = [f"Surah {i}" for i in range(1, 115)]
    
    def load_data(self):
        """Load Quran data for all analyzers."""
        logger.info("Loading Quran data...")
        self.normalized_analyzer.extract_quran_data()
        self.advanced_analyzer.load_quran_data()
        self.asymmetric_sts_analyzer.load_data()
        logger.info("✓ Data loaded for all analyzers")
    
    def compute_all_matrices(self):
        """Compute all similarity matrices using different methods."""
        logger.info("\n" + "="*70)
        logger.info("COMPUTING ALL SIMILARITY MATRICES")
        logger.info("="*70)
        
        # 1. KL Divergence (normalized)
        logger.info("\n1. Computing KL Divergence similarity...")
        self.normalized_analyzer.preprocess_all_surahs()
        self.normalized_analyzer.compute_word_frequencies()
        kl_matrix = self.normalized_analyzer.compute_normalized_similarity()
        self.matrices['kl_divergence'] = kl_matrix
        logger.info(f"   ✓ KL Divergence: mean={np.mean(kl_matrix):.2f}%")
        
        # 2. Bigram similarity - try to load from cache first
        logger.info("\n2. Computing Bigram similarity...")
        bigram_cache = "advanced_results/2gram_matrix.npy"
        if os.path.exists(bigram_cache):
            bigram_matrix = np.load(bigram_cache)
            logger.info(f"   Loaded from cache: {bigram_cache}")
        else:
            bigram_matrix = self.advanced_analyzer.compute_ngram_similarity(n=2)
        # Convert to percentage and scale to 0-100
        bigram_matrix_pct = bigram_matrix * 100
        self.matrices['bigram'] = bigram_matrix_pct
        logger.info(f"   ✓ Bigram: mean={np.mean(bigram_matrix_pct):.2f}%")
        
        # 3. Trigram similarity - try to load from cache first
        logger.info("\n3. Computing Trigram similarity...")
        trigram_cache = "advanced_results/3gram_matrix.npy"
        if os.path.exists(trigram_cache):
            trigram_matrix = np.load(trigram_cache)
            logger.info(f"   Loaded from cache: {trigram_cache}")
        else:
            trigram_matrix = self.advanced_analyzer.compute_ngram_similarity(n=3)
        # Convert to percentage
        trigram_matrix_pct = trigram_matrix * 100
        self.matrices['trigram'] = trigram_matrix_pct
        logger.info(f"   ✓ Trigram: mean={np.mean(trigram_matrix_pct):.2f}%")
        
        # 4. Sentence Embeddings (multilingual) - try to load from cache first
        logger.info("\n4. Computing Sentence Embeddings...")
        embedding_cache = "advanced_results/multilingual_embedding_matrix.npy"
        if os.path.exists(embedding_cache):
            embedding_matrix = np.load(embedding_cache)
            logger.info(f"   Loaded from cache: {embedding_cache}")
        else:
            try:
                embedding_matrix = self.advanced_analyzer.compute_embedding_similarity('multilingual')
            except Exception as e:
                logger.error(f"   ✗ Could not compute embeddings: {e}")
                logger.warning(f"   Using KL divergence as fallback for embeddings")
                embedding_matrix = kl_matrix / 100.0  # Back to 0-1 scale
                self.weights['kl_divergence'] += self.weights['embeddings']
                self.weights['embeddings'] = 0.0
        # Convert to percentage
        embedding_matrix_pct = embedding_matrix * 100
        self.matrices['embeddings'] = embedding_matrix_pct
        logger.info(f"   ✓ Embeddings: mean={np.mean(embedding_matrix_pct):.2f}%")
        
        # 5. AraBERT
        logger.info("\n5. Computing AraBERT similarity...")
        try:
            arabert_matrix = self.advanced_analyzer.compute_embedding_similarity('arabert')
            # Convert to percentage
            arabert_matrix_pct = arabert_matrix * 100
            self.matrices['arabert'] = arabert_matrix_pct
            logger.info(f"   ✓ AraBERT: mean={np.mean(arabert_matrix_pct):.2f}%")
        except Exception as e:
            logger.warning(f"   ⚠ AraBERT computation failed: {e}")
            logger.warning(f"   Using embeddings as fallback for AraBERT")
            self.matrices['arabert'] = embedding_matrix_pct
            self.weights['embeddings'] += self.weights['arabert']
            self.weights['arabert'] = 0.0
        
        # 6. Asymmetric STS (verse-level with AraBERT)
        logger.info("\n6. Computing Asymmetric STS similarity...")
        asymsts_cache = "results/matrices/asymmetric_sts_arabert_similarity_matrix.csv"
        if os.path.exists(asymsts_cache):
            logger.info(f"   Loading from cache: {asymsts_cache}")
            df = pd.read_csv(asymsts_cache, index_col=0)
            asymsts_matrix = df.values
        else:
            logger.info("   Computing Asym-STS (this may take 30-45 minutes)...")
            try:
                self.asymmetric_sts_analyzer.load_model()
                asymsts_matrix = self.asymmetric_sts_analyzer.compute_asymmetric_sts_matrix()
                logger.info(f"   Saving to cache: {asymsts_cache}")
                # Save for future use
                os.makedirs("results/matrices", exist_ok=True)
                df = pd.DataFrame(asymsts_matrix, 
                                index=[f"Surah {i}" for i in range(1, 115)],
                                columns=[f"Surah {i}" for i in range(1, 115)])
                df.to_csv(asymsts_cache)
            except Exception as e:
                logger.warning(f"   ⚠ Asym-STS computation failed: {e}")
                logger.warning(f"   Using AraBERT as fallback for Asym-STS")
                asymsts_matrix = arabert_matrix_pct
                self.weights['arabert'] += self.weights['asymmetric_sts']
                self.weights['asymmetric_sts'] = 0.0
        
        self.matrices['asymmetric_sts'] = asymsts_matrix
        logger.info(f"   ✓ Asym-STS: mean={np.mean(asymsts_matrix):.2f}%")
        
        logger.info("\n✓ All 6 matrices computed successfully")
        return self.matrices
    
    def compute_unified_matrix(self):
        """
        Combine all matrices into a unified similarity matrix.
        Uses weighted average of all methods.
        """
        logger.info("\n" + "="*70)
        logger.info("COMPUTING UNIFIED MATRIX")
        logger.info("="*70)
        
        if not self.matrices:
            raise ValueError("Matrices not computed. Call compute_all_matrices() first.")
        
        # Initialize unified matrix
        n = 114  # Number of surahs
        unified = np.zeros((n, n))
        
        # Weighted combination
        for method, weight in self.weights.items():
            if weight > 0 and method in self.matrices:
                matrix = self.matrices[method]
                unified += weight * matrix
                logger.info(f"  Adding {method}: weight={weight:.2f} (mean={np.mean(matrix):.2f}%)")
        
        # Set diagonal to 100% (perfect self-similarity)
        np.fill_diagonal(unified, 100.0)
        
        self.unified_matrix = unified
        
        # Statistics
        upper_tri = unified[np.triu_indices(n, k=1)]
        logger.info("\n" + "="*70)
        logger.info("UNIFIED MATRIX STATISTICS")
        logger.info("="*70)
        logger.info(f"Mean Similarity: {np.mean(upper_tri):.2f}%")
        logger.info(f"Std Deviation: {np.std(upper_tri):.2f}%")
        logger.info(f"Min Similarity: {np.min(upper_tri):.2f}%")
        logger.info(f"Max Similarity: {np.max(upper_tri):.2f}%")
        logger.info(f"Median Similarity: {np.median(upper_tri):.2f}%")
        
        return unified
    
    def analyze_unified_relationships(self, top_n=10):
        """Analyze relationships in the unified matrix."""
        if self.unified_matrix is None:
            raise ValueError("Unified matrix not computed. Call compute_unified_matrix() first.")
        
        matrix = self.unified_matrix
        n_surahs = matrix.shape[0]
        
        # Find most similar pairs
        relationships = []
        for i in range(n_surahs):
            for j in range(n_surahs):
                if i != j:
                    relationships.append({
                        'source': i + 1,
                        'target': j + 1,
                        'source_name': f"Surah {i+1}",
                        'target_name': f"Surah {j+1}",
                        'similarity': matrix[i, j],
                        'asymmetry': matrix[i, j] - matrix[j, i]
                    })
        
        # Sort by similarity
        relationships.sort(key=lambda x: x['similarity'], reverse=True)
        most_similar = relationships[:top_n]
        
        # Sort by dissimilarity
        relationships.sort(key=lambda x: x['similarity'])
        most_different = relationships[:top_n]
        
        # Sort by asymmetry
        relationships.sort(key=lambda x: abs(x['asymmetry']), reverse=True)
        most_asymmetric = relationships[:top_n]
        
        # Statistics
        upper_tri = matrix[np.triu_indices(n_surahs, k=1)]
        
        return {
            'most_similar': most_similar,
            'most_different': most_different,
            'most_asymmetric': most_asymmetric,
            'statistics': {
                'mean_similarity': np.mean(upper_tri),
                'std_similarity': np.std(upper_tri),
                'min_similarity': np.min(upper_tri),
                'max_similarity': np.max(upper_tri),
                'median_similarity': np.median(upper_tri)
            }
        }
    
    def visualize_unified_matrix(self, save_path="unified_similarity_matrix.png"):
        """Create comprehensive visualization of the unified matrix."""
        if self.unified_matrix is None:
            raise ValueError("Unified matrix not computed.")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main unified matrix
        ax_main = fig.add_subplot(gs[:2, :2])
        im_main = ax_main.imshow(self.unified_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        ax_main.set_title('Unified Multi-Method Similarity Matrix (0-100%)', 
                         fontsize=18, fontweight='bold', pad=20)
        ax_main.set_xlabel('Target Surah', fontsize=14)
        ax_main.set_ylabel('Source Surah', fontsize=14)
        ax_main.set_xticks(np.arange(0, 114, 10))
        ax_main.set_yticks(np.arange(0, 114, 10))
        ax_main.set_xticklabels(np.arange(1, 115, 10))
        ax_main.set_yticklabels(np.arange(1, 115, 10))
        ax_main.grid(True, alpha=0.3, linewidth=0.5)
        cbar_main = plt.colorbar(im_main, ax=ax_main)
        cbar_main.set_label('Similarity %', fontsize=12)
        
        # Asymmetry matrix
        ax_asym = fig.add_subplot(gs[:2, 2])
        asymmetry = self.unified_matrix - self.unified_matrix.T
        im_asym = ax_asym.imshow(asymmetry, cmap='RdBu_r', vmin=-30, vmax=30, aspect='auto')
        ax_asym.set_title('Asymmetry Matrix', fontsize=14, fontweight='bold')
        ax_asym.set_xlabel('Target Surah', fontsize=12)
        ax_asym.set_ylabel('Source Surah', fontsize=12)
        ax_asym.set_xticks(np.arange(0, 114, 20))
        ax_asym.set_yticks(np.arange(0, 114, 20))
        ax_asym.set_xticklabels(np.arange(1, 115, 20))
        ax_asym.set_yticklabels(np.arange(1, 115, 20))
        ax_asym.grid(True, alpha=0.3, linewidth=0.5)
        cbar_asym = plt.colorbar(im_asym, ax=ax_asym)
        cbar_asym.set_label('Asymmetry', fontsize=10)
        
        # Distribution histogram
        ax_hist = fig.add_subplot(gs[2, 0])
        upper_tri = self.unified_matrix[np.triu_indices(114, k=1)]
        ax_hist.hist(upper_tri, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax_hist.set_title('Similarity Distribution', fontsize=12, fontweight='bold')
        ax_hist.set_xlabel('Similarity %', fontsize=10)
        ax_hist.set_ylabel('Frequency', fontsize=10)
        ax_hist.axvline(np.mean(upper_tri), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(upper_tri):.1f}%')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Method weights
        ax_weights = fig.add_subplot(gs[2, 1])
        methods = [k.replace('_', ' ').title() for k in self.weights.keys()]
        weights = [v*100 for v in self.weights.values()]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        ax_weights.bar(range(len(methods)), weights, color=colors[:len(methods)], alpha=0.7, edgecolor='black')
        ax_weights.set_title('Method Weights', fontsize=12, fontweight='bold')
        ax_weights.set_ylabel('Weight %', fontsize=10)
        ax_weights.set_xticks(range(len(methods)))
        ax_weights.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax_weights.grid(True, alpha=0.3, axis='y')
        
        # Statistics table
        ax_stats = fig.add_subplot(gs[2, 2])
        ax_stats.axis('off')
        stats_text = f"""
        UNIFIED MATRIX STATISTICS
        {'─'*35}
        Mean Similarity:     {np.mean(upper_tri):.2f}%
        Std Deviation:       {np.std(upper_tri):.2f}%
        Min Similarity:      {np.min(upper_tri):.2f}%
        Max Similarity:      {np.max(upper_tri):.2f}%
        Median:              {np.median(upper_tri):.2f}%
        
        METHODS COMBINED
        {'─'*35}
        • KL Divergence      {self.weights['kl_divergence']*100:.0f}%
        • Bigrams            {self.weights['bigram']*100:.0f}%
        • Trigrams           {self.weights['trigram']*100:.0f}%
        • Embeddings         {self.weights['embeddings']*100:.0f}%
        • AraBERT            {self.weights['arabert']*100:.0f}%
        """
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Unified visualization saved to {save_path}")
        
        return fig
    
    def compare_methods(self):
        """Create visualization comparing all individual methods."""
        if not self.matrices:
            raise ValueError("Matrices not computed.")
        
        n_methods = len([k for k, v in self.weights.items() if v > 0])
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()
        
        method_names = {
            'kl_divergence': 'KL Divergence',
            'bigram': 'Bigram Similarity',
            'trigram': 'Trigram Similarity',
            'embeddings': 'Sentence Embeddings',
            'arabert': 'AraBERT'
        }
        
        idx = 0
        for method, matrix in self.matrices.items():
            if self.weights[method] > 0:
                ax = axes[idx]
                im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=100, aspect='auto')
                ax.set_title(f'{method_names[method]}\n(Weight: {self.weights[method]*100:.0f}%)',
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Target Surah', fontsize=10)
                ax.set_ylabel('Source Surah', fontsize=10)
                ax.set_xticks(np.arange(0, 114, 20))
                ax.set_yticks(np.arange(0, 114, 20))
                ax.set_xticklabels(np.arange(1, 115, 20))
                ax.set_yticklabels(np.arange(1, 115, 20))
                plt.colorbar(im, ax=ax, label='Similarity %')
                idx += 1
        
        # Add unified matrix in last subplot
        ax = axes[idx]
        im = ax.imshow(self.unified_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        ax.set_title('UNIFIED (Weighted Average)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Surah', fontsize=10)
        ax.set_ylabel('Source Surah', fontsize=10)
        ax.set_xticks(np.arange(0, 114, 20))
        ax.set_yticks(np.arange(0, 114, 20))
        ax.set_xticklabels(np.arange(1, 115, 20))
        ax.set_yticklabels(np.arange(1, 115, 20))
        plt.colorbar(im, ax=ax, label='Similarity %')
        
        plt.tight_layout()
        plt.savefig('method_comparison_all.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Method comparison saved to method_comparison_all.png")
        
        return fig
    
    def save_results(self, output_dir="unified_results"):
        """Save all unified analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save unified matrix
        np.save(os.path.join(output_dir, "unified_similarity_matrix.npy"), self.unified_matrix)
        
        # Save as CSV
        df = pd.DataFrame(self.unified_matrix, 
                         index=self.surah_names,
                         columns=self.surah_names)
        df.to_csv(os.path.join(output_dir, "unified_similarity_matrix.csv"))
        
        # Save individual matrices
        for method, matrix in self.matrices.items():
            df_method = pd.DataFrame(matrix,
                                    index=self.surah_names,
                                    columns=self.surah_names)
            df_method.to_csv(os.path.join(output_dir, f"{method}_matrix.csv"))
        
        # Save analysis
        analysis = self.analyze_unified_relationships()
        
        with open(os.path.join(output_dir, "unified_analysis_results.txt"), 'w', encoding='utf-8') as f:
            f.write("UNIFIED MULTI-METHOD QURAN SEMANTIC ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            f.write("Methods Combined:\n")
            f.write("-"*70 + "\n")
            for method, weight in self.weights.items():
                f.write(f"  • {method.replace('_', ' ').title()}: {weight*100:.1f}%\n")
            
            f.write("\n\nStatistics:\n")
            f.write("-"*70 + "\n")
            stats = analysis['statistics']
            f.write(f"Mean Similarity: {stats['mean_similarity']:.2f}%\n")
            f.write(f"Std Deviation: {stats['std_similarity']:.2f}%\n")
            f.write(f"Min Similarity: {stats['min_similarity']:.2f}%\n")
            f.write(f"Max Similarity: {stats['max_similarity']:.2f}%\n")
            f.write(f"Median Similarity: {stats['median_similarity']:.2f}%\n")
            
            f.write("\n\nTop 10 Most Similar Pairs:\n")
            f.write("-"*70 + "\n")
            for i, rel in enumerate(analysis['most_similar'][:10], 1):
                f.write(f"{i}. {rel['source_name']} ↔ {rel['target_name']}: {rel['similarity']:.2f}%\n")
            
            f.write("\n\nTop 10 Most Different Pairs:\n")
            f.write("-"*70 + "\n")
            for i, rel in enumerate(analysis['most_different'][:10], 1):
                f.write(f"{i}. {rel['source_name']} ↔ {rel['target_name']}: {rel['similarity']:.2f}%\n")
            
            f.write("\n\nTop 10 Most Asymmetric Relationships:\n")
            f.write("-"*70 + "\n")
            for i, rel in enumerate(analysis['most_asymmetric'][:10], 1):
                f.write(f"{i}. {rel['source_name']} → {rel['target_name']}: ")
                f.write(f"similarity={rel['similarity']:.2f}%, asymmetry={rel['asymmetry']:.2f}%\n")
        
        # Save configuration
        config = {
            'weights': self.weights,
            'methods_used': list(self.matrices.keys()),
            'statistics': {k: float(v) for k, v in analysis['statistics'].items()}
        }
        
        with open(os.path.join(output_dir, "unified_config.json"), 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"✓ All results saved to {output_dir}/")


def main():
    """Run unified multi-method analysis."""
    logger.info("="*70)
    logger.info("UNIFIED MULTI-METHOD QURAN SEMANTIC ANALYSIS")
    logger.info("="*70)
    
    # Initialize with custom weights (optional)
    # You can adjust these weights based on your preferences
    custom_weights = {
        'kl_divergence': 0.30,  # Statistical foundation
        'bigram': 0.10,          # Local phrase patterns
        'trigram': 0.10,         # Longer phrases
        'embeddings': 0.35,      # Deep semantic meaning (increased)
        'arabert': 0.15          # Arabic-specific context
    }
    
    analyzer = UnifiedQuranAnalyzer(weights=custom_weights)
    
    # 1. Load data
    logger.info("\n1. Loading Quran data...")
    analyzer.load_data()
    
    # 2. Compute all matrices
    logger.info("\n2. Computing all similarity matrices...")
    analyzer.compute_all_matrices()
    
    # 3. Compute unified matrix
    logger.info("\n3. Computing unified matrix...")
    unified_matrix = analyzer.compute_unified_matrix()
    
    # 4. Analyze relationships
    logger.info("\n4. Analyzing relationships...")
    analysis = analyzer.analyze_unified_relationships()
    
    # 5. Visualize
    logger.info("\n5. Creating visualizations...")
    analyzer.visualize_unified_matrix()
    analyzer.compare_methods()
    
    # 6. Save results
    logger.info("\n6. Saving results...")
    analyzer.save_results()
    
    # Print summary
    print("\n" + "="*70)
    print("UNIFIED ANALYSIS COMPLETE")
    print("="*70)
    
    stats = analysis['statistics']
    print(f"\nUnified Similarity Statistics:")
    print(f"  Mean: {stats['mean_similarity']:.2f}%")
    print(f"  Range: {stats['min_similarity']:.2f}% - {stats['max_similarity']:.2f}%")
    print(f"  Std Dev: {stats['std_similarity']:.2f}%")
    
    print(f"\nTop 5 Most Similar Pairs (Unified):")
    for i, rel in enumerate(analysis['most_similar'][:5], 1):
        print(f"  {i}. {rel['source_name']} ↔ {rel['target_name']}: {rel['similarity']:.2f}%")
    
    print(f"\nTop 5 Most Different Pairs (Unified):")
    for i, rel in enumerate(analysis['most_different'][:5], 1):
        print(f"  {i}. {rel['source_name']} ↔ {rel['target_name']}: {rel['similarity']:.2f}%")
    
    print("\n" + "="*70)
    print("Results saved to 'unified_results/' directory")
    print("Visualizations:")
    print("  - unified_similarity_matrix.png")
    print("  - method_comparison_all.png")
    print("="*70)


if __name__ == "__main__":
    main()

