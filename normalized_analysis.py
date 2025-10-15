#!/usr/bin/env python3
"""
Normalized Quran Analysis with 0-100% Similarity Scores
======================================================

This script converts KL divergence (dissimilarity) to similarity scores
in the range 0-100% for easier interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simple_analysis import SimpleQuranAnalyzer
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NormalizedQuranAnalyzer(SimpleQuranAnalyzer):
    """Analyzer with normalized 0-100% similarity scores."""
    
    def __init__(self):
        super().__init__()
        self.similarity_matrix = None
        self.normalized_matrix = None
    
    def compute_normalized_similarity(self) -> np.ndarray:
        """
        Compute normalized similarity matrix (0-100%).
        
        Converts KL divergence (dissimilarity) to similarity percentage.
        Formula: similarity = 100 * exp(-kl_divergence / scale)
        """
        logger.info("Computing normalized similarity matrix (0-100%)...")
        
        # First compute KL divergence matrix
        kl_matrix = self.compute_asymmetric_matrix()
        
        # Convert KL divergence to similarity
        # Use exponential decay: higher KL = lower similarity
        # Scale factor chosen to map typical KL range to meaningful similarities
        scale_factor = 10.0  # Adjust this to control sensitivity
        
        # Convert to similarity: 100% = identical, 0% = very different
        similarity_matrix = 100 * np.exp(-kl_matrix / scale_factor)
        
        # Set diagonal to 100% (perfect self-similarity)
        np.fill_diagonal(similarity_matrix, 100.0)
        
        self.similarity_matrix = similarity_matrix
        self.normalized_matrix = similarity_matrix
        
        logger.info("✓ Normalized similarity matrix computed")
        return similarity_matrix
    
    def compute_alternative_normalization(self) -> np.ndarray:
        """
        Alternative normalization: Min-Max scaling to 0-100%.
        
        This maps the minimum KL divergence to 100% and maximum to 0%.
        """
        logger.info("Computing alternative normalized similarity (min-max)...")
        
        # Get KL divergence matrix
        kl_matrix = self.compute_asymmetric_matrix()
        
        # Get min and max (excluding diagonal)
        non_diag_mask = ~np.eye(kl_matrix.shape[0], dtype=bool)
        min_kl = np.min(kl_matrix[non_diag_mask])
        max_kl = np.max(kl_matrix[non_diag_mask])
        
        # Min-max normalization: invert so high KL = low similarity
        similarity_matrix = 100 * (1 - (kl_matrix - min_kl) / (max_kl - min_kl))
        
        # Set diagonal to 100%
        np.fill_diagonal(similarity_matrix, 100.0)
        
        logger.info(f"  KL range: {min_kl:.2f} to {max_kl:.2f}")
        logger.info(f"  Mapped to similarity: 100% to 0%")
        
        return similarity_matrix
    
    def analyze_normalized_relationships(self, top_n: int = 10) -> dict:
        """Analyze relationships using normalized similarity scores."""
        if self.similarity_matrix is None:
            self.compute_normalized_similarity()
        
        matrix = self.similarity_matrix
        n_surahs = matrix.shape[0]
        
        # Find most similar pairs (highest similarity)
        relationships = []
        for i in range(n_surahs):
            for j in range(n_surahs):
                if i != j:
                    relationships.append({
                        'source': f"Surah {i+1}",
                        'target': f"Surah {j+1}",
                        'similarity': matrix[i, j],
                        'asymmetry': matrix[i, j] - matrix[j, i]
                    })
        
        # Sort by similarity (highest first)
        relationships.sort(key=lambda x: x['similarity'], reverse=True)
        most_similar = relationships[:top_n]
        
        # Sort by dissimilarity (lowest first)
        relationships.sort(key=lambda x: x['similarity'])
        most_different = relationships[:top_n]
        
        # Most asymmetric
        relationships.sort(key=lambda x: abs(x['asymmetry']), reverse=True)
        most_asymmetric = relationships[:top_n]
        
        # Statistics
        upper_tri = matrix[np.triu_indices(n_surahs, k=1)]
        
        analysis = {
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
        
        return analysis
    
    def visualize_normalized_matrix(self, save_path: str = "normalized_similarity_matrix.png"):
        """Visualize normalized similarity matrix."""
        if self.similarity_matrix is None:
            self.compute_normalized_similarity()
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Similarity matrix
        ax1 = axes[0]
        im1 = ax1.imshow(self.similarity_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        ax1.set_title('Normalized Similarity Matrix (0-100%)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Target Surah', fontsize=12)
        ax1.set_ylabel('Source Surah', fontsize=12)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Similarity %', fontsize=12)
        
        # Add grid
        ax1.set_xticks(np.arange(0, 114, 10))
        ax1.set_yticks(np.arange(0, 114, 10))
        ax1.set_xticklabels(np.arange(1, 115, 10))
        ax1.set_yticklabels(np.arange(1, 115, 10))
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        
        # Plot 2: Asymmetry matrix
        ax2 = axes[1]
        asymmetry = self.similarity_matrix - self.similarity_matrix.T
        im2 = ax2.imshow(asymmetry, cmap='RdBu_r', vmin=-50, vmax=50, aspect='auto')
        ax2.set_title('Asymmetry Matrix (A→B - B→A)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Target Surah', fontsize=12)
        ax2.set_ylabel('Source Surah', fontsize=12)
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Asymmetry (% difference)', fontsize=12)
        
        # Add grid
        ax2.set_xticks(np.arange(0, 114, 10))
        ax2.set_yticks(np.arange(0, 114, 10))
        ax2.set_xticklabels(np.arange(1, 115, 10))
        ax2.set_yticklabels(np.arange(1, 115, 10))
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Visualization saved to {save_path}")
        
        return fig
    
    def save_normalized_results(self, output_dir: str = "normalized_results"):
        """Save normalized results."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.similarity_matrix is None:
            self.compute_normalized_similarity()
        
        # Save matrix
        np.save(os.path.join(output_dir, "similarity_matrix_normalized.npy"), self.similarity_matrix)
        
        # Save as CSV with percentage formatting
        df = pd.DataFrame(self.similarity_matrix,
                         index=self.surah_names,
                         columns=self.surah_names)
        df.to_csv(os.path.join(output_dir, "similarity_matrix_normalized.csv"))
        
        # Save analysis
        analysis = self.analyze_normalized_relationships()
        
        with open(os.path.join(output_dir, "normalized_analysis_results.txt"), 'w', encoding='utf-8') as f:
            f.write("Normalized Quran Semantic Analysis Results (0-100%)\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Top 10 Most Similar Pairs:\n")
            f.write("-" * 60 + "\n")
            for i, rel in enumerate(analysis['most_similar'][:10], 1):
                f.write(f"{i}. {rel['source']} ↔ {rel['target']}: {rel['similarity']:.2f}%\n")
            
            f.write("\nTop 10 Most Different Pairs:\n")
            f.write("-" * 60 + "\n")
            for i, rel in enumerate(analysis['most_different'][:10], 1):
                f.write(f"{i}. {rel['source']} ↔ {rel['target']}: {rel['similarity']:.2f}%\n")
            
            f.write("\nTop 10 Most Asymmetric Relationships:\n")
            f.write("-" * 60 + "\n")
            for i, rel in enumerate(analysis['most_asymmetric'][:10], 1):
                f.write(f"{i}. {rel['source']} → {rel['target']}: ")
                f.write(f"similarity={rel['similarity']:.2f}%, asymmetry={rel['asymmetry']:.2f}%\n")
            
            f.write("\nStatistics:\n")
            f.write("-" * 60 + "\n")
            stats = analysis['statistics']
            f.write(f"Mean Similarity: {stats['mean_similarity']:.2f}%\n")
            f.write(f"Std Deviation: {stats['std_similarity']:.2f}%\n")
            f.write(f"Min Similarity: {stats['min_similarity']:.2f}%\n")
            f.write(f"Max Similarity: {stats['max_similarity']:.2f}%\n")
            f.write(f"Median Similarity: {stats['median_similarity']:.2f}%\n")
        
        logger.info(f"✓ Normalized results saved to {output_dir}/")


def main():
    """Run normalized analysis."""
    logger.info("=" * 70)
    logger.info("NORMALIZED QURAN SEMANTIC ANALYSIS (0-100% Scale)")
    logger.info("=" * 70)
    
    # Initialize analyzer
    analyzer = NormalizedQuranAnalyzer()
    
    # Extract data
    logger.info("\n1. Extracting Quran data...")
    analyzer.extract_quran_data()
    
    # Preprocess
    logger.info("\n2. Preprocessing text...")
    analyzer.preprocess_all_surahs()
    
    # Compute word frequencies
    logger.info("\n3. Computing word frequencies...")
    analyzer.compute_word_frequencies()
    
    # Compute normalized similarity
    logger.info("\n4. Computing normalized similarity matrix...")
    similarity_matrix = analyzer.compute_normalized_similarity()
    
    # Analyze
    logger.info("\n5. Analyzing relationships...")
    analysis = analyzer.analyze_normalized_relationships()
    
    # Visualize
    logger.info("\n6. Creating visualizations...")
    analyzer.visualize_normalized_matrix()
    
    # Save results
    logger.info("\n7. Saving results...")
    analyzer.save_normalized_results()
    
    # Print summary
    print("\n" + "=" * 70)
    print("NORMALIZED ANALYSIS COMPLETE")
    print("=" * 70)
    
    stats = analysis['statistics']
    print(f"\nSimilarity Statistics (0-100% scale):")
    print(f"  Mean Similarity: {stats['mean_similarity']:.2f}%")
    print(f"  Std Deviation: {stats['std_similarity']:.2f}%")
    print(f"  Min Similarity: {stats['min_similarity']:.2f}%")
    print(f"  Max Similarity: {stats['max_similarity']:.2f}%")
    
    print(f"\nTop 5 Most Similar Pairs:")
    for i, rel in enumerate(analysis['most_similar'][:5], 1):
        print(f"  {i}. {rel['source']} ↔ {rel['target']}: {rel['similarity']:.2f}%")
    
    print(f"\nTop 5 Most Different Pairs:")
    for i, rel in enumerate(analysis['most_different'][:5], 1):
        print(f"  {i}. {rel['source']} ↔ {rel['target']}: {rel['similarity']:.2f}%")
    
    print("\n" + "=" * 70)
    print("Results saved to 'normalized_results/' directory")
    print("Visualization saved as 'normalized_similarity_matrix.png'")
    print("=" * 70)


if __name__ == "__main__":
    main()
