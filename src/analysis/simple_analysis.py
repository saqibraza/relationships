#!/usr/bin/env python3
"""
Simplified Quran Semantic Analysis without Gensim dependency.
This version uses basic text analysis and statistical methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import rel_entr
from scipy.spatial.distance import cosine
from collections import Counter
import re
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleQuranAnalyzer:
    """
    Simplified Quran analyzer using basic text analysis methods.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.surahs = {}
        self.preprocessed_surahs = {}
        self.word_frequencies = {}
        self.relationship_matrix = None
        self.surah_names = []
        
    def extract_quran_data(self):
        """
        Extract real Quran data from cached file or QuranExtractor.
        """
        import json
        from quran_extractor import QuranExtractor
        
        # Try to load from cached file first
        cached_file = "data/quran_surahs.json"
        if os.path.exists(cached_file):
            try:
                logger.info(f"Loading Quran from cached file: {cached_file}")
                with open(cached_file, 'r', encoding='utf-8') as f:
                    surahs_data = json.load(f)
                    # Convert string keys to integers
                    surahs = {int(k): v for k, v in surahs_data.items()}
                    
                if len(surahs) == 114:
                    self.surahs = surahs
                    self.surah_names = [f"Surah {i}" for i in range(1, 115)]
                    logger.info(f"Successfully loaded {len(surahs)} surahs from cache")
                    return surahs
            except Exception as e:
                logger.warning(f"Could not load from cache: {e}")
        
        # Use QuranExtractor to download and extract
        try:
            logger.info("Extracting Quran using QuranExtractor...")
            extractor = QuranExtractor()
            surahs = extractor.extract_all_surahs()
            
            # Verify extraction
            if extractor.verify_extraction():
                self.surahs = surahs
                self.surah_names = [f"Surah {i}" for i in range(1, 115)]
                logger.info(f"Successfully extracted {len(surahs)} surahs")
                return surahs
            else:
                raise Exception("Quran extraction verification failed")
                
        except Exception as e:
            logger.error(f"Error extracting Quran text: {e}")
            raise Exception("Could not extract Quran text. Please run quran_extractor.py first.")
    
    def preprocess_arabic_text(self, text):
        """Preprocess Arabic text."""
        # Remove diacritics
        diacritics = ['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ', 'ٰ', 'ٱ', 'ۤ', 'ۥ', 'ۦ', 'ۧ', 'ۨ', '۩']
        for diacritic in diacritics:
            text = text.replace(diacritic, '')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common Arabic stop words
        stop_words = {
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل', 'عند', 'لدى',
            'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'التي', 'الذين', 'اللاتي',
            'كان', 'كانت', 'يكون', 'تكون', 'كانوا', 'كن', 'كنت', 'كنتم', 'كنتما',
            'له', 'لها', 'لهما', 'لهم', 'لهن', 'ال', 'و', 'أو', 'لكن', 'إلا', 'إن', 'أن'
        }
        
        # Tokenize and filter
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_all_surahs(self):
        """Preprocess all surahs."""
        if not self.surahs:
            self.extract_quran_data()
        
        preprocessed = {}
        for surah_num, text in self.surahs.items():
            preprocessed[surah_num] = self.preprocess_arabic_text(text)
            logger.info(f"Preprocessed surah {surah_num}")
        
        self.preprocessed_surahs = preprocessed
        return preprocessed
    
    def compute_word_frequencies(self):
        """Compute word frequencies for each surah."""
        if not self.preprocessed_surahs:
            self.preprocess_all_surahs()
        
        word_frequencies = {}
        all_words = set()
        
        # Collect all unique words
        for surah_num, text in self.preprocessed_surahs.items():
            words = text.split()
            all_words.update(words)
        
        # Compute frequencies for each surah
        for surah_num, text in self.preprocessed_surahs.items():
            words = text.split()
            word_count = Counter(words)
            
            # Create frequency vector
            freq_vector = {}
            for word in all_words:
                freq_vector[word] = word_count.get(word, 0)
            
            word_frequencies[surah_num] = freq_vector
        
        self.word_frequencies = word_frequencies
        return word_frequencies
    
    def compute_similarity_matrix(self):
        """Compute similarity matrix using cosine similarity."""
        if not self.word_frequencies:
            self.compute_word_frequencies()
        
        surah_nums = sorted(self.word_frequencies.keys())
        n_surahs = len(surah_nums)
        similarity_matrix = np.zeros((n_surahs, n_surahs))
        
        for i, surah_i in enumerate(surah_nums):
            for j, surah_j in enumerate(surah_nums):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Get frequency vectors
                    freq_i = list(self.word_frequencies[surah_i].values())
                    freq_j = list(self.word_frequencies[surah_j].values())
                    
                    # Compute cosine similarity
                    similarity = 1 - cosine(freq_i, freq_j)
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def compute_asymmetric_matrix(self):
        """Compute asymmetric relationship matrix using KL divergence."""
        if not self.word_frequencies:
            self.compute_word_frequencies()
        
        surah_nums = sorted(self.word_frequencies.keys())
        n_surahs = len(surah_nums)
        matrix = np.zeros((n_surahs, n_surahs))
        
        for i, surah_i in enumerate(surah_nums):
            for j, surah_j in enumerate(surah_nums):
                if i != j:
                    # Get frequency vectors
                    freq_i = np.array(list(self.word_frequencies[surah_i].values()))
                    freq_j = np.array(list(self.word_frequencies[surah_j].values()))
                    
                    # Add small epsilon to avoid log(0)
                    epsilon = 1e-10
                    freq_i = freq_i + epsilon
                    freq_j = freq_j + epsilon
                    
                    # Normalize to probability distributions
                    freq_i = freq_i / np.sum(freq_i)
                    freq_j = freq_j / np.sum(freq_j)
                    
                    # Compute KL divergence
                    kl_div = np.sum(rel_entr(freq_j, freq_i))
                    matrix[i, j] = kl_div
        
        self.relationship_matrix = matrix
        return matrix
    
    def visualize_matrix(self, save_path=None):
        """Visualize the relationship matrix."""
        if self.relationship_matrix is None:
            self.compute_asymmetric_matrix()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Heatmap of the full matrix
        sns.heatmap(self.relationship_matrix, 
                   xticklabels=self.surah_names[::10],  # Show every 10th label
                   yticklabels=self.surah_names[::10],
                   cmap='viridis', ax=ax1, cbar_kws={'label': 'KL Divergence'})
        ax1.set_title('Asymmetric Relationship Matrix (KL Divergence)')
        ax1.set_xlabel('Target Surah')
        ax1.set_ylabel('Source Surah')
        
        # Difference matrix (asymmetry visualization)
        diff_matrix = self.relationship_matrix - self.relationship_matrix.T
        sns.heatmap(diff_matrix,
                   xticklabels=self.surah_names[::10],
                   yticklabels=self.surah_names[::10],
                   cmap='RdBu_r', center=0, ax=ax2,
                   cbar_kws={'label': 'Asymmetry (A→B - B→A)'})
        ax2.set_title('Asymmetry Visualization')
        ax2.set_xlabel('Target Surah')
        ax2.set_ylabel('Source Surah')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matrix visualization saved to {save_path}")
        
        return fig
    
    def analyze_relationships(self, top_n=10):
        """Analyze the most significant relationships."""
        if self.relationship_matrix is None:
            self.compute_asymmetric_matrix()
        
        # Find strongest relationships
        surah_nums = sorted(self.word_frequencies.keys())
        relationships = []
        
        for i in range(len(surah_nums)):
            for j in range(len(surah_nums)):
                if i != j:
                    relationships.append({
                        'source': f"Surah {surah_nums[i]}",
                        'target': f"Surah {surah_nums[j]}",
                        'kl_divergence': self.relationship_matrix[i, j],
                        'asymmetry': self.relationship_matrix[i, j] - self.relationship_matrix[j, i]
                    })
        
        # Sort by KL divergence
        relationships.sort(key=lambda x: x['kl_divergence'], reverse=True)
        top_relationships = relationships[:top_n]
        
        # Find most asymmetric relationships
        asymmetric_relationships = sorted(relationships, 
                                       key=lambda x: abs(x['asymmetry']), 
                                       reverse=True)[:top_n]
        
        analysis = {
            'top_relationships': top_relationships,
            'most_asymmetric': asymmetric_relationships,
            'matrix_stats': {
                'mean_kl_divergence': np.mean(self.relationship_matrix[np.triu_indices(len(surah_nums), k=1)]),
                'std_kl_divergence': np.std(self.relationship_matrix[np.triu_indices(len(surah_nums), k=1)]),
                'max_kl_divergence': np.max(self.relationship_matrix),
                'min_kl_divergence': np.min(self.relationship_matrix[np.triu_indices(len(surah_nums), k=1)])
            }
        }
        
        return analysis
    
    def save_results(self, output_dir="results"):
        """Save all results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save relationship matrix
        if self.relationship_matrix is not None:
            np.save(os.path.join(output_dir, "relationship_matrix.npy"), self.relationship_matrix)
            
            # Save as CSV
            df = pd.DataFrame(self.relationship_matrix, 
                            index=self.surah_names, 
                            columns=self.surah_names)
            df.to_csv(os.path.join(output_dir, "relationship_matrix.csv"))
        
        # Save analysis results
        analysis = self.analyze_relationships()
        with open(os.path.join(output_dir, "analysis_results.txt"), 'w', encoding='utf-8') as f:
            f.write("Quran Semantic Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Top Relationships (by KL Divergence):\n")
            for i, rel in enumerate(analysis['top_relationships'][:10]):
                f.write(f"{i+1}. {rel['source']} → {rel['target']}: {rel['kl_divergence']:.4f}\n")
            
            f.write("\nMost Asymmetric Relationships:\n")
            for i, rel in enumerate(analysis['most_asymmetric'][:10]):
                f.write(f"{i+1}. {rel['source']} → {rel['target']}: asymmetry = {rel['asymmetry']:.4f}\n")
            
            f.write(f"\nMatrix Statistics:\n")
            for key, value in analysis['matrix_stats'].items():
                f.write(f"{key}: {value:.4f}\n")
        
        logger.info(f"Results saved to {output_dir}")


def main():
    """Main execution function."""
    logger.info("Starting Simplified Quran Semantic Analysis")
    
    # Initialize analyzer
    analyzer = SimpleQuranAnalyzer()
    
    # Extract Quran data
    logger.info("Extracting Quran data...")
    surahs = analyzer.extract_quran_data()
    
    # Preprocess text
    logger.info("Preprocessing Arabic text...")
    analyzer.preprocess_all_surahs()
    
    # Compute word frequencies
    logger.info("Computing word frequencies...")
    analyzer.compute_word_frequencies()
    
    # Compute asymmetric matrix
    logger.info("Computing asymmetric relationship matrix...")
    matrix = analyzer.compute_asymmetric_matrix()
    
    # Visualize results
    logger.info("Creating visualizations...")
    fig = analyzer.visualize_matrix(save_path="simple_relationship_matrix.png")
    
    # Analyze relationships
    logger.info("Analyzing relationships...")
    analysis = analyzer.analyze_relationships()
    
    # Save results
    logger.info("Saving results...")
    analyzer.save_results()
    
    # Print summary
    print("\n" + "="*60)
    print("SIMPLIFIED QURAN SEMANTIC ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analyzed {len(surahs)} surahs")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Mean KL divergence: {np.mean(matrix[np.triu_indices(114, k=1)]):.4f}")
    print(f"Max KL divergence: {np.max(matrix):.4f}")
    print("\nTop 5 Relationships:")
    for i, rel in enumerate(analysis['top_relationships'][:5]):
        print(f"{i+1}. {rel['source']} → {rel['target']}: {rel['kl_divergence']:.4f}")
    
    print("\nResults saved to 'results/' directory")
    print("Visualization saved as 'simple_relationship_matrix.png'")


if __name__ == "__main__":
    main()
