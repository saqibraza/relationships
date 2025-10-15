"""
Quran Semantic Analysis: Asymmetric Relationship Matrix
=====================================================

This module implements a comprehensive analysis of the Quranic corpus to determine
semantic and thematic relationships between surahs using advanced NLP techniques.

The solution uses:
- JQuranTree library for Arabic text extraction
- CAMeL Tools for Arabic preprocessing
- LDA topic modeling for thematic analysis
- KL divergence for asymmetric relationship computation
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
import jpype
import jpype.imports
from gensim import corpora, models
from gensim.models import LdaModel
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.normalize import normalize_whitespace
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import MorphologyAnalyzer
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.stemmers.arabic_stemmer import ArabicStemmer
import arabic_reshaper
from bidi.algorithm import get_display
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuranAnalyzer:
    """
    Main class for analyzing Quranic surahs and computing asymmetric relationships.
    """
    
    def __init__(self, jqurantree_jar_path: str = None):
        """
        Initialize the Quran analyzer.
        
        Args:
            jqurantree_jar_path: Path to JQuranTree JAR file
        """
        self.jqurantree_jar_path = jqurantree_jar_path
        self.surahs = {}
        self.preprocessed_surahs = {}
        self.topic_model = None
        self.topic_distributions = {}
        self.relationship_matrix = None
        self.surah_names = []
        
        # Initialize Arabic NLP tools
        self._setup_arabic_nlp()
        
    def _setup_arabic_nlp(self):
        """Initialize Arabic NLP tools."""
        try:
            # Initialize CAMeL Tools morphology database
            self.morphology_db = MorphologyDB.pretrained()
            self.morphology_analyzer = MorphologyAnalyzer(self.morphology_db)
            self.arabic_stemmer = ArabicStemmer()
            logger.info("Arabic NLP tools initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Arabic NLP tools: {e}")
            self.morphology_db = None
            self.morphology_analyzer = None
            self.arabic_stemmer = None
    
    def _setup_jqurantree(self):
        """Initialize JQuranTree library."""
        if not jpype.isJVMStarted():
            if self.jqurantree_jar_path and os.path.exists(self.jqurantree_jar_path):
                jpype.startJVM(classpath=[self.jqurantree_jar_path])
            else:
                # Try to find JQuranTree in common locations
                possible_paths = [
                    "jqurantree.jar",
                    "lib/jqurantree.jar",
                    "/usr/local/lib/jqurantree.jar"
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        jpype.startJVM(classpath=[path])
                        break
                else:
                    raise FileNotFoundError("JQuranTree JAR file not found. Please provide the path.")
        
        # Import JQuranTree classes
        from org.jqurantree.analysis import AnalysisTable
        from org.jqurantree.analysis import AnalysisTableReader
        from org.jqurantree.analysis import AnalysisTableWriter
        from org.jqurantree.analysis import ArabicText
        from org.jqurantree.analysis import DataTable
        from org.jqurantree.analysis import DataTableReader
        from org.jqurantree.analysis import DataTableWriter
        from org.jqurantree.analysis import EnglishText
        from org.jqurantree.analysis import Transliteration
        from org.jqurantree.analysis import TransliterationTable
        from org.jqurantree.analysis import TransliterationTableReader
        from org.jqurantree.analysis import TransliterationTableWriter
        from org.jqurantree.analysis import UnicodeText
        from org.jqurantree.analysis import UnicodeTextReader
        from org.jqurantree.analysis import UnicodeTextWriter
        from org.jqurantree.analysis import Verse
        from org.jqurantree.analysis import VerseTable
        from org.jqurantree.analysis import VerseTableReader
        from org.jqurantree.analysis import VerseTableWriter
        from org.jqurantree.analysis import Word
        from org.jqurantree.analysis import WordTable
        from org.jqurantree.analysis import WordTableReader
        from org.jqurantree.analysis import WordTableWriter
        from org.jqurantree.analysis import ArabicText
        from org.jqurantree.analysis import ArabicTextReader
        from org.jqurantree.analysis import ArabicTextWriter
        
        logger.info("JQuranTree library initialized successfully")
    
    def extract_quran_text(self) -> Dict[int, str]:
        """
        Extract Arabic text for all 114 surahs.
        First tries to load from cached file, then from QuranExtractor.
        
        Returns:
            Dictionary mapping surah numbers to their Arabic text
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
    
    
    def preprocess_arabic_text(self, text: str) -> str:
        """
        Preprocess Arabic text for analysis.
        
        Args:
            text: Raw Arabic text
            
        Returns:
            Preprocessed text
        """
        try:
            # Step 1: Normalize Unicode
            text = normalize_unicode(text)
            
            # Step 2: Normalize Alef variations
            text = normalize_alef_maksura_ar(text)
            text = normalize_teh_marbuta_ar(text)
            
            # Step 3: Normalize whitespace
            text = normalize_whitespace(text)
            
            # Step 4: Remove diacritics (tashkeel)
            text = self._remove_diacritics(text)
            
            # Step 5: Tokenize
            tokens = simple_word_tokenize(text)
            
            # Step 6: Remove stop words
            tokens = self._remove_arabic_stopwords(tokens)
            
            # Step 7: Stemming/Lemmatization
            if self.arabic_stemmer:
                tokens = [self.arabic_stemmer.stem(token) for token in tokens if token.strip()]
            
            # Step 8: Filter out short tokens
            tokens = [token for token in tokens if len(token) > 2]
            
            return ' '.join(tokens)
            
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return text
    
    def _remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics (tashkeel) from text."""
        diacritics = ['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ', 'ٰ', 'ٱ', 'ۤ', 'ۥ', 'ۦ', 'ۧ', 'ۨ', '۩']
        for diacritic in diacritics:
            text = text.replace(diacritic, '')
        return text
    
    def _remove_arabic_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove Arabic stop words."""
        arabic_stopwords = {
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل', 'عند', 'لدى',
            'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'التي', 'الذين', 'اللاتي',
            'كان', 'كانت', 'يكون', 'تكون', 'كانوا', 'كن', 'كنت', 'كنتم', 'كنتما',
            'له', 'لها', 'لهما', 'لهم', 'لهن', 'له', 'لها', 'لهما', 'لهم', 'لهن',
            'ال', 'و', 'أو', 'لكن', 'إلا', 'إن', 'أن', 'أن', 'أن', 'أن', 'أن'
        }
        return [token for token in tokens if token not in arabic_stopwords]
    
    def preprocess_all_surahs(self) -> Dict[int, str]:
        """
        Preprocess all surahs.
        
        Returns:
            Dictionary of preprocessed surah texts
        """
        if not self.surahs:
            self.extract_quran_text()
        
        preprocessed = {}
        for surah_num, text in self.surahs.items():
            preprocessed[surah_num] = self.preprocess_arabic_text(text)
            logger.info(f"Preprocessed surah {surah_num}")
        
        self.preprocessed_surahs = preprocessed
        return preprocessed
    
    def train_topic_model(self, num_topics: int = 20, passes: int = 10) -> LdaModel:
        """
        Train LDA topic model on preprocessed surahs.
        
        Args:
            num_topics: Number of topics to extract
            passes: Number of passes through the corpus
            
        Returns:
            Trained LDA model
        """
        if not self.preprocessed_surahs:
            self.preprocess_all_surahs()
        
        # Prepare documents for LDA
        documents = []
        for surah_num in sorted(self.preprocessed_surahs.keys()):
            text = self.preprocessed_surahs[surah_num]
            # Tokenize the preprocessed text
            tokens = text.split()
            documents.append(tokens)
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        
        # Train LDA model
        self.topic_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            alpha='auto',
            eta='auto',
            random_state=42
        )
        
        # Get topic distributions for each surah
        for i, surah_num in enumerate(sorted(self.preprocessed_surahs.keys())):
            topic_dist = self.topic_model.get_document_topics(corpus[i])
            # Convert to probability vector
            topic_vector = np.zeros(num_topics)
            for topic_id, prob in topic_dist:
                topic_vector[topic_id] = prob
            self.topic_distributions[surah_num] = topic_vector
        
        logger.info(f"Trained LDA model with {num_topics} topics")
        return self.topic_model
    
    def compute_asymmetric_matrix(self) -> np.ndarray:
        """
        Compute asymmetric relationship matrix using KL divergence.
        
        Returns:
            114x114 asymmetric relationship matrix
        """
        if not self.topic_distributions:
            self.train_topic_model()
        
        num_surahs = 114
        matrix = np.zeros((num_surahs, num_surahs))
        
        surah_nums = sorted(self.topic_distributions.keys())
        
        for i, surah_i in enumerate(surah_nums):
            for j, surah_j in enumerate(surah_nums):
                if i != j:
                    # Compute KL divergence from surah_j to surah_i
                    dist_i = self.topic_distributions[surah_i]
                    dist_j = self.topic_distributions[surah_j]
                    
                    # Add small epsilon to avoid log(0)
                    epsilon = 1e-10
                    dist_i = dist_i + epsilon
                    dist_j = dist_j + epsilon
                    
                    # Normalize to probability distributions
                    dist_i = dist_i / np.sum(dist_i)
                    dist_j = dist_j / np.sum(dist_j)
                    
                    # Compute KL divergence
                    kl_div = np.sum(rel_entr(dist_j, dist_i))
                    matrix[i, j] = kl_div
        
        self.relationship_matrix = matrix
        logger.info("Computed asymmetric relationship matrix")
        return matrix
    
    def visualize_matrix(self, figsize: Tuple[int, int] = (15, 12), 
                        save_path: str = None) -> plt.Figure:
        """
        Visualize the asymmetric relationship matrix.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if self.relationship_matrix is None:
            self.compute_asymmetric_matrix()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Heatmap of the full matrix
        sns.heatmap(self.relationship_matrix, 
                   xticklabels=self.surah_names[::5],  # Show every 5th label
                   yticklabels=self.surah_names[::5],
                   cmap='viridis', ax=ax1, cbar_kws={'label': 'KL Divergence'})
        ax1.set_title('Asymmetric Relationship Matrix (KL Divergence)')
        ax1.set_xlabel('Target Surah')
        ax1.set_ylabel('Source Surah')
        
        # Difference matrix (asymmetry visualization)
        diff_matrix = self.relationship_matrix - self.relationship_matrix.T
        sns.heatmap(diff_matrix,
                   xticklabels=self.surah_names[::5],
                   yticklabels=self.surah_names[::5],
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
    
    def analyze_relationships(self, top_n: int = 10) -> Dict:
        """
        Analyze the most significant relationships.
        
        Args:
            top_n: Number of top relationships to return
            
        Returns:
            Dictionary with analysis results
        """
        if self.relationship_matrix is None:
            self.compute_asymmetric_matrix()
        
        # Find strongest relationships
        surah_nums = sorted(self.topic_distributions.keys())
        
        # Get top relationships (excluding diagonal)
        relationships = []
        for i in range(114):
            for j in range(114):
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
                'mean_kl_divergence': np.mean(self.relationship_matrix[np.triu_indices(114, k=1)]),
                'std_kl_divergence': np.std(self.relationship_matrix[np.triu_indices(114, k=1)]),
                'max_kl_divergence': np.max(self.relationship_matrix),
                'min_kl_divergence': np.min(self.relationship_matrix[np.triu_indices(114, k=1)])
            }
        }
        
        return analysis
    
    def save_results(self, output_dir: str = "results"):
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
        
        # Save topic distributions
        if self.topic_distributions:
            topic_df = pd.DataFrame(self.topic_distributions).T
            topic_df.to_csv(os.path.join(output_dir, "topic_distributions.csv"))
        
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
    logger.info("Starting Quran Semantic Analysis")
    
    # Initialize analyzer
    analyzer = QuranAnalyzer()
    
    # Extract Quran text
    logger.info("Extracting Quran text...")
    surahs = analyzer.extract_quran_text()
    
    # Preprocess text
    logger.info("Preprocessing Arabic text...")
    analyzer.preprocess_all_surahs()
    
    # Train topic model
    logger.info("Training LDA topic model...")
    analyzer.train_topic_model(num_topics=20, passes=10)
    
    # Compute asymmetric matrix
    logger.info("Computing asymmetric relationship matrix...")
    matrix = analyzer.compute_asymmetric_matrix()
    
    # Visualize results
    logger.info("Creating visualizations...")
    fig = analyzer.visualize_matrix(save_path="quran_relationship_matrix.png")
    
    # Analyze relationships
    logger.info("Analyzing relationships...")
    analysis = analyzer.analyze_relationships()
    
    # Save results
    logger.info("Saving results...")
    analyzer.save_results()
    
    # Print summary
    print("\n" + "="*60)
    print("QURAN SEMANTIC ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analyzed {len(surahs)} surahs")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Mean KL divergence: {np.mean(matrix[np.triu_indices(114, k=1)]):.4f}")
    print(f"Max KL divergence: {np.max(matrix):.4f}")
    print("\nTop 5 Relationships:")
    for i, rel in enumerate(analysis['top_relationships'][:5]):
        print(f"{i+1}. {rel['source']} → {rel['target']}: {rel['kl_divergence']:.4f}")
    
    print("\nResults saved to 'results/' directory")
    print("Visualization saved as 'quran_relationship_matrix.png'")


if __name__ == "__main__":
    main()
