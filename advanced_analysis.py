#!/usr/bin/env python3
"""
Advanced Quran Semantic Analysis with N-grams, Embeddings, and Semantic Similarity
================================================================================

This module implements advanced NLP techniques including:
- N-gram models (bigrams, trigrams)
- Sentence embeddings (AraBERT, multilingual transformers)
- Syntactic parsing (dependency trees)
- Semantic similarity measures (word embeddings, contextual models)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from collections import Counter
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedQuranAnalyzer:
    """
    Advanced analyzer with N-grams, embeddings, and semantic similarity.
    """
    
    def __init__(self):
        """Initialize the advanced analyzer."""
        self.surahs = {}
        self.preprocessed_surahs = {}
        self.sentence_embeddings = {}
        self.word_embeddings = {}
        self.ngram_models = {}
        self.dependency_trees = {}
        self.similarity_matrices = {}
        self.surah_names = []
        
        # Try to load advanced models
        self._load_models()
    
    def _load_models(self):
        """Load advanced NLP models."""
        try:
            # Try to load sentence transformers
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            logger.info("✓ Loaded sentence transformer model")
        except Exception as e:
            logger.warning(f"Could not load sentence transformers: {e}")
            self.sentence_model = None
        
        try:
            # Try to load AraBERT
            from transformers import AutoTokenizer, AutoModel
            self.arabert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
            self.arabert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
            logger.info("✓ Loaded AraBERT model")
        except Exception as e:
            logger.warning(f"Could not load AraBERT: {e}")
            self.arabert_tokenizer = None
            self.arabert_model = None
        
        try:
            # Try to load spaCy for dependency parsing
            import spacy
            # Try Arabic model first, fall back to multilingual
            try:
                self.nlp = spacy.load("xx_ent_wiki_sm")  # Multilingual model
                logger.info("✓ Loaded spaCy multilingual model")
            except:
                logger.warning("spaCy model not available for dependency parsing")
                self.nlp = None
        except Exception as e:
            logger.warning(f"Could not load spaCy: {e}")
            self.nlp = None
    
    def load_quran_data(self, data_path: str = "data/quran_surahs.json"):
        """Load Quran data from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            surahs_data = json.load(f)
            self.surahs = {int(k): v for k, v in surahs_data.items()}
        
        self.surah_names = [f"Surah {i}" for i in range(1, 115)]
        logger.info(f"Loaded {len(self.surahs)} surahs")
        return self.surahs
    
    def preprocess_text(self, text: str) -> str:
        """Basic preprocessing for Arabic text."""
        import re
        # Remove diacritics
        diacritics = ['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ', 'ٰ', 'ٱ', 'ۤ', 'ۥ', 'ۦ', 'ۧ', 'ۨ', '۩']
        for d in diacritics:
            text = text.replace(d, '')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # ============================================================================
    # 1. N-GRAM ANALYSIS
    # ============================================================================
    
    def extract_ngrams(self, text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """
        Extract n-grams from text.
        
        Args:
            text: Input text
            n: Size of n-gram (2=bigram, 3=trigram, etc.)
            
        Returns:
            List of n-gram tuples
        """
        words = text.split()
        ngrams = []
        
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def compute_ngram_similarity(self, n: int = 2) -> np.ndarray:
        """
        Compute similarity matrix based on n-gram overlap.
        
        Args:
            n: Size of n-gram (2=bigram, 3=trigram)
            
        Returns:
            Similarity matrix based on n-gram Jaccard similarity
        """
        logger.info(f"Computing {n}-gram similarity matrix...")
        
        if not self.surahs:
            self.load_quran_data()
        
        # Extract n-grams for each surah
        surah_ngrams = {}
        for surah_num, text in self.surahs.items():
            preprocessed = self.preprocess_text(text)
            ngrams = self.extract_ngrams(preprocessed, n=n)
            surah_ngrams[surah_num] = set(ngrams)
        
        # Compute Jaccard similarity between all pairs
        n_surahs = len(self.surahs)
        similarity_matrix = np.zeros((n_surahs, n_surahs))
        
        surah_nums = sorted(self.surahs.keys())
        
        for i, surah_i in enumerate(surah_nums):
            for j, surah_j in enumerate(surah_nums):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    ngrams_i = surah_ngrams[surah_i]
                    ngrams_j = surah_ngrams[surah_j]
                    
                    # Jaccard similarity
                    intersection = len(ngrams_i & ngrams_j)
                    union = len(ngrams_i | ngrams_j)
                    
                    if union > 0:
                        similarity = intersection / union
                        similarity_matrix[i, j] = similarity
        
        self.similarity_matrices[f'{n}gram'] = similarity_matrix
        logger.info(f"✓ {n}-gram similarity matrix computed")
        return similarity_matrix
    
    def analyze_common_ngrams(self, n: int = 2, top_k: int = 20) -> Dict:
        """
        Analyze most common n-grams across all surahs.
        
        Args:
            n: Size of n-gram
            top_k: Number of top n-grams to return
            
        Returns:
            Dictionary with n-gram statistics
        """
        logger.info(f"Analyzing common {n}-grams...")
        
        if not self.surahs:
            self.load_quran_data()
        
        all_ngrams = []
        surah_ngram_counts = {}
        
        for surah_num, text in self.surahs.items():
            preprocessed = self.preprocess_text(text)
            ngrams = self.extract_ngrams(preprocessed, n=n)
            all_ngrams.extend(ngrams)
            surah_ngram_counts[surah_num] = Counter(ngrams)
        
        # Overall most common n-grams
        ngram_counter = Counter(all_ngrams)
        most_common = ngram_counter.most_common(top_k)
        
        results = {
            'total_ngrams': len(all_ngrams),
            'unique_ngrams': len(ngram_counter),
            'most_common': most_common,
            'surah_counts': surah_ngram_counts
        }
        
        logger.info(f"✓ Found {len(ngram_counter)} unique {n}-grams")
        return results
    
    # ============================================================================
    # 2. SENTENCE EMBEDDINGS
    # ============================================================================
    
    def compute_sentence_embeddings(self, model_type: str = 'multilingual') -> Dict:
        """
        Compute sentence embeddings for each surah using transformers.
        
        Args:
            model_type: 'multilingual' or 'arabert'
            
        Returns:
            Dictionary of embeddings for each surah
        """
        logger.info(f"Computing sentence embeddings using {model_type}...")
        
        if not self.surahs:
            self.load_quran_data()
        
        embeddings = {}
        
        if model_type == 'multilingual' and self.sentence_model:
            # Use sentence-transformers (multilingual)
            for surah_num, text in self.surahs.items():
                try:
                    # Truncate if too long
                    text_clean = self.preprocess_text(text)
                    if len(text_clean) > 5000:
                        text_clean = text_clean[:5000]
                    
                    embedding = self.sentence_model.encode(text_clean)
                    embeddings[surah_num] = embedding
                    
                    if surah_num % 20 == 0:
                        logger.info(f"  Encoded {surah_num}/114 surahs")
                        
                except Exception as e:
                    logger.warning(f"Could not encode surah {surah_num}: {e}")
                    embeddings[surah_num] = None
        
        elif model_type == 'arabert' and self.arabert_model:
            # Use AraBERT
            import torch
            
            for surah_num, text in self.surahs.items():
                try:
                    text_clean = self.preprocess_text(text)
                    if len(text_clean) > 5000:
                        text_clean = text_clean[:5000]
                    
                    # Tokenize and encode
                    inputs = self.arabert_tokenizer(text_clean, return_tensors="pt", 
                                                    truncation=True, max_length=512)
                    
                    with torch.no_grad():
                        outputs = self.arabert_model(**inputs)
                    
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[0, 0, :].numpy()
                    embeddings[surah_num] = embedding
                    
                    if surah_num % 20 == 0:
                        logger.info(f"  Encoded {surah_num}/114 surahs")
                        
                except Exception as e:
                    logger.warning(f"Could not encode surah {surah_num}: {e}")
                    embeddings[surah_num] = None
        
        else:
            logger.error(f"Model {model_type} not available")
            return None
        
        self.sentence_embeddings[model_type] = embeddings
        logger.info(f"✓ Computed embeddings for {len(embeddings)} surahs")
        return embeddings
    
    def compute_embedding_similarity(self, model_type: str = 'multilingual') -> np.ndarray:
        """
        Compute similarity matrix based on sentence embeddings.
        
        Args:
            model_type: Which embedding model to use
            
        Returns:
            Cosine similarity matrix
        """
        logger.info(f"Computing embedding similarity matrix ({model_type})...")
        
        if model_type not in self.sentence_embeddings:
            self.compute_sentence_embeddings(model_type)
        
        embeddings = self.sentence_embeddings[model_type]
        
        # Create embedding matrix
        surah_nums = sorted([k for k in embeddings.keys() if embeddings[k] is not None])
        embedding_matrix = np.array([embeddings[num] for num in surah_nums])
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        self.similarity_matrices[f'{model_type}_embedding'] = similarity_matrix
        logger.info(f"✓ Embedding similarity matrix computed")
        return similarity_matrix
    
    # ============================================================================
    # 3. DEPENDENCY PARSING
    # ============================================================================
    
    def parse_dependencies(self, text: str, max_sentences: int = 10):
        """
        Parse dependency trees for sentences in text.
        
        Args:
            text: Input text
            max_sentences: Maximum number of sentences to parse
            
        Returns:
            List of dependency parses
        """
        if not self.nlp:
            logger.warning("spaCy not available for dependency parsing")
            return None
        
        try:
            doc = self.nlp(text[:5000])  # Limit length
            
            dependency_info = []
            for sent_idx, sent in enumerate(doc.sents):
                if sent_idx >= max_sentences:
                    break
                
                sent_deps = []
                for token in sent:
                    sent_deps.append({
                        'text': token.text,
                        'pos': token.pos_,
                        'dep': token.dep_,
                        'head': token.head.text
                    })
                
                dependency_info.append(sent_deps)
            
            return dependency_info
            
        except Exception as e:
            logger.warning(f"Dependency parsing failed: {e}")
            return None
    
    def analyze_syntactic_patterns(self) -> Dict:
        """
        Analyze syntactic patterns across surahs.
        
        Returns:
            Dictionary with syntactic pattern statistics
        """
        logger.info("Analyzing syntactic patterns...")
        
        if not self.nlp:
            logger.warning("spaCy not available")
            return None
        
        if not self.surahs:
            self.load_quran_data()
        
        pattern_stats = {
            'pos_distributions': {},
            'dependency_patterns': {},
            'common_structures': []
        }
        
        for surah_num, text in list(self.surahs.items())[:10]:  # Sample first 10
            try:
                preprocessed = self.preprocess_text(text)
                dependencies = self.parse_dependencies(preprocessed, max_sentences=5)
                
                if dependencies:
                    self.dependency_trees[surah_num] = dependencies
                    logger.info(f"  Parsed surah {surah_num}")
                    
            except Exception as e:
                logger.warning(f"Could not parse surah {surah_num}: {e}")
        
        logger.info(f"✓ Parsed {len(self.dependency_trees)} surahs")
        return pattern_stats
    
    # ============================================================================
    # 4. SEMANTIC SIMILARITY
    # ============================================================================
    
    def compute_semantic_similarity_matrix(self) -> Dict[str, np.ndarray]:
        """
        Compute comprehensive semantic similarity matrices using multiple methods.
        
        Returns:
            Dictionary of similarity matrices from different methods
        """
        logger.info("Computing comprehensive semantic similarity matrices...")
        
        matrices = {}
        
        # 1. Bigram similarity
        try:
            matrices['bigram'] = self.compute_ngram_similarity(n=2)
            logger.info("✓ Bigram similarity computed")
        except Exception as e:
            logger.warning(f"Bigram similarity failed: {e}")
        
        # 2. Trigram similarity
        try:
            matrices['trigram'] = self.compute_ngram_similarity(n=3)
            logger.info("✓ Trigram similarity computed")
        except Exception as e:
            logger.warning(f"Trigram similarity failed: {e}")
        
        # 3. Sentence embedding similarity (multilingual)
        try:
            if self.sentence_model:
                matrices['sentence_embedding'] = self.compute_embedding_similarity('multilingual')
                logger.info("✓ Sentence embedding similarity computed")
        except Exception as e:
            logger.warning(f"Sentence embedding similarity failed: {e}")
        
        # 4. AraBERT similarity
        try:
            if self.arabert_model:
                matrices['arabert'] = self.compute_embedding_similarity('arabert')
                logger.info("✓ AraBERT similarity computed")
        except Exception as e:
            logger.warning(f"AraBERT similarity failed: {e}")
        
        self.similarity_matrices.update(matrices)
        logger.info(f"✓ Computed {len(matrices)} similarity matrices")
        
        return matrices
    
    # ============================================================================
    # 5. VISUALIZATION AND ANALYSIS
    # ============================================================================
    
    def visualize_similarity_comparison(self, save_path: str = "advanced_similarity_comparison.png"):
        """Create comparison visualization of different similarity measures."""
        
        if not self.similarity_matrices:
            logger.warning("No similarity matrices to visualize")
            return
        
        n_methods = len(self.similarity_matrices)
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(20, 12))
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        for idx, (method_name, matrix) in enumerate(self.similarity_matrices.items()):
            ax = axes[idx]
            
            # Plot heatmap
            sns.heatmap(matrix, 
                       xticklabels=False,
                       yticklabels=False,
                       cmap='viridis',
                       ax=ax,
                       cbar_kws={'label': 'Similarity'})
            
            ax.set_title(f'{method_name.replace("_", " ").title()} Similarity')
            ax.set_xlabel('Surahs')
            ax.set_ylabel('Surahs')
        
        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Visualization saved to {save_path}")
        
        return fig
    
    def compare_methods(self) -> pd.DataFrame:
        """
        Compare different similarity methods.
        
        Returns:
            DataFrame with comparison statistics
        """
        logger.info("Comparing similarity methods...")
        
        if not self.similarity_matrices:
            logger.warning("No similarity matrices available")
            return None
        
        comparison = []
        
        for method_name, matrix in self.similarity_matrices.items():
            # Get upper triangle (excluding diagonal)
            upper_tri = matrix[np.triu_indices(matrix.shape[0], k=1)]
            
            stats = {
                'Method': method_name,
                'Mean Similarity': np.mean(upper_tri),
                'Std Similarity': np.std(upper_tri),
                'Min Similarity': np.min(upper_tri),
                'Max Similarity': np.max(upper_tri),
                'Median Similarity': np.median(upper_tri)
            }
            
            comparison.append(stats)
        
        df = pd.DataFrame(comparison)
        logger.info("✓ Method comparison complete")
        
        return df
    
    def find_most_similar_pairs(self, method: str = 'bigram', top_k: int = 10) -> List[Tuple]:
        """
        Find most similar surah pairs using specified method.
        
        Args:
            method: Similarity method to use
            top_k: Number of top pairs to return
            
        Returns:
            List of (surah_i, surah_j, similarity) tuples
        """
        if method not in self.similarity_matrices:
            logger.warning(f"Method {method} not available")
            return []
        
        matrix = self.similarity_matrices[method]
        n = matrix.shape[0]
        
        # Get all pairs with similarities
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i + 1, j + 1, matrix[i, j]))
        
        # Sort by similarity
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs[:top_k]
    
    def save_results(self, output_dir: str = "advanced_results"):
        """Save all analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save similarity matrices
        for method_name, matrix in self.similarity_matrices.items():
            np.save(os.path.join(output_dir, f"{method_name}_matrix.npy"), matrix)
            
            df = pd.DataFrame(matrix, 
                            index=self.surah_names,
                            columns=self.surah_names)
            df.to_csv(os.path.join(output_dir, f"{method_name}_matrix.csv"))
        
        # Save comparison
        comparison_df = self.compare_methods()
        if comparison_df is not None:
            comparison_df.to_csv(os.path.join(output_dir, "method_comparison.csv"), index=False)
        
        # Save top similar pairs for each method
        with open(os.path.join(output_dir, "top_similar_pairs.txt"), 'w', encoding='utf-8') as f:
            for method in self.similarity_matrices.keys():
                f.write(f"\n{'='*60}\n")
                f.write(f"Top 10 Similar Pairs - {method.upper()}\n")
                f.write(f"{'='*60}\n\n")
                
                pairs = self.find_most_similar_pairs(method, top_k=10)
                for rank, (i, j, sim) in enumerate(pairs, 1):
                    f.write(f"{rank}. Surah {i} ↔ Surah {j}: {sim:.4f}\n")
        
        logger.info(f"✓ Results saved to {output_dir}/")


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("ADVANCED QURAN SEMANTIC ANALYSIS")
    logger.info("=" * 70)
    
    # Initialize analyzer
    analyzer = AdvancedQuranAnalyzer()
    
    # Load data
    logger.info("\n1. Loading Quran data...")
    analyzer.load_quran_data()
    
    # Compute N-gram similarities
    logger.info("\n2. Computing N-gram similarities...")
    analyzer.compute_ngram_similarity(n=2)  # Bigrams
    analyzer.compute_ngram_similarity(n=3)  # Trigrams
    
    # Analyze common n-grams
    logger.info("\n3. Analyzing common n-grams...")
    bigram_analysis = analyzer.analyze_common_ngrams(n=2, top_k=20)
    trigram_analysis = analyzer.analyze_common_ngrams(n=3, top_k=20)
    
    print(f"\nMost common bigrams:")
    for ngram, count in bigram_analysis['most_common'][:5]:
        print(f"  {' '.join(ngram)}: {count}")
    
    print(f"\nMost common trigrams:")
    for ngram, count in trigram_analysis['most_common'][:5]:
        print(f"  {' '.join(ngram)}: {count}")
    
    # Compute sentence embeddings
    logger.info("\n4. Computing sentence embeddings...")
    if analyzer.sentence_model:
        analyzer.compute_embedding_similarity('multilingual')
    
    # Visualize comparisons
    logger.info("\n5. Creating visualizations...")
    analyzer.visualize_similarity_comparison()
    
    # Compare methods
    logger.info("\n6. Comparing methods...")
    comparison = analyzer.compare_methods()
    if comparison is not None:
        print("\n" + "=" * 70)
        print("METHOD COMPARISON")
        print("=" * 70)
        print(comparison.to_string(index=False))
    
    # Save results
    logger.info("\n7. Saving results...")
    analyzer.save_results()
    
    print("\n" + "=" * 70)
    print("ADVANCED ANALYSIS COMPLETE")
    print("=" * 70)
    print("Results saved to 'advanced_results/' directory")
    print("Visualization saved as 'advanced_similarity_comparison.png'")


if __name__ == "__main__":
    main()
