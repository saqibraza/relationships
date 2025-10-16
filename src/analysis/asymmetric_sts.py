#!/usr/bin/env python3
"""
Asymmetric Semantic Textual Similarity (Asym-STS)
==================================================

Implements directional sentence-level semantic matching to capture
asymmetric relationships between surahs.

Method:
-------
For each sentence in source surah, find its maximum similarity to any
sentence in target surah. Average these max similarities.

Asym-STS(B→A) = (1/|S_B|) * Σ_{s_b ∈ S_B} max_{s_a ∈ S_A} CosineSim(s_b, s_a)

Key Properties:
- Directional: Asym-STS(A→B) ≠ Asym-STS(B→A)
- Sentence-level: Finer granularity than document-level embeddings
- Interpretable: Measures "coverage" - how much of source is found in target

Example:
--------
If Surah B (short, focused) has all themes in Surah A (long, broad):
  - Asym-STS(B→A) = HIGH (all of B's content matches A)
  - Asym-STS(A→B) = LOWER (much of A's content not in B)

Author: Quran Semantic Analysis Project
Date: October 16, 2025
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsymmetricSTSAnalyzer:
    """
    Asymmetric Semantic Textual Similarity analyzer for Quranic surahs.
    
    Supports multiple embedding models:
    - 'arabert': Arabic-specific BERT trained on Classical Arabic & Quranic texts
    - 'multilingual': General multilingual sentence transformer
    """
    
    def __init__(self, model_type='arabert'):
        """
        Initialize the Asym-STS analyzer.
        
        Args:
            model_type: Type of model to use
                - 'arabert': aubmindlab/bert-base-arabertv02 (recommended for Quranic text)
                - 'multilingual': paraphrase-multilingual-mpnet-base-v2
        """
        self.model_type = model_type
        self.model = None
        self.tokenizer = None  # For AraBERT
        self.quran_data = None
        self.sentence_embeddings = {}  # Cache: {surah_id: [sentence_vectors]}
        
        # Set model names based on type
        if model_type == 'arabert':
            self.model_name = 'aubmindlab/bert-base-arabertv02'
        elif model_type == 'multilingual':
            self.model_name = 'paraphrase-multilingual-mpnet-base-v2'
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'arabert' or 'multilingual'")
        
    def load_data(self, data_path='data/quran.json'):
        """Load Quranic text data with verse-by-verse information."""
        logger.info(f"Loading Quran data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Process verse-by-verse data
        if 'verses' in data:
            # Format: {verses: [{id, verse_key, text_uthmani}, ...]}
            verses = data['verses']
            
            # Group verses by surah
            self.quran_data = {}
            for verse in verses:
                verse_key = verse.get('verse_key', '')
                text = verse.get('text_uthmani', '')
                
                # Parse verse_key like "1:1" -> surah_id=1, verse_num=1
                if ':' in verse_key:
                    surah_id, verse_num = verse_key.split(':')
                    surah_id = int(surah_id)
                    
                    if surah_id not in self.quran_data:
                        self.quran_data[surah_id] = {'verses': []}
                    
                    self.quran_data[surah_id]['verses'].append(text)
            
            logger.info(f"Loaded {len(self.quran_data)} surahs with verse-by-verse data")
            
            # Log some statistics
            total_verses = sum(len(s['verses']) for s in self.quran_data.values())
            logger.info(f"Total verses: {total_verses}")
        else:
            # Fallback to old format
            self.quran_data = data
            logger.info(f"Loaded {len(self.quran_data)} surahs")
        
    def load_model(self):
        """Load the embedding model (AraBERT or multilingual)."""
        if self.model is None:
            logger.info(f"Loading {self.model_type} model: {self.model_name}")
            
            if self.model_type == 'arabert':
                # Load AraBERT with tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                elif torch.backends.mps.is_available():
                    self.model = self.model.to('mps')
                logger.info(f"AraBERT loaded successfully on {self.model.device}")
                
            elif self.model_type == 'multilingual':
                # Load sentence transformer
                self.model = SentenceTransformer(self.model_name)
                logger.info("Multilingual model loaded successfully")
    
    def get_verses(self, surah_id: int) -> List[str]:
        """
        Get list of verses for a surah.
        
        Important: Skips the first verse (Bismillah) for all surahs except Surah 9,
        as it's the same verse repeated across all surahs.
        
        Args:
            surah_id: Surah number (1-114)
            
        Returns:
            List of verse strings
        """
        surah_data = self.quran_data.get(surah_id, {})
        
        # If we have verse-by-verse data
        if 'verses' in surah_data:
            verses = surah_data['verses']
            # Filter out empty verses
            verses = [v.strip() for v in verses if v.strip()]
            
            # Skip first verse (Bismillah) for all surahs except Surah 9
            # Bismillah is repeated at the start of every surah except At-Tawbah
            if surah_id != 9 and len(verses) > 1:
                verses = verses[1:]
            
            return verses
        
        # Fallback: if we have full text, try to split it
        elif 'text' in surah_data:
            text = surah_data['text']
            # Split by common separators
            separators = ['۞', '\n', '﴿', '﴾']
            verses = [text]
            for sep in separators:
                new_verses = []
                for v in verses:
                    new_verses.extend(v.split(sep))
                verses = new_verses
            
            verses = [v.strip() for v in verses if v.strip()]
            
            # If still just one long string, chunk by words
            if len(verses) == 1 and len(text) > 100:
                words = text.split()
                chunk_size = 50
                verses = [
                    ' '.join(words[i:i+chunk_size]) 
                    for i in range(0, len(words), chunk_size)
                ]
            
            return verses
        
        return []
    
    def compute_sentence_embeddings(self, surah_id: int, force_recompute=False):
        """
        Compute and cache sentence embeddings for a surah.
        
        Args:
            surah_id: Surah number (1-114)
            force_recompute: If True, recompute even if cached
            
        Returns:
            numpy array of sentence embeddings (num_verses, embedding_dim)
        """
        if not force_recompute and surah_id in self.sentence_embeddings:
            return self.sentence_embeddings[surah_id]
        
        # Get verses
        verses = self.get_verses(surah_id)
        
        if not verses:
            logger.warning(f"No verses found for surah {surah_id}")
            return np.array([])
        
        # Encode verses based on model type
        if self.model_type == 'arabert':
            embeddings = self._encode_with_arabert(verses)
        elif self.model_type == 'multilingual':
            embeddings = self._encode_with_sentence_transformer(verses)
        
        # Cache
        self.sentence_embeddings[surah_id] = embeddings
        
        return embeddings
    
    def _encode_with_arabert(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using AraBERT.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings (num_texts, 768)
        """
        embeddings = []
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # Move to same device as model
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding (first token)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])
        
        return np.array(embeddings)
    
    def _encode_with_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using Sentence Transformer.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings (num_texts, 768)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings
    
    def compute_asymmetric_sts(
        self, 
        source_id: int, 
        target_id: int
    ) -> float:
        """
        Compute Asymmetric STS from source to target.
        
        Asym-STS(source→target) measures how much of source's content
        can be found in target.
        
        Args:
            source_id: Source surah ID
            target_id: Target surah ID
            
        Returns:
            Asymmetric STS score (0-1, higher means more coverage)
        """
        # Get sentence embeddings
        source_embeddings = self.compute_sentence_embeddings(source_id)
        target_embeddings = self.compute_sentence_embeddings(target_id)
        
        # For each sentence in source, find max similarity to target sentences
        max_similarities = []
        
        for source_sent in source_embeddings:
            # Compute cosine similarity to all target sentences
            # Cosine similarity: dot(a, b) / (norm(a) * norm(b))
            similarities = np.dot(target_embeddings, source_sent) / (
                np.linalg.norm(target_embeddings, axis=1) * 
                np.linalg.norm(source_sent)
            )
            
            # Take maximum similarity
            max_sim = np.max(similarities)
            max_similarities.append(max_sim)
        
        # Average the max similarities
        asymmetric_sts = np.mean(max_similarities)
        
        return asymmetric_sts
    
    def compute_full_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute full 114x114 asymmetric STS matrix.
        
        Returns:
            Tuple of (similarity_matrix, asymmetry_matrix)
            - similarity_matrix[i][j]: Asym-STS from surah i+1 to surah j+1
            - asymmetry_matrix[i][j]: |Asym-STS(i→j) - Asym-STS(j→i)|
        """
        n_surahs = 114
        similarity_matrix = np.zeros((n_surahs, n_surahs))
        
        logger.info("Computing Asymmetric STS matrix (114x114)...")
        
        # Pre-compute all sentence embeddings
        logger.info("Pre-computing sentence embeddings for all surahs...")
        for surah_id in tqdm(range(1, n_surahs + 1), desc="Encoding surahs"):
            self.compute_sentence_embeddings(surah_id)
        
        # Compute pairwise similarities
        logger.info("Computing pairwise Asym-STS scores...")
        total_pairs = n_surahs * n_surahs
        
        with tqdm(total=total_pairs, desc="Computing Asym-STS") as pbar:
            for i in range(n_surahs):
                for j in range(n_surahs):
                    if i == j:
                        similarity_matrix[i][j] = 100.0  # Perfect self-similarity
                    else:
                        # Compute Asym-STS(i+1 → j+1)
                        asym_sts = self.compute_asymmetric_sts(i + 1, j + 1)
                        # Convert to 0-100% scale
                        similarity_matrix[i][j] = asym_sts * 100.0
                    
                    pbar.update(1)
        
        # Compute asymmetry matrix
        asymmetry_matrix = np.zeros((n_surahs, n_surahs))
        for i in range(n_surahs):
            for j in range(n_surahs):
                asymmetry_matrix[i][j] = abs(
                    similarity_matrix[i][j] - similarity_matrix[j][i]
                )
        
        logger.info("Asym-STS matrix computation complete!")
        logger.info(f"Similarity range: {similarity_matrix.min():.2f}% - {similarity_matrix.max():.2f}%")
        logger.info(f"Mean similarity: {similarity_matrix.mean():.2f}%")
        logger.info(f"Mean asymmetry: {asymmetry_matrix.mean():.2f}%")
        
        return similarity_matrix, asymmetry_matrix
    
    def save_results(self, output_dir='results/matrices'):
        """
        Save Asym-STS results to CSV files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving Asym-STS results to {output_dir}")
        
        # Compute matrices
        similarity_matrix, asymmetry_matrix = self.compute_full_matrix()
        
        # Create dataframes
        surah_ids = [str(i) for i in range(1, 115)]
        
        df_similarity = pd.DataFrame(
            similarity_matrix,
            index=surah_ids,
            columns=surah_ids
        )
        
        df_asymmetry = pd.DataFrame(
            asymmetry_matrix,
            index=surah_ids,
            columns=surah_ids
        )
        
        # Save to CSV with model type in filename
        sim_path = output_path / f'asymmetric_sts_{self.model_type}_similarity_matrix.csv'
        asym_path = output_path / f'asymmetric_sts_{self.model_type}_asymmetry_matrix.csv'
        
        df_similarity.to_csv(sim_path, float_format='%.2f')
        df_asymmetry.to_csv(asym_path, float_format='%.2f')
        
        logger.info(f"✓ Saved similarity matrix: {sim_path}")
        logger.info(f"✓ Saved asymmetry matrix: {asym_path}")
        
        # Save statistics
        stats = {
            'method': 'Asymmetric Semantic Textual Similarity (Asym-STS)',
            'model_type': self.model_type,
            'model_name': self.model_name,
            'n_surahs': 114,
            'total_pairs': 114 * 114,
            'similarity_stats': {
                'min': float(similarity_matrix.min()),
                'max': float(similarity_matrix.max()),
                'mean': float(similarity_matrix.mean()),
                'median': float(np.median(similarity_matrix)),
                'std': float(similarity_matrix.std())
            },
            'asymmetry_stats': {
                'min': float(asymmetry_matrix.min()),
                'max': float(asymmetry_matrix.max()),
                'mean': float(asymmetry_matrix.mean()),
                'median': float(np.median(asymmetry_matrix)),
                'std': float(asymmetry_matrix.std())
            }
        }
        
        stats_path = output_path / f'asymmetric_sts_{self.model_type}_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✓ Saved statistics: {stats_path}")
        
        return df_similarity, df_asymmetry
    
    def analyze_pair(self, surah_a: int, surah_b: int) -> Dict:
        """
        Detailed analysis of a specific surah pair.
        
        Args:
            surah_a: First surah ID
            surah_b: Second surah ID
            
        Returns:
            Dictionary with detailed analysis
        """
        # Compute bidirectional scores
        a_to_b = self.compute_asymmetric_sts(surah_a, surah_b) * 100
        b_to_a = self.compute_asymmetric_sts(surah_b, surah_a) * 100
        
        # Get sentence counts
        embeddings_a = self.compute_sentence_embeddings(surah_a)
        embeddings_b = self.compute_sentence_embeddings(surah_b)
        
        n_sentences_a = len(embeddings_a)
        n_sentences_b = len(embeddings_b)
        
        # Compute statistics
        asymmetry = abs(a_to_b - b_to_a)
        average = (a_to_b + b_to_a) / 2
        
        # Interpretation
        if a_to_b > b_to_a + 5:
            interpretation = f"Surah {surah_a} → {surah_b}: HIGH coverage (most of {surah_a}'s content found in {surah_b})"
        elif b_to_a > a_to_b + 5:
            interpretation = f"Surah {surah_b} → {surah_a}: HIGH coverage (most of {surah_b}'s content found in {surah_a})"
        else:
            interpretation = "Relatively symmetric relationship"
        
        return {
            'surah_a': surah_a,
            'surah_b': surah_b,
            'a_to_b': round(a_to_b, 2),
            'b_to_a': round(b_to_a, 2),
            'asymmetry': round(asymmetry, 2),
            'average': round(average, 2),
            'n_sentences_a': n_sentences_a,
            'n_sentences_b': n_sentences_b,
            'interpretation': interpretation
        }


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("ASYMMETRIC SEMANTIC TEXTUAL SIMILARITY (Asym-STS)")
    logger.info("="*70)
    
    # Initialize analyzer with AraBERT (recommended for Classical Arabic)
    logger.info("\nUsing AraBERT model (optimized for Classical Arabic & Quranic text)")
    analyzer = AsymmetricSTSAnalyzer(model_type='arabert')
    
    # Load data and model
    analyzer.load_data()
    analyzer.load_model()
    
    # Compute and save full matrix
    analyzer.save_results()
    
    # Example: Analyze specific pairs
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE PAIR ANALYSIS")
    logger.info("="*70)
    
    example_pairs = [
        (2, 65),   # Al-Baqarah (long) vs At-Talaq (short, focused)
        (113, 114), # Similar short surahs
        (10, 11),   # Similar prophetic narratives
    ]
    
    for surah_a, surah_b in example_pairs:
        result = analyzer.analyze_pair(surah_a, surah_b)
        
        logger.info(f"\nSurah {surah_a} ↔ Surah {surah_b}:")
        logger.info(f"  Sentences: {result['n_sentences_a']} vs {result['n_sentences_b']}")
        logger.info(f"  {surah_a}→{surah_b}: {result['a_to_b']:.2f}%")
        logger.info(f"  {surah_b}→{surah_a}: {result['b_to_a']:.2f}%")
        logger.info(f"  Asymmetry: {result['asymmetry']:.2f}%")
        logger.info(f"  Average: {result['average']:.2f}%")
        logger.info(f"  {result['interpretation']}")
    
    logger.info("\n" + "="*70)
    logger.info("✓ Asym-STS analysis complete!")
    logger.info("="*70)


if __name__ == '__main__':
    main()

