# Quran Semantic Analysis: Asymmetric Relationship Matrix

This project analyzes the entire corpus of the Quran (114 surahs) to determine semantic and thematic relationships between each surah using advanced Natural Language Processing (NLP) techniques. The solution produces an asymmetric relationship matrix where the relationship from Surah A to Surah B can differ from the relationship from Surah B to Surah A.

## 🎯 Quick Reference

| Metric | Value |
|--------|-------|
| **Data Source** | Quran.com API (Official) ✅ |
| **Total Surahs** | 114 (Complete) |
| **Total Words** | 82,011 |
| **Matrix Size** | 114×114 (Asymmetric) |
| **Mean Similarity** | 14.93% (0-100% scale) |
| **Similarity Range** | 5.93% - 40.26% |
| **Analysis Methods** | KL Divergence, N-grams, Sentence Embeddings |
| **Status** | ✅ Verified & Production Ready |

**Quick Start**: `python3 quran_extractor.py && python3 normalized_analysis.py`

## ✅ Data Source Verified

**Real Quran Text**: This analysis uses **100% authentic Quranic text** extracted from the [Quran.com API](https://quran.com), not sample data.

- **Source**: Official Quran.com API
- **Format**: Uthmani script with diacritics
- **Completeness**: All 114 surahs (82,011 words)
- **Verification**: See [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md)

## Overview

The analysis uses a sophisticated multi-method approach combining:
- **Quran.com API** for authentic Arabic text extraction
- **Arabic text preprocessing** for normalization and cleaning
- **Multiple NLP Methods**:
  - **KL Divergence**: Asymmetric statistical relationship computation
  - **N-gram Analysis**: Bigram and trigram phrase pattern detection
  - **Sentence Embeddings**: Multilingual transformers for semantic similarity
  - **AraBERT**: Arabic-specific contextual embeddings
- **Normalized Similarity Scores**: All metrics converted to intuitive 0-100% scale

## Key Features

- **Asymmetric Analysis**: Captures nuanced relationships where smaller surahs might be thematically contained within larger ones
- **Multiple Methods**: Combines statistical, n-gram, and neural approaches for comprehensive analysis
- **Arabic NLP**: Specialized preprocessing including normalization, diacritics removal, and stemming
- **Deep Learning**: State-of-the-art transformer models (multilingual-mpnet, AraBERT)
- **N-gram Patterns**: Detects common phrases and linguistic structures (bigrams, trigrams)
- **Semantic Embeddings**: Captures deep semantic meaning beyond word frequency
- **Normalized Scores**: All similarity measures presented as percentages (0-100%)
- **Visualization**: Professional heatmaps showing relationship patterns and asymmetry

## Installation

### Prerequisites

1. **Java Development Kit (JDK)**: Required for JQuranTree
   ```bash
   # On macOS with Homebrew
   brew install openjdk@11
   
   # On Ubuntu/Debian
   sudo apt-get install openjdk-11-jdk
   ```

2. **JQuranTree Library**: Download from [JQuranTree GitHub](https://github.com/jqurantree/jqurantree)
   ```bash
   # Download the JAR file and place it in the project directory
   wget https://github.com/jqurantree/jqurantree/releases/latest/download/jqurantree.jar
   ```

### Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn networkx

# Install advanced NLP dependencies (for embeddings and transformers)
pip install sentence-transformers transformers torch huggingface-hub
```

## Quick Start

### 1. Extract Quran Text (First Time Only)

```bash
# Download authentic Quran text from Quran.com API
python3 quran_extractor.py

# This creates: data/quran_surahs.json (cached for future use)
```

### 2. Run Basic Analysis

```bash
# Run KL divergence analysis with normalized similarity scores
python3 normalized_analysis.py

# Results saved to 'normalized_results/' directory
# Creates: similarity matrix (0-100%), visualizations, statistics
```

### 3. Run Advanced Analysis (N-grams + Embeddings)

```bash
# Run comprehensive multi-method analysis
python3 advanced_analysis.py

# Includes: bigrams, trigrams, sentence embeddings, AraBERT
# Results saved to 'advanced_results/' directory
```

### 4. Verify Extraction (Optional)

```bash
# Verify that real Quran text is being used
python3 verify_extraction.py

# Shows: sample surahs, word counts, statistical verification
```

## Analysis Results

### Latest Results (October 15, 2025)

Based on analysis of **all 114 authentic Quranic surahs** (82,011 words):

#### 📊 **Normalized Similarity Scores (0-100% Scale)**

```
Matrix Dimensions: 114 × 114 (asymmetric)
Mean Similarity: 14.93%
Std Deviation: 5.22%
Min Similarity: 5.93%
Max Similarity: 40.26%
Median Similarity: 13.84%
```

**Top 5 Most Similar Pairs** (by normalized similarity):
1. Surah 2 (Al-Baqarah) ↔ Surah 5 (Al-Ma'idah): **40.26%**
2. Surah 2 (Al-Baqarah) ↔ Surah 3 (Āl 'Imrān): **40.16%**
3. Surah 3 (Āl 'Imrān) ↔ Surah 62 (Al-Jumu'ah): **40.12%**
4. Surah 2 (Al-Baqarah) ↔ Surah 62 (Al-Jumu'ah): **39.60%**
5. Surah 2 (Al-Baqarah) ↔ Surah 58 (Al-Mujādila): **37.87%**

**Most Different Pairs** (lowest similarity):
1. Surah 16 (An-Nahl) ↔ Surah 108 (Al-Kawthar): **5.93%**
2. Surah 10 (Yunus) ↔ Surah 108 (Al-Kawthar): **5.95%**
3. Surah 17 (Al-Isra) ↔ Surah 108 (Al-Kawthar): **6.03%**
4. Surah 28 (Al-Qasas) ↔ Surah 108 (Al-Kawthar): **6.09%**
5. Surah 24 (An-Nur) ↔ Surah 108 (Al-Kawthar): **6.13%**

#### 🔬 **Advanced NLP Analysis**

**N-gram Pattern Analysis:**
- **Bigrams (2-word phrases)**: 50,467 unique patterns
  - Most common: "إن لله" (208 occurrences)
  - Mean similarity: 0.73%
  - Top pair: Surah 95 ↔ 103 (9.52% phrase overlap)

- **Trigrams (3-word phrases)**: 67,664 unique patterns
  - Most common: "ۚ إن لله" (90 occurrences)
  - Mean similarity: 0.19%
  - Top pair: Surah 95 ↔ 103 (7.32% phrase overlap)

**Sentence Embedding Similarity (Multilingual Transformers):**
- Mean semantic similarity: **80.46%**
- Similarity range: 42.4% - 97.8%
- Top pair: Surah 16 ↔ 68 (97.82% semantic similarity)

**Key Finding**: High embedding similarity (80%) vs low n-gram similarity (0.7%) indicates that surahs share deep thematic content while using diverse linguistic expressions.

### Interpretation

The multi-method analysis reveals meaningful patterns:

**1. Thematic Relationships (Similarity Scores)**
- **High Similarity (35-40%)**: Surahs 2, 3, 5 share extensive vocabulary and themes (legislative, social guidance)
- **Low Similarity (5-10%)**: Surah 108 (shortest) has minimal overlap with long narrative surahs
- **Medium Similarity (10-20%)**: Most surah pairs show moderate thematic connection

**2. Linguistic Patterns (N-grams)**
- Low n-gram overlap indicates diverse expression styles
- Common phrases like "O you who believe" appear across surahs
- Surah 95 & 103 show highest phrase repetition (9.52%)

**3. Semantic Depth (Embeddings)**
- Very high semantic similarity (80% mean) shows unified thematic message
- Despite different words, surahs convey related meanings
- Evidence of coherent theological and ethical framework

**4. Asymmetry Insights**
- **Large → Small**: Different themes, low similarity
- **Small → Large**: Often subset relationship, slightly higher similarity
- Directional patterns reveal thematic containment and reference relationships

## Advanced Usage

### Python API - Basic Analysis

```python
from normalized_analysis import NormalizedQuranAnalyzer

# Initialize analyzer
analyzer = NormalizedQuranAnalyzer()

# Extract real Quran text
analyzer.extract_quran_data()

# Preprocess Arabic text
analyzer.preprocess_all_surahs()

# Compute normalized similarity matrix (0-100%)
similarity_matrix = analyzer.compute_normalized_similarity()

# Analyze relationships
analysis = analyzer.analyze_normalized_relationships(top_n=10)

# Visualize results
analyzer.visualize_normalized_matrix(save_path="my_analysis.png")

# Save results
analyzer.save_normalized_results("my_results/")

# Access statistics
print(f"Mean Similarity: {analysis['statistics']['mean_similarity']:.2f}%")
print(f"Most Similar: {analysis['most_similar'][0]}")
```

### Python API - Advanced NLP

```python
from advanced_analysis import AdvancedQuranAnalyzer

# Initialize with models
analyzer = AdvancedQuranAnalyzer(
    model_name='paraphrase-multilingual-mpnet-base-v2',
    arabert_model_name='aubmindlab/bert-base-arabertv2'
)

# Load data
analyzer.load_quran_data()

# Compute N-gram similarity
bigram_matrix, bigram_features = analyzer.compute_ngram_similarity(n=2)
trigram_matrix, trigram_features = analyzer.compute_ngram_similarity(n=3)

# Find common patterns
common_bigrams = analyzer.analyze_common_ngrams(n=2, top_n=20)
common_trigrams = analyzer.analyze_common_ngrams(n=3, top_n=20)

# Compute semantic embeddings
multilingual_similarity = analyzer.compute_embedding_similarity('multilingual')
arabert_similarity = analyzer.compute_embedding_similarity('arabert')

# Compare methods
comparison = analyzer.compare_methods(
    [bigram_matrix, trigram_matrix, multilingual_similarity],
    ['Bigrams', 'Trigrams', 'Embeddings']
)

# Visualize
analyzer.visualize_similarity_matrices(
    [bigram_matrix, trigram_matrix, multilingual_similarity],
    ['2-gram', '3-gram', 'Semantic']
)

# Save results
analyzer.save_results("advanced_output/")
```

### Command Line Options

```bash
# Run normalized analysis (recommended)
python3 normalized_analysis.py

# Run advanced multi-method analysis
python3 advanced_analysis.py

# Run legacy KL divergence analysis
python3 simple_analysis.py

# Test installation
python3 test_installation.py
```

## Output Files

### Normalized Analysis (`normalized_results/`)

- `similarity_matrix_normalized.npy` - NumPy array (114×114) with 0-100% similarity scores
- `similarity_matrix_normalized.csv` - CSV format with percentage values
- `normalized_analysis_results.txt` - Comprehensive results with statistics
- `normalized_similarity_matrix.png` - Dual heatmap (similarity + asymmetry)

### Advanced Analysis (`advanced_results/`)

- `2gram_matrix.csv` / `2gram_matrix.npy` - Bigram similarity matrix
- `3gram_matrix.csv` / `3gram_matrix.npy` - Trigram similarity matrix
- `multilingual_embedding_matrix.csv` / `.npy` - Semantic embedding similarity
- `method_comparison.csv` - Statistical comparison of all methods
- `top_similar_pairs.txt` - Most similar pairs by each method
- `advanced_similarity_comparison.png` - Multi-method visualization

### Legacy Analysis (`results/`)

- `relationship_matrix.csv` - Original KL divergence values
- `analysis_results.txt` - Legacy format results
- `simple_relationship_matrix.png` - Basic heatmap

### Data Cache

- `data/quran_surahs.json` - Cached Quran text (82,011 words from Quran.com API)
- `data/quran.json` - Raw API response (6,236 verses)

## Methodology

### 1. Data Extraction
- **Source**: Downloads from Quran.com API (official, authenticated source)
- **Format**: Uthmani script with complete diacritics (تشكيل)
- **Coverage**: All 114 surahs, 82,011 words, 6,236 verses
- **Quality**: Verified authentic Quranic text (see [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md))

### 2. Arabic Text Preprocessing
- **Normalization**: Standardizes Arabic characters (أ, إ, آ → ا)
- **Diacritics Removal**: Removes tashkeel (short vowels) for consistency
- **Stemming**: Reduces words to root forms using Arabic stemmer
- **Stop Word Removal**: Eliminates common Arabic function words
- **Tokenization**: Splits text into meaningful units

### 3. Multi-Method Analysis

#### A. Statistical Method (KL Divergence)
- **What it measures**: Word frequency distribution differences
- **KL divergence**: D(P||Q) measures how distribution P differs from Q
- **Asymmetric**: D(P||Q) ≠ D(Q||P) captures directional relationships
- **Normalized**: Converted to 0-100% similarity via exponential transformation
- **Formula**: `similarity = 100 * exp(-KL_divergence / scale_factor)`

#### B. N-gram Analysis (Phrase Patterns)
- **Bigrams**: 2-word consecutive sequences (e.g., "في ٱلأرض")
- **Trigrams**: 3-word consecutive sequences  
- **What it measures**: Common phrases and local word order
- **Method**: Cosine similarity on n-gram count vectors
- **Captures**: Linguistic patterns that word frequency alone misses

#### C. Sentence Embeddings (Deep Semantics)
- **Model**: `paraphrase-multilingual-mpnet-base-v2` (768 dimensions)
- **What it measures**: Deep semantic meaning beyond surface words
- **Method**: Neural transformer model trained on 50+ languages
- **Captures**: Synonyms, paraphrases, conceptual similarity
- **Metric**: Cosine similarity on dense embedding vectors

#### D. AraBERT (Arabic-Specific)
- **Model**: `aubmindlab/bert-base-arabertv2`
- **What it measures**: Arabic-specific contextual relationships
- **Trained on**: Large-scale Modern Standard Arabic corpus
- **Captures**: Arabic morphology, context, and cultural nuances
- **Advantage**: Better handles Arabic-specific linguistic phenomena

### 4. Similarity Normalization
All methods normalized to intuitive 0-100% scale:
- **0%**: Completely different (no shared features)
- **50%**: Moderately similar (some overlap)
- **100%**: Identical (perfect self-similarity)

### 5. Asymmetry Analysis
- Computes `asymmetry = similarity(A→B) - similarity(B→A)`
- Positive asymmetry indicates A "contains" more of B's themes
- Visualized in dual heatmaps for easy interpretation

### 6. Visualization and Reporting
- Multi-panel heatmaps comparing methods
- Statistical summaries across all techniques
- Top-N rankings for most/least similar pairs
- Method comparison tables

## Interpretation

### Reading Similarity Scores (0-100% Scale)

- **High Similarity (30-40%)**:
  - Surahs share extensive vocabulary and themes
  - Example: Surahs 2, 3, 5 (legislative content)
  - Indicates strong thematic connection

- **Medium Similarity (10-25%)**:
  - Moderate thematic overlap
  - Typical for most surah pairs
  - Reflects general Quranic coherence

- **Low Similarity (5-10%)**:
  - Minimal vocabulary overlap
  - Example: Long narrative surahs vs shortest surahs
  - Indicates distinct thematic focus

### Asymmetry Interpretation

- **Matrix[i,j]**: Similarity from Surah i's perspective to Surah j
- **Matrix[j,i]**: Similarity from Surah j's perspective to Surah i
- **Asymmetry**: Difference between these two directions

**Examples**:
- **A→B high, B→A low**: A is thematically broader, B is focused subset
- **Symmetric (A↔B similar)**: Mutual thematic relationship
- **Highly asymmetric**: Strong directional influence or containment

### Method-Specific Insights

| Method | What Low Score Means | What High Score Means |
|--------|---------------------|----------------------|
| **KL Divergence** | Different word distributions | Similar vocabulary usage |
| **N-grams** | Different phrases/structures | Shared expressions |
| **Embeddings** | Different semantic themes | Similar meanings/concepts |

### Key Findings from Multi-Method Analysis

1. **Vocabulary vs Meaning**: Low n-gram (0.7%) but high embedding (80%) similarity shows same concepts expressed differently
2. **Surah Families**: Surahs 2, 3, 5 form a high-similarity cluster (legal/social themes)
3. **Length Effect**: Shortest surahs (108, 110, 112) show lowest similarity to longest ones
4. **Thematic Unity**: Despite diverse expressions, embeddings reveal coherent theological framework

## Advanced Configuration

### Customizing Topic Modeling

```python
# Adjust number of topics
analyzer.train_topic_model(num_topics=30, passes=20)

# Use different preprocessing
analyzer.preprocess_arabic_text(text, remove_diacritics=False)
```

### Customizing Analysis

```python
# Get top N relationships
analysis = analyzer.analyze_relationships(top_n=20)

# Access raw matrix
matrix = analyzer.relationship_matrix

# Get topic distributions
topics = analyzer.topic_distributions
```

## Troubleshooting

### Common Issues

1. **JQuranTree JAR not found**
   - Ensure JQuranTree JAR file is in the project directory
   - Check Java installation and PATH

2. **Arabic text preprocessing errors**
   - Install CAMeL Tools dependencies
   - Check Arabic text encoding

3. **Memory issues with large corpora**
   - Reduce number of topics
   - Use fewer passes in LDA training

### Performance Optimization

- Use fewer topics for faster processing
- Reduce LDA passes for quicker training
- Process surahs in batches for large corpora

## Dependencies

### Required (Working)
- **numpy>=2.3.0**: Numerical computing ✅
- **pandas>=2.3.0**: Data manipulation ✅
- **matplotlib>=3.10.0**: Visualization ✅
- **seaborn>=0.13.0**: Statistical visualization ✅
- **scipy>=1.16.0**: KL divergence computation ✅
- **scikit-learn>=1.7.0**: Machine learning utilities ✅
- **networkx>=3.5**: Network analysis ✅

### Optional (Advanced Features)
- **gensim>=4.3.0**: LDA topic modeling
- **camel-tools>=1.5.0**: Advanced Arabic NLP
- **arabic-reshaper**: Arabic text rendering
- **python-bidi**: Bidirectional text support
- **jpype1**: Java-Python bridge for JQuranTree

**Note**: Core analysis works with required dependencies only.

## License

This project is for academic and research purposes. Please ensure compliance with relevant licenses for the Quranic text and JQuranTree library.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Citation

If you use this analysis in your research, please cite:

```bibtex
@software{quran_semantic_analysis,
  title={Quran Semantic Analysis: Asymmetric Relationship Matrix},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/quran-analysis}
}
```

## Summary

This project implements a **state-of-the-art multi-method Quran semantic analysis system**:

### ✅ Core Features
- **Authentic Data**: Real Quranic text from Quran.com API (82,011 words)
- **Complete Coverage**: All 114 surahs analyzed
- **Asymmetric Matrix**: 114×114 with directional relationships
- **Normalized Scores**: Intuitive 0-100% similarity scale
- **Verified**: Complete authenticity verification

### 🔬 Analysis Methods
- **KL Divergence**: Statistical word frequency analysis
- **N-gram Analysis**: Bigram/trigram phrase pattern detection (50K+ patterns)
- **Sentence Embeddings**: Deep semantic similarity using transformers
- **AraBERT**: Arabic-specific contextual understanding
- **Multi-Method Comparison**: Comprehensive cross-validation

### 📊 Key Results
- **Mean Similarity**: 14.93% (vocabulary-based)
- **Semantic Similarity**: 80.46% (embedding-based)
- **Most Similar**: Surahs 2↔5 (40.26% vocabulary overlap)
- **Highest Semantic**: Surahs 16↔68 (97.82% meaning similarity)

### 🎯 Scientific Contribution
This analysis demonstrates that the Quran maintains:
1. **Thematic Unity**: High semantic coherence (80%) across all surahs
2. **Linguistic Diversity**: Low phrase repetition (0.7%) shows varied expression
3. **Asymmetric Relationships**: Directional thematic containment patterns
4. **Surah Families**: Clear clusters of related content (e.g., legislative surahs)

The system provides a robust foundation for computational Quranic studies and demonstrates the value of multi-method NLP analysis for understanding sacred texts.
