# Methodology - Technical Details

This document provides comprehensive technical details on how the Quran semantic analysis is performed.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Extraction](#data-extraction)
3. [Preprocessing](#preprocessing)
4. [Analysis Methods](#analysis-methods)
5. [Unified Matrices](#unified-matrices)
6. [Medinan/Makkan Classification](#medinanmakkan-classification)
7. [Asymmetric Matrix Design](#asymmetric-matrix-design)
8. [Normalization](#normalization)
9. [Evaluation Metrics](#evaluation-metrics)

---

## Overview

This project computes **asymmetric similarity matrices** (114×114) for all Quranic surahs using multiple NLP techniques:

- **Unified-All**: Combines 5 methods (KL divergence, bigrams, trigrams, sentence embeddings, AraBERT)
- **Unified-Semantic**: Focuses on semantic methods only (sentence embeddings, AraBERT)

### Key Characteristics

- **Asymmetric**: Similarity from Surah A→B may differ from B→A
- **Normalized**: All scores are 0-100% for intuitive interpretation
- **Comprehensive**: Analyzes all 6,441 unique surah pairs (114 × 113 / 2)
- **Classical Arabic Aware**: Uses AraBERT trained on Quranic texts

---

## Data Extraction

### Source

**Quran.com API**: `https://api.quran.com/api/v4/`

### Process

```python
# Extract all 114 surahs
for surah_id in range(1, 115):
    url = f"https://api.quran.com/api/v4/quran/verses/uthmani"
    params = {"chapter_number": surah_id}
    response = requests.get(url, params=params)
    
    # Store verse text
    verses = [verse['text_uthmani'] for verse in data['verses']]
    surah_text = ' '.join(verses)
```

### Output

- **File**: `quran_data/quran_arabic.json`
- **Format**: JSON with surah numbers, names, and full Arabic text
- **Script**: `uthmani` (standard Uthmanic orthography without diacritics)

### Verification

The `verify_extraction.py` script checks:
- ✅ All 114 surahs extracted
- ✅ Known verse counts match (e.g., Al-Fatiha: 7 verses, Al-Baqarah: 286 verses)
- ✅ Arabic text integrity (proper encoding)
- ✅ No missing or corrupted data

---

## Preprocessing

### Arabic Text Normalization

All Arabic text undergoes the following preprocessing:

```python
def preprocess_arabic(text):
    # 1. Remove diacritics (tashkeel)
    text = remove_diacritics(text)
    
    # 2. Normalize Arabic characters
    text = text.replace('أ', 'ا')  # Alif with hamza above
    text = text.replace('إ', 'ا')  # Alif with hamza below
    text = text.replace('آ', 'ا')  # Alif with madda
    text = text.replace('ة', 'ه')  # Ta marbuta to ha
    text = text.replace('ى', 'ي')  # Alif maksura to ya
    
    # 3. Remove punctuation
    text = remove_punctuation(text)
    
    # 4. Remove Arabic stop words
    text = remove_stop_words(text, arabic_stop_words)
    
    # 5. Tokenize
    tokens = text.split()
    
    return tokens
```

### Why This Preprocessing?

- **Diacritics removal**: Focuses on root meanings, not grammatical inflections
- **Normalization**: Groups morphological variants (e.g., different Alif forms)
- **Stop words**: Removes common function words that don't carry thematic meaning
- **No stemming/lemmatization**: Preserves semantic richness of Classical Arabic

---

## Analysis Methods

### 1. KL Divergence (30% weight in Unified-All)

**Purpose**: Measures difference in word frequency distributions.

**Algorithm**:
```python
def compute_kl_divergence(surah_a, surah_b):
    # Count word frequencies
    freq_a = word_frequency(surah_a)
    freq_b = word_frequency(surah_b)
    
    # Get vocabulary union
    vocab = set(freq_a.keys()) | set(freq_b.keys())
    
    # Convert to probability distributions (with smoothing)
    P = [freq_a.get(w, 0) + 1e-10 for w in vocab]
    Q = [freq_b.get(w, 0) + 1e-10 for w in vocab]
    
    # Normalize
    P = P / sum(P)
    Q = Q / sum(Q)
    
    # Compute KL(P||Q)
    kl = sum(p * log(p/q) for p, q in zip(P, Q))
    
    return kl
```

**Interpretation**:
- Lower KL divergence = more similar word distributions
- **Asymmetric**: KL(A||B) ≠ KL(B||A)
- Captures thematic vocabulary overlap

**Normalization to 0-100%**:
```python
# KL divergence ranges from 0 to ~10
# Convert to similarity percentage
similarity = 100 * exp(-kl / max_kl)
```

**Limitations**:
- Doesn't capture word order or sentence structure
- Treats words as independent (bag-of-words assumption)
- Doesn't understand semantic meaning beyond word frequency

### 2. N-gram Analysis (10% bigrams + 10% trigrams in Unified-All)

**Purpose**: Captures local word order and common phrases.

**Bigrams (2-word sequences)**:
```python
def extract_bigrams(text):
    words = text.split()
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    return Counter(bigrams)
```

**Trigrams (3-word sequences)**:
```python
def extract_trigrams(text):
    words = text.split()
    trigrams = [(words[i], words[i+1], words[i+2]) 
                for i in range(len(words)-2)]
    return Counter(trigrams)
```

**Similarity Computation**:
```python
def ngram_similarity(bigrams_a, bigrams_b):
    # Jaccard similarity
    set_a = set(bigrams_a.keys())
    set_b = set(bigrams_b.keys())
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return 100 * intersection / union if union > 0 else 0
```

**What N-grams Detect**:
- Repeated phrases (e.g., "قل أعوذ برب" - "Say: I seek refuge in the Lord")
- Common verse structures
- Parallel linguistic patterns

**Limitations**:
- Still local (doesn't capture long-range dependencies)
- Sensitive to exact word order (rigid matching)

### 3. Sentence Embeddings (35% in Unified-All, 70% in Unified-Semantic)

**Model**: `paraphrase-multilingual-mpnet-base-v2` (Sentence Transformers)

**Purpose**: Captures deep semantic meaning across languages.

**Process**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def compute_sentence_embeddings(surah_text):
    # Split into sentences (by verse markers)
    sentences = surah_text.split('۞')
    
    # Encode each sentence
    embeddings = model.encode(sentences)
    
    # Aggregate: mean pooling
    surah_embedding = np.mean(embeddings, axis=0)
    
    return surah_embedding
```

**Similarity Computation**:
```python
def embedding_similarity(embed_a, embed_b):
    # Cosine similarity
    similarity = np.dot(embed_a, embed_b) / (
        np.linalg.norm(embed_a) * np.linalg.norm(embed_b)
    )
    
    # Convert to percentage (cosine ranges from -1 to 1)
    return 50 * (similarity + 1)  # Maps to 0-100%
```

**What Embeddings Detect**:
- **Semantic similarity**: Understanding meaning beyond exact words
- **Conceptual overlap**: Similar themes with different vocabulary
- **Cross-lingual patterns**: Model trained on 50+ languages
- **Contextual understanding**: Considers surrounding words

**Advantages**:
- Captures synonyms and paraphrases
- Language-agnostic semantic understanding
- Pre-trained on billions of sentence pairs

**Limitations**:
- May miss Classical Arabic nuances (trained mostly on modern text)
- Computationally expensive (768-dimensional vectors)

### 4. AraBERT (15% in Unified-All, 30% in Unified-Semantic)

**Model**: `aubmindlab/bert-base-arabertv02` (Hugging Face)

**Purpose**: Arabic-specific contextual embeddings, aware of Classical Arabic.

**Process**:
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02")

def compute_arabert_embedding(surah_text):
    # Tokenize
    inputs = tokenizer(surah_text, return_tensors='pt', 
                      truncation=True, max_length=512)
    
    # Get contextualized embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embedding
```

**What AraBERT Detects**:
- **Arabic morphology**: Understands root-based word structure
- **Classical Arabic patterns**: Trained on Quranic texts
- **Contextual meaning**: Same word in different contexts
- **Arabic-specific semantics**: Nuances lost in multilingual models

**Training Data**:
- Arabic Wikipedia
- News articles
- **Classical Arabic corpus** (including Quranic texts)
- Modern Standard Arabic

**Why AraBERT + Multilingual?**:
- AraBERT: Classical Arabic expertise
- Multilingual: Broader semantic understanding
- Together: Complementary strengths

---

## Unified Matrices

### Unified-All (Comprehensive)

Combines all 5 methods with optimized weights:

```python
weights = {
    'kl_divergence': 0.30,  # Thematic vocabulary
    'bigrams': 0.10,        # Short phrases
    'trigrams': 0.10,       # Longer phrases
    'embeddings': 0.35,     # Semantic meaning (highest!)
    'arabert': 0.15         # Arabic-specific
}

unified_all[i][j] = sum(
    weights[method] * normalized_matrix[method][i][j]
    for method in weights
)
```

**Rationale for Weights**:
- **Embeddings (35%)**: Most powerful semantic understanding
- **KL Divergence (30%)**: Captures thematic vocabulary overlap
- **AraBERT (15%)**: Classical Arabic nuances
- **N-grams (10% each)**: Phrase patterns (less critical)

### Unified-Semantic (Pure Conceptual)

Focuses only on deep semantic methods:

```python
weights = {
    'embeddings': 0.70,  # Universal semantic patterns
    'arabert': 0.30      # Arabic-specific semantics
}

unified_semantic[i][j] = (
    0.70 * embeddings_matrix[i][j] +
    0.30 * arabert_matrix[i][j]
)
```

**Use Case**:
- Emphasizes **meaning over structure**
- Ignores word frequency and phrase patterns
- Best for discovering **conceptual relationships**

---

## Medinan/Makkan Classification

### Important: NOT Computed by the System!

**The Medinan/Makkan labels are HARDCODED from Islamic scholarship, NOT detected by semantic analysis.**

### Source Code

```python
# This list is manually encoded based on scholarly consensus
MEDINAN_SURAHS = [
    2, 3, 4, 5, 8, 9, 13, 22, 24, 33, 47, 48, 49, 
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 98, 110
]

MAKKAN_SURAHS = [
    i for i in range(1, 115) if i not in MEDINAN_SURAHS
]
```

### How Labels Are Used

**In explanations and results**, when I write:

> "Both Medinan surahs with extensive legislative guidance"

**I am doing**:
1. Looking up the surah numbers in the hardcoded `MEDINAN_SURAHS` list
2. Finding they are both Medinan (from Islamic scholarship)
3. Adding this as **contextual information** for human understanding
4. Describing the **themes** (what the models actually detected)

**What the semantic system does**:
- ✅ Detects thematic content (legislation, stories, nature, etc.)
- ✅ Computes similarity based on themes
- ❌ Does NOT know or detect historical period
- ❌ Does NOT use Medinan/Makkan in any computation

### Why This Matters

**Statistical Evidence** (from analysis of all 6,441 pairs):

| Pair Type | Count | Mean Similarity | Std Dev |
|-----------|-------|-----------------|---------|
| Same Period (Med-Med or Mak-Mak) | 4,216 | 79.87% | 11.53% |
| Mixed Period (Med-Mak) | 2,225 | **83.15%** | 10.82% |

**Key Finding**: Mixed-period pairs are actually **3.28% MORE similar** than same-period pairs!

**Conclusion**: The semantic models are **theme-based, not period-based**.

### How Themes Correlate with Period

While models don't detect period, **thematic content correlates** with Medinan/Makkan:

**Medinan Themes** (detected by models):
- Legal vocabulary: طلاق (divorce), نكاح (marriage), ميراث (inheritance)
- Community terms: مؤمنين (believers), أزواج (wives)
- Interfaith: أهل الكتاب (People of the Book)
- Detailed procedural language

**Makkan Themes** (detected by models):
- Nature vocabulary: سماء (sky), جبال (mountains), نحل (bees)
- Prophetic stories: نوح (Noah), موسى (Moses)
- Afterlife: جنة (Paradise), نار (Fire)
- Rhetorical style, shorter verses

**The Causality**:
```
Medinan period → Community needs → Legislative themes
                                          ↓
                                   Models detect themes
                                          ↓
                                   High similarity with other
                                   legislative-themed surahs
```

**NOT**:
```
Medinan period → Models detect period → High similarity
```

### Example: Top Similar Pair

**Surah 16 (An-Nahl, Makkan) ↔ Surah 33 (Al-Ahzab, Medinan): 97.49%**

- **Different periods**: Makkan vs Medinan
- **Highest similarity**: 97.49% (top pair!)
- **Why?**: Both discuss Allah's blessings, community conduct, guidance
- **Proof**: Models don't use period information (otherwise mixed pairs would have lower similarity)

### Source of Classification

The Medinan/Makkan classification comes from:
- Traditional Islamic scholarship (1400+ years of consensus)
- Historical records of revelation context
- Well-established in Quranic studies

**References**:
- Ibn Kathir's Tafsir
- Jalalayn's Tafsir
- Modern scholarly works (e.g., Angelika Neuwirth, Muhammad Abdel Haleem)

---

## Asymmetric Matrix Design

### Why Asymmetric?

**Traditional similarity matrices** are symmetric: `Sim(A, B) = Sim(B, A)`

**This project uses asymmetric matrices**: `Sim(A→B) ≠ Sim(B→A)`

### Rationale

**KL Divergence** is inherently asymmetric:
```python
KL(P||Q) ≠ KL(Q||P)
```

**Interpretation**:
- `Sim(A→B)`: How much of A's themes are present in B?
- `Sim(B→A)`: How much of B's themes are present in A?

### Example

**Surah 2 (Al-Baqarah, 286 verses) vs Surah 65 (At-Talaq, 12 verses)**

- `Sim(2→65) = 85%`: Most of Talaq's themes are in Baqarah
- `Sim(65→2) = 92%`: Talaq is highly focused on themes also in Baqarah
- **Asymmetry**: 7% difference

**Why?**
- Surah 2 covers many themes (broad)
- Surah 65 focuses on divorce law (narrow subset of Surah 2's topics)
- Direction matters!

### Computing Asymmetry

```python
# Forward direction
similarity_A_to_B = compute_similarity(A, B)

# Reverse direction
similarity_B_to_A = compute_similarity(B, A)

# Asymmetry measure
asymmetry = abs(similarity_A_to_B - similarity_B_to_A)

# Average (for symmetric view)
average_similarity = (similarity_A_to_B + similarity_B_to_A) / 2
```

### Use Cases

- **Network analysis**: Directed graphs of thematic influence
- **Causality**: Does A "contain" B's themes more than vice versa?
- **Hierarchical relationships**: Parent/child thematic structures

---

## Normalization

### Why Normalize?

Different methods produce different scales:
- KL divergence: 0 to ~10 (lower = more similar)
- Cosine similarity: -1 to +1 (higher = more similar)
- Jaccard index: 0 to 1 (higher = more similar)

**Solution**: Map all to **0-100%** for intuitive interpretation.

### Method-Specific Normalization

#### KL Divergence
```python
# Original: lower is better
kl_raw = compute_kl(A, B)  # Range: 0 to ~10

# Invert and normalize
kl_similarity = 100 * exp(-kl_raw / 3.0)  # Maps to 0-100%
```

#### Cosine Similarity
```python
# Original: -1 to +1
cosine = dot(A, B) / (norm(A) * norm(B))

# Shift and scale
similarity = 50 * (cosine + 1)  # Maps to 0-100%
```

#### Jaccard Index
```python
# Original: 0 to 1
jaccard = len(A & B) / len(A | B)

# Scale
similarity = 100 * jaccard  # Maps to 0-100%
```

### Why 0-100%?

- ✅ Intuitive (percentage interpretation)
- ✅ Consistent across methods
- ✅ Easy to weight and combine
- ✅ Human-readable in visualizations

---

## Evaluation Metrics

### Individual Method Metrics

For each analysis method, we report:

1. **Similarity Score** (0-100%)
2. **Asymmetry** (difference between A→B and B→A)
3. **Average Similarity** (bidirectional mean)

### Unified Matrix Metrics

1. **Unified-All Score**: Weighted combination of all 5 methods
2. **Unified-Semantic Score**: Semantic methods only (embeddings + AraBERT)
3. **Semantic Boost**: Difference between Unified-Semantic and Unified-All

### Sample Pair Analysis

For 10 selected surah pairs, we compute:

- Bidirectional scores for all methods
- Thematic explanations (manually curated)
- Linguistic patterns (Classical Arabic roots and structures)
- Structural similarities (verse patterns, length, style)

### Statistical Summary

From the full 114×114 matrix:

- **Top 10 Most Similar Pairs**: Highest semantic overlap
- **Top 10 Least Similar Pairs**: Greatest thematic diversity
- **Distribution Statistics**: Mean, median, std dev, range
- **Period Analysis**: Same-period vs mixed-period similarity

---

## Computational Details

### Hardware Requirements

- **CPU**: Multi-core recommended (for parallel embedding computation)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~500MB (including models and results)

### Runtime

On a typical machine (MacBook Pro, M1):

- **Data extraction**: ~2 minutes
- **Preprocessing**: ~1 minute
- **KL Divergence**: ~5 minutes
- **N-grams**: ~3 minutes
- **Sentence Embeddings**: ~15 minutes (first run downloads 420MB model)
- **AraBERT**: ~20 minutes (first run downloads 543MB model)
- **Unified computation**: ~2 minutes
- **Visualization**: ~3 minutes

**Total**: ~50 minutes (first run), ~30 minutes (subsequent runs with cached models)

### Caching

Results are cached to avoid recomputation:

- `results/matrices/*.csv`: Individual method matrices
- `results/matrices/unified_*.csv`: Unified matrices
- Model weights cached by Hugging Face/Sentence Transformers

---

## References

### NLP Libraries

- **Sentence Transformers**: Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **AraBERT**: Antoun et al. (2020). "AraBERT: Transformer-based Model for Arabic Language Understanding"
- **Transformers**: Hugging Face (2020). "Transformers: State-of-the-art Natural Language Processing"

### Quranic Studies

- **Quran.com API**: Open-source Quranic text API (2015-present)
- **Uthmanic Script**: Standard orthography for digital Quran texts
- **Medinan/Makkan Classification**: Traditional Islamic scholarship (Ibn Kathir, Jalalayn, et al.)

### Statistical Methods

- **KL Divergence**: Kullback & Leibler (1951). "On Information and Sufficiency"
- **Cosine Similarity**: Salton & McGill (1983). "Introduction to Modern Information Retrieval"
- **Jaccard Index**: Jaccard (1912). "The Distribution of the Flora in the Alpine Zone"

---

## Reproducibility

### Exact Package Versions

See `requirements.txt` and `requirements_advanced.txt` for pinned versions.

Key dependencies:
- `transformers==4.30.0`
- `sentence-transformers==2.2.2`
- `torch==2.0.1`
- `numpy==1.24.3`
- `pandas==2.0.2`

### Random Seed

Set for reproducibility:
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

### Data Source

All data extracted from **Quran.com API** on the date of analysis.

Verification script ensures consistency: `verify_extraction.py`

---

## Future Enhancements

### Planned Features

1. **Syntactic Parsing**: Dependency trees for structural analysis
2. **Topic Modeling**: LDA for automated theme discovery
3. **Temporal Analysis**: Chronological ordering effects
4. **Word Embeddings**: Word2Vec/FastText for Arabic
5. **Interactive Visualization**: Web-based matrix explorer

### Research Directions

1. **Transfer Learning**: Fine-tune AraBERT specifically on Quran
2. **Attention Analysis**: Visualize what BERT attends to
3. **Comparative Analysis**: Quran vs Hadith vs classical Arabic poetry
4. **Causal Inference**: Directional thematic influence

---

**Last Updated**: October 16, 2025  
**Author**: Quran Semantic Analysis Project  
**License**: MIT

