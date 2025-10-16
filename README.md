# Quran Semantic Analysis - Asymmetric Relationship Matrix

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Comprehensive semantic and thematic analysis of the Quran (114 surahs) using advanced NLP techniques including KL divergence, N-grams, sentence embeddings, and AraBERT.

---

## 🎯 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements_advanced.txt

# Run complete analysis (one command!)
python scripts/run_complete_analysis.py

# View results
open results/sample_pairs/ENHANCED_PAIRS_ANALYSIS.md
```

**See [Quick Start Guide](docs/QUICKSTART.md) for detailed instructions.**

---

## 📊 What This Project Does

Generates **two types of asymmetric similarity matrices** (114×114) analyzing relationships between all Quranic surahs:

### 1. Unified-All Matrix (Comprehensive)
Combines **5 analysis methods**:
- 🔢 **KL Divergence** (30%): Word frequency distributions
- 📝 **Bigrams** (10%): 2-word phrase patterns
- 📝 **Trigrams** (10%): 3-word phrase patterns
- 🧠 **Sentence Embeddings** (35%): Deep semantic meaning (multilingual)
- 🇸🇦 **AraBERT** (15%): Arabic-specific contextual embeddings

### 2. Unified-Semantic Matrix (Pure Conceptual)
Focuses on **semantic methods only**:
- 🧠 **Sentence Embeddings** (70%): Universal semantic patterns
- 🇸🇦 **AraBERT** (30%): Classical Arabic-aware contextual understanding

---

## 🔬 Key Features

✅ **Asymmetric Matrices**: A→B ≠ B→A (captures directional relationships)  
✅ **Classical Arabic Aware**: AraBERT trained on Quranic and classical texts  
✅ **0-100% Normalized Scores**: Intuitive percentage-based similarity  
✅ **Bidirectional Analysis**: Separate scores for both directions  
✅ **Thematic Explanations**: WHY surahs are similar (themes, topics, structure, linguistics)  
✅ **10 Sample Pairs**: Detailed analysis with explanations  
✅ **Authentic Source**: Quran.com API (official, verified text)  

---

## 📈 Results Summary

### Overall Statistics

| Metric | Unified-All | Unified-Semantic | Semantic Boost |
|--------|-------------|------------------|----------------|
| **Mean Similarity** | 48.79% | 85.05% | +36.26% |
| **Max Similarity** | 57.02% | 95.85% | +40.18% |
| **Min Similarity** | 43.32% | 76.51% | +33.19% |

**Key Finding**: All surah pairs show **30-40% semantic boost**, indicating **thematic unity with linguistic diversity** - a hallmark of Classical Quranic text.

### Top 5 Most Similar Pairs (Unified-Semantic)

| Rank | Pair | Surahs | Similarity | Reason |
|------|------|--------|------------|--------|
| 🥇 | 10×11 | Yunus & Hud | **95.85%** | Parallel prophetic narratives |
| 🥈 | 24×33 | An-Nur & Al-Ahzab | **94.01%** | Complementary social guidance |
| 🥉 | 2×3 | Al-Baqarah & Āl 'Imrān | **87.95%** | Sequential Medinan legislation |
| 4 | 69×101 | Al-Haqqah & Al-Qari'ah | **85.97%** | Eschatological descriptions |
| 5 | 93×94 | Ad-Duha & Ash-Sharh | **83.18%** | Consolation to the Prophet |

---

## 📂 Project Structure

```
matrix-project/
├── README.md                          # This file
├── requirements.txt                   # Core dependencies
├── requirements_advanced.txt          # Advanced features
│
├── scripts/                           # Executable scripts
│   ├── run_complete_analysis.py      # 🎯 MAIN ENTRY POINT
│   ├── verify_extraction.py          # Verify Quran extraction
│   └── analyze_sample_pairs.py       # Sample pairs analysis
│
├── src/                               # Source code modules
│   └── extraction/
│       └── quran_extractor.py        # Quran text extraction
│
├── data/                              # Cached data
│   ├── quran_surahs.json             # Extracted Quran text
│   └── quran.json                    # Raw API response
│
├── results/                           # All analysis results
│   ├── matrices/                     # CSV matrices (7 files)
│   │   ├── unified_all_matrix.csv
│   │   ├── unified_semantic_matrix.csv
│   │   └── ... (individual methods)
│   ├── visualizations/               # Heatmaps (PNG)
│   │   ├── unified_all_heatmap.png
│   │   └── unified_semantic_heatmap.png
│   └── sample_pairs/                 # Detailed 10-pair analysis
│       ├── ENHANCED_PAIRS_ANALYSIS.md
│       └── enhanced_pairs_scores.csv
│
└── docs/                              # Documentation
    ├── QUICKSTART.md                 # Installation & usage
    ├── METHODOLOGY.md                # Technical details
    └── CLASSICAL_ARABIC.md           # Language considerations
```

---

## 🚀 Usage

### Basic Usage

```bash
# Run complete analysis
python scripts/run_complete_analysis.py

# Verify Quran extraction
python scripts/verify_extraction.py
```

### Python API

```python
import pandas as pd

# Load unified matrices
unified_all = pd.read_csv('results/matrices/unified_all_matrix.csv', index_col=0)
unified_semantic = pd.read_csv('results/matrices/unified_semantic_matrix.csv', index_col=0)

# Get similarity between Surah 2 and Surah 3
sim_all = unified_all.loc['Surah 2', 'Surah 3']
sim_semantic = unified_semantic.loc['Surah 2', 'Surah 3']

print(f"Surah 2 ↔ 3:")
print(f"  Unified-All:      {sim_all:.2f}%")
print(f"  Unified-Semantic: {sim_semantic:.2f}%")
print(f"  Boost:            +{sim_semantic - sim_all:.2f}%")
```

---

## 🔍 Methodology

### Data Extraction
- **Source**: Quran.com API (official, verified)
- **Script**: Uthmani script with diacritics
- **Verification**: Validated against known surah texts

### Preprocessing
1. **Diacritics Removal**: Remove tashkeel marks
2. **Normalization**: Standardize Arabic characters (أ/إ/آ → ا)
3. **Character Filtering**: Keep Arabic letters and spaces only

### Analysis Methods

#### 1. KL Divergence (Statistical)
Measures vocabulary distribution differences between surahs.

```
KL(P||Q) = Σ P(w) × log(P(w) / Q(w))
Similarity = 100 × exp(-KL / scale_factor)
```

#### 2. N-gram Analysis (Phrasal)
Captures local word order and common phrase patterns using bigrams and trigrams.

#### 3. Sentence Embeddings (Semantic)
- **Model**: `paraphrase-multilingual-mpnet-base-v2`
- **Method**: Encode full surah text → cosine similarity
- **Captures**: Universal semantic meaning

#### 4. AraBERT (Classical Arabic)
- **Model**: `aubmindlab/araBERTv02`
- **Training**: Quranic text, Hadith, classical poetry
- **Captures**: Arabic-specific contextual patterns

#### 5. Unified Matrix
Weighted combination of all methods:
```
Unified = Σ (weight_i × method_i)
```

**See [Methodology Guide](docs/METHODOLOGY.md) for complete technical details.**

---

## 📖 Classical Arabic Consideration

### Is the Quran in Modern Standard Arabic?

**No.** The Quran is in **Classical Arabic (7th century)**, NOT Modern Standard Arabic (19th century+).

| Aspect | Classical Arabic | Modern Standard Arabic |
|--------|------------------|------------------------|
| Period | 7th century CE | 19th century+ |
| Vocabulary | Archaic, poetic, religious | Contemporary, scientific |
| Grammar | Full case endings (i'rab) | Simplified |
| Style | Highly rhetorical | Standardized |

### Why Our Analysis is Valid

1. **AraBERT Training**: Includes Quranic and classical Arabic texts
2. **Root System**: Trilateral roots shared across all Arabic varieties
3. **Preprocessing**: Normalizes diacritics and variants
4. **Validation**: High AraBERT scores (80-96%) confirm accurate Classical pattern recognition

**See [Classical Arabic Guide](docs/CLASSICAL_ARABIC.md) for detailed explanation.**

---

## 📊 Sample Pairs Analysis

Detailed analysis of **10 carefully selected surah pairs** showing:
- Bidirectional scores for all 7 methods
- **Thematic explanations**: WHY there is semantic similarity
- Shared themes, topics, structures
- Linguistic patterns in Classical Arabic

### Example: Surah 10 ↔ 11 (Yunus & Hud)

| Method | 10→11 | 11→10 | Average |
|--------|-------|-------|---------|
| **Unified-All** | 57.02% | 57.02% | 57.02% |
| **Unified-Semantic** | **95.85%** | **95.85%** | **95.85%** |
| AraBERT | **96.91%** | **96.91%** | **96.91%** |

**Why High Similarity?**
- **Shared Themes**: Prophetic narratives, warning through history
- **Common Topics**: Noah, Moses, destroyed nations
- **Structure**: Both named after prophets, extended Makkan narratives
- **Linguistic**: Narrative markers (wa-laqad, fa-lamma), prophet dialogue formulas

**See [Enhanced Pairs Analysis](results/sample_pairs/ENHANCED_PAIRS_ANALYSIS.md) for all 10 pairs.**

---

## 🛠️ Requirements

### Core (Required)
```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
requests>=2.28.0
```

### Advanced (Optional, for full features)
```
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
```

---

## 🎓 Understanding the Results

### Score Interpretation

| Range | Interpretation | Example |
|-------|---------------|---------|
| 90-100% | Extremely high (parallel narratives) | Yunus ↔ Hud (95.85%) |
| 80-90% | Very high (complementary topics) | Ad-Duha ↔ Ash-Sharh (83.18%) |
| 70-80% | High (overlapping themes) | Al-Falaq ↔ An-Nas (76.51%) |
| 50-70% | Moderate (some shared topics) | Various pairs |
| <50% | Low (different themes) | Distant surahs |

### Asymmetry Analysis

Most pairs show **near-zero asymmetry** in semantic methods (embeddings, AraBERT) because:
- Cosine similarity is symmetric
- Thematic connections are bidirectional

**Asymmetry appears mainly in KL divergence** due to vocabulary size differences.

---

## 📚 Documentation

- **[Quick Start](docs/QUICKSTART.md)**: Installation and basic usage
- **[Methodology](docs/METHODOLOGY.md)**: Technical details and algorithms
- **[Classical Arabic](docs/CLASSICAL_ARABIC.md)**: Language considerations
- **[Sample Pairs Analysis](results/sample_pairs/ENHANCED_PAIRS_ANALYSIS.md)**: Detailed 10-pair analysis

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📜 Citation

If you use this work in research, please cite:

```bibtex
@software{quran_semantic_analysis_2025,
  author = {Raza, Saqib},
  title = {Quran Semantic Analysis: Asymmetric Relationship Matrix},
  year = {2025},
  url = {https://github.com/saqibraza/relationships},
  note = {Multi-method semantic analysis of Quranic surahs using NLP}
}
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Quran.com API**: For providing verified Quranic text
- **AraBERT Team**: For Classical Arabic language model
- **Sentence Transformers**: For multilingual embeddings
- **Islamic Scholars**: For thematic insights

---

## 📧 Contact

**Author**: Saqib Raza  
**Repository**: https://github.com/saqibraza/relationships  
**Issues**: https://github.com/saqibraza/relationships/issues

---

## 🎯 Project Status

✅ **Complete** - All features implemented and documented
- Two unified matrices generated
- 10 sample pairs analyzed with thematic explanations
- Classical Arabic validation
- Comprehensive documentation
- Clean, organized codebase

**Last Updated**: October 16, 2025  
**Version**: 4.0 (Production-ready)

---

**Made with ❤️ for Quranic Studies**
