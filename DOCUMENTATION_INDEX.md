# Documentation Index

Complete guide to all project documentation.

---

## 📚 Quick Links

| Document | Purpose | Size |
|----------|---------|------|
| [README.md](README.md) | Project overview & results | Main file |
| [QUICKSTART.md](docs/QUICKSTART.md) | Installation & basic usage | 176 lines |
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Technical details & algorithms | 709 lines |
| [CLASSICAL_ARABIC.md](docs/CLASSICAL_ARABIC.md) | Language considerations | 271 lines |
| [SEMANTIC_VS_HISTORICAL.md](docs/SEMANTIC_VS_HISTORICAL.md) | Theme vs period detection | 298 lines |
| [MEDINAN_MAKKAN_LABELING.md](docs/MEDINAN_MAKKAN_LABELING.md) | How labels are added | 396 lines |

---

## 🎯 Where to Start

### New Users
1. Start with [README.md](README.md) for project overview
2. Follow [QUICKSTART.md](docs/QUICKSTART.md) to install and run
3. Check results in `results/sample_pairs/ENHANCED_PAIRS_ANALYSIS.md`

### Technical Users
1. Read [METHODOLOGY.md](docs/METHODOLOGY.md) for complete technical details
2. Understand [CLASSICAL_ARABIC.md](docs/CLASSICAL_ARABIC.md) for language context
3. Review code in `src/analysis/` directory

### Researchers
1. Study [METHODOLOGY.md](docs/METHODOLOGY.md) for algorithms and references
2. Read [SEMANTIC_VS_HISTORICAL.md](docs/SEMANTIC_VS_HISTORICAL.md) for key findings
3. Examine [MEDINAN_MAKKAN_LABELING.md](docs/MEDINAN_MAKKAN_LABELING.md) for data provenance

---

## 📖 Document Descriptions

### README.md
**Main project documentation**
- What the project does
- Analysis results (top pairs)
- Quick start instructions
- Key features
- Project structure

### docs/QUICKSTART.md
**Installation & usage guide**
- Prerequisites
- Step-by-step installation
- Running the analysis
- Viewing results
- Troubleshooting

### docs/METHODOLOGY.md ⭐
**Comprehensive technical documentation**

Covers:
- Data extraction (Quran.com API)
- Arabic text preprocessing
- All 5 analysis methods:
  - KL Divergence (with formulas)
  - N-gram analysis (bigrams, trigrams)
  - Sentence embeddings (multilingual)
  - AraBERT (Classical Arabic-aware)
- Unified matrix computation
- **Medinan/Makkan classification source** (hardcoded from scholarship!)
- Asymmetric matrix design
- Normalization to 0-100%
- Evaluation metrics
- Computational details
- Academic references

### docs/CLASSICAL_ARABIC.md
**Language considerations**
- Classical Arabic vs Modern Standard Arabic
- Why AraBERT works for Quranic text
- Preprocessing justification
- Linguistic features

### docs/SEMANTIC_VS_HISTORICAL.md ⭐
**Key finding: Semantic similarity ≠ period detection**

Proves:
- Models detect **themes**, not **historical periods**
- Mixed-period pairs are 3.28% MORE similar than same-period
- Top pair (97.49%) is Makkan ↔ Medinan (different periods!)
- Statistical evidence from all 6,441 pairs

### docs/MEDINAN_MAKKAN_LABELING.md ⭐
**How Medinan/Makkan labels are added**

Explains:
- Labels are **hardcoded** from Islamic scholarship
- **NOT computed** by semantic analysis
- Two-step process (analysis + manual labeling)
- Where labels appear in results
- Full source code locations
- References to traditional scholarship

---

## 🔍 Frequently Asked Questions

### Q: How does the system detect if surahs are Medinan or Makkan?
**A**: It doesn't! The classification is hardcoded from Islamic scholarship (25 Medinan, 89 Makkan).

**See**: [MEDINAN_MAKKAN_LABELING.md](docs/MEDINAN_MAKKAN_LABELING.md)

### Q: Why are some Makkan and Medinan surahs highly similar?
**A**: Because semantic models detect **themes** (blessings, guidance, community), not **periods**. Themes transcend historical context.

**See**: [SEMANTIC_VS_HISTORICAL.md](docs/SEMANTIC_VS_HISTORICAL.md)

### Q: What does "97.49% similarity" mean?
**A**: A normalized score (0-100%) indicating thematic and semantic overlap based on 5 NLP methods.

**See**: [METHODOLOGY.md](docs/METHODOLOGY.md) → Normalization section

### Q: Why is the matrix asymmetric?
**A**: Because KL divergence is directional: how much of A's themes are in B ≠ how much of B's themes are in A.

**See**: [METHODOLOGY.md](docs/METHODOLOGY.md) → Asymmetric Matrix Design

### Q: Can the system handle Classical Arabic?
**A**: Yes! AraBERT is trained on Quranic texts and classical Arabic corpus.

**See**: [CLASSICAL_ARABIC.md](docs/CLASSICAL_ARABIC.md)

### Q: What are the weights for Unified-Semantic?
**A**: 70% Sentence Embeddings + 30% AraBERT (pure semantic, no word frequency or n-grams).

**See**: [METHODOLOGY.md](docs/METHODOLOGY.md) → Unified Matrices

---

## 📊 Analysis Results

### Primary Outputs
1. **Unified-All Matrix** (5 methods): `results/matrices/unified_all_matrix.csv`
2. **Unified-Semantic Matrix** (2 methods): `results/matrices/unified_semantic_matrix.csv`
3. **Sample Pairs Analysis**: `results/sample_pairs/ENHANCED_PAIRS_ANALYSIS.md`

### Key Findings
- **Highest similarity**: Surah 16 ↔ 33 (97.49%) - Makkan & Medinan!
- **Lowest similarity**: Surah 92 ↔ 114 (35.34%)
- **Average similarity**: 81.00%
- **Total pairs analyzed**: 6,441

**See**: [README.md](README.md) → Analysis Results

---

## 🔬 Methodology Summary

### 5 Analysis Methods

| Method | Weight (All) | Weight (Semantic) | Purpose |
|--------|--------------|-------------------|---------|
| KL Divergence | 30% | - | Word frequency distributions |
| Bigrams | 10% | - | 2-word phrase patterns |
| Trigrams | 10% | - | 3-word phrase patterns |
| Sentence Embeddings | 35% | 70% | Universal semantic meaning |
| AraBERT | 15% | 30% | Arabic-specific context |

### Process Flow
```
1. Extract Quranic text (Quran.com API)
   ↓
2. Preprocess Arabic (normalize, tokenize)
   ↓
3. Compute 5 similarity matrices (114×114 each)
   ↓
4. Normalize all to 0-100%
   ↓
5. Combine into unified matrices (weighted)
   ↓
6. Generate visualizations and reports
```

**Full details**: [METHODOLOGY.md](docs/METHODOLOGY.md)

---

## 🎓 Academic Context

### What This Project Demonstrates

1. **Computational Quranic Studies**: Applying modern NLP to sacred texts
2. **Classical Arabic NLP**: Handling language not in most training corpora
3. **Asymmetric Similarity**: Directional thematic relationships
4. **Multi-Method Integration**: Combining frequency, syntactic, and semantic approaches
5. **Transparent Methodology**: Clear about what's computed vs what's known

### Limitations
- No human tafsir (interpretation) validation
- Limited to lexical and semantic features
- Doesn't capture rhetorical devices (metaphor, allegory)
- Context of revelation not modeled
- Inter-verse relationships not analyzed

**See**: [METHODOLOGY.md](docs/METHODOLOGY.md) → Future Enhancements

---

## 💻 Code Organization

```
matrix-project/
├── docs/                          # All documentation
│   ├── QUICKSTART.md
│   ├── METHODOLOGY.md             # Technical details
│   ├── CLASSICAL_ARABIC.md
│   ├── SEMANTIC_VS_HISTORICAL.md
│   └── MEDINAN_MAKKAN_LABELING.md
├── src/
│   └── analysis/                  # Main Python modules
│       ├── simple_analysis.py     # KL divergence
│       ├── normalized_analysis.py # Normalization
│       ├── advanced_analysis.py   # N-grams, embeddings, AraBERT
│       ├── unified_analysis.py    # Unified matrices
│       ├── sample_pairs_analysis.py
│       └── enhanced_pairs_analysis.py
├── scripts/
│   └── run_complete_analysis.py   # Main execution script
├── results/                       # All outputs
│   ├── matrices/                  # CSV matrices
│   ├── visualizations/            # PNG heatmaps
│   └── sample_pairs/              # Detailed analysis
└── quran_data/                    # Extracted Quran text
```

---

## 🚀 Running the Analysis

```bash
# One command to run everything
python scripts/run_complete_analysis.py

# What it does:
# 1. Computes Unified-All matrix (5 methods)
# 2. Computes Unified-Semantic matrix (2 methods)
# 3. Analyzes 10 sample pairs in detail
# 4. Formats all CSVs to 2 decimal places
# 5. Generates visualizations
```

**Full instructions**: [QUICKSTART.md](docs/QUICKSTART.md)

---

## 📝 Citation

If you use this project in research, please cite:

```bibtex
@software{quran_semantic_analysis_2025,
  title = {Quran Semantic Analysis: Asymmetric Relationship Matrix},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/saqibraza/relationships}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## 🤝 Contributing

This is a research project. For questions or contributions:
1. Open an issue on GitHub
2. Read [METHODOLOGY.md](docs/METHODOLOGY.md) first
3. Follow coding standards in existing modules

---

## 🔗 External Resources

### Data Sources
- [Quran.com API](https://quran.com/api) - Quranic text source
- [Tanzil.net](https://tanzil.net) - Alternative Quran database

### Models Used
- [Sentence-BERT](https://www.sbert.net/) - Multilingual embeddings
- [AraBERT](https://github.com/aub-mind/arabert) - Arabic BERT models
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Model hub

### Islamic Scholarship
- Ibn Kathir's Tafsir (traditional Medinan/Makkan classification)
- Jalalayn's Tafsir
- Modern scholarly consensus

---

**Last Updated**: October 16, 2025  
**Total Documentation**: 55KB, 1,850+ lines  
**Status**: Complete and verified ✅
