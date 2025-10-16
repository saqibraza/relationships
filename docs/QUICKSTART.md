# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/saqibraza/relationships.git
cd relationships

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements_advanced.txt  # Optional: for advanced features

# 4. Verify installation
python scripts/verify_extraction.py
```

## Run Analysis (1 command)

```bash
python scripts/run_complete_analysis.py
```

This generates:
- ✅ Two unified similarity matrices (All & Semantic)
- ✅ Individual method matrices (KL, bigrams, trigrams, embeddings, AraBERT)
- ✅ Sample pairs analysis with thematic explanations
- ✅ Visualizations (heatmaps)

**Duration**: 15-30 minutes (depending on hardware and whether models need downloading)

## View Results

```bash
# Main results directory
ls results/

results/
├── matrices/                    # All CSV matrices (7 files)
│   ├── unified_all_matrix.csv
│   ├── unified_semantic_matrix.csv
│   ├── kl_divergence_matrix.csv
│   ├── bigrams_matrix.csv
│   ├── trigrams_matrix.csv
│   ├── embeddings_matrix.csv
│   └── arabert_matrix.csv
│
├── visualizations/              # Heatmaps
│   ├── unified_all_heatmap.png
│   └── unified_semantic_heatmap.png
│
└── sample_pairs/                # Detailed 10-pair analysis
    ├── ENHANCED_PAIRS_ANALYSIS.md
    └── enhanced_pairs_scores.csv
```

## Quick Python Usage

```python
import pandas as pd
import numpy as np

# Load unified matrices
unified_all = pd.read_csv('results/matrices/unified_all_matrix.csv', index_col=0)
unified_semantic = pd.read_csv('results/matrices/unified_semantic_matrix.csv', index_col=0)

# Get similarity between Surah 2 and Surah 3
sim_all = unified_all.loc['Surah 2', 'Surah 3']
sim_semantic = unified_semantic.loc['Surah 2', 'Surah 3']

print(f"Surah 2 ↔ 3:")
print(f"  Unified-All:      {sim_all:.2f}%")
print(f"  Unified-Semantic: {sim_semantic:.2f}%")
print(f"  Semantic Boost:   +{sim_semantic - sim_all:.2f}%")

# Find most similar surahs to Surah 1
similarities = unified_semantic.loc['Surah 1'].sort_values(ascending=False)
print("\nTop 5 most similar to Surah 1:")
print(similarities.head(6))  # 6 because first is self-similarity (100%)
```

## What Do the Numbers Mean?

| Score Range | Interpretation |
|-------------|----------------|
| **90-100%** | Extremely high similarity (parallel narratives, same themes) |
| **80-90%** | Very high similarity (related topics, complementary) |
| **70-80%** | High similarity (overlapping themes) |
| **50-70%** | Moderate similarity (some shared topics) |
| **30-50%** | Low-moderate similarity (distant connections) |
| **0-30%** | Low similarity (different themes/topics) |

## Understanding the Two Unified Matrices

### Unified-All (Comprehensive)
- Combines **all 5 methods**: KL divergence (30%), bigrams (10%), trigrams (10%), embeddings (35%), AraBERT (15%)
- Best for: Overall similarity considering vocabulary + semantics
- Use when: You want a balanced, comprehensive measure

### Unified-Semantic (Conceptual)
- Combines **semantic methods only**: Embeddings (70%), AraBERT (30%)
- Best for: Pure thematic/conceptual similarity regardless of vocabulary
- Use when: You want to find surahs with similar meaning using different words

## Common Tasks

### Find Top Similar Pairs
```python
import pandas as pd
import numpy as np

matrix = pd.read_csv('results/matrices/unified_semantic_matrix.csv', index_col=0)
values = matrix.values

# Get upper triangle (avoid self-similarity and duplicates)
n = len(matrix)
similarities = []
for i in range(n):
    for j in range(i+1, n):
        similarities.append({
            'Surah_A': i+1,
            'Surah_B': j+1,
            'Similarity': values[i, j]
        })

df = pd.DataFrame(similarities)
print(df.nlargest(10, 'Similarity'))
```

### Compare Methods for Specific Pair
```python
pair = (2, 3)  # Al-Baqarah and Āl 'Imrān

methods = ['unified_all', 'unified_semantic', 'kl_divergence', 
           'bigrams', 'trigrams', 'embeddings', 'arabert']

for method in methods:
    matrix = pd.read_csv(f'results/matrices/{method}_matrix.csv', index_col=0)
    score = matrix.loc[f'Surah {pair[0]}', f'Surah {pair[1]}']
    print(f"{method:20s}: {score:.2f}%")
```

## Troubleshooting

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### "CUDA out of memory" or slow processing
The analysis will automatically use:
- CUDA if available (fastest)
- MPS (Apple Silicon) if available (fast)
- CPU otherwise (slower but works)

### "ModuleNotFoundError"
Make sure you're in the project root and virtual environment is activated:
```bash
source venv/bin/activate
python scripts/run_complete_analysis.py
```

## Next Steps

- Read the methodology: `docs/METHODOLOGY.md`
- Understand Classical Arabic handling: `docs/CLASSICAL_ARABIC.md`
- View sample pairs analysis: `results/sample_pairs/ENHANCED_PAIRS_ANALYSIS.md`
- Explore the main README: `README.md`

## Questions?

See the full README or open an issue on GitHub.
