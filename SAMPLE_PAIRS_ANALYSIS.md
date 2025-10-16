# Detailed Analysis of Sample Surah Pairs

This document provides comprehensive analysis of specific surah pairs across all methods, including **two types of unified scores**:

1. **Unified-All**: Combines all 5 methods (KL divergence, bigrams, trigrams, embeddings, AraBERT)
2. **Unified-Semantic**: Combines only semantic methods (embeddings 70% + AraBERT 30%)

## Analysis Methods

| Method | Weight in Unified-All | Weight in Unified-Semantic | What It Measures |
|--------|---------------------|---------------------------|------------------|
| **KL Divergence** | 30% | 0% | Word frequency distributions |
| **Bigrams** | 10% | 0% | 2-word phrase patterns |
| **Trigrams** | 10% | 0% | 3-word phrase patterns |
| **Sentence Embeddings** | 35% | 70% | Deep semantic meaning |
| **AraBERT** | 15% | 30% | Arabic-specific contextual embeddings |

## Reading the Results

For each pair, we show **bidirectional scores** (A→B and B→A separately):
- **A→B**: Similarity from Surah A to Surah B
- **B→A**: Similarity from Surah B to Surah A
- **Asymmetry**: Difference (A→B minus B→A)
- **Average**: Mean of both directions

---

## Pair 1: Surah 113 (Al-Falaq) ↔ Surah 114 (An-Nas)

### Summary Table

| Method | A→B | B→A | Asymmetry | Average | Interpretation |
|--------|-----|-----|-----------|---------|----------------|
| **UNIFIED-ALL** | **43.31%** | **43.33%** | **-0.02%** | **43.32%** | **Symmetric** |
| **UNIFIED-SEMANTIC** | **76.51%** | **76.51%** | **+0.00%** | **76.51%** | **Symmetric** |
| KL Divergence | 13.10% | 13.18% | -0.08% | 13.14% | Symmetric |
| Bigrams | 8.57% | 8.57% | +0.00% | 8.57% | Symmetric |
| Trigrams | 2.63% | 2.63% | +0.00% | 2.63% | Symmetric |
| Embeddings | 71.31% | 71.31% | +0.00% | 71.31% | Symmetric |
| AraBERT | 88.65% | 88.65% | +0.00% | 88.65% | Symmetric |

### Detailed Analysis

**Unified-All Similarity**: 43.32%  
**Unified-Semantic Similarity**: 76.51%

**Key Insight**: Semantic similarity (76.51%) is **33.19% higher** than overall similarity (43.32%). This indicates strong semantic connection despite lower vocabulary/phrase overlap.

---

## Pair 2: Surah 93 (Ad-Duha) ↔ Surah 94 (Ash-Sharh)

### Summary Table

| Method | A→B | B→A | Asymmetry | Average | Interpretation |
|--------|-----|-----|-----------|---------|----------------|
| **UNIFIED-ALL** | **45.08%** | **45.56%** | **-0.48%** | **45.32%** | **Symmetric** |
| **UNIFIED-SEMANTIC** | **83.18%** | **83.18%** | **+0.00%** | **83.18%** | **Symmetric** |
| KL Divergence | 11.62% | 13.24% | -1.62% | 12.43% | Symmetric |
| Bigrams | 0.00% | 0.00% | +0.00% | 0.00% | Symmetric |
| Trigrams | 0.00% | 0.00% | +0.00% | 0.00% | Symmetric |
| Embeddings | 84.86% | 84.86% | +0.00% | 84.86% | Symmetric |
| AraBERT | 79.27% | 79.27% | +0.00% | 79.27% | Symmetric |

### Detailed Analysis

**Unified-All Similarity**: 45.32%  
**Unified-Semantic Similarity**: 83.18%

**Key Insight**: Semantic similarity (83.18%) is **37.86% higher** than overall similarity (45.32%). This indicates strong semantic connection despite lower vocabulary/phrase overlap.

---

## Pair 3: Surah 69 (Al-Haqqah) ↔ Surah 101 (Al-Qari'ah)

### Summary Table

| Method | A→B | B→A | Asymmetry | Average | Interpretation |
|--------|-----|-----|-----------|---------|----------------|
| **UNIFIED-ALL** | **48.23%** | **47.30%** | **+0.93%** | **47.77%** | **Symmetric** |
| **UNIFIED-SEMANTIC** | **85.97%** | **85.97%** | **+0.00%** | **85.97%** | **Symmetric** |
| KL Divergence | 16.32% | 13.19% | +3.13% | 14.75% | Al-Haqqah → Al-Qari'ah |
| Bigrams | 2.49% | 2.49% | +0.00% | 2.49% | Symmetric |
| Trigrams | 1.04% | 1.04% | +0.00% | 1.04% | Symmetric |
| Embeddings | 89.80% | 89.80% | +0.00% | 89.80% | Symmetric |
| AraBERT | 77.05% | 77.05% | +0.00% | 77.05% | Symmetric |

### Detailed Analysis

**Unified-All Similarity**: 47.77%  
**Unified-Semantic Similarity**: 85.97%

**Key Insight**: Semantic similarity (85.97%) is **38.20% higher** than overall similarity (47.77%). This indicates strong semantic connection despite lower vocabulary/phrase overlap.

---

## Pair 4: Surah 2 (Al-Baqarah) ↔ Surah 3 (Āl 'Imrān)

### Summary Table

| Method | A→B | B→A | Asymmetry | Average | Interpretation |
|--------|-----|-----|-----------|---------|----------------|
| **UNIFIED-ALL** | **57.07%** | **55.62%** | **+1.45%** | **56.34%** | **Symmetric** |
| **UNIFIED-SEMANTIC** | **87.95%** | **87.95%** | **+0.00%** | **87.95%** | **Symmetric** |
| KL Divergence | 40.16% | 35.32% | +4.84% | 37.74% | Al-Baqarah → Āl 'Imrān |
| Bigrams | 7.14% | 7.14% | +0.00% | 7.14% | Symmetric |
| Trigrams | 3.36% | 3.36% | +0.00% | 3.36% | Symmetric |
| Embeddings | 84.29% | 84.29% | +0.00% | 84.29% | Symmetric |
| AraBERT | 96.47% | 96.47% | +0.00% | 96.47% | Symmetric |

### Detailed Analysis

**Unified-All Similarity**: 56.34%  
**Unified-Semantic Similarity**: 87.95%

**Key Insight**: Semantic similarity (87.95%) is **31.61% higher** than overall similarity (56.34%). This indicates strong semantic connection despite lower vocabulary/phrase overlap.

---

## Pair 5: Surah 2 (Al-Baqarah) ↔ Surah 65 (At-Talaq)

### Summary Table

| Method | A→B | B→A | Asymmetry | Average | Interpretation |
|--------|-----|-----|-----------|---------|----------------|
| **UNIFIED-ALL** | **50.96%** | **47.29%** | **+3.67%** | **49.12%** | **Al-Baqarah → At-Talaq** |
| **UNIFIED-SEMANTIC** | **83.34%** | **83.34%** | **+0.00%** | **83.34%** | **Symmetric** |
| KL Divergence | 30.36% | 18.14% | +12.22% | 24.25% | Al-Baqarah → At-Talaq |
| Bigrams | 1.34% | 1.34% | +0.00% | 1.34% | Symmetric |
| Trigrams | 0.46% | 0.46% | +0.00% | 0.46% | Symmetric |
| Embeddings | 78.70% | 78.70% | +0.00% | 78.70% | Symmetric |
| AraBERT | 94.18% | 94.18% | +0.00% | 94.18% | Symmetric |

### Detailed Analysis

**Unified-All Similarity**: 49.12%  
**Unified-Semantic Similarity**: 83.34%

**Key Insight**: Semantic similarity (83.34%) is **34.22% higher** than overall similarity (49.12%). This indicates strong semantic connection despite lower vocabulary/phrase overlap.

**Asymmetry Note**: Surah 2 shows 3.67% higher similarity toward Surah 65 (50.96%) than vice versa (47.29%). This suggests Surah 2 may encompass more of Surah 65's themes.

---

## Pair 6: Surah 24 (An-Nur) ↔ Surah 33 (Al-Ahzab)

### Summary Table

| Method | A→B | B→A | Asymmetry | Average | Interpretation |
|--------|-----|-----|-----------|---------|----------------|
| **UNIFIED-ALL** | **53.78%** | **53.88%** | **-0.10%** | **53.83%** | **Symmetric** |
| **UNIFIED-SEMANTIC** | **94.01%** | **94.01%** | **+0.00%** | **94.01%** | **Symmetric** |
| KL Divergence | 21.31% | 21.66% | -0.35% | 21.48% | Symmetric |
| Bigrams | 3.10% | 3.10% | +0.00% | 3.10% | Symmetric |
| Trigrams | 0.65% | 0.65% | +0.00% | 0.65% | Symmetric |
| Embeddings | 92.85% | 92.85% | +0.00% | 92.85% | Symmetric |
| AraBERT | 96.72% | 96.72% | +0.00% | 96.72% | Symmetric |

### Detailed Analysis

**Unified-All Similarity**: 53.83%  
**Unified-Semantic Similarity**: 94.01%

**Key Insight**: Semantic similarity (94.01%) is **40.18% higher** than overall similarity (53.83%). This indicates strong semantic connection despite lower vocabulary/phrase overlap.

---

## Summary Comparison

### Rankings by Unified-All Score

| Rank | Pair | Unified-All | Unified-Semantic | Difference |
|------|------|-------------|------------------|------------|
| 1 | 2×3 (Al-Baqarah ↔ Āl 'Imrān) | 56.34% | 87.95% | +31.61% |
| 2 | 24×33 (An-Nur ↔ Al-Ahzab) | 53.83% | 94.01% | +40.18% |
| 3 | 2×65 (Al-Baqarah ↔ At-Talaq) | 49.12% | 83.34% | +34.22% |
| 4 | 69×101 (Al-Haqqah ↔ Al-Qari'ah) | 47.77% | 85.97% | +38.20% |
| 5 | 93×94 (Ad-Duha ↔ Ash-Sharh) | 45.32% | 83.18% | +37.86% |
| 6 | 113×114 (Al-Falaq ↔ An-Nas) | 43.32% | 76.51% | +33.19% |

---

**Analysis Date**: October 16, 2025  
**Sample Pairs**: 6  
**Methods**: 7 (5 individual + 2 unified types)  
**Score Format**: Bidirectional (A→B and B→A shown separately)  
