# How Medinan/Makkan Labels Are Added to Results

**Key Point**: The Medinan/Makkan classification is **NOT computed** by the semantic analysis system. It's **manually added** from Islamic scholarship for contextual understanding.

---

## The Two-Step Process

### Step 1: Semantic Analysis (What the System Does)

The NLP models analyze surah text and compute similarity:

```python
# Input: Just the Arabic text
surah_16_text = "بسم الله الرحمن الرحيم..."
surah_33_text = "بسم الله الرحمن الرحيم..."

# Semantic analysis
similarity = compute_semantic_similarity(surah_16_text, surah_33_text)
# Output: 97.49%

# The system has NO IDEA about:
# - When these surahs were revealed (Mecca/Medina)
# - Historical context
# - Revelation circumstances
```

**What the system detects**:
- ✅ Vocabulary overlap (shared words)
- ✅ Thematic patterns (blessings, guidance, community)
- ✅ Semantic meaning (deep contextual understanding)
- ✅ Linguistic structures (sentence patterns)

**What the system does NOT detect**:
- ❌ Historical period (Medinan/Makkan)
- ❌ Revelation location
- ❌ Chronological order
- ❌ Historical circumstances

---

### Step 2: Adding Labels (What I Do Manually)

After the system computes similarity, I add contextual labels:

```python
# Hardcoded classification from Islamic scholarship
MEDINAN_SURAHS = [2, 3, 4, 5, 8, 9, 13, 22, 24, 33, 47, 48, 49, 
                  57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 98, 110]

def add_period_label(surah_number):
    """Look up period from hardcoded list"""
    if surah_number in MEDINAN_SURAHS:
        return "Medinan"
    else:
        return "Makkan"

# Example
surah_16_period = add_period_label(16)  # → "Makkan"
surah_33_period = add_period_label(33)  # → "Medinan"

# Now I can write in results:
print(f"Surah 16 (An-Nahl, {surah_16_period}) ↔ "
      f"Surah 33 (Al-Ahzab, {surah_33_period}): 97.49%")
# Output: "Surah 16 (An-Nahl, Makkan) ↔ Surah 33 (Al-Ahzab, Medinan): 97.49%"
```

---

## Where Labels Appear in Results

### 1. README.md - Top Pairs

```markdown
### Top 10 Most Similar Pairs

1. **Surah 16 (An-Nahl) ↔ Surah 33 (Al-Ahzab): 97.49%**
   - Both discuss Allah's blessings, gratitude, and community conduct.
   - An-Nahl is Makkan (nature and blessings theme)
   - Al-Ahzab is Medinan (community guidance)
   - Shared vocabulary: blessings, guidance, believers
```

**How this is generated**:
```python
def generate_top_pairs_explanation(surah_a, surah_b, similarity):
    # Get names (from Quran.com API)
    name_a = get_surah_name(surah_a)  # "An-Nahl"
    name_b = get_surah_name(surah_b)  # "Al-Ahzab"
    
    # Get periods (from hardcoded list)
    period_a = add_period_label(surah_a)  # "Makkan"
    period_b = add_period_label(surah_b)  # "Medinan"
    
    # Describe themes (from semantic analysis)
    themes = get_detected_themes(surah_a, surah_b)
    # ["blessings", "guidance", "community conduct"]
    
    # Write explanation
    return f"""
**Surah {surah_a} ({name_a}) ↔ Surah {surah_b} ({name_b}): {similarity}%**
- Both discuss {', '.join(themes)}
- {name_a} is {period_a} (nature and blessings theme)
- {name_b} is {period_b} (community guidance)
- Shared vocabulary: {', '.join(common_words)}
"""
```

### 2. Sample Pairs Analysis

In `results/sample_pairs/ENHANCED_PAIRS_ANALYSIS.md`:

```markdown
### Surah 2 (Al-Baqarah) ↔ Surah 3 (Āl 'Imrān)

**Structural Similarities:**
Sequential long Medinan surahs (286 and 200 verses). Both begin with 
'Alif Lam Mim', contain extensive legislation.
```

**How this is generated**:
```python
# In src/analysis/enhanced_pairs_analysis.py

THEMATIC_EXPLANATIONS = {
    (2, 3): {
        "themes": ["Legislation", "Faith and disbelief", "Community guidance"],
        "topics": ["Salah, fasting, hajj", "Battle of Badr/Uhud"],
        # I manually wrote this based on my knowledge:
        "structure": "Sequential long Medinan surahs (286 and 200 verses). "
                     "Both begin with 'Alif Lam Mim', contain extensive legislation",
        "semantic_reason": "Continuous discourse addressing Medinan community. "
                          "Surah 3 references events in Surah 2.",
        "linguistic": "Legal terminology: حكم (hukm, ruling), فرض (fard, obligation)"
    }
}
```

**Notice**: The phrase "Medinan surahs" is **manually written** by me in the explanation dictionary, not computed by the system.

---

## Why Add These Labels?

### 1. **Contextual Understanding**

Without labels:
> "Surah 16 and 33 are 97.49% similar due to shared themes."

With labels:
> "Surah 16 (Makkan) and 33 (Medinan) are 97.49% similar despite different revelation periods, proving theme transcends historical context."

**More informative!**

### 2. **Educational Value**

Readers learn:
- Which surahs are Medinan/Makkan
- That similar themes can appear in different periods
- That semantic similarity is content-based, not period-based

### 3. **Scholarly Relevance**

Connects computational analysis to traditional Islamic scholarship:
- Validates that certain themes are typical of Medinan period (legislation)
- Shows which themes transcend period boundaries (blessings, guidance)
- Provides academic context for findings

---

## The Hardcoded Classification

### Full List

```python
# From Islamic scholarly consensus
# Source: Traditional tafsir (Ibn Kathir, Jalalayn, etc.)

MEDINAN_SURAHS = [
    2,   # Al-Baqarah (The Cow)
    3,   # Āl 'Imrān (The Family of Imran)
    4,   # An-Nisa (The Women)
    5,   # Al-Ma'idah (The Table Spread)
    8,   # Al-Anfal (The Spoils of War)
    9,   # At-Tawbah (The Repentance)
    13,  # Ar-Ra'd (The Thunder)
    22,  # Al-Hajj (The Pilgrimage)
    24,  # An-Nur (The Light)
    33,  # Al-Ahzab (The Confederates)
    47,  # Muhammad
    48,  # Al-Fath (The Victory)
    49,  # Al-Hujurat (The Rooms)
    57,  # Al-Hadid (The Iron)
    58,  # Al-Mujadilah (The Pleading Woman)
    59,  # Al-Hashr (The Exile)
    60,  # Al-Mumtahanah (The Examined One)
    61,  # As-Saff (The Ranks)
    62,  # Al-Jumu'ah (The Friday)
    63,  # Al-Munafiqun (The Hypocrites)
    64,  # At-Taghabun (The Mutual Disillusion)
    65,  # At-Talaq (The Divorce)
    66,  # At-Tahrim (The Prohibition)
    98,  # Al-Bayyinah (The Clear Evidence)
    110, # An-Nasr (The Divine Support)
]

MAKKAN_SURAHS = [1, 6, 7, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 
                 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 
                 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 55, 56, 67, 68, 
                 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 
                 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 
                 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 
                 113, 114]
```

### Total Counts

- **Medinan**: 25 surahs (~22%)
- **Makkan**: 89 surahs (~78%)

### Characteristics (For Reference)

**Medinan Surahs** (revealed 622-632 CE in Medina):
- Typically longer
- Focus on legislation (family, criminal, financial law)
- Community organization (prayer, fasting, hajj)
- Interfaith relations (Jews, Christians)
- Social ethics and governance

**Makkan Surahs** (revealed 610-622 CE in Mecca):
- Typically shorter
- Focus on faith fundamentals (monotheism, afterlife, prophethood)
- Prophetic stories (Noah, Moses, Abraham)
- Nature signs (creation, cosmos)
- Opposition to idolatry
- Rhetorical style

---

## Proof That Models Don't Detect Period

### Statistical Evidence

From analysis of ALL 6,441 surah pairs:

| Pair Type | Mean Similarity | Count |
|-----------|-----------------|-------|
| **Same Period** (Med-Med or Mak-Mak) | 79.87% | 4,216 |
| **Mixed Period** (Med-Mak) | **83.15%** | 2,225 |

**Mixed pairs are 3.28% MORE similar!**

**If models detected period**:
- Same-period pairs would be more similar
- Mixed pairs would be less similar
- Period would be a clustering factor

**Actual result**:
- Mixed pairs are slightly MORE similar
- Period is NOT a factor in similarity
- Only themes matter

### Highest Similarity Pair

**Surah 16 (Makkan) ↔ Surah 33 (Medinan): 97.49%**

- **Different periods**
- **Highest similarity in entire corpus**
- **Proves**: Models don't know or care about period

### Example Breakdown

**Surah 16 (An-Nahl, Makkan)** - What models see:
- Words: بركة (blessing), شكر (gratitude), نعمة (favor)
- Themes: Allah's blessings, natural signs, gratitude
- Structure: Enumerating divine gifts

**Surah 33 (Al-Ahzab, Medinan)** - What models see:
- Words: مؤمنين (believers), هدى (guidance), رحمة (mercy)
- Themes: Community guidance, believers' conduct, divine support
- Structure: Prescriptive instructions

**Similarity**: Both emphasize Allah's favor, guidance, and proper response
**NOT**: Both are Medinan (they're not!)

---

## Where to Find the Code

### Main Classification

File: `src/analysis/enhanced_pairs_analysis.py` (lines ~40-60)

```python
def get_surah_period(surah_number):
    """
    Returns 'Medinan' or 'Makkan' based on scholarly consensus.
    
    This is NOT computed from the text!
    It's hardcoded from traditional Islamic scholarship.
    """
    MEDINAN = [2, 3, 4, 5, 8, 9, 13, 22, 24, 33, 47, 48, 49, 
               57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 98, 110]
    
    if surah_number in MEDINAN:
        return "Medinan"
    else:
        return "Makkan"
```

### Usage in Explanations

File: `src/analysis/enhanced_pairs_analysis.py` (lines ~100-150)

```python
# Manually written explanations
THEMATIC_EXPLANATIONS = {
    (2, 3): {
        # ...
        "structure": "Sequential long Medinan surahs...",
        #                          ^^^^^^^ manually added by me
    },
    (113, 114): {
        # ...
        "structure": "Both are short Makkan surahs...",
        #                          ^^^^^^ manually added by me
    }
}
```

---

## Summary

### What the Semantic System Does

1. ✅ Reads Arabic text
2. ✅ Preprocesses (normalize, tokenize)
3. ✅ Extracts features (word frequencies, embeddings)
4. ✅ Computes similarity (KL divergence, cosine similarity)
5. ✅ Outputs: **97.49% similarity**

**Input**: Text only  
**Output**: Similarity score only

### What I Do Manually

1. ✅ Look up surah numbers in hardcoded `MEDINAN_SURAHS` list
2. ✅ Label as "Medinan" or "Makkan"
3. ✅ Write explanations mentioning the period
4. ✅ Add contextual information for readers

**Input**: Similarity scores + surah numbers  
**Output**: Explanations with period labels

### Why This Is Important

- **Transparency**: Users should know labels are from scholarship, not computation
- **Accuracy**: Prevents misunderstanding that models detect period
- **Educational**: Shows how computational + traditional scholarship combine
- **Scientific**: Clear about what is measured vs what is known

---

## References

### Islamic Scholarship Sources for Classification

1. **Ibn Kathir** (1301-1373). *Tafsir al-Quran al-Azim*
   - Traditional classification of Medinan/Makkan surahs

2. **Jalalayn** (15th-16th century). *Tafsir al-Jalalayn*
   - Consensus classification

3. **Muhammad Abdel Haleem** (2004). *The Qur'an: A New Translation*
   - Modern scholarly consensus

4. **Angelika Neuwirth** (2010). *Der Koran als Text der Spätantike*
   - Academic historical analysis

5. **Traditional Islamic Transmission** (7th century onwards)
   - Oral and written transmission of revelation circumstances
   - *Asbab al-Nuzul* (Occasions of Revelation) literature

### Digital Resources

- **Quran.com**: Provides Medinan/Makkan labels in metadata
- **Tanzil.net**: Machine-readable Quran with classification
- **Corpus Quran**: Academic project with comprehensive metadata

---

**Last Updated**: October 16, 2025  
**Purpose**: Clarify that Medinan/Makkan labels are from scholarship, not semantic analysis  
**See Also**: `docs/METHODOLOGY.md`, `docs/SEMANTIC_VS_HISTORICAL.md`

