# Bismillah Exclusion Update

**Date**: October 16, 2025  
**Change**: Exclude Bismillah verse from all surahs except Surah 9

---

## 📋 Summary

Updated the Asymmetric STS implementation to skip the first verse (Bismillah: "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ") for all surahs except Surah 9 (At-Tawbah).

### Rationale

**Problem**: Bismillah is identical in 113 out of 114 surahs
- This would artificially inflate similarity scores
- Not semantically meaningful for distinguishing surahs
- Would bias the analysis towards finding all surahs similar

**Solution**: Skip the first verse for all surahs except Surah 9
- Surah 9 (At-Tawbah) is the only surah that doesn't start with Bismillah
- It begins directly with "بَرَآءَةٌ مِّنَ ٱللَّهِ" (A declaration from Allah)

---

## 🔢 Impact on Verse Counts

### Before (with Bismillah)
```
Total verses: 6,236
All surahs included first verse
```

### After (Bismillah excluded)
```
Total verses analyzed: 6,123
Calculation: 6,236 - 113 Bismillahs = 6,123
```

### Example Verse Counts

| Surah | Name | Original | After Exclusion | Notes |
|-------|------|----------|-----------------|-------|
| 1 | Al-Fatiha | 7 | **6** | Bismillah skipped |
| 2 | Al-Baqarah | 286 | **285** | Bismillah skipped |
| 9 | At-Tawbah | 129 | **129** | No Bismillah (no skip!) |
| 65 | At-Talaq | 12 | **11** | Bismillah skipped |
| 113 | Al-Falaq | 5 | **4** | Bismillah skipped |
| 114 | An-Nas | 6 | **5** | Bismillah skipped |

---

## 💻 Code Changes

### File: `src/analysis/asymmetric_sts.py`

**Location**: `get_verses()` method

```python
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
    
    if 'verses' in surah_data:
        verses = surah_data['verses']
        verses = [v.strip() for v in verses if v.strip()]
        
        # Skip first verse (Bismillah) for all surahs except Surah 9
        # Bismillah is repeated at the start of every surah except At-Tawbah
        if surah_id != 9 and len(verses) > 1:
            verses = verses[1:]  # ← NEW: Skip first verse
        
        return verses
```

**Key Logic**:
- Check if `surah_id != 9` (not At-Tawbah)
- Check if `len(verses) > 1` (has more than just Bismillah)
- If both true: `verses = verses[1:]` (skip first element)

---

## 📊 Updated Test Results

### Before (with Bismillah)
```
Al-Falaq ↔ An-Nas:
  Verse counts: 5 vs 6
  113→114: 91.38%
  114→113: 89.19%

Al-Baqarah ↔ At-Talaq:
  Verse counts: 286 vs 12
  2→65: 94.13%
  65→2: 95.63%
```

### After (Bismillah excluded)
```
Al-Falaq ↔ An-Nas:
  Verse counts: 4 vs 5 ✓
  113→114: 91.39%
  114→113: 87.62%

Al-Baqarah ↔ At-Talaq:
  Verse counts: 285 vs 11 ✓
  2→65: 94.21%
  65→2: 95.44%
```

**Changes**:
- Verse counts reduced by 1 (except Surah 9)
- Similarity scores slightly different (more accurate)
- Asymmetry patterns preserved

---

## 📝 Documentation Updates

### Files Updated

1. **`docs/ASYMMETRIC_STS_GUIDE.md`**
   - Added Bismillah exclusion explanation in "Split Surahs into Sentences" section
   - Updated all example verse counts (286→285, 12→11)
   - Added "Important Exclusion" subsection in Technical Details
   - Updated total verses: 6,236 → 6,123

2. **`ASYMMETRIC_STS_SUMMARY.md`**
   - Updated key features to mention "excludes repeated Bismillah"
   - Updated example verse counts
   - Added note about Bismillah exclusion
   - Updated test results section

3. **`src/analysis/asymmetric_sts.py`**
   - Updated docstring for `get_verses()` method
   - Implemented exclusion logic

### Key Sections Updated

**Sentence Splitting**:
```markdown
**Important Exclusion**:
- The first verse (Bismillah: "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ") 
  is **skipped for all surahs except Surah 9**
- Reason: Bismillah is identical across 113 surahs and would 
  artificially inflate similarity
- Surah 9 (At-Tawbah) doesn't start with Bismillah, so no exclusion needed
- Total verses analyzed: **6,123** (6,236 - 113 Bismillahs)
```

**Example Code**:
```python
# Surah A (Al-Baqarah, originally 286 verses, 285 after skipping Bismillah)
sentences_A = [
    # "بسم الله الرحمن الرحيم",  # SKIPPED (Bismillah)
    "ذَٰلِكَ ٱلْكِتَـٰبُ لَا رَيْبَ...",  # Verse 1 after Bismillah
    # ... 283 more verses
]
```

---

## ✅ Validation

### Verification Tests

All tests pass with Bismillah exclusion:

```
✅ Verse counts correct:
   Al-Fatiha (1):    6 verses (was 7)
   Al-Baqarah (2):   285 verses (was 286)
   At-Tawbah (9):    129 verses (unchanged - no Bismillah)
   At-Talaq (65):    11 verses (was 12)
   Al-Falaq (113):   4 verses (was 5)
   An-Nas (114):     5 verses (was 6)

✅ Self-similarity: 100%
✅ Asymmetry patterns: Correct (65→2 > 2→65)
✅ Symmetric pairs: Low asymmetry (<5%)
✅ Total verses: 6,123 (6,236 - 113)
```

### Run Tests

```bash
# Quick test (updated with new counts)
python scripts/test_asymmetric_sts.py

# Full analysis (114×114 matrix, currently running)
python src/analysis/asymmetric_sts.py
```

---

## 🎯 Why This Matters

### 1. **Accuracy**
- Removes artificial similarity from repeated verse
- Focuses on unique content of each surah
- More meaningful comparison

### 2. **Fairness**
- All surahs compared on equal footing
- No bias from formulaic opening
- Captures true thematic relationships

### 3. **Scholarly Alignment**
- Bismillah is considered a marker, not core content
- Traditional Quranic scholarship treats it separately
- Many verse numbering systems exclude it from count

### 4. **Statistical Soundness**
- Removes constant term from all comparisons
- Increases variance between surahs
- Better discriminative power

---

## 📈 Expected Impact on Results

### Similarity Scores

**Minor changes expected**:
- Most scores within ±1-2% of previous values
- Slight decrease in similarity for very short surahs (more impact from excluding 1 verse)
- Minimal change for long surahs (285/286 ≈ 99.7% of original)

**Asymmetry patterns**:
- Should remain largely unchanged
- Same directional relationships (A→B vs B→A)
- Possibly slightly stronger asymmetry signals

### Top Pairs

**Unlikely to change significantly**:
- Top similar pairs should remain similar
- Order might shift slightly
- Core thematic relationships preserved

### Surah 9 (At-Tawbah)

**Special case**:
- Only surah with no Bismillah exclusion
- Might show slightly different patterns vs other surahs
- Scientifically accurate (reflects actual text)

---

## 🔄 Analysis Status

### Current Run

**Started**: October 16, 2025  
**Status**: Running in background  
**Expected Duration**: 20-30 minutes  
**Command**: `python src/analysis/asymmetric_sts.py`

**Output Files** (will be generated):
- `results/matrices/asymmetric_sts_similarity_matrix.csv`
- `results/matrices/asymmetric_sts_asymmetry_matrix.csv`
- `results/matrices/asymmetric_sts_statistics.json`

**Progress Tracking**:
```bash
# Check progress
tail -f asymmetric_sts_run.log

# Check if complete
ls -lh results/matrices/asymmetric_sts_*.csv
```

---

## 📚 References

### Islamic Scholarship

**Bismillah as Separator**:
- Traditionally considered a verse marker, not part of surah content
- Some scholars count it as verse 1, others don't
- Our approach: Exclude for similarity analysis (standard in NLP)

**Surah 9 Exception**:
- Well-documented: At-Tawbah is the only surah without Bismillah
- Begins with "بَرَآءَةٌ" (A declaration of disassociation)
- Historical context: Revealed about breaking treaties with idolaters

### Technical References

**NLP Best Practices**:
- Remove formulaic text that appears across all documents
- Focus on discriminative features
- Standard preprocessing for similarity analysis

**Similar Approaches**:
- Email analysis: Remove "Dear X" / "Sincerely Y"
- Legal documents: Remove standard headers/footers
- Books: Remove publisher information

---

## ✨ Summary

**Change**: Exclude Bismillah from all surahs except Surah 9  
**Impact**: 113 fewer verses analyzed (6,236 → 6,123)  
**Benefit**: More accurate, focused semantic comparison  
**Status**: Implemented, tested, documented, running full analysis  

**Files Updated**:
- ✅ `src/analysis/asymmetric_sts.py` (implementation)
- ✅ `docs/ASYMMETRIC_STS_GUIDE.md` (documentation)
- ✅ `ASYMMETRIC_STS_SUMMARY.md` (quick reference)
- ⏳ Results files (generating...)

---

**Last Updated**: October 16, 2025  
**Status**: Complete (awaiting full analysis results)

