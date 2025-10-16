# How Semantic Similarity Works - Medinan vs Makkan

**Key Question**: How does semantic similarity figure out if surahs are Medinan or Makkan?

**Short Answer**: **It doesn't!** Semantic similarity detects **thematic content**, not historical period.

---

## The Surprising Finding

When we analyzed all 6,441 surah pairs:

| Pair Type | Count | Mean Similarity | Std Dev |
|-----------|-------|-----------------|---------|
| **Same Period** (Med-Med or Mak-Mak) | 4,216 | 79.87% | 11.53% |
| **Mixed Period** (Med-Mak) | 2,225 | **83.15%** | 10.82% |

**Result**: Mixed-period pairs are actually **3.28% MORE similar** than same-period pairs!

**Conclusion**: Semantic similarity is **theme-based, not period-based**.

---

## What's Actually Happening

### 1. **Semantic Models Don't Know History**

The models (Sentence Transformers, AraBERT) were trained on:
- ✅ **Linguistic patterns** (words, grammar, structure)
- ✅ **Semantic meaning** (concepts, themes)
- ❌ **Historical metadata** (when/where revealed)

They have **zero knowledge** that:
- Surah 2 was revealed in Medina
- Surah 11 was revealed in Mecca
- There's even a concept of "Medinan vs Makkan"

### 2. **Thematic Overlap Across Periods**

Many themes appear in **both** Medinan and Makkan surahs:

| Theme | Medinan Example | Makkan Example | Similarity |
|-------|-----------------|----------------|------------|
| **Allah's Blessings** | Surah 33 (Al-Ahzab) | Surah 16 (An-Nahl) | **97.49%** 🏆 |
| **Prophetic Stories** | Surah 3 (Āl 'Imrān) | Surah 11 (Hud) | ~87% |
| **Divine Guidance** | Surah 24 (An-Nur) | Surah 35 (Fatir) | ~94% |

**Example**: Surah 16 (An-Nahl, Makkan) ↔ Surah 33 (Al-Ahzab, Medinan) = **97.49%**
- Why? Both discuss Allah's blessings, community conduct, gratitude
- Theme matters more than period!

### 3. **What Models Actually Detect**

```
NOT: "This surah is from Medina" → "Similar to other Medinan surahs"

BUT: "This surah talks about X, Y, Z" → "Similar to other surahs with X, Y, Z"
```

The model sees:
```
Surah 33: [community, guidance, Prophet's household, believers, conduct, ...]
Surah 16: [blessings, gratitude, guidance, believers, conduct, creation, ...]

Similarity computed from: Overlapping concepts (guidance, believers, conduct)
NOT from: Both being "Medinan/Makkan"
```

---

## Why the Confusion?

### Case 1: When Period Seems to Matter

**Top pair #2**: Surah 33 ↔ 58 (Both Medinan) = 97.49%
- ✓ Both Medinan
- ✓ High similarity

**But**: This is correlation, not causation!
- They're similar because: Community issues, women's rights, social conduct
- NOT because: They're both Medinan

### Case 2: When Period Doesn't Matter

**Top pair #1**: Surah 16 (Makkan) ↔ 33 (Medinan) = 97.49%
- ✗ Different periods
- ✓ Still highest similarity

**Why?**: Shared themes (blessings, conduct, guidance) transcend period.

---

## Medinan vs Makkan: Content Characteristics

The models detect **content patterns** that *happen to correlate* with period:

### Medinan Content Patterns (What Models See)

**Vocabulary patterns**:
- Legal terms: طلاق (divorce), نكاح (marriage), ميراث (inheritance)
- Community: أزواج (wives), يتامى (orphans), مؤمنين (believers)
- Interfaith: أهل الكتاب (People of the Book), يهود (Jews), نصارى (Christians)

**Semantic patterns**:
- Detailed procedural instructions
- Conditional statements (if X, then Y)
- Social regulations
- Community organization

**Example - Surah 2 (Medinan)**:
```
"Divorce is twice. Then, either keep [her] in an acceptable manner or release [her] 
with good treatment..."
→ Detailed legal procedure
→ Conditional logic
→ Specific regulations
```

### Makkan Content Patterns (What Models See)

**Vocabulary patterns**:
- Nature: سماء (sky), أرض (earth), جبال (mountains), نحل (bees)
- Stories: نوح (Noah), موسى (Moses), فرعون (Pharaoh)
- Afterlife: جنة (Paradise), نار (Fire), يوم القيامة (Day of Resurrection)

**Semantic patterns**:
- Rhetorical questions
- Oaths and emphatic statements
- Narrative storytelling
- Existential arguments

**Example - Surah 16 (Makkan)**:
```
"And He has subjected to you the night and day and the sun and moon, 
and the stars are subjected by His command..."
→ Enumerating natural signs
→ Emphasis on Allah's power
→ Call to gratitude
```

---

## The Real Mechanism

### Step 1: Preprocessing (No Period Info)
```python
# Input: Surah text (just words)
text = "بسم الله الرحمن الرحيم..."

# Preprocessing
preprocessed = remove_diacritics(text)
preprocessed = normalize_arabic(preprocessed)

# NO period information added!
```

### Step 2: Semantic Encoding (Theme Detection)
```python
# AraBERT/Embeddings see only content
embedding = model.encode(preprocessed)

# Embedding captures:
# - Word meanings
# - Semantic relationships  
# - Thematic concepts

# Does NOT capture:
# - Historical period
# - Revelation location
# - Chronological order
```

### Step 3: Similarity Computation (Pure Theme Comparison)
```python
# Compare two embeddings
similarity = cosine_similarity(embed_A, embed_B)

# High similarity means:
# ✓ Similar vocabulary
# ✓ Similar themes
# ✓ Similar semantic content

# Does NOT mean:
# ✗ Same historical period
# ✗ Same revelation context
```

---

## Evidence from Top Pairs

Looking at top 10 most similar pairs:

| Rank | Surah A | Period A | Surah B | Period B | Similarity | Same Period? |
|------|---------|----------|---------|----------|------------|--------------|
| 1 | 16 | **Makkan** | 33 | **Medinan** | 97.49% | ❌ No |
| 2 | 33 | Medinan | 58 | Medinan | 97.49% | ✅ Yes |
| 3 | 16 | **Makkan** | 68 | **Makkan** | 97.36% | ✅ Yes |
| 4 | 11 | Makkan | 16 | Makkan | 97.36% | ✅ Yes |
| 5 | 35 | Makkan | 42 | Makkan | 97.23% | ✅ Yes |
| 6 | 13 | **Medinan** | 16 | **Makkan** | 97.20% | ❌ No |
| 7 | 11 | **Makkan** | 33 | **Medinan** | 97.20% | ❌ No |
| 8 | 32 | Makkan | 45 | Makkan | 97.11% | ✅ Yes |
| 9 | 11 | Makkan | 68 | Makkan | 97.10% | ✅ Yes |
| 10 | 11 | **Makkan** | 40 | **Makkan** | 97.03% | ✅ Yes |

**Result**: 3 of top 10 (30%) are **mixed period** pairs!
- If models detected period, mixed pairs would have lower similarity
- But they don't - because **theme matters more than period**

---

## Why My Explanation Used "Medinan/Makkan"

In the README, I wrote things like:
> "Both Medinan surahs with extensive legislative and social guidance"

**This is shorthand for**:
> "Both surahs contain themes that are TYPICAL of Medinan revelation (legislation, 
> community guidance) - and we know from Islamic scholarship that they happen to 
> BE Medinan surahs."

**NOT saying**:
> "The model detected they are Medinan and therefore found them similar"

---

## The Correct Understanding

### What Semantic Similarity Does:
1. ✅ Reads surah content (words, meanings, structure)
2. ✅ Identifies themes (legislation, stories, nature, afterlife)
3. ✅ Compares themes between surahs
4. ✅ Reports: "These surahs discuss similar topics"

### What It Doesn't Do:
1. ❌ Know historical period (Medinan/Makkan)
2. ❌ Use revelation context
3. ❌ Factor in chronological order
4. ❌ Consider location or audience

### Why High Similarity?

**Not because**: "Both are Medinan"  
**But because**: "Both discuss community guidance, social laws, and believers' conduct"

**The fact that both happen to be Medinan** is because:
- Medinan period = established Muslim community
- Community needs = legislation and social guidance
- Therefore: Medinan surahs often contain similar themes

**Causality**:
```
Medinan period → Need for laws → Legislative themes → High similarity
        ↑                                                    ↑
        └────────────────────────────────────────────────────┘
               Models detect themes, not period!
```

---

## Conclusion

**Question**: How does semantic similarity figure out if surahs are Medinan or Makkan?

**Answer**: **It doesn't!**

Semantic similarity:
- ✅ Detects **thematic content**
- ✅ Finds **conceptual overlap**
- ❌ Doesn't know **historical period**
- ❌ Doesn't use **revelation context**

When I say "both Medinan surahs" in explanations:
- It's **descriptive** (they happen to be Medinan)
- NOT **causative** (high similarity BECAUSE Medinan)

The high similarity comes from **shared themes**, which *correlate with* but are *not determined by* the historical period.

---

## Statistical Proof

```
Same-period pairs:  79.87% mean similarity
Mixed-period pairs: 83.15% mean similarity  (-3.28%)

If models detected period → Same-period pairs would be MORE similar
Actual result → Mixed pairs slightly MORE similar
Conclusion → Period is NOT a factor in similarity computation
```

**The models are purely thematic, not historical.** 🎯

---

**Last Updated**: October 16, 2025  
**Status**: Explanation validated with statistical analysis
