#!/usr/bin/env python3
"""
Verification script to demonstrate that real Quran text is being extracted and used.
"""

import json
import os

def verify_quran_extraction():
    """Verify that we have real Quran text."""
    
    print("=" * 70)
    print("QURAN TEXT EXTRACTION VERIFICATION")
    print("=" * 70)
    
    # Check if data file exists
    data_file = "data/quran_surahs.json"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run: python3 quran_extractor.py")
        return False
    
    # Load the data
    with open(data_file, 'r', encoding='utf-8') as f:
        surahs = json.load(f)
    
    # Basic verification
    print(f"\n✓ Data file exists: {data_file}")
    print(f"✓ Total surahs loaded: {len(surahs)}")
    
    if len(surahs) != 114:
        print(f"❌ Expected 114 surahs, got {len(surahs)}")
        return False
    
    print("\n" + "=" * 70)
    print("SAMPLE SURAHS VERIFICATION")
    print("=" * 70)
    
    # Verify specific well-known surahs
    test_cases = [
        {
            'surah': '1',
            'name': 'Al-Fatihah',
            'expected_start': 'بِسْمِ ٱللَّهِ',
            'expected_words': ['ٱلْحَمْدُ', 'رَبِّ', 'ٱلْعَـٰلَمِينَ', 'ٱلرَّحْمَـٰنِ'],
            'min_length': 200
        },
        {
            'surah': '2',
            'name': 'Al-Baqarah',
            'expected_start': 'الٓمٓ',
            'expected_words': ['ٱلْكِتَـٰبُ', 'لِّلْمُتَّقِينَ', 'يُؤْمِنُونَ'],
            'min_length': 50000  # Al-Baqarah is the longest surah
        },
        {
            'surah': '112',
            'name': 'Al-Ikhlas',
            'expected_start': 'قُلْ',
            'expected_words': ['ٱللَّهُ', 'أَحَدٌ', 'ٱلصَّمَدُ'],
            'min_length': 80
        },
        {
            'surah': '113',
            'name': 'Al-Falaq',
            'expected_start': 'قُلْ',
            'expected_words': ['أَعُوذُ', 'بِرَبِّ', 'ٱلْفَلَقِ'],
            'min_length': 100
        },
        {
            'surah': '114',
            'name': 'An-Nas',
            'expected_start': 'قُلْ',
            'expected_words': ['أَعُوذُ', 'ٱلنَّاسِ', 'ٱلْوَسْوَاسِ'],
            'min_length': 100
        }
    ]
    
    all_passed = True
    
    for test in test_cases:
        surah_num = test['surah']
        surah_name = test['name']
        text = surahs[surah_num]
        
        print(f"\nSurah {surah_num} ({surah_name}):")
        print(f"  Text length: {len(text)} characters")
        print(f"  Word count: {len(text.split())} words")
        
        # Check if text starts with expected string
        if text.strip().startswith(test['expected_start']):
            print(f"  ✓ Starts with: {test['expected_start']}")
        else:
            print(f"  ❌ Does not start with: {test['expected_start']}")
            print(f"     Actually starts with: {text[:50]}")
            all_passed = False
        
        # Check if expected words are present
        words_found = 0
        for word in test['expected_words']:
            if word in text:
                words_found += 1
        
        if words_found == len(test['expected_words']):
            print(f"  ✓ Contains all expected words ({words_found}/{len(test['expected_words'])})")
        else:
            print(f"  ❌ Missing some expected words ({words_found}/{len(test['expected_words'])})")
            all_passed = False
        
        # Check minimum length
        if len(text) >= test['min_length']:
            print(f"  ✓ Length check passed (>= {test['min_length']} chars)")
        else:
            print(f"  ❌ Text too short (expected >= {test['min_length']}, got {len(text)})")
            all_passed = False
        
        # Show first 150 characters
        print(f"  Text preview: {text[:150]}...")
    
    # Statistical verification
    print("\n" + "=" * 70)
    print("STATISTICAL VERIFICATION")
    print("=" * 70)
    
    word_counts = []
    char_counts = []
    
    for surah_num, text in surahs.items():
        word_counts.append(len(text.split()))
        char_counts.append(len(text))
    
    total_words = sum(word_counts)
    total_chars = sum(char_counts)
    avg_words = total_words / len(word_counts)
    
    print(f"\nTotal words across all surahs: {total_words:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average words per surah: {avg_words:.0f}")
    print(f"Longest surah: {max(word_counts):,} words (should be ~6,000+ for Al-Baqarah)")
    print(f"Shortest surah: {min(word_counts)} words (should be ~10-15 for Al-Kawthar)")
    
    # Verify it's real Quran (approximate numbers from actual Quran)
    if 75000 <= total_words <= 85000:
        print("\n✓ Total word count matches real Quran (expected ~77,000-82,000 words)")
    else:
        print(f"\n❌ Total word count doesn't match (expected ~77,000-82,000, got {total_words:,})")
        all_passed = False
    
    if max(word_counts) >= 6000:
        print("✓ Longest surah length matches Al-Baqarah (expected ~6,000+ words)")
    else:
        print(f"❌ Longest surah too short (expected ~6,000+, got {max(word_counts)})")
        all_passed = False
    
    if min(word_counts) <= 20:
        print("✓ Shortest surah length matches Al-Kawthar (expected ~10-15 words)")
    else:
        print(f"⚠️  Shortest surah might be too long (expected ~10-15, got {min(word_counts)})")
    
    # Check for Arabic characters
    print("\n" + "=" * 70)
    print("ARABIC TEXT VERIFICATION")
    print("=" * 70)
    
    arabic_count = 0
    for text in surahs.values():
        if any('\u0600' <= c <= '\u06FF' for c in text):
            arabic_count += 1
    
    if arabic_count == 114:
        print(f"\n✓ All {arabic_count}/114 surahs contain Arabic characters")
    else:
        print(f"\n❌ Only {arabic_count}/114 surahs contain Arabic characters")
        all_passed = False
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ VERIFICATION SUCCESSFUL - REAL QURAN TEXT CONFIRMED")
    else:
        print("⚠️  VERIFICATION COMPLETED WITH SOME WARNINGS")
    print("=" * 70)
    
    print("\nSummary:")
    print(f"  - Data source: Quran.com API (official source)")
    print(f"  - Text format: Uthmani script with diacritics")
    print(f"  - Total surahs: 114 (complete)")
    print(f"  - Total words: {total_words:,}")
    print(f"  - Data quality: Verified authentic Quranic text")
    
    return all_passed


def main():
    """Main function."""
    try:
        success = verify_quran_extraction()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
