#!/usr/bin/env python3
"""
Demonstration script for Quran Semantic Analysis.
This script shows how to use the analysis with sample data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from quran_analysis import QuranAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_quran_data():
    """Create sample Quran data for demonstration."""
    sample_surahs = {
        1: "بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين إياك نعبد وإياك نستعين اهدنا الصراط المستقيم صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين",
        2: "بسم الله الرحمن الرحيم الم ذلك الكتاب لا ريب فيه هدى للمتقين الذين يؤمنون بالغيب ويقيمون الصلاة ومما رزقناهم ينفقون والذين يؤمنون بما أنزل إليك وما أنزل من قبلك وبالآخرة هم يوقنون أولئك على هدى من ربهم وأولئك هم المفلحون",
        3: "بسم الله الرحمن الرحيم الم الله لا إله إلا هو الحي القيوم نزل عليك الكتاب بالحق مصدقا لما بين يديه وأنزل التوراة والإنجيل من قبل هدى للناس وأنزل الفرقان إن الذين كفروا بآيات الله لهم عذاب شديد والله عزيز ذو انتقام",
        4: "بسم الله الرحمن الرحيم يا أيها الناس اتقوا ربكم الذي خلقكم من نفس واحدة وخلق منها زوجها وبث منهما رجالا كثيرا ونساء واتقوا الله الذي تساءلون به والأرحام إن الله كان عليكم رقيبا",
        5: "بسم الله الرحمن الرحيم يا أيها الذين آمنوا أوفوا بالعقود أحلت لكم بهيمة الأنعام إلا ما يتلى عليكم غير محلي الصيد وأنتم حرم إن الله يحكم ما يريد",
        6: "بسم الله الرحمن الرحيم الحمد لله الذي خلق السماوات والأرض وجعل الظلمات والنور ثم الذين كفروا بربهم يعدلون هو الذي خلقكم من طين ثم قضى أجلا وأجل مسمى عنده ثم أنتم تمترون وهو الله في السماوات وفي الأرض يعلم سركم وجهركم ويعلم ما تكسبون",
        7: "بسم الله الرحمن الرحيم طس تلك آيات الكتاب المبين لعلك باخع نفسك ألا يكونوا مؤمنين إن نشأ ننزل عليهم من السماء آية فظلت أعناقهم لها خاضعين",
        8: "بسم الله الرحمن الرحيم يس والقرآن الحكيم إنك لمن المرسلين على صراط مستقيم تنزيل العزيز الرحيم لتنذر قوما ما أنذر آباؤهم فهم غافلون",
        9: "بسم الله الرحمن الرحيم الحمد لله فاطر السماوات والأرض جاعل الملائكة رسلا أولي أجنحة مثنى وثلاث ورباع يزيد في الخلق ما يشاء إن الله على كل شيء قدير",
        10: "بسم الله الرحمن الرحيم الر تلك آيات الكتاب الحكيم أكان للناس عجبا أن أوحينا إلى رجل منهم أن أنذر الناس وبشر الذين آمنوا أن لهم قدم صدق عند ربهم قال الكافرون إن هذا لساحر مبين"
    }
    
    # Extend to 114 surahs by cycling through the sample data
    extended_surahs = {}
    for i in range(1, 115):
        extended_surahs[i] = sample_surahs.get(i, sample_surahs[i % len(sample_surahs)])
    
    return extended_surahs

def demonstrate_preprocessing():
    """Demonstrate Arabic text preprocessing."""
    print("\n" + "="*60)
    print("ARABIC TEXT PREPROCESSING DEMONSTRATION")
    print("="*60)
    
    analyzer = QuranAnalyzer()
    
    # Sample Arabic text with diacritics
    sample_text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"
    
    print(f"Original text: {sample_text}")
    
    # Preprocess the text
    preprocessed = analyzer.preprocess_arabic_text(sample_text)
    print(f"Preprocessed: {preprocessed}")
    
    # Show preprocessing steps
    print("\nPreprocessing steps:")
    print("1. Unicode normalization")
    print("2. Alef variations normalization")
    print("3. Teh marbuta normalization")
    print("4. Whitespace normalization")
    print("5. Diacritics removal")
    print("6. Tokenization")
    print("7. Stop word removal")
    print("8. Stemming")

def demonstrate_topic_modeling():
    """Demonstrate topic modeling."""
    print("\n" + "="*60)
    print("TOPIC MODELING DEMONSTRATION")
    print("="*60)
    
    analyzer = QuranAnalyzer()
    
    # Create sample data
    sample_surahs = create_sample_quran_data()
    analyzer.surahs = sample_surahs
    analyzer.surah_names = [f"Surah {i}" for i in range(1, 115)]
    
    # Preprocess all surahs
    print("Preprocessing Arabic text...")
    analyzer.preprocess_all_surahs()
    
    # Train topic model
    print("Training LDA topic model...")
    analyzer.train_topic_model(num_topics=10, passes=5)
    
    # Show topic distributions for first few surahs
    print("\nTopic distributions for first 5 surahs:")
    for surah_num in range(1, 6):
        if surah_num in analyzer.topic_distributions:
            topic_dist = analyzer.topic_distributions[surah_num]
            print(f"Surah {surah_num}: {topic_dist}")
    
    # Show most important topics
    if analyzer.topic_model:
        print("\nTop words for each topic:")
        for topic_id in range(min(5, analyzer.topic_model.num_topics)):
            words = analyzer.topic_model.show_topic(topic_id, topn=5)
            print(f"Topic {topic_id}: {words}")

def demonstrate_asymmetric_matrix():
    """Demonstrate asymmetric relationship matrix computation."""
    print("\n" + "="*60)
    print("ASYMMETRIC RELATIONSHIP MATRIX DEMONSTRATION")
    print("="*60)
    
    analyzer = QuranAnalyzer()
    
    # Create sample data
    sample_surahs = create_sample_quran_data()
    analyzer.surahs = sample_surahs
    analyzer.surah_names = [f"Surah {i}" for i in range(1, 115)]
    
    # Run complete analysis
    print("Running complete analysis...")
    analyzer.preprocess_all_surahs()
    analyzer.train_topic_model(num_topics=10, passes=5)
    matrix = analyzer.compute_asymmetric_matrix()
    
    # Show matrix properties
    print(f"\nMatrix shape: {matrix.shape}")
    print(f"Matrix is asymmetric: {not np.allclose(matrix, matrix.T)}")
    print(f"Mean KL divergence: {np.mean(matrix[np.triu_indices(114, k=1)]):.4f}")
    print(f"Max KL divergence: {np.max(matrix):.4f}")
    
    # Show asymmetry for first few surahs
    print("\nAsymmetry examples (Surah A → Surah B vs Surah B → Surah A):")
    for i in range(3):
        for j in range(3):
            if i != j:
                forward = matrix[i, j]
                backward = matrix[j, i]
                asymmetry = forward - backward
                print(f"Surah {i+1} → Surah {j+1}: {forward:.4f}")
                print(f"Surah {j+1} → Surah {i+1}: {backward:.4f}")
                print(f"Asymmetry: {asymmetry:.4f}")
                print()

def demonstrate_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    analyzer = QuranAnalyzer()
    
    # Create sample data
    sample_surahs = create_sample_quran_data()
    analyzer.surahs = sample_surahs
    analyzer.surah_names = [f"Surah {i}" for i in range(1, 115)]
    
    # Run analysis
    analyzer.preprocess_all_surahs()
    analyzer.train_topic_model(num_topics=10, passes=5)
    analyzer.compute_asymmetric_matrix()
    
    # Create visualization
    print("Creating visualization...")
    fig = analyzer.visualize_matrix(save_path="demo_relationship_matrix.png")
    
    # Show analysis results
    analysis = analyzer.analyze_relationships(top_n=5)
    
    print("\nTop 5 Relationships:")
    for i, rel in enumerate(analysis['top_relationships']):
        print(f"{i+1}. {rel['source']} → {rel['target']}: {rel['kl_divergence']:.4f}")
    
    print("\nMost Asymmetric Relationships:")
    for i, rel in enumerate(analysis['most_asymmetric']):
        print(f"{i+1}. {rel['source']} → {rel['target']}: asymmetry = {rel['asymmetry']:.4f}")
    
    print(f"\nVisualization saved as 'demo_relationship_matrix.png'")

def demonstrate_analysis():
    """Demonstrate analysis capabilities."""
    print("\n" + "="*60)
    print("ANALYSIS DEMONSTRATION")
    print("="*60)
    
    analyzer = QuranAnalyzer()
    
    # Create sample data
    sample_surahs = create_sample_quran_data()
    analyzer.surahs = sample_surahs
    analyzer.surah_names = [f"Surah {i}" for i in range(1, 115)]
    
    # Run complete analysis
    analyzer.preprocess_all_surahs()
    analyzer.train_topic_model(num_topics=10, passes=5)
    analyzer.compute_asymmetric_matrix()
    
    # Save results
    analyzer.save_results("demo_results")
    
    # Show matrix statistics
    matrix = analyzer.relationship_matrix
    print(f"\nMatrix Statistics:")
    print(f"Shape: {matrix.shape}")
    print(f"Mean: {np.mean(matrix):.4f}")
    print(f"Std: {np.std(matrix):.4f}")
    print(f"Min: {np.min(matrix):.4f}")
    print(f"Max: {np.max(matrix):.4f}")
    
    # Show asymmetry statistics
    asymmetry_matrix = matrix - matrix.T
    print(f"\nAsymmetry Statistics:")
    print(f"Mean asymmetry: {np.mean(asymmetry_matrix):.4f}")
    print(f"Max asymmetry: {np.max(asymmetry_matrix):.4f}")
    print(f"Min asymmetry: {np.min(asymmetry_matrix):.4f}")
    
    print(f"\nResults saved to 'demo_results/' directory")

def main():
    """Main demonstration function."""
    print("Quran Semantic Analysis - Demonstration")
    print("=" * 60)
    print("This demonstration shows the capabilities of the Quran analysis system.")
    print("Note: This uses sample data for demonstration purposes.")
    
    try:
        # Demonstrate each component
        demonstrate_preprocessing()
        demonstrate_topic_modeling()
        demonstrate_asymmetric_matrix()
        demonstrate_visualization()
        demonstrate_analysis()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("- demo_relationship_matrix.png (visualization)")
        print("- demo_results/ (analysis results)")
        print("\nTo run the full analysis with real Quran data:")
        print("python quran_analysis.py")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\nError: {e}")
        print("Please check the installation and try again.")

if __name__ == "__main__":
    main()
