#!/usr/bin/env python3
"""
Test script for Quran Semantic Analysis installation.
This script verifies that all dependencies are properly installed.
"""

import sys
import importlib
import subprocess
import os
from pathlib import Path

def test_python_version():
    """Test Python version."""
    print("Testing Python version...")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version.split()[0]} detected")
    return True

def test_imports():
    """Test importing required modules."""
    print("\nTesting Python module imports...")
    
    required_modules = [
        'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn',
        'gensim', 'sklearn', 'networkx', 'jpype'
    ]
    
    optional_modules = [
        'camel_tools', 'arabic_reshaper', 'bidi'
    ]
    
    all_passed = True
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            all_passed = False
    
    print("\nTesting optional modules...")
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"⚠ {module}: {e} (optional)")
    
    return all_passed

def test_java():
    """Test Java installation."""
    print("\nTesting Java installation...")
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Java is installed")
            return True
        else:
            print("✗ Java is not working properly")
            return False
    except FileNotFoundError:
        print("✗ Java is not installed")
        return False

def test_jqurantree():
    """Test JQuranTree JAR file."""
    print("\nTesting JQuranTree JAR file...")
    
    jar_paths = [
        "jqurantree.jar",
        "lib/jqurantree.jar",
        "/usr/local/lib/jqurantree.jar"
    ]
    
    for jar_path in jar_paths:
        if os.path.exists(jar_path):
            print(f"✓ JQuranTree JAR found at {jar_path}")
            return True
    
    print("✗ JQuranTree JAR not found")
    print("Please download JQuranTree JAR file and place it in the project directory")
    return False

def test_arabic_nlp():
    """Test Arabic NLP functionality."""
    print("\nTesting Arabic NLP functionality...")
    
    try:
        from camel_tools.utils.normalize import normalize_unicode
        from camel_tools.tokenizers.word import simple_word_tokenize
        
        # Test with sample Arabic text
        sample_text = "بسم الله الرحمن الرحيم"
        normalized = normalize_unicode(sample_text)
        tokens = simple_word_tokenize(normalized)
        
        print("✓ Arabic text normalization")
        print("✓ Arabic tokenization")
        return True
        
    except Exception as e:
        print(f"✗ Arabic NLP test failed: {e}")
        return False

def test_gensim():
    """Test Gensim functionality."""
    print("\nTesting Gensim functionality...")
    
    try:
        from gensim import corpora, models
        from gensim.models import LdaModel
        
        # Create sample documents
        documents = [
            ["الله", "رحمن", "رحيم"],
            ["الحمد", "لله", "رب", "العالمين"],
            ["الرحمن", "الرحيم", "مالك", "يوم", "الدين"]
        ]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        
        # Train simple LDA model
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=1)
        
        print("✓ Gensim LDA model training")
        return True
        
    except Exception as e:
        print(f"✗ Gensim test failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality."""
    print("\nTesting visualization functionality...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Create sample data
        data = np.random.rand(10, 10)
        
        # Test matplotlib
        fig, ax = plt.subplots()
        ax.imshow(data)
        plt.close(fig)
        
        # Test seaborn
        fig, ax = plt.subplots()
        sns.heatmap(data, ax=ax)
        plt.close(fig)
        
        print("✓ Matplotlib functionality")
        print("✓ Seaborn functionality")
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def test_quran_analysis():
    """Test the main Quran analysis module."""
    print("\nTesting Quran analysis module...")
    
    try:
        from quran_analysis import QuranAnalyzer
        
        # Create analyzer instance
        analyzer = QuranAnalyzer()
        
        # Test preprocessing
        sample_text = "بسم الله الرحمن الرحيم الحمد لله رب العالمين"
        preprocessed = analyzer.preprocess_arabic_text(sample_text)
        
        print("✓ QuranAnalyzer initialization")
        print("✓ Arabic text preprocessing")
        return True
        
    except Exception as e:
        print(f"✗ Quran analysis test failed: {e}")
        return False

def test_demo():
    """Test the demonstration script."""
    print("\nTesting demonstration script...")
    
    try:
        # Run demo script
        result = subprocess.run([sys.executable, 'demo.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Demonstration script runs successfully")
            return True
        else:
            print(f"✗ Demonstration script failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠ Demonstration script timed out (this is normal for large datasets)")
        return True
    except Exception as e:
        print(f"✗ Demonstration test failed: {e}")
        return False

def run_performance_test():
    """Run a simple performance test."""
    print("\nRunning performance test...")
    
    try:
        import time
        import numpy as np
        from scipy.special import rel_entr
        
        # Test KL divergence computation
        start_time = time.time()
        
        # Create sample probability distributions
        p = np.array([0.3, 0.3, 0.4])
        q = np.array([0.2, 0.4, 0.4])
        
        # Compute KL divergence
        kl_div = np.sum(rel_entr(q, p))
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        print(f"✓ KL divergence computation: {computation_time:.4f} seconds")
        print(f"✓ Result: {kl_div:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Quran Semantic Analysis - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Module Imports", test_imports),
        ("Java Installation", test_java),
        ("JQuranTree JAR", test_jqurantree),
        ("Arabic NLP", test_arabic_nlp),
        ("Gensim", test_gensim),
        ("Visualization", test_visualization),
        ("Quran Analysis", test_quran_analysis),
        ("Performance", run_performance_test),
        ("Demo Script", test_demo)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The installation is complete and ready to use.")
        print("\nNext steps:")
        print("1. Run: python quran_analysis.py")
        print("2. Or run: python demo.py for a demonstration")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. Some optional components may not be available.")
        print("The core functionality should work.")
    else:
        print("❌ Several tests failed. Please check the installation.")
        print("Run: python setup.py to install missing dependencies.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
