# Installation Guide for Quran Semantic Analysis

This guide provides step-by-step instructions for setting up the Quran Semantic Analysis project.

## Quick Start

### 1. Prerequisites

- **Python 3.8+** (Python 3.13 recommended)
- **Java Development Kit (JDK) 11+** (for JQuranTree library)
- **Git** (for cloning the repository)

### 2. Installation Steps

#### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd matrix-project
```

#### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scipy scikit-learn networkx
```

#### Step 4: Test Installation
```bash
python3 simple_analysis.py
```

## Detailed Installation

### System Requirements

#### macOS
```bash
# Install Java
brew install openjdk@11

# Install Python dependencies
brew install python@3.13
```

#### Ubuntu/Debian
```bash
# Install Java
sudo apt-get update
sudo apt-get install openjdk-11-jdk

# Install Python dependencies
sudo apt-get install python3-dev libxml2-dev libxslt1-dev
```

#### Windows
1. Download and install Python 3.13 from [python.org](https://python.org)
2. Download and install OpenJDK 11 from [Adoptium](https://adoptium.net)
3. Add both to your system PATH

### Python Dependencies

#### Core Dependencies (Required)
- `numpy>=1.24.0` - Numerical computing
- `pandas>=1.5.0` - Data manipulation
- `matplotlib>=3.6.0` - Plotting and visualization
- `seaborn>=0.12.0` - Statistical visualization
- `scipy>=1.10.0` - Scientific computing
- `scikit-learn>=1.2.0` - Machine learning

#### Optional Dependencies
- `gensim>=4.3.0` - Topic modeling (may require compilation)
- `camel-tools>=1.5.0` - Arabic NLP tools
- `arabic-reshaper>=3.0.0` - Arabic text rendering
- `python-bidi>=0.4.2` - Bidirectional text support
- `jpype1>=1.4.1` - Java-Python bridge

### Installation Methods

#### Method 1: Basic Installation (Recommended)
```bash
# Install core dependencies only
pip install numpy pandas matplotlib seaborn scipy scikit-learn networkx

# Run simplified analysis
python3 simple_analysis.py
```

#### Method 2: Full Installation (Advanced)
```bash
# Install all dependencies
pip install -r requirements.txt

# Run full analysis
python3 quran_analysis.py
```

#### Method 3: Development Installation
```bash
# Install in development mode
pip install -e .

# Run tests
python3 test_installation.py
```

## Troubleshooting

### Common Issues

#### 1. Scipy Installation Fails
**Error**: `ERROR: Unknown compiler(s): [['gfortran']]`

**Solution**: Install Fortran compiler
```bash
# macOS
brew install gcc

# Ubuntu/Debian
sudo apt-get install gfortran

# Or use pre-compiled packages
pip install --only-binary=scipy scipy
```

#### 2. Gensim Installation Fails
**Error**: Compilation errors with Cython

**Solution**: Use alternative or skip gensim
```bash
# Try specific version
pip install gensim==4.2.0

# Or use simplified analysis without gensim
python3 simple_analysis.py
```

#### 3. Java/JQuranTree Issues
**Error**: `FileNotFoundError: JQuranTree JAR file not found`

**Solution**: Download JQuranTree manually
```bash
# Download JQuranTree JAR
wget https://github.com/jqurantree/jqurantree/releases/latest/download/jqurantree.jar

# Or use sample data
python3 simple_analysis.py  # Uses sample data automatically
```

#### 4. Arabic Text Issues
**Error**: Arabic text not displaying correctly

**Solution**: Install Arabic text support
```bash
pip install arabic-reshaper python-bidi
```

### Performance Issues

#### Memory Issues
- Reduce number of topics in LDA model
- Use fewer passes in training
- Process surahs in batches

#### Slow Processing
- Use simplified analysis for quick results
- Reduce matrix size for testing
- Use pre-computed results

## Verification

### Test Installation
```bash
python3 test_installation.py
```

### Run Demo
```bash
python3 demo.py
```

### Check Results
```bash
ls results/
# Should contain:
# - relationship_matrix.csv
# - relationship_matrix.npy
# - analysis_results.txt
# - simple_relationship_matrix.png
```

## Alternative Installations

### Using Conda
```bash
conda create -n quran-analysis python=3.13
conda activate quran-analysis
conda install numpy pandas matplotlib seaborn scipy scikit-learn
pip install networkx
```

### Using Docker
```dockerfile
FROM python:3.13-slim
RUN apt-get update && apt-get install -y openjdk-11-jdk
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python3", "simple_analysis.py"]
```

## Getting Help

### Documentation
- README.md - Project overview
- config.py - Configuration options
- utils.py - Utility functions

### Support
- Check GitHub issues
- Review error logs
- Use simplified analysis as fallback

### Examples
```bash
# Basic analysis
python3 simple_analysis.py

# Advanced analysis (if gensim is available)
python3 quran_analysis.py

# Custom configuration
python3 run_analysis.py --mode standard --topics 20
```

## Next Steps

1. **Run Analysis**: Execute `python3 simple_analysis.py`
2. **View Results**: Check the `results/` directory
3. **Customize**: Modify `config.py` for different settings
4. **Extend**: Add your own analysis methods in `utils.py`

## Notes

- The simplified analysis works without JQuranTree and gensim
- Sample data is used for demonstration purposes
- For production use, install JQuranTree for real Quran text
- Advanced features require full dependency installation
