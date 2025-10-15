"""
Configuration file for Quran Semantic Analysis.
Contains all configurable parameters for the analysis.
"""

# Analysis Parameters
ANALYSIS_CONFIG = {
    # Topic Modeling
    'num_topics': 20,  # Number of topics for LDA
    'lda_passes': 10,  # Number of passes for LDA training
    'lda_alpha': 'auto',  # LDA alpha parameter
    'lda_eta': 'auto',  # LDA eta parameter
    'lda_random_state': 42,  # Random state for reproducibility
    
    # Preprocessing
    'remove_diacritics': True,  # Remove Arabic diacritics
    'normalize_unicode': True,  # Normalize Unicode characters
    'normalize_alef': True,  # Normalize Alef variations
    'normalize_teh': True,  # Normalize Teh marbuta
    'normalize_whitespace': True,  # Normalize whitespace
    'remove_stopwords': True,  # Remove Arabic stop words
    'stem_words': True,  # Apply Arabic stemming
    'min_token_length': 2,  # Minimum token length
    
    # Analysis
    'top_n_relationships': 10,  # Number of top relationships to show
    'kl_divergence_epsilon': 1e-10,  # Small value to avoid log(0)
    'normalize_probabilities': True,  # Normalize topic distributions
    
    # Visualization
    'figure_size': (15, 12),  # Figure size for visualizations
    'dpi': 300,  # DPI for saved figures
    'colormap': 'viridis',  # Colormap for heatmaps
    'asymmetry_colormap': 'RdBu_r',  # Colormap for asymmetry visualization
    'show_every_nth_label': 5,  # Show every nth label on axes
    
    # Output
    'output_directory': 'results',  # Output directory
    'save_matrix_npy': True,  # Save matrix as NumPy file
    'save_matrix_csv': True,  # Save matrix as CSV
    'save_topic_distributions': True,  # Save topic distributions
    'save_analysis_results': True,  # Save analysis results
    'save_visualization': True,  # Save visualization
}

# Arabic Stop Words
ARABIC_STOPWORDS = {
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل', 'عند', 'لدى',
    'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'التي', 'الذين', 'اللاتي',
    'كان', 'كانت', 'يكون', 'تكون', 'كانوا', 'كن', 'كنت', 'كنتم', 'كنتما',
    'له', 'لها', 'لهما', 'لهم', 'لهن', 'له', 'لها', 'لهما', 'لهم', 'لهن',
    'ال', 'و', 'أو', 'لكن', 'إلا', 'إن', 'أن', 'أن', 'أن', 'أن', 'أن',
    'هو', 'هي', 'هم', 'هن', 'هما', 'أنت', 'أنتم', 'أنتن', 'أنتما',
    'أنا', 'نحن', 'هذا', 'هذه', 'هؤلاء', 'ذلك', 'تلك', 'أولئك',
    'كل', 'بعض', 'أي', 'أيها', 'أيتها', 'يا', 'أي', 'أيتها',
    'إذا', 'إذ', 'حيث', 'حيثما', 'أين', 'متى', 'كيف', 'لماذا',
    'لأن', 'لكن', 'بل', 'أم', 'أو', 'إما', 'أو', 'إما', 'أو'
}

# Arabic Diacritics to Remove
ARABIC_DIACRITICS = [
    'َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ', 'ٰ', 'ٱ', 'ۤ', 'ۥ', 'ۦ', 'ۧ', 'ۨ', '۩'
]

# Surah Names (for reference)
SURAH_NAMES = [
    "Al-Fatihah", "Al-Baqarah", "Ali 'Imran", "An-Nisa", "Al-Ma'idah",
    "Al-An'am", "Al-A'raf", "Al-Anfal", "At-Tawbah", "Yunus",
    "Hud", "Yusuf", "Ar-Ra'd", "Ibrahim", "Al-Hijr",
    "An-Nahl", "Al-Isra", "Al-Kahf", "Maryam", "Taha",
    "Al-Anbiya", "Al-Hajj", "Al-Mu'minun", "An-Nur", "Al-Furqan",
    "Ash-Shu'ara", "An-Naml", "Al-Qasas", "Al-Ankabut", "Ar-Rum",
    "Luqman", "As-Sajdah", "Al-Ahzab", "Saba", "Fatir",
    "Ya-Sin", "As-Saffat", "Sad", "Az-Zumar", "Ghafir",
    "Fussilat", "Ash-Shura", "Az-Zukhruf", "Ad-Dukhan", "Al-Jathiyah",
    "Al-Ahqaf", "Muhammad", "Al-Fath", "Al-Hujurat", "Qaf",
    "Adh-Dhariyat", "At-Tur", "An-Najm", "Al-Qamar", "Ar-Rahman",
    "Al-Waqi'ah", "Al-Hadid", "Al-Mujadilah", "Al-Hashr", "Al-Mumtahanah",
    "As-Saff", "Al-Jumu'ah", "Al-Munafiqun", "At-Taghabun", "At-Talaq",
    "At-Tahrim", "Al-Mulk", "Al-Qalam", "Al-Haqqah", "Al-Ma'arij",
    "Nuh", "Al-Jinn", "Al-Muzzammil", "Al-Muddaththir", "Al-Qiyamah",
    "Al-Insan", "Al-Mursalat", "An-Naba", "An-Nazi'at", "Abasa",
    "At-Takwir", "Al-Infitar", "Al-Mutaffifin", "Al-Inshiqaq", "Al-Buruj",
    "At-Tariq", "Al-A'la", "Al-Ghashiyah", "Al-Fajr", "Al-Balad",
    "Ash-Shams", "Al-Layl", "Ad-Duha", "Ash-Sharh", "At-Tin",
    "Al-Alaq", "Al-Qadr", "Al-Bayyinah", "Az-Zalzalah", "Al-Adiyat",
    "Al-Qari'ah", "At-Takathur", "Al-Asr", "Al-Humazah", "Al-Fil",
    "Quraysh", "Al-Ma'un", "Al-Kawthar", "Al-Kafirun", "An-Nasr",
    "Al-Masad", "Al-Ikhlas", "Al-Falaq", "An-Nas"
]

# Analysis Modes
ANALYSIS_MODES = {
    'quick': {
        'num_topics': 10,
        'lda_passes': 5,
        'description': 'Quick analysis for testing'
    },
    'standard': {
        'num_topics': 20,
        'lda_passes': 10,
        'description': 'Standard analysis for general use'
    },
    'detailed': {
        'num_topics': 30,
        'lda_passes': 20,
        'description': 'Detailed analysis for research'
    },
    'comprehensive': {
        'num_topics': 50,
        'lda_passes': 30,
        'description': 'Comprehensive analysis for deep research'
    }
}

# Visualization Themes
VISUALIZATION_THEMES = {
    'default': {
        'colormap': 'viridis',
        'asymmetry_colormap': 'RdBu_r',
        'figure_size': (15, 12),
        'dpi': 300
    },
    'colorblind_friendly': {
        'colormap': 'plasma',
        'asymmetry_colormap': 'PiYG',
        'figure_size': (15, 12),
        'dpi': 300
    },
    'print_friendly': {
        'colormap': 'gray',
        'asymmetry_colormap': 'RdBu_r',
        'figure_size': (12, 10),
        'dpi': 600
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'quran_analysis.log'
}

# Performance Settings
PERFORMANCE_CONFIG = {
    'batch_size': 100,  # Process surahs in batches
    'memory_limit_mb': 2048,  # Memory limit for processing
    'use_multiprocessing': True,  # Use multiprocessing for parallel processing
    'num_workers': 4,  # Number of worker processes
    'chunk_size': 50,  # Chunk size for batch processing
}

# Output Formats
OUTPUT_FORMATS = {
    'matrix': ['npy', 'csv', 'json'],
    'topics': ['csv', 'json'],
    'analysis': ['txt', 'json', 'html'],
    'visualization': ['png', 'pdf', 'svg']
}

# Validation Settings
VALIDATION_CONFIG = {
    'check_matrix_symmetry': True,  # Check that matrix is asymmetric
    'validate_probability_distributions': True,  # Validate topic distributions
    'check_kl_divergence_properties': True,  # Check KL divergence properties
    'validate_input_text': True,  # Validate input Arabic text
}

# Error Handling
ERROR_HANDLING = {
    'max_retries': 3,  # Maximum number of retries for failed operations
    'retry_delay': 1,  # Delay between retries (seconds)
    'fallback_to_sample_data': True,  # Use sample data if extraction fails
    'log_errors': True,  # Log errors to file
    'continue_on_error': True,  # Continue processing even if some operations fail
}
