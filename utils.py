"""
Utility functions for Quran Semantic Analysis.
Contains helper functions for advanced analysis and data processing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from scipy.spatial.distance import jensenshannon, wasserstein_distance
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from config import ANALYSIS_CONFIG, SURAH_NAMES

def compute_symmetric_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Compute symmetric similarity matrix from asymmetric relationship matrix.
    
    Args:
        matrix: Asymmetric relationship matrix
        
    Returns:
        Symmetric similarity matrix
    """
    # Convert KL divergence to similarity using exponential
    similarity = np.exp(-matrix)
    
    # Make symmetric by taking average
    symmetric_similarity = (similarity + similarity.T) / 2
    
    return symmetric_similarity

def find_thematic_clusters(matrix: np.ndarray, n_clusters: int = 5) -> Dict:
    """
    Find thematic clusters in the surahs using the relationship matrix.
    
    Args:
        matrix: Relationship matrix
        n_clusters: Number of clusters to find
        
    Returns:
        Dictionary with cluster information
    """
    from sklearn.cluster import KMeans
    
    # Convert to similarity matrix
    similarity = compute_symmetric_similarity(matrix)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(similarity)
    
    # Group surahs by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(f"Surah {i+1}")
    
    return {
        'clusters': clusters,
        'labels': cluster_labels,
        'centers': kmeans.cluster_centers_
    }

def compute_centrality_measures(matrix: np.ndarray) -> Dict:
    """
    Compute centrality measures for surahs based on the relationship matrix.
    
    Args:
        matrix: Relationship matrix
        
    Returns:
        Dictionary with centrality measures
    """
    # Convert to similarity matrix
    similarity = compute_symmetric_similarity(matrix)
    
    # Create network graph
    G = nx.from_numpy_array(similarity)
    
    # Compute centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    
    return {
        'degree': degree_centrality,
        'betweenness': betweenness_centrality,
        'closeness': closeness_centrality,
        'eigenvector': eigenvector_centrality
    }

def analyze_thematic_flow(matrix: np.ndarray, surah_names: List[str]) -> Dict:
    """
    Analyze the thematic flow between surahs.
    
    Args:
        matrix: Relationship matrix
        surah_names: List of surah names
        
    Returns:
        Dictionary with thematic flow analysis
    """
    # Find strongest connections
    n_surahs = len(surah_names)
    connections = []
    
    for i in range(n_surahs):
        for j in range(n_surahs):
            if i != j:
                connections.append({
                    'source': surah_names[i],
                    'target': surah_names[j],
                    'strength': matrix[i, j],
                    'asymmetry': matrix[i, j] - matrix[j, i]
                })
    
    # Sort by strength
    connections.sort(key=lambda x: x['strength'], reverse=True)
    
    # Find thematic hubs (surahs with many strong connections)
    hub_scores = {}
    for i, surah in enumerate(surah_names):
        # Count strong connections (top 10% of all connections)
        threshold = np.percentile([c['strength'] for c in connections], 90)
        strong_connections = [c for c in connections if c['source'] == surah and c['strength'] > threshold]
        hub_scores[surah] = len(strong_connections)
    
    # Find thematic bridges (surahs connecting different clusters)
    bridge_scores = {}
    for i, surah in enumerate(surah_names):
        # Calculate how well this surah connects to others
        outgoing_strength = np.mean([matrix[i, j] for j in range(n_surahs) if j != i])
        incoming_strength = np.mean([matrix[j, i] for j in range(n_surahs) if j != i])
        bridge_scores[surah] = (outgoing_strength + incoming_strength) / 2
    
    return {
        'top_connections': connections[:20],
        'hub_scores': hub_scores,
        'bridge_scores': bridge_scores,
        'thematic_hubs': sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:10],
        'thematic_bridges': sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    }

def create_network_visualization(matrix: np.ndarray, surah_names: List[str], 
                                top_connections: int = 50, 
                                save_path: str = None) -> plt.Figure:
    """
    Create a network visualization of surah relationships.
    
    Args:
        matrix: Relationship matrix
        surah_names: List of surah names
        top_connections: Number of top connections to show
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Convert to similarity matrix
    similarity = compute_symmetric_similarity(matrix)
    
    # Create network graph
    G = nx.from_numpy_array(similarity)
    
    # Get top connections
    edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    edges.sort(key=lambda x: x[2], reverse=True)
    top_edges = edges[:top_connections]
    
    # Create subgraph with top connections
    top_nodes = set()
    for u, v, w in top_edges:
        top_nodes.add(u)
        top_nodes.add(v)
    
    subgraph = G.subgraph(top_nodes)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Calculate layout
    pos = nx.spring_layout(subgraph, k=3, iterations=50)
    
    # Draw nodes
    node_sizes = [G.degree(node) * 50 for node in subgraph.nodes()]
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7, ax=ax)
    
    # Draw edges
    edge_weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]
    nx.draw_networkx_edges(subgraph, pos, width=edge_weights, 
                          alpha=0.6, edge_color='gray', ax=ax)
    
    # Draw labels
    labels = {i: surah_names[i] for i in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(f'Quran Surah Network (Top {top_connections} Connections)', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def perform_dimensionality_reduction(matrix: np.ndarray, method: str = 'tsne') -> np.ndarray:
    """
    Perform dimensionality reduction on the relationship matrix.
    
    Args:
        matrix: Relationship matrix
        method: Reduction method ('pca' or 'tsne')
        
    Returns:
        Reduced coordinates
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Convert to similarity matrix
    similarity = compute_symmetric_similarity(matrix)
    
    # Apply dimensionality reduction
    reduced = reducer.fit_transform(similarity)
    
    return reduced

def create_dimensionality_plot(matrix: np.ndarray, surah_names: List[str], 
                             method: str = 'tsne', save_path: str = None) -> plt.Figure:
    """
    Create a 2D plot of surahs using dimensionality reduction.
    
    Args:
        matrix: Relationship matrix
        surah_names: List of surah names
        method: Reduction method ('pca' or 'tsne')
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Perform dimensionality reduction
    reduced = perform_dimensionality_reduction(matrix, method)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot points
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=100)
    
    # Add labels
    for i, surah in enumerate(surah_names):
        ax.annotate(surah, (reduced[i, 0], reduced[i, 1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_title(f'Quran Surahs in 2D Space ({method.upper()})', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def compute_statistical_measures(matrix: np.ndarray) -> Dict:
    """
    Compute various statistical measures for the relationship matrix.
    
    Args:
        matrix: Relationship matrix
        
    Returns:
        Dictionary with statistical measures
    """
    # Basic statistics
    stats = {
        'mean': np.mean(matrix),
        'std': np.std(matrix),
        'min': np.min(matrix),
        'max': np.max(matrix),
        'median': np.median(matrix)
    }
    
    # Asymmetry statistics
    asymmetry = matrix - matrix.T
    stats.update({
        'mean_asymmetry': np.mean(asymmetry),
        'std_asymmetry': np.std(asymmetry),
        'max_asymmetry': np.max(asymmetry),
        'min_asymmetry': np.min(asymmetry)
    })
    
    # Distribution statistics
    upper_triangle = matrix[np.triu_indices(matrix.shape[0], k=1)]
    stats.update({
        'mean_upper_triangle': np.mean(upper_triangle),
        'std_upper_triangle': np.std(upper_triangle)
    })
    
    # Correlation with position
    n_surahs = matrix.shape[0]
    positions = np.arange(n_surahs)
    correlations = []
    
    for i in range(n_surahs):
        corr = np.corrcoef(positions, matrix[i, :])[0, 1]
        correlations.append(corr)
    
    stats.update({
        'mean_position_correlation': np.mean(correlations),
        'std_position_correlation': np.std(correlations)
    })
    
    return stats

def export_analysis_results(matrix: np.ndarray, surah_names: List[str], 
                           output_dir: str = "advanced_results") -> None:
    """
    Export comprehensive analysis results.
    
    Args:
        matrix: Relationship matrix
        surah_names: List of surah names
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute all analyses
    clusters = find_thematic_clusters(matrix)
    centrality = compute_centrality_measures(matrix)
    thematic_flow = analyze_thematic_flow(matrix, surah_names)
    stats = compute_statistical_measures(matrix)
    
    # Export cluster information
    cluster_df = pd.DataFrame({
        'surah': surah_names,
        'cluster': clusters['labels']
    })
    cluster_df.to_csv(os.path.join(output_dir, "clusters.csv"), index=False)
    
    # Export centrality measures
    centrality_df = pd.DataFrame({
        'surah': surah_names,
        'degree_centrality': [centrality['degree'][i] for i in range(len(surah_names))],
        'betweenness_centrality': [centrality['betweenness'][i] for i in range(len(surah_names))],
        'closeness_centrality': [centrality['closeness'][i] for i in range(len(surah_names))],
        'eigenvector_centrality': [centrality['eigenvector'][i] for i in range(len(surah_names))]
    })
    centrality_df.to_csv(os.path.join(output_dir, "centrality_measures.csv"), index=False)
    
    # Export thematic flow analysis
    with open(os.path.join(output_dir, "thematic_flow_analysis.txt"), 'w', encoding='utf-8') as f:
        f.write("Thematic Flow Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Top Connections:\n")
        for i, conn in enumerate(thematic_flow['top_connections'][:20]):
            f.write(f"{i+1}. {conn['source']} → {conn['target']}: {conn['strength']:.4f}\n")
        
        f.write("\nThematic Hubs:\n")
        for surah, score in thematic_flow['thematic_hubs']:
            f.write(f"{surah}: {score}\n")
        
        f.write("\nThematic Bridges:\n")
        for surah, score in thematic_flow['thematic_bridges']:
            f.write(f"{surah}: {score:.4f}\n")
    
    # Export statistical measures
    with open(os.path.join(output_dir, "statistical_measures.txt"), 'w', encoding='utf-8') as f:
        f.write("Statistical Measures\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in stats.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Create network visualization
    network_fig = create_network_visualization(matrix, surah_names, save_path=os.path.join(output_dir, "network_visualization.png"))
    
    # Create dimensionality reduction plot
    tsne_fig = create_dimensionality_plot(matrix, surah_names, method='tsne', save_path=os.path.join(output_dir, "tsne_plot.png"))
    pca_fig = create_dimensionality_plot(matrix, surah_names, method='pca', save_path=os.path.join(output_dir, "pca_plot.png"))
    
    print(f"Advanced analysis results exported to {output_dir}/")

def compare_analysis_methods(matrix1: np.ndarray, matrix2: np.ndarray, 
                           method_names: List[str] = None) -> Dict:
    """
    Compare two analysis methods by comparing their matrices.
    
    Args:
        matrix1: First relationship matrix
        matrix2: Second relationship matrix
        method_names: Names of the methods
        
    Returns:
        Dictionary with comparison results
    """
    if method_names is None:
        method_names = ['Method 1', 'Method 2']
    
    # Compute correlation between matrices
    correlation = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]
    
    # Compute mean absolute difference
    mean_diff = np.mean(np.abs(matrix1 - matrix2))
    
    # Compute relative difference
    relative_diff = np.mean(np.abs(matrix1 - matrix2) / (matrix1 + 1e-10))
    
    # Find most different relationships
    diff_matrix = np.abs(matrix1 - matrix2)
    max_diff_idx = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
    
    return {
        'correlation': correlation,
        'mean_absolute_difference': mean_diff,
        'relative_difference': relative_diff,
        'max_difference': np.max(diff_matrix),
        'max_difference_location': max_diff_idx,
        'method_names': method_names
    }
