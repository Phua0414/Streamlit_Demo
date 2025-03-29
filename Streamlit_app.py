import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth, AgglomerativeClustering, OPTICS, AffinityPropagation, Birch, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import skew
import umap
import hdbscan

def convert_less_than(value):
    if isinstance(value, str) and value.startswith("<"):
        try:
            return float(re.sub(r'[^\d.]', '', value)) / 2
        except ValueError:
            return None
    return value

def preprocess_data(df):
    #Remove the Sample No and Dates
    df = df.drop(columns=['Sample No', 'Dates'], errors='ignore')
    
    #Encoded the categorical column
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_df = pd.DataFrame(encoder.fit_transform(df[['Water Control Zone']]), columns=encoder.get_feature_names_out(['Water Control Zone']))
    df = df.drop(columns=['Water Control Zone']).join(encoded_df)
    df['Station'] = df['Station'].map(df['Station'].value_counts(normalize=True))
    depth_order = {'Surface Water': 0, 'Middle Water': 1, 'Bottom Water': 2}
    df['Depth'] = df['Depth'].map(depth_order)

    #Transformation : Log and MinMaxScaler
    numeric_cols = df.select_dtypes(include=['object']).columns
    for col in numeric_cols:
        df[col] = df[col].apply(convert_less_than)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    skew_values_before = df_numeric.apply(skew)
    high_skew_cols = skew_values_before[skew_values_before > 1].index
    df_numeric[high_skew_cols] = np.log1p(df_numeric[high_skew_cols])
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)
    return df_scaled

def perform_clustering(df, algorithm, k=None, eps=None, min_samples=None, damping=None, preference=None, n_components=None):
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)
    
    if algorithm == "K-Means":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == "Mean Shift":
        bandwidth = estimate_bandwidth(df_pca, quantile=0.2)
        model = MeanShift(bandwidth=bandwidth)
    elif algorithm == "Gaussian Mixture":
        model = GaussianMixture(n_components=k, random_state=42)
    elif algorithm == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=k)
    elif algorithm == "OPTICS":
        model = OPTICS(min_samples=min_samples)
    elif algorithm == "HDBSCAN":
        model = hdbscan.HDBSCAN(min_cluster_size=min_samples)
    elif algorithm == "Affinity Propagation":
        model = AffinityPropagation(damping=damping, preference=preference)
    elif algorithm == "BIRCH":
        model = Birch(n_clusters=k)
    elif algorithm == "Spectral Clustering":
        model = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors')
    else:
        return None, None, None, None
    
    labels = model.fit_predict(df_pca)
    
    if len(set(labels)) > 1:
        silhouette = silhouette_score(df_pca, labels)
        db_index = davies_bouldin_score(df_pca, labels)
    else:
        silhouette, db_index = -1, -1
    
    return df_pca, labels, silhouette, db_index

def plot_clusters(df_pca, labels, title):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
    if cluster_centers is not None:
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
        plt.legend()
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    st.pyplot(plt)

def main():
    st.title("Machine Learning Clustering App")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview")
        st.write(df.head())
        
        df_scaled = preprocess_data(df)
        st.write("### Processed Data Preview")
        st.write(df_scaled.head())
        
        n_components = st.slider("Select Number of PCA Components", 2, 5, 2)
        algorithm = st.selectbox("Select Clustering Algorithm", ["K-Means", "DBSCAN", "Mean Shift", "Gaussian Mixture", "Agglomerative Clustering", "OPTICS", "HDBSCAN", "Affinity Propagation", "BIRCH", "Spectral Clustering"])
        
        k = st.slider("Select Number of Clusters", 2, 10, 4) if algorithm in ["K-Means", "Gaussian Mixture", "Agglomerative Clustering", "BIRCH", "Spectral Clustering"] else None
        eps = st.slider("Select Epsilon (eps) Value", 0.1, 5.0, 0.5) if algorithm == "DBSCAN" else None
        min_samples = st.slider("Select Min Samples", 1, 20, 10) if algorithm in ["DBSCAN", "OPTICS", "HDBSCAN"] else None
        damping = st.slider("Select Damping Value", 0.5, 1.0, 0.9) if algorithm == "Affinity Propagation" else None
        preference = st.slider("Select Preference Value", -100, 100, -50) if algorithm == "Affinity Propagation" else None
        
        if st.button("Run Clustering"):
            df_pca, labels, silhouette, db_index = perform_clustering(df_scaled, algorithm, k, eps, min_samples, damping, preference, n_components)
            
            st.write(f"### {algorithm} Clustering Results")
            st.write(f"Silhouette Score: {silhouette:.6f}")
            st.write(f"Davies-Bouldin Index: {db_index:.6f}")
            plot_clusters(df_pca, labels, f"{algorithm} Clustering")

if __name__ == "__main__":
    main()
