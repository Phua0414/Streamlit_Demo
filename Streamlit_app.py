import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import skew
import umap

def convert_less_than(value):
    if isinstance(value, str) and value.startswith("<"):
        try:
            return float(re.sub(r'[^\d.]', '', value)) / 2
        except ValueError:
            return None
    return value

def preprocess_data(df):
    df = df.drop(columns=['Sample No', 'Dates'], errors='ignore')
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_df = pd.DataFrame(encoder.fit_transform(df[['Water Control Zone']]), columns=encoder.get_feature_names_out(['Water Control Zone']))
    df = df.drop(columns=['Water Control Zone']).join(encoded_df)
    df['Station'] = df['Station'].map(df['Station'].value_counts(normalize=True))
    depth_order = {'Surface Water': 0, 'Middle Water': 1, 'Bottom Water': 2}
    df['Depth'] = df['Depth'].map(depth_order)
    numeric_cols = df.select_dtypes(include=['object']).columns
    for col in numeric_cols:
        df[col] = df[col].apply(convert_less_than)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)
    return df_scaled

def perform_clustering(df, algorithm, k=4, eps=0.5, min_samples=10):
    pca = PCA(n_components=2)
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
    else:
        return None, None
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
        algorithm = st.selectbox("Select Clustering Algorithm", ["K-Means", "DBSCAN", "Mean Shift", "Gaussian Mixture"])
        k = st.slider("Select Number of Clusters (for K-Means & GMM)", 2, 10, 4) if algorithm in ["K-Means", "Gaussian Mixture"] else None
        eps = st.slider("Select Epsilon (eps) Value", 0.1, 5.0, 0.5) if algorithm == "DBSCAN" else None
        min_samples = st.slider("Select Min Samples", 1, 20, 10) if algorithm == "DBSCAN" else None
        if st.button("Run Clustering"):
            df_pca, labels, silhouette, db_index = perform_clustering(df_scaled, algorithm, k, eps, min_samples)
            st.write(f"### {algorithm} Clustering Results")
            st.write(f"Silhouette Score: {silhouette:.6f}")
            st.write(f"Davies-Bouldin Index: {db_index:.6f}")
            plot_clusters(df_pca, labels, f"{algorithm} Clustering")

if __name__ == "__main__":
    main()
