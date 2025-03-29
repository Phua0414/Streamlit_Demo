import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import skew
import umap

# Streamlit App Title
st.title("Marine Water Quality Data Analysis")

# Upload CSV File
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Overview")
    st.write(df.head())

    st.subheader("Checking Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cmap="coolwarm", cbar=False, ax=ax)
    st.pyplot(fig)

    # Data Cleaning: Drop Unnecessary Columns
    df = df.drop(columns=['Sample No', 'Dates'], errors='ignore')

    # Encode Categorical Data
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_df = pd.DataFrame(encoder.fit_transform(df[['Water Control Zone']]), 
                              columns=encoder.get_feature_names_out(['Water Control Zone']))
    df = df.drop(columns=['Water Control Zone']).join(encoded_df)

    # Apply PCA for Dimensionality Reduction
    st.subheader("PCA for Dimensionality Reduction")
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=[np.number])), columns=df.select_dtypes(include=[np.number]).columns)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_pca[:, 0], df_pca[:, 1], alpha=0.6)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Result (2D Projection)")
    st.pyplot(fig)

    # K-Means Clustering
    st.subheader("K-Means Clustering")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_pca)
    
    silhouette = silhouette_score(df_pca, labels)
    db_index = davies_bouldin_score(df_pca, labels)

    st.write(f"Silhouette Score: {silhouette:.6f}")
    st.write(f"Davies-Bouldin Index: {db_index:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    ax.set_title("K-Means Clustering")
    ax.set_xlabel("First Component")
    ax.set_ylabel("Second Component")
    st.pyplot(fig)
