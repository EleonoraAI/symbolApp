import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd 
import pdb 
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, SpectralClustering
from sklearn.mixture import GaussianMixture


def applyClustering(symbols_detected, object_info, num_clusters, img):
    if len(object_info) >= num_clusters:
        # Crea gli algoritmi di clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        dbscan = DBSCAN(eps=0.1, min_samples=5)
        agg_clustering = AgglomerativeClustering(n_clusters=num_clusters)
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        birch = Birch(n_clusters=num_clusters)
        spectral = SpectralClustering(n_clusters=num_clusters, random_state=42)

        # Lista di algoritmi e nomi per ciclare attraverso di essi
        algorithms = [kmeans, dbscan, agg_clustering, gmm, birch, spectral]
        algorithm_names = ['K-Means', 'DBSCAN', 'Agglomerative Clustering', 'Gaussian Mixture Model', 'Birch', 'Spectral Clustering']

        # Titolo dell'app
        st.title("Clustering algorithms")

        # Dividi la pagina in 6 colonne
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        for i in range(len(algorithms)):
            algorithm = algorithms[i]
            algorithm_name = algorithm_names[i]

            # Seleziona la colonna corretta in base all'indice dell'iterazione
            if i == 0:
                col = col1
            elif i == 1:
                col = col2
            elif i == 2:
                col = col3
            elif i == 3:
                col = col4
            elif i == 4:
                col = col5
            else:
                col = col6

            # Chiamata alla funzione plot_clusters e visualizzazione del plot tramite st.pyplot nella colonna selezionata
            with col:
                plot_clusters(symbols_detected, algorithm, algorithm_name, object_info, num_clusters, img)


def plot_clusters(symbols_detected, algorithm, algorithm_name, object_info, num_clusters, img):
    object_info = np.array(object_info)  # Convert object_info to a NumPy array
    if len(object_info) >= num_clusters:
        cluster_labels = algorithm.fit_predict(object_info)

        # Crea un plot per visualizzare i cluster
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying the original image
        plt.imshow(cv2.cvtColor(symbols_detected, cv2.COLOR_BGR2RGB), alpha=0.6)
        for cluster_label in np.unique(cluster_labels):
            if cluster_label == -1:  # DBSCAN assigns noise points to cluster label -1
                color = 'gray'
            else:
                color = plt.cm.tab20(cluster_label % 20)  # Usa una mappa di colori per i cluster
            plt.scatter(object_info[cluster_labels == cluster_label, -2], object_info[cluster_labels == cluster_label, -1], c=color, s=1, label=f"Cluster {cluster_label}")

        plt.legend()
        plt.title(f"{algorithm_name}")
        plt.xlabel("Coord X")
        plt.ylabel("Coord Y")
        st.pyplot(plt)
            