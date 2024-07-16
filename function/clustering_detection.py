import os
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from PIL import Image
import shutil
import pdb
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def plot_elbow_method(X, max_clusters, method, progress_bar):
    distortions = []
    
    for n_clusters in range(2, max_clusters + 1):
        if method == 'K-Means':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'DBSCAN':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'Agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)

        cluster_labels = clusterer.fit_predict(X)
        distortion = clusterer.inertia_
        distortions.append(distortion)
        
        progress_bar.progress((n_clusters - 2) / (max_clusters - 1))

    plt.plot(range(2, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Numero di Cluster')
    plt.ylabel('Distorsione (Inertia)')
    plt.title(f'Metodo del Gomito per {method}')
    
    return plt

def calculate_silhouette_score(X, labels):
    score = silhouette_score(X, labels)
    return score

def plot_silhouette_scores(X, max_clusters, method, progress_bar):
    scores = []
    # st.write(f'Numero massimo di cluster: {max_clusters}')
    
    for n_clusters in range(2, max_clusters + 1):
        if method == 'K-Means':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'DBSCAN':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'Agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)

        cluster_labels = clusterer.fit_predict(X)
        score = calculate_silhouette_score(X, cluster_labels)
        scores.append(score)
        # st.write(f'Silhouette Score per {n_clusters} cluster: {score}')
        progress_bar.progress((n_clusters - 2) / (max_clusters - 1))

    plt.plot(range(2, max_clusters + 1), scores, marker='o')
    plt.xlabel('Numero di Cluster')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Score per {method}')
    
    return plt

def apply_binarization(image, method, threshold=128):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'Global':
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    elif method == 'Adaptive':
        binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == 'Otsu':
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'Gaussian':
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'Inverse':
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)

    return binary_image

def perform_clustering(symbols_folder, num_clusters, clustering_algorithm, binarization_method, is_pca, silscore, elbowscore):
    # Crea una lista di percorsi delle immagini nella cartella 'symbols'
    image_paths = [os.path.join(symbols_folder, filename) for filename in os.listdir(symbols_folder) if filename.endswith(('.jpg', '.png'))]

    st.subheader('Loading...')
    progress_bar_im = st.progress(0)
    
    # Carica le immagini e converte in array
    images = [Image.open(image_path).convert('RGB').resize((100, 100)) for image_path in image_paths]

    col1, col2 = st.columns(2)
    
    with col1:
        # visualizza le prime 5 immagini
        st.subheader('First 5 images')
        st.image(images[:5], width=50)

    # Applica la binarizzazione alle immagini
    binarized_images = [apply_binarization(np.array(image), binarization_method) for image in images]
    
    with col2:
        # Visualizza le prime 5 immagini binarizzate
        st.subheader('First 5 binarized images:')
        st.image(binarized_images[:5], width=50)

    image_arrays = [np.array(image) for image in binarized_images]
    progress_bar_im.progress(1.0)

    # Flattening degli array delle immagini
    flattened_images = [image.flatten() for image in image_arrays]

    # Scegli l'algoritmo di clustering
    if clustering_algorithm == 'K-Means':
        cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
    # elif clustering_algorithm == 'DBSCAN':
    #     cluster_model = DBSCAN(eps=0.8, min_samples=25)  # Modifica eps e min_samples secondo le tue esigenze
    elif clustering_algorithm == 'Agglomerative':
        cluster_model = AgglomerativeClustering(n_clusters=num_clusters)

    # add checkbox for PCA
    if is_pca:
        st.subheader('Riduzione delle dimensioni utilizzando PCA...')
        progress_bar_pca = st.progress(0)
        num_components = min(len(flattened_images), len(flattened_images[0]))
        pca = PCA(n_components=num_components)
        reduced_images = pca.fit_transform(flattened_images)
        progress_bar_pca.progress(1.0)
    else:
        reduced_images = flattened_images

    # pdb.set_trace()
    col1, col2 = st.columns(2)
    with col1:
        if silscore:
            st.subheader('Silhouette Score:')
            progress_bar = st.progress(1)
            fig = plot_silhouette_scores(reduced_images, num_clusters, clustering_algorithm, progress_bar)
            st.pyplot(fig)
    with col2:
        if elbowscore:
            st.subheader(' Elbow Method:')
            progress_bar_elbow = st.progress(1)
            fig_elbow = plot_elbow_method(reduced_images, num_clusters, clustering_algorithm, progress_bar_elbow)
            st.pyplot(fig_elbow)
    
    # Add progress bar for clustering fitting
    st.subheader('Clustering...')
    progress_bar_cluster = st.progress(0)
    
    # Esegui il clustering
    cluster_labels = cluster_model.fit_predict(reduced_images)
    progress_bar_cluster.progress(1.0)  # Update progress bar to indicate completion of clustering

    # Visualizza i cluster individuati
    st.subheader('Clusters:')
    cluster_symbols = {}
    for cluster in set(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_images = [Image.open(image_paths[i]) for i in cluster_indices]

        cluster_symbols[cluster] = [os.path.basename(image_paths[i]) for i in cluster_indices]
        st.write(f'**Cluster {cluster + 1}:**')
        st.image(cluster_images, width=50)
        cluster_symbols[cluster] = [os.path.basename(image_paths[i]) for i in cluster_indices]

    return cluster_symbols



# Funzione per salvare la classificazione del cluster
def save_cluster_classification(symbols_folder, cluster, symbols_list):
    cluster_folder = os.path.join(symbols_folder, f'cluster_{cluster + 1}')
    os.makedirs(cluster_folder, exist_ok=True)
    
    for symbol in symbols_list:
        symbol_path = os.path.join(symbols_folder, symbol)
        destination_path = os.path.join(cluster_folder, symbol)
        
        try:
            shutil.copy(symbol_path, destination_path)

            # Salvataggio delle classificazioni in formato JPG
            img = Image.open(destination_path).convert('RGB')
            img.save(os.path.join(cluster_folder, f'{symbol.split(".")[0]}.jpg'))

        except Exception as e:
            st.error(f'Errore nel salvataggio della classificazione del Cluster {cluster + 1}: {str(e)}')

    st.success(f'Classificazione del Cluster {cluster + 1} salvata con successo.')

def create_dataset(symbols_folder, num_clusters, clustering_algorithm, binarization_method, is_pca, silscore, elbowscore):
    cluster_symbols = perform_clustering(symbols_folder, num_clusters, clustering_algorithm, binarization_method, is_pca, silscore, elbowscore)

    # Creazione della cartella del dataset
    dataset_folder = os.path.join(os.getcwd(), 'dataset')
    os.makedirs(dataset_folder, exist_ok=True)

    for cluster in range(num_clusters):
        cluster_folder = os.path.join(dataset_folder, f'cluster_{cluster + 1}')
        os.makedirs(cluster_folder, exist_ok=True)

        # Sposta le immagini del cluster nella rispettiva cartella
        for symbol in cluster_symbols[cluster]:
            symbol_path = os.path.join(symbols_folder, symbol)
            destination_path = os.path.join(cluster_folder, symbol)
            shutil.copy(symbol_path, destination_path)

def main():
    st.title('Symbol Clustering App')

    # Widget for selecting the symbols folder
    symbols_folder = st.sidebar.selectbox('Select symbols folder', os.listdir('.'), index=8)

    # Widget for enabling PCA
    is_pca = st.sidebar.checkbox("Dimensionality reduction using PCA")

    # Widget for selecting the number of clusters
    num_clusters = st.sidebar.slider('Select the number of clusters', 2, 100, 5)

    # Widget for selecting the clustering algorithm
    clustering_algorithm = st.sidebar.selectbox('Select the clustering algorithm', ['K-Means', 'Agglomerative'])

    # Checkbox for Silhouette Score
    silcscore = st.checkbox('Silhouette Score')

    # Checkbox for Elbow Method
    elbowscore = st.checkbox('Elbow Method')

    # Widget for selecting the binarization method, set default to gaussian
    binarization_method = st.sidebar.selectbox('Select the binarization method', ['Global', 'Adaptive', 'Otsu', 'Gaussian', 'Inverse'], index=3)

    # Button widget to perform clustering
    if st.sidebar.button('Perform Clustering'):
        perform_clustering(symbols_folder, num_clusters, clustering_algorithm, binarization_method, is_pca, silcscore, elbowscore)

    # Button widget for "Create Dataset"
    if st.sidebar.button('Create Dataset'):
        create_dataset(symbols_folder, num_clusters, clustering_algorithm, binarization_method, is_pca, silcscore, elbowscore)
        st.success(f'Dataset created successfully.')


if __name__ == "__main__":
    main()
