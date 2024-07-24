import os
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import silhouette_score

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
        if hasattr(clusterer, 'inertia_'):
            distortion = clusterer.inertia_
        else:
            distortion = np.sum(np.min(clusterer.transform(X), axis=1)) / X.shape[0]
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
    image_paths = [os.path.join(symbols_folder, filename) for filename in os.listdir(symbols_folder) if filename.endswith(('.jpg', '.png'))]

    st.subheader('Loading...')
    progress_bar_im = st.progress(0)
    
    images = [Image.open(image_path).convert('RGB').resize((100, 100)) for image_path in image_paths]

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('First 5 images')
        st.image(images[:5], width=50)

    binarized_images = [apply_binarization(np.array(image), binarization_method) for image in images]
    
    with col2:
        st.subheader('First 5 binarized images:')
        st.image(binarized_images[:5], width=50)

    image_arrays = [np.array(image) for image in binarized_images]
    progress_bar_im.progress(1.0)

    flattened_images = [image.flatten() for image in image_arrays]

    if clustering_algorithm == 'K-Means':
        cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
    elif clustering_algorithm == 'Agglomerative':
        cluster_model = AgglomerativeClustering(n_clusters=num_clusters)

    if is_pca:
        st.subheader('Dimensionality reduction using PCA...')
        progress_bar_pca = st.progress(0)
        num_components = min(len(flattened_images), len(flattened_images[0]))
        pca = PCA(n_components=num_components)
        reduced_images = pca.fit_transform(flattened_images)
        progress_bar_pca.progress(1.0)
    else:
        reduced_images = flattened_images

    col1, col2 = st.columns(2)
    with col1:
        if silscore:
            st.subheader('Silhouette Score:')
            progress_bar = st.progress(1)
            fig = plot_silhouette_scores(reduced_images, num_clusters, clustering_algorithm, progress_bar)
            st.pyplot(fig)
    with col2:
        if elbowscore:
            st.subheader('Elbow Method:')
            progress_bar_elbow = st.progress(1)
            fig_elbow = plot_elbow_method(reduced_images, num_clusters, clustering_algorithm, progress_bar_elbow)
            st.pyplot(fig_elbow)
    
    st.subheader('Clustering...')
    progress_bar_cluster = st.progress(0)
    
    cluster_labels = cluster_model.fit_predict(reduced_images)
    progress_bar_cluster.progress(1.0)

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

def save_cluster_classification(symbols_folder, cluster, symbols_list):
    cluster_folder = os.path.join(symbols_folder, f'cluster_{cluster + 1}')
    os.makedirs(cluster_folder, exist_ok=True)
    
    for symbol in symbols_list:
        symbol_path = os.path.join(symbols_folder, symbol)
        destination_path = os.path.join(cluster_folder, symbol)
        
        try:
            shutil.copy(symbol_path, destination_path)

            img = Image.open(destination_path).convert('RGB')
            img.save(os.path.join(cluster_folder, f'{symbol.split(".")[0]}.jpg'))

        except Exception as e:
            st.error(f'Errore nel salvataggio della classificazione del Cluster {cluster + 1}: {str(e)}')

    st.success(f'Classificazione del Cluster {cluster + 1} salvata con successo.')

def create_dataset(symbols_folder, num_clusters, clustering_algorithm, binarization_method, is_pca, silscore, elbowscore):
    cluster_symbols = perform_clustering(symbols_folder, num_clusters, clustering_algorithm, binarization_method, is_pca, silscore, elbowscore)

    dataset_folder = os.path.join(os.getcwd(), 'dataset')
    os.makedirs(dataset_folder, exist_ok=True)

    for cluster in range(num_clusters):
        cluster_folder = os.path.join(dataset_folder, f'cluster_{cluster + 1}')
        os.makedirs(cluster_folder, exist_ok=True)

        for symbol in cluster_symbols[cluster]:
            symbol_path = os.path.join(symbols_folder, symbol)
            destination_path = os.path.join(cluster_folder, symbol)
            shutil.copy(symbol_path, destination_path)

def main():
    st.title('Symbol clustering')

    symbols_folder = "symbols"

    is_pca = st.sidebar.checkbox("Dimensionality reduction using PCA")
    num_clusters = st.sidebar.slider('Select the number of clusters', 2, 100, 5)
    clustering_algorithm = st.sidebar.selectbox('Select the clustering algorithm', ['K-Means', 'Agglomerative'])

    silcscore = st.sidebar.checkbox('Silhouette Score')
    elbowscore = st.sidebar.checkbox('Elbow Method')

    binarization_method = st.sidebar.selectbox('Select the binarization method', ['Global', 'Adaptive', 'Otsu', 'Gaussian', 'Inverse'], index=3)

    st.markdown("""

    This step helps in organizing and grouping similar symbols together, which enhances the quality and efficiency of the training process. Here's how clustering contributes to this process:

    ### 1. Data organization and cleaning:
    - **Purpose**: Clustering helps in organizing large sets of symbols into meaningful groups.
    - **Description**: By grouping similar symbols together, clustering helps in identifying and removing outliers or irrelevant data. This ensures that the dataset is clean and representative of the target symbols, which is essential for effective training of the neural network.

    ### 2. Enhanced training efficiency:
    - **Purpose**: Improve the efficiency of the training process by providing well-organized data.
    - **Description**: Clustering reduces the complexity of the dataset by grouping similar symbols. This allows the CNN to learn patterns and features more effectively, leading to faster convergence and improved accuracy.

    ### 3. Better feature extraction:
    - **Purpose**: Facilitate better feature extraction by the neural network.
    - **Description**: When symbols are clustered based on their visual similarities, it becomes easier for the CNN to extract relevant features. This is because the network can focus on the distinct characteristics of each cluster, enhancing its ability to recognize symbols accurately.

    ### 4. Balanced dataset creation:
    - **Purpose**: Ensure a balanced representation of different symbol classes in the dataset.
    - **Description**: Clustering helps in identifying underrepresented or overrepresented classes of symbols. This information can be used to balance the dataset, ensuring that the neural network is trained on a diverse set of symbols, which improves its generalization capabilities.

    ### 5. Improved model generalization:
    - **Purpose**: Enhance the generalization ability of the neural network.
    - **Description**: By training on clusters of similar symbols, the CNN can learn to generalize better across different variations of symbols. This is particularly important in real-world applications where symbols may vary in size, shape, or orientation.

    """)

    if st.sidebar.button('Perform clustering'):
        perform_clustering(symbols_folder, num_clusters, clustering_algorithm, binarization_method, is_pca, silcscore, elbowscore)

    if st.sidebar.button('Create dataset'):
        create_dataset(symbols_folder, num_clusters, clustering_algorithm, binarization_method, is_pca, silcscore, elbowscore)
        st.success(f'Dataset created successfully.')

if __name__ == "__main__":
    main()
