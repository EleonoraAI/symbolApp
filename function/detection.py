import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
import pandas as pd 
import function.clustering as clustering
import function.detection as detection
import function.desc as desc

def binary_threshold(edges, threshold_value):
    _, binary_image = cv2.threshold(edges, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def useBinarization(edges):
    col1, col2, col3 = st.columns(3)

    # Colonna 1: Visualizza il titolo
    col1.title("Original Edges")

    # Colonna 2: Visualizza l'immagine originale
    col2.image(edges, width=200, caption="Original Edges")

    # Colonna 3: Aggiungi uno slider per regolare il valore di soglia
    threshold_value = col1.slider("Threshold Value", min_value=0, max_value=255, value=128)

    # Applica la sogliatura binaria utilizzando la funzione binary_threshold
    binary_img = binary_threshold(edges, threshold_value)

    # Visualizza l'immagine binarizzata
    col3.image(binary_img, width=200, caption="Binarized Image")

    return binary_img

def cannyContours(img, low_threshold, high_threshold, bin_on):
    # Check if the input image is grayscale
    if len(img.shape) == 2:
        edges = cv2.Canny(img, low_threshold, high_threshold)
    else:
        # Convert colored image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, low_threshold, high_threshold)
    if bin_on:
        edges = useBinarization(edges)
    # Trova i contorni nell'immagine edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def siftContours(img, bin_on):
    # Converti l'immagine in scala di grigi
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Applica la binarizzazione se richiesta
    if bin_on:
        img_gray = useBinarization(img_gray)
    
    # Crea l'oggetto SIFT
    sift = cv2.SIFT_create()
    
    # Trova i punti chiave e i descrittori
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    
    # Estrai i punti chiave come contorni
    contours = np.array([np.int32(kp.pt) for kp in keypoints]).reshape((-1, 1, 2))
    
    return contours

def sift_with_canny_contours(img, low_threshold, high_threshold, bin_on):
    # Converti l'immagine in scala di grigi
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Crea l'oggetto SIFT
    sift = cv2.SIFT_create()

    # Trova i punti chiave e i descrittori con SIFT
    keypoints, _ = sift.detectAndCompute(img_gray, None)

    # Ottieni le coordinate dei punti chiave
    keypoints_coords = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)

    contours = cannyContours(img, low_threshold, high_threshold, bin_on)
    
    # Trova i contorni corrispondenti ai punti chiave
    sift_contours = []
    for contour in contours:
        for point in keypoints_coords:
            if cv2.pointPolygonTest(contour, tuple(point[0]), False) == 1:
                sift_contours.append(contour)
                break

    return contours

def sobelContours(img, bin_on):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    edges = np.uint8(edges)
    if bin_on:
        edges = useBinarization(edges)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    return contours

def scharrContours(img, bin_on):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
    edges = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    edges = np.uint8(edges)
    if bin_on:
        edges = useBinarization(edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def laplacianContours(img, bin_on):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(img_gray, cv2.CV_64F)
    edges = np.uint8(edges)
    if bin_on:
        edges = useBinarization(edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_centroid(contour):
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def object_detection(img):
    col1, col2 = st.columns(2)
    with col1:
        st.title("Contour Detection App")
        # subtitle
        st.subheader("Contour Detection Options")
        options = ["SIFT", "Canny", "Sobel", "Scharr", "Laplacian", "SIFT & Canny"]
        selected_option = st.selectbox("Choose a contour detection method:", options)
        min_contour_area = st.slider("Min Contour Area", min_value=0, max_value=1000, value=8)
        if selected_option == "Canny" or selected_option == "SIFT & Canny":
            
            low_threshold = st.slider("Low Threshold", min_value=0, max_value=255, value=150)
            high_threshold = st.slider("High Threshold", min_value=0, max_value=255, value=250)

        # Aggiungi un checkbox per abilitare/disabilitare la binarizzazione
        bin_on = st.sidebar.checkbox("Enable Binarization")
        if bin_on:
            st.title("Binarization of Edges")
            st.write("Select a threshold value to binarize the edges.")

    if selected_option == "SIFT":
        contours = siftContours(img, bin_on)
        desc_txt = desc.description("SIFT")
    elif selected_option == "SIFT & Canny":
        contours = sift_with_canny_contours(img, low_threshold, high_threshold, bin_on)
        desc_txt = desc.description("SIFT_Canny")
    elif selected_option == "Canny":
        contours = cannyContours(img, low_threshold, high_threshold, bin_on)
        desc_txt = desc.description("Canny")
    elif selected_option == "Sobel":
        contours = sobelContours(img, bin_on)
        desc_txt = desc.description("Sobel")
    elif selected_option == "Scharr":
        contours = scharrContours(img, bin_on)
        desc_txt = desc.description("Scharr")
    elif selected_option == "Laplacian":
        contours = laplacianContours(img, bin_on)
        desc_txt = desc.description("Laplacian")
        
    # Filtra i contorni piccoli e disegna rettangoli intorno ai simboli    
    symbols_detected = img.copy()
    object_info = []  # Lista per memorizzare le informazioni sugli oggetti rilevati

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(symbols_detected, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calcola il colore medio della bbox
            bbox_color = np.mean(img[y:y+h, x:x+w], axis=(0, 1))

            # Calcola il rapporto altezza/larghezza della bbox
            aspect_ratio = h / float(w)

            # Calcola il centroide del contorno
            cx, cy = calculate_centroid(contour)
            
            image = img[y:y+h, x:x+w] 
            # Aggiungi le informazioni sugli oggetti rilevati alla lista
            object_info.append([area, *bbox_color, aspect_ratio, cx, cy, image])

    # Calcola l'area media degli oggetti rilevati
    avg_area = np.mean([info[0] for info in object_info]) if object_info else 0.0
    
    with col2:
        cv2.drawContours(symbols_detected, contours, -1, (0, 255, 0), 7)
        # Mostra l'immagine con i contorni approssimati
        st.image(symbols_detected, channels="BGR", width=500)
    with col1:
        st.write(desc_txt)

    return symbols_detected, avg_area, object_info


def main():
    imgFiltered = cv2.imread("processing_dataset/uploadedCropped.jpg")
  
    image_np = np.array(imgFiltered)
    symbols_detected, avg_area, object_info = detection.object_detection(image_np)

    # Create a DataFrame from object_info
    df_columns = ['Area', 'R', 'G', 'B', 'Aspect Ratio', 'Center X', 'Center Y', 'Image']
    df = pd.DataFrame(object_info, columns=df_columns)
    
    
    if not df.empty:
        st.write("Object Information:")
        st.dataframe(df)


    # feature_extraction_option = st.checkbox("Feature Extraction")
    clusteringOption = st.sidebar.checkbox("Clustering", value=False)

    if clusteringOption:
        # Parametri per clustering
        num_clusters = st.slider("Seleziona il numero di cluster", min_value=2, max_value=10, value=5)
        clustering.applyClustering(symbols_detected, object_info, num_clusters, image_np)
        # Display the result in two columns
        col1, col2 = st.columns(2)
        # Column 1: Symbol Detection Result
        with col1:
            st.image([image_np], caption=['Immagine originale'], width=200)

        # Column 2: Histogram of Object Areas
        with col2:
            st.image([symbols_detected], caption=['Simboli rilevati'], width=200)

        
        # Display the result in two columns
        col1, col2 = st.columns(2)

        # Column 1: Symbol Detection Result
        with col1:
            st.subheader("Symbol Detection Result")
            # st.image(symbols_detected, channels="BGR")
            st.image(symbols_detected , width=200)
            st.write(f"Average Area of Detected Objects: {avg_area:.2f}")

        # Column 2: Histogram of Object Areas
        with col2:
            plt.figure(figsize=(8, 5))
            plt.hist(object_info, bins=20, edgecolor='black')
            plt.xlabel("Object Area")
            plt.ylabel("Frequency")
            plt.title("Distribution of Object Areas")
            st.pyplot(plt)

        
# Function to apply Hough Circle Transform for object detection
def object_detection_hough_circle(img, dp, min_dist, param1, param2, min_radius, max_radius):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp, minDist=min_dist,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(img, center, radius, (0, 255, 0), 2)
    return img

# Function for HOG (Histogram of Oriented Gradients) feature extraction
def feature_extraction_hog(img, pixels_per_cell, cells_per_block):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_img, hog_vis = hog(img_gray, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)
    hog_img = (hog_img - np.min(hog_img)) / (np.max(hog_img) - np.min(hog_img))  # Normalize image data
    # Resize HOG image to match the original image dimensions
    hog_img_resized = cv2.resize(hog_img, (img_gray.shape[1], img_gray.shape[0]))
    return hog_img_resized

# Function for Local Binary Patterns (LBP) feature extraction
def feature_extraction_lbp(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp_img = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    lbp_img = lbp_img.astype(np.float32)
    lbp_img = (lbp_img - np.min(lbp_img)) / (np.max(lbp_img) - np.min(lbp_img))  # Normalize image data
    # Resize LBP image to match the original image dimensions
    lbp_img_resized = cv2.resize(lbp_img, (img.shape[1], img.shape[0]))
    return lbp_img_resized

# Function to plot the Histogram of Gradient Orientation (HOG)
def plot_hog_histogram(hog_img, num_bins=9):
    angles = np.linspace(0, np.pi, num_bins + 1)
    histogram, _ = np.histogram(hog_img, bins=num_bins, range=(0, np.pi))

    fig, ax = plt.subplots()
    ax.bar(angles[:-1], histogram, width=0.8 * np.pi / num_bins)
    ax.set_xticks(angles[:-1] + 0.4 * np.pi / num_bins)
    ax.set_xticklabels([f"{int(angle * 180 / np.pi)}°" for angle in angles[:-1]])
    ax.set_xlabel("Gradient Orientation")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
          

