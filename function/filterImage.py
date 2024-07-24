import os
import cv2
import streamlit as st

# Function to apply different image filtering methods
def apply_image_filter(img, filter_type, **kwargs):
    if filter_type == 'Canny':
        return cv2.Canny(img, kwargs.get('low_threshold', 50), kwargs.get('high_threshold', 150))
    elif filter_type == 'Gaussian':
        kernel_size = kwargs.get('kernel_size', 5)
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif filter_type == 'Bilateral':
        d = kwargs.get('diameter', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    elif filter_type == 'Median':
        kernel_size = kwargs.get('kernel_size', 5)
        return cv2.medianBlur(img, kernel_size)
    elif filter_type == 'CLAHE':
        clahe = cv2.createCLAHE(clipLimit=kwargs.get('clip_limit', 2.0), tileGridSize=(kwargs.get('tile_grid_size', 8), kwargs.get('tile_grid_size', 8)))
        return clahe.apply(img)
    elif filter_type == 'Histogram Equalization':
        return cv2.equalizeHist(img)
    elif filter_type == 'Unsharp Masking':
        return cv2.addWeighted(img, kwargs.get('alpha', 1.5), cv2.GaussianBlur(img, (0,0), kwargs.get('sigma', 10)), kwargs.get('beta', -0.5), 0)
    else:
        raise ValueError("Invalid filter type selected.")

def main(img):
    st.subheader("Image Filtering")
    col1, col2 = st.columns(2)
    with col1:
        selected_filter = st.selectbox("Choose an Image Filtering Method", ["Original","Canny", "Gaussian", "Bilateral", "Median", "CLAHE", "Histogram Equalization", "Unsharp Masking"])

    if selected_filter == "Original":
        filtered_img = img
        with col2:
            st.image(filtered_img, channels="BGR", width=400)
        with col1:
            st.markdown("**Original**: The original image without any filtering.")
    
    else:
        # Additional filters
        with col1:
            if selected_filter == "CLAHE":
                clip_limit = st.slider("Clip Limit", min_value=0.0, max_value=10.0, value=2.0)
                tile_grid_size = st.slider("Tile Grid Size", min_value=2, max_value=16, value=8)
            elif selected_filter == "Unsharp Masking":
                alpha = st.slider("Alpha", min_value=0.0, max_value=10.0, value=1.5)
                beta = st.slider("Beta", min_value=-10.0, max_value=10.0, value=-0.5)
                sigma = st.slider("Sigma", min_value=0, max_value=100, value=10)
            elif selected_filter == "Canny":
                low_threshold = st.slider("Low Threshold", min_value=0, max_value=255, value=50)
                high_threshold = st.slider("High Threshold", min_value=0, max_value=255, value=150)
            elif selected_filter == "Histogram Equalization":
                pass
            elif selected_filter == "Gaussian":
                kernel_size = st.slider("Kernel Size", min_value=1, max_value=21, step=2, value=5)  
            elif selected_filter == "Bilateral":
                diameter = st.slider("Diameter", min_value=1, max_value=21, step=2, value=9)
                sigma_color = st.slider("Sigma Color", min_value=1, max_value=150, value=75)
                sigma_space = st.slider("Sigma Space", min_value=1, max_value=150, value=75)
            elif selected_filter == "Median":
                kernel_size = st.slider("Kernel Size", min_value=1, max_value=21, step=2, value=5)

        with col2:
            # Converti l'immagine in Scala di Grigi se ha più di 3 canali
            if len(img.shape) > 2 and img.shape[2] > 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Applying selected additional filters
            if selected_filter == "CLAHE":
                if len(img.shape) > 2:  # Controlla se l'immagine è a colori
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converti in scala di grigi
                filtered_img = apply_image_filter(img, selected_filter, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
                with col1:
                    st.markdown("**CLAHE**: Contrast Limited Adaptive Histogram Equalization enhances the contrast of the image, making symbols more distinguishable. This helps the neural network to detect features more effectively in low contrast images.")
            elif selected_filter == "Histogram Equalization":
                if len(img.shape) > 2:  # Controlla se l'immagine è a colori
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converti in scala di grigi
                filtered_img = apply_image_filter(img, selected_filter)
                with col1:
                    st.markdown("**Histogram Equalization**: This method equalizes the histogram of the image to improve contrast. It helps in highlighting the features of symbols by distributing intensities evenly.")
            elif selected_filter == "Canny":
                if len(img.shape) > 2:  # Controlla se l'immagine è a colori
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converti in scala di grigi
                filtered_img = apply_image_filter(img, selected_filter, low_threshold=low_threshold, high_threshold=high_threshold)
                with col1:
                    st.markdown("**Canny**: Canny Edge Detection highlights the edges of objects in the image. This helps the neural network to recognize the shapes and outlines of symbols more accurately.")
            elif selected_filter == "Unsharp Masking":
                filtered_img = apply_image_filter(img, selected_filter, alpha=alpha, beta=beta, sigma=sigma)
                with col1:
                    st.markdown("**Unsharp Masking**: This filter sharpens the image by enhancing edges and fine details. It can make the symbols more pronounced, aiding the neural network in recognizing finer features.")
            elif selected_filter == "Gaussian":
                filtered_img = apply_image_filter(img, selected_filter, kernel_size=kernel_size)
                with col1:
                    st.markdown("**Gaussian**: Gaussian Blur reduces noise and detail in the image. This can help the neural network focus on larger, more significant features of the symbols by smoothing out irrelevant details.")
            elif selected_filter == "Bilateral":
                filtered_img = apply_image_filter(img, selected_filter, diameter=diameter, sigma_color=sigma_color, sigma_space=sigma_space)
                with col1:
                    st.markdown("**Bilateral**: Bilateral Filtering smooths the image while preserving edges. This reduces noise and makes edges clearer, which is useful for symbol recognition by maintaining important details.")
            elif selected_filter == "Median":
                filtered_img = apply_image_filter(img, selected_filter, kernel_size=kernel_size)
                with col1:
                    st.markdown("**Median**: Median Filtering removes salt-and-pepper noise while preserving edges. This helps the neural network by reducing noise and keeping important edge information intact.")

            # Visualizza l'immagine con Streamlit
            st.image(filtered_img, channels="GRAY" if len(filtered_img.shape) < 3 else "BGR", width=1500)

    with col1:
        if st.button("Select filter"):
            # Save filtered image after user has selected the filter
            cv2.imwrite("processing_dataset/uploadedFiltered.jpg", filtered_img)
            # Success saving image
            st.success("Filter applied successfully")

    return filtered_img
