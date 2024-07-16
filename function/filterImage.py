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
            st.image(filtered_img, channels="BGR", width=500)
    
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
            elif selected_filter == "Histogram Equalization":
                if len(img.shape) > 2:  # Controlla se l'immagine è a colori
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converti in scala di grigi
                filtered_img = apply_image_filter(img, selected_filter)
            elif selected_filter == "Canny":
                if len(img.shape) > 2:  # Controlla se l'immagine è a colori
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converti in scala di grigi
                filtered_img = apply_image_filter(img, selected_filter)
            elif selected_filter == "Unsharp Masking":
                filtered_img = apply_image_filter(img, selected_filter, alpha=alpha, beta=beta, sigma=sigma)
            else:
                filtered_img = img

            # Visualizza l'immagine con Streamlit
            st.image(filtered_img, channels="GRAY" if len(filtered_img.shape) < 3 else "BGR", width=500)

    with col1:
        if st.button("Select filter"):
            # Save filtered image after user has selected the filter
            cv2.imwrite("processing_dataset/uploadedFiltered.jpg", filtered_img)
            # Success saving image
            st.success("Image saved to disk")

    return filtered_img
