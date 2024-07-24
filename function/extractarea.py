import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)

def main():
    st.header("Area Extraction")
    st.markdown("""
    This phase allows for the selection of specific parts of an image that contain the symbols of interest, with two main objectives:

    #### Neural network training:

    **Purpose**: Create a training dataset (set of data for training) using specific areas with symbols.  
    **Description**: During the training of a neural network, it is essential to have a carefully labeled dataset. By manually selecting areas that contain relevant symbols, we can create a dataset that the neural network will use to learn to recognize these symbols. This process ensures that the neural network is exposed to clear and relevant examples, improving its ability to correctly identify symbols in future images.

    #### Testing and recognition:

    **Purpose**: Verify the accuracy of the classifier (trained neural network) by recognizing symbols in specific areas.  
    **Description**: Once trained, the neural network is tested on new images to evaluate its performance. By selecting areas where symbols are to be recognized, we can check if the classifier can correctly identify the symbols and assign the corresponding label. This step is essential to understand if the neural network is ready to be used in real-world applications or if it requires further improvements.
    """)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        # Set options for demo purposes
        realtime_update = st.checkbox(label="Update in Real Time", value=True)
        box_color = st.color_picker(label="Box Color", value='#0000FF')
        aspect_choice = st.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
        aspect_dict = {
            "1:1": (1, 1),
            "16:9": (16, 9),
            "4:3": (4, 3),
            "2:3": (2, 3),
            "Free": None
        }
        aspect_ratio = aspect_dict[aspect_choice]
    with col2:
        img_path = "processing_dataset/uploadedFiltered.jpg"
        img = Image.open(img_path)

        if not realtime_update:
            st.write("Double click to save crop")

        # Get a cropped image from the frontend mantain high resolution
        cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    with col3:
        # Manipulate cropped image at will
        st.write("Preview")
        _ = cropped_img.thumbnail((150, 150))
        st.image(cropped_img, width=250)

    with col1:  
        # save cropped_img
        if st.button("Cut image"):
            cropped_img.save(f"processing_dataset/uploadedCropped.jpg")
            st.success("Action completed successfully")

if __name__ == "__main__":
    main()
