import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)

def main():
    st.header("Cropper")

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

    img_path = "processing_dataset/uploadedFiltered.jpg"
    img = Image.open(img_path)

    if not realtime_update:
        st.write("Double click to save crop")

    # Get a cropped image from the frontend mantain high resolution
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                             aspect_ratio=aspect_ratio)

    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img.thumbnail((1200, 1200))
    st.image(cropped_img)

    # save cropped_img
    if st.button("Save Image"):
        cropped_img.save(f"processing_dataset/uploadedCropped.jpg")
        st.success("Image saved to disk")

if __name__ == "__main__":
    main()
