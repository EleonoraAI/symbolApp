import streamlit as st
from PIL import Image

def main():
    st.title("Import image with symbols")

    file = st.file_uploader("Upload file", type=["jpg", "png"])
    if file is not None:
        try:
            image = Image.open(file)
            st.image(image, caption="Uploaded Image", width=200)
        
            # set a file name and save to disk
            if st.button("Save Image"):
                # Save the image using the file name specified by the user
                image.save(f"processing_dataset/uploaded.jpg")
                st.success("Image saved to disk")
                # insert image path set as default the saved path and the image name
                img_path = f"processing_dataset/uploaded.jpg"
                st.write(img_path)
              
        except OSError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()