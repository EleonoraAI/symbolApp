import streamlit as st
from PIL import Image
import os

def main():
    st.title("Import image with symbols")
    col1, col2 = st.columns([1, 2])
    files = os.listdir("extracted")
    for i in range(len(files)):
        files[i] = "extracted/" + files[i]

    with col1:
        file = st.selectbox("Select an image file", files)


    if file is not None:
        try:
            image = Image.open(file)
            with col2:
                st.image(image, caption="Uploaded Image", width=400)
            with col1:
                # set a file name and save to disk
                if st.button(f"Select {file.replace('extracted/', '')}"):
                    # Save the image using the file name specified by the user
                    image.save(f"processing_dataset/uploaded.jpg")
                    st.success("Image selected successfully")
                    # insert image path set as default the saved path and the image name
              
        except OSError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()