import streamlit as st
from PIL import Image
import os

def main():
    st.title("Import image with symbols")
    col1, col2 = st.columns([1, 2])

    # Ottieni la lista dei file nella directory "extracted"
    files = os.listdir("extracted")

    # Crea una mappatura dai nomi dei file ai percorsi completi
    file_paths = {os.path.basename(file).replace("extracted_", ""): os.path.join("extracted", file) for file in files}

    # Estrai solo i nomi dei file per il selectbox
    file_names = list(file_paths.keys())

    with col1:
        selected_file_name = st.selectbox("Select a file", file_names, index=3)

    # Ottieni il percorso completo del file selezionato
    selected_file_path = file_paths[selected_file_name]
    
    if selected_file_path is not None:
        file = selected_file_path
        try:
            image = Image.open(file)
            with col2:
                st.image(image, caption="Uploaded Image", width=400)
            with col1:
                # set a file name and save to disk
                
                if st.button(f"Select {selected_file_name}"):
                    # Save the image using the file name specified by the user
                    image.save(f"processing_dataset/uploaded.jpg")
                    st.success("Image selected successfully")
                    # insert image path set as default the saved path and the image name
              
        except OSError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()