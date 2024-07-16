import streamlit as st
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
# Carica il modello
model = tf.keras.models.load_model('my_model.h5')

def draw_tile(image, tile_size):
    # Disegna un tassello sull'immagine
    img = Image.open(image).convert('RGB')
    draw = ImageDraw.Draw(img)

    for i in range(0, img.size[1], tile_size):
        for j in range(0, img.size[0], tile_size):
            symbol_box = (j, i, j + tile_size, i + tile_size)
            draw.rectangle(symbol_box, outline="red", width=2)

    return img


def predict(image, tile_size):
    stride = tile_size // 2
    # Preprocessa l'immagine
    img = Image.open(image).convert('L')  # Converti in scala di grigi
    # img = img.resize((224, 224))
 
    img_array = np.array(img) / 255.0

    # Effettua la previsione
    predictions = []

    # Define progress bar
    progress_bar = st.progress(0)

    total_tiles = ((img_array.shape[0] - tile_size) // stride + 1) * ((img_array.shape[1] - tile_size) // stride + 1)
    st.write(f"Image shape 0:{img_array.shape[0]}, Image shape 1:{img_array.shape[1]}")
    st.write(f"Total tiles: {total_tiles}")

    for i in range(0, img_array.shape[0] - tile_size + 1, stride):
        for j in range(0, img_array.shape[1] - tile_size + 1, stride):
            # pdb.set_trace()
            tile = img_array[i:i + tile_size, j:j + tile_size]
            # Resize the tile to match the model input shape
            tile = tf.image.resize(np.expand_dims(tile, axis=-1), (224, 224))
            tile = np.expand_dims(tile, axis=0)

            tile_predictions = model.predict(tile)
            predictions.append(tile_predictions)

            # Update progress bar
            progress = len(predictions) / total_tiles
            progress_bar.progress(progress)

    # Clear the progress bar when done
    progress_bar.empty()

    return predictions

def generate_statistics(predictions):
    # Elabora statistiche sui risultati delle previsioni
    categories = {}  # Dizionario per conteggiare le categorie

    for tile_predictions in predictions:
        for symbol_probabilities in tile_predictions:
            category = np.argmax(symbol_probabilities)  # Ottiene l'indice della categoria con la probabilità massima
            if category not in categories:
                categories[category] = 1
            else:
                categories[category] += 1

    return categories

def display_statistics(categories):
    # Mostra le statistiche
    st.write("**Statistiche delle categorie rilevate:**")
    for category, count in categories.items():
        st.write(f"Categoria {category}: {count} simboli rilevati")


def main():
    st.title("Test Handwritten Recognition")

    # carica un immagine di default
    uploaded_file = "extracted/extracted_image_3.png"

    if uploaded_file is not None:
        # Slider per regolare la dimensione del tassello
        tile_size = st.slider('Seleziona la dimensione del tassello:', min_value=10, max_value=100, value=43)

        # Disegna il tassello sull'immagine
        img_with_tile = draw_tile(uploaded_file, tile_size)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img_with_tile, caption=f'Immagine con tassello {tile_size}x{tile_size}', width=1000)

        # Esegui la previsione
        predictions = predict(uploaded_file, tile_size)

        # Disegna i risultati sull'immagine originale
        img = Image.open(uploaded_file).convert('RGB')
        draw = ImageDraw.Draw(img)

        st.write(f"\n\n**Risultati della previsione con tassello di dimensione {tile_size}x{tile_size}:**")

        for i, tile_predictions in enumerate(predictions):
            for j, prob in enumerate(tile_predictions):
                # Calcola le coordinate del rettangolo sul simbolo individuato
                symbol_box = (j * tile_size, i * tile_size, (j + 1) * tile_size, (i + 1) * tile_size)
                draw.rectangle(symbol_box, outline="red", width=2)

                # Aggiungi il testo con la percentuale di accuratezza
                text_position = ((j + 0.5) * tile_size, (i + 0.5) * tile_size)
                draw.text(text_position, f"{prob[0] * 100:.2f}%", fill="red", anchor="mm")

        # Mostra l'immagine con i risultati sovrapposti
        with col2: 
            st.image(img, caption='Risultati sovrapposti.', width=800)

    categories_count = generate_statistics(predictions)

    # Mostra l'immagine con i risultati sovrapposti
    with col2: 
        # st.image(img, caption='Risultati sovrapposti.', width=800)
        display_statistics(categories_count)  # Mostra le statistiche

if __name__ == '__main__':
    main()
