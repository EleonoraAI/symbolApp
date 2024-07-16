import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image, UnidentifiedImageError
import io
# Funzione per eseguire il rilevamento dei simboli sull'immagine
def detect_symbols(image, symbols_folder, num_clusters):
    # Dimensioni della finestra mobile
    window_size = (100, 100)  # Regola le dimensioni della finestra in base alle tue esigenze

    # Scorri l'immagine con una finestra mobile
    for y in range(0, image.shape[0] - window_size[0] + 1):
        for x in range(0, image.shape[1] - window_size[1] + 1):
            # Estrai la regione della finestra mobile
            window = image[y:y + window_size[0], x:x + window_size[1]]

            # Esegui il rilevamento del simbolo sulla finestra
            # Usa il tuo metodo di rilevamento del simbolo qui
            # Ad esempio, puoi utilizzare il template matching

            # Se hai rilevato un simbolo, puoi salvare le informazioni relative
            # Ad esempio, potresti ottenere la label del cluster associato

    # Visualizza l'immagine originale con le regioni dei simboli evidenziate
    st.image(image, channels="GRAY", use_column_width=True, caption='Original Image with Symbols Highlighted')

# Funzione per eseguire il template matching
def template_matching(uploaded_file, symbols_folder, num_clusters):
    # Converti BytesIO in un oggetto immagine
    image_bytes = uploaded_file.read()
    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    input_image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # Esegui il template matching per ogni simbolo nella cartella 'symbols'
    results = []
    for cluster in range(1, num_clusters + 1):
        symbol_folder = os.path.join(symbols_folder, f'cluster_{cluster}')
        for symbol_file in os.listdir(symbol_folder):
            symbol_path = os.path.join(symbol_folder, symbol_file)
            template = cv2.imread(symbol_path, cv2.IMREAD_GRAYSCALE)

            # Esegui il template matching
            result = cv2.matchTemplate(input_image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Salva il risultato
            results.append({
                'symbol': symbol_file,
                'cluster': cluster,
                'confidence': max_val,
                'location': max_loc
            })

    return results

# Funzione principale per l'app Streamlit
def main():
    st.title('Template Matching App')

    # Widget per la selezione della cartella dei simboli
    symbols_folder = st.sidebar.selectbox('Seleziona la cartella dei simboli', os.listdir('.'))
    num_clusters = len(os.listdir(symbols_folder))

    # Widget per il caricamento dell'immagine
    uploaded_file = st.sidebar.file_uploader("Carica un'immagine", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Esegui il template matching
        results = template_matching(uploaded_file, symbols_folder, num_clusters)

        # Visualizza i risultati
        st.subheader('Risultati del Template Matching:')
        for result in results:
            st.write(f"Simbolo: {result['symbol']}, Cluster: {result['cluster']}, Confidenza: {result['confidence']:.2f}")

            # Disegna il rettangolo intorno al simbolo nell'immagine originale
            image_with_rectangle = draw_rectangle(uploaded_file, result['location'])
            st.image(image_with_rectangle, caption=f"Simbolo: {result['symbol']}", use_column_width=True)

# Funzione per disegnare un rettangolo sull'immagine
def draw_rectangle(uploaded_file, location):
    # Converti BytesIO in un oggetto immagine
    image_bytes = uploaded_file.read()

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as e:
        st.error(f"Errore nell'apertura dell'immagine: {e}")
        return

    if pil_image is None:
        st.error("L'immagine non può essere aperta.")
        return

    image_array = np.array(pil_image)

    # Estrai le coordinate del rettangolo
    top_left = location
    bottom_right = (top_left[0] + 100, top_left[1] + 100)  # Assumendo una dimensione fissa di 100x100 per il simbolo

    # Disegna il rettangolo sull'immagine
    color = (0, 255, 0)  # Colore del rettangolo in formato BGR (verde)
    thickness = 2
    image_with_rectangle = cv2.rectangle(image_array, top_left, bottom_right, color, thickness)

    # Visualizza l'immagine con il rettangolo utilizzando Streamlit
    st.image(image_with_rectangle, channels="BGR", use_column_width=True, caption='Symbol Detection')

if __name__ == "__main__":
    main()
