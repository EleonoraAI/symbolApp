import os
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from itertools import groupby

def find_recurrent_sequences(sequence, sequence_length):
    subsequences = [tuple(sequence[i:i+sequence_length]) for i in range(len(sequence) - sequence_length + 1)]
    recurring_sequences = Counter(subsequences)
    min_frequency = 2  # Frequenza minima desiderata
    recurring_sequences = {seq: freq for seq, freq in recurring_sequences.items() if freq >= min_frequency}
    return recurring_sequences

def display_predicted_images(folder_path, predicted_classes):
    img_pred = []
    predicted_sequence = []

    for image_name, predicted_class in zip(os.listdir(folder_path), predicted_classes):
        image_path = os.path.join(folder_path, image_name)
        image = Image.open(image_path)
        img_pred.append([image, predicted_class[0]])
        predicted_sequence.append(predicted_class[0])

    class_labels = [str(label) for label in predicted_sequence]
    unique_labels, label_counts = np.unique(class_labels, return_counts=True)
    fig, ax = plt.subplots()
    ax.bar(unique_labels, label_counts)
    ax.set_xlabel('Predicted Classes')
    ax.set_ylabel('Count')
    ax.set_title('Count of Predicted Classes')

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)

    seq = []
    with col2:
        st.write("Sequence of Predictions:")
        for car in range(2, 10):            
            recurring_sequences = find_recurrent_sequences(predicted_sequence, car)
            for sequence, frequency in recurring_sequences.items():
                # st.write(f"Sequence: {sequence}, Frequency: {frequency}")
                seq.append([sequence, frequency])

    return seq, img_pred
    

def predict_symbol_class(image_folder, model):
    model_path = model
    model = load_model(model_path)
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    predictions = []
    
    for img_path in image_files:      
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = img.convert('L')
        img_array = np.array(img)
        img_array = img_array.reshape((224, 224, 1))
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predictions.append([np.argmax(prediction), img_array])
        
    return predictions

def showPredictedSequence(predicted_sequence, img_pred):
    for i in range(len(img_pred)):
        # img_pred[i][1] = str(img_pred[i][1])

        for entry in predicted_sequence:
            sequenzaPredizioni = entry[0]
            frequenzaPredizioni = entry[1]

            lenSeq = len(sequenzaPredizioni)

            # for i in range(lenSeq):
            #     # st.write(sequenzaPredizioni[i])
            #     if sequenzaPredizioni[i] == img_pred[i][1]:
                    # st.image(img_pred[i][0], width=25)
            
                    # st.image(img_pred[i][0])

            # for img in img_pred:
            #     singolaPredizione = img[1] # 3 numero intero che rappresenta la classe predetta
            #     immaginePillow = img[0] # imagePillow del simbolo riconosciuto

# esempio predicted_sequence = [("3,5,6", 2), ("2,3", 1)]
# esempio img_pred = [(image1, image2, image3, image4, image5, image6, image7), (3, 5, 6, 1, 2, 3, 3)]
# voglio ottenere =[(["3,5,6",2][image1,image2,image3][3,5,6]]),(["2,3",1][image5, image6])]
            
                
        # st.write(sequenzaPredizioni, frequenzaPredizioni)


#     # img_pred ha (pilImage e predicted_class) # img_pred_example = ([<PIL.PngImagePlugin.PngImageFile image mode=RGB size=9x11 at 0x7FD6B9E99EF0>, 1],[<PIL.PngImagePlugin.PngImageFile image mode=RGB size=9x11 at 0x7FD6B9E99EF0>, 2],[<PIL.PngImagePlugin.PngImageFile image mode=RGB size=9x11 at 0x7FD6B9E99EF0>, 3])

#     #devo trovare tutte le successione dei singoli valori di predicted_class che siano identiche a ciascuna Sequence elencata in dataframe. Per esempio se Seq = 1,2,3, allora devo registrare i casi in cui predicted_class[0] = 1, predicted_class[1] = 2 e predicted_class[2] = 3

#     #quando trovo la corrispondenza allora mostro le immagini PILLOW in successione insieme alla classe predetta e la frequenza: quindi per esempio [predicted_class[0] - img[0], predicted_class[1] - img[1], predicted_class[2] - img[2]] - frequency
#     return None
            
def main():
    st.title("Symbol Classifier")
    st.sidebar.header("Upload Model and Image Folder")
    
    uploaded_model = "./my_model.h5"
    # Ottieni la lista delle cartelle nella directory corrente
    folders = os.listdir("./")

    # Se "symbols" è presente nella lista, imposta quella come opzione predefinita
    if "symbols" in folders:
        uploaded_folder = st.sidebar.selectbox("Select Image Folder", folders, index=folders.index("symbols"))
    else:
        # Altrimenti, mostra la lista senza opzione predefinita
        uploaded_folder = st.sidebar.selectbox("Select Image Folder", folders)

    predicted_classes = predict_symbol_class(uploaded_folder, uploaded_model) 
    seq, img_pred = display_predicted_images(uploaded_folder, predicted_classes)

    dataframe = pd.DataFrame(seq, columns=['Sequence', 'Frequency'])
    dataframe = dataframe.sort_values(by='Frequency', ascending=False)
    st.dataframe(dataframe)    
 
    # showPredictedSequence(seq, img_pred)
    

if __name__ == "__main__":
    main()
