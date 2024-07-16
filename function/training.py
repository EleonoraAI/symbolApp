import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, UpSampling2D, concatenate

from sklearn.metrics import confusion_matrix
import seaborn as sns

def main():
    st.title("Symbol Recognition Application")

    # Imposta il percorso della cartella 'symbols'
    default_symbols_folder = 'dataset'
    
    # Sezione per impostare il percorso della cartella del dataset
    st.sidebar.header("Dataset Settings")
    symbols_folder = st.sidebar.text_input("Enter dataset path", default_symbols_folder)
    st.sidebar.text("Default Symbols:dataset etc.")  # Aggiungi eventuali altri simboli di default
    
    # Sezione per selezionare le opzioni di divisione del dataset
    st.sidebar.header("Dataset Splitting Options")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, step=0.01)
    random_state = st.sidebar.slider("Random State", 0, 100, 42, step=1)
    
    st.subheader("Dataset")
    images, labels_encoded = load_images(symbols_folder)
    X_train, X_test, y_train, y_test = split_dataset(images, labels_encoded, test_size, random_state)
    
    if st.checkbox("Show Dataset Info"):
        col1, col2 = st.columns(2)
        with col1:
            show_dataset_info(images, labels_encoded)
        with col2:
            plot_class_distribution(labels_encoded)

        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Set")
            show_dataset_info(X_train, y_train)

        with col2:
            st.subheader("Test Set")
            show_dataset_info(X_test, y_test)
    
    if st.checkbox("Show model"):
        try:
            # Utilizzo della funzione con parametri personalizzati
            my_model = createModel(labels_encoded)
            my_model.summary(print_fn=lambda x: st.text(x))
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    if st.checkbox("Train model"):
        my_model = createModel(labels_encoded)
        history = trainModel(my_model, X_train, X_test, y_train, y_test)
    


def load_images(symbols_folder):
    try:
        images = []
        labels = []
        for class_name in os.listdir(symbols_folder):
            class_folder = os.path.join(symbols_folder, class_name)
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(class_name)

        # Conversione in array numpy
        images = np.array(images)
        labels = np.array(labels)

        # Codifica delle etichette
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)

        st.write("Shape of images: ", images.shape)

        # Default MaxPoolingOp only supports NHWC on device type CPU
        if images.shape[-1] == 3:
            images = images.reshape(images.shape[0], 224, 224, 3)
        else:
            images = images.reshape(images.shape[0], 224, 224, 1)
        
        st.write("Images reshape: ", images.shape)
    except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    return images, labels_encoded

def split_dataset(images, labels_encoded, test_size, random_state):
    # Split dei dati in training e test set
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def show_dataset_info(images, labels):
    col1, col2 = st.columns(2)
    with col1:
        st.write('Images shape:', images.shape)
    with col2:
        if images.shape[-1] == 3:
            st.write('Color images')
        else:
            st.write('Grayscale images')
    st.write('Labels shape:', labels.shape)
    st.write('Number of classes:', len(np.unique(labels)))
    st.write('Avg of images per class:', len(labels) / len(np.unique(labels)))   

def plot_class_distribution(labels):
    # Plot della distribuzione delle immagini nelle classi
    class_counts = np.unique(labels, return_counts=True)
    classes, counts = class_counts[0], class_counts[1]

    fig, ax = plt.subplots(figsize=(10, 5))
    # replace classes names
    # classes = np.array([re.sub('cluster_', '', class_name) for class_name in classes])
    ax.bar(classes, counts)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Images')
    ax.set_title('Class Distribution after dataset editing')

    st.pyplot(fig)

def createModel(labels):
    conv_filters=16
    dense_units=512
    dropout_rate=0.5
    num_classes = len(np.unique(labels))

    model = Sequential([
        Conv2D(conv_filters, (3, 3), activation='relu', input_shape=(224, 224, 1), data_format='channels_last'),
        MaxPooling2D(2, 2, data_format='channels_last'),
        Conv2D(2 * conv_filters, (3, 3), activation='relu', data_format='channels_last'),
        MaxPooling2D(2, 2, data_format='channels_last'),
        Conv2D(4 * conv_filters, (3, 3), activation='relu', data_format='channels_last'),
        MaxPooling2D(2, 2, data_format='channels_last'),
        Flatten(),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    return model

def trainModel(model, X_train, X_test, y_train, y_test):
    
    try:
        # Create an empty placeholder for real-time updates
        status_placeholder = st.empty()
        status_placeholder.text("Compiling the model...")

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Update the status
        status_placeholder.text("Model compiled successfully. Training in progress...")

        history = model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))

        # save model
        model.save('my_model.h5')
        # save X_test, y_test
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)

        status_placeholder.text("Model exported successfully.")
        # save history
        np.save('history.npy', history.history)
        status_placeholder.text("History exported successfully.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
