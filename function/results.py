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
    # Show history results
    st.subheader("Results")
    # Load saved history
    history = np.load('history.npy', allow_pickle=True).item()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Accuracy")
        st.write(history['accuracy'][-1])
    with col2:
        st.subheader("Loss")
        st.write(history['loss'][-1])
    with col3:
        # precisione, recall e F1-score
        # Plot della loss e dell'accuracy
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(history['accuracy'], label='train accuracy')
        ax[0].plot(history['val_accuracy'], label='test accuracy')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_title('Accuracy over epochs')
        ax[0].legend()

        ax[1].plot(history['loss'], label='train loss')
        ax[1].plot(history['val_loss'], label='test loss')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].set_title('Loss over epochs')
        ax[1].legend()

    st.pyplot(fig)


    # Load the model
    model = tf.keras.models.load_model('my_model.h5')

    # Load test data
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    st.write('X_test shape:', X_test.shape)

    col1, col2, col3, col4 = st.columns(4)
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    with col1:
        st.write(f'Test loss: {test_loss:.4f}')
        st.write(f'Test accuracy: {test_acc:.4f}')

    # Make predictions
    predictions = model.predict(X_test)

    # Display some predictions and true labels
    with col2:
        st.write('Sample predictions:')
        for i in range(5):  # Display the predictions for the first 5 samples with plot
            st.write(f"Sample {i + 1}: Predicted class: {np.argmax(predictions[i])}, True class: {y_test[i]}")

    # Plot a sample image with its predicted class
    plt.figure()
    plt.imshow(X_test[0].reshape(224, 224), cmap='gray')  # Assuming it's a grayscale image
    plt.title(f'Predicted class: {np.argmax(predictions[0])}, True class: {y_test[0]}')
    plt.axis('off')
    with col3:
        # Create a single plot with subplots for each image
        fig, axs = plt.subplots(2, 5, figsize=(15, 5))
        for i in range(10):
            ax = axs[i // 5, i % 5]
            ax.imshow(X_test[i].reshape(224, 224), cmap='gray')
            ax.set_title(f'Predicted class: {np.argmax(predictions[i])}\nTrue class: {y_test[i]}')
            ax.axis('off')
        st.pyplot(fig)

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, np.argmax(predictions, axis=1))
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    with col4:
        st.pyplot(plt)

if __name__ == "__main__":
    main()
 