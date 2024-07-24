import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_history(file_path):
    try:
        history = np.load(file_path, allow_pickle=True).item()
        return history
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return None

def display_history(history):
    st.subheader("Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Accuracy")
        st.write(history['accuracy'][-1])
    with col2:
        st.subheader("Loss")
        st.write(history['loss'][-1])
    with col3:
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

def load_model_and_data(model_path, x_test_path, y_test_path):
    try:
        model = load_model(model_path)
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        return model, X_test, y_test
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None, None

def display_evaluation_results(model, X_test, y_test):
    col1, col2, col3, col4 = st.columns(4)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    with col1:
        st.write(f'Test loss: {test_loss:.4f}')
        st.write(f'Test accuracy: {test_acc:.4f}')

    predictions = model.predict(X_test)
    with col2:
        st.write('Sample predictions:')
        for i in range(5):
            st.write(f"Sample {i + 1}: Predicted class: {np.argmax(predictions[i])}, True class: {y_test[i]}")

    with col3:
        fig, axs = plt.subplots(2, 5, figsize=(15, 5))
        for i in range(10):
            ax = axs[i // 5, i % 5]
            ax.imshow(X_test[i].reshape(224, 224), cmap='gray')
            ax.set_title(f'Predicted: {np.argmax(predictions[i])}\nTrue: {y_test[i]}')
            ax.axis('off')
        st.pyplot(fig)

    with col4:
        cm = confusion_matrix(y_test, np.argmax(predictions, axis=1))
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        st.pyplot(plt)

    

def main():
    st.title("Model Evaluation Dashboard")

    st.markdown("""

    ### Model Performance Metrics
    - **Accuracy**: The proportion of true results (both true positives and true negatives) among the total number of cases examined. 
    - **Loss**: A measure of how well (or poorly) the model is performing. It quantifies the difference between the predicted values and the actual values.

    ### Visualization of Results
    - The plots show the accuracy and loss over epochs for both training and validation data, which helps in understanding the model's learning process over time.

    ### Confusion Matrix
    - A **confusion matrix** is a table used to describe the performance of a classification model on a set of test data for which the true values are known. It helps in understanding the types of errors the model is making.
    
    - **True Positives (TP)**: The model correctly predicted the positive class.
    - **True Negatives (TN)**: The model correctly predicted the negative class.
    - **False Positives (FP)**: The model incorrectly predicted the positive class (Type I error).
    - **False Negatives (FN)**: The model incorrectly predicted the negative class (Type II error).
    
    The confusion matrix gives a detailed breakdown of how well your model is performing on each class, which is crucial for understanding the strengths and weaknesses of the model.
    """)

    history = load_history('history.npy')
    if history:
        display_history(history)

    model, X_test, y_test = load_model_and_data('my_model.h5', 'X_test.npy', 'y_test.npy')
    if model and X_test is not None and y_test is not None:
        st.write('X_test shape:', X_test.shape)
        display_evaluation_results(model, X_test, y_test)

if __name__ == "__main__":
    main()
