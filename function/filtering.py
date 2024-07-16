
import streamlit as st
import cv2

import function.filterImage as filterImage

def main():
    img = cv2.imread("processing_dataset/uploaded.jpg")
    filterImage.main(img)
