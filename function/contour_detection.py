import cv2
import numpy as np
import streamlit as st
import pdb
import os
import shutil
import tensorflow as tf
from PIL import Image
import pdb

def find_symbol_contours(img_path):
    # Read the image
    img = cv2.imread(img_path)

    # Create an image for thin contours
    img_contours = img.copy()

    # Create an image for labeling with IDs
    img_with_ids = img.copy()

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, 50, 150)

    # Find contours in the Canny edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to save images of each contour
    contour_images = []

    for idx, contour in enumerate(contours):
        # Generate a random color for each contour
        color = list(np.random.random(size=3) * 256)
        color = [int(c) for c in color]

        # Draw the thin contour on the image with assigned color
        cv2.drawContours(img_contours, [contour], -1, color, 1)

        cv2.drawContours(img_with_ids, [contour], -1, color, 1)

        # Add ID to the contour on the image with IDs with the same color
        x, y, _, _ = cv2.boundingRect(contour)
        number_position = (x + 10, y + 10)
        cv2.putText(img_with_ids, str(idx + 1), number_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Crop the image with the contour
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = float(w) / h
        # Calcola il colore medio della bbox
        bbox_color = np.mean(img[y:y+h, x:x+w], axis=(0, 1))

        # Calcola il centroide del contorno
        cx, cy = calculate_centroid(contour)

        contour_image = img[y:y+h, x:x+w]
        contour_images.append((contour_image, idx + 1, area, aspect_ratio, cx, cy, *bbox_color))

    return img_contours, img_with_ids, contour_images

def calculate_centroid(contour):
    M = cv2.moments(contour)
    try:
        cx = (int(M["m10"])/int(M["m00"]))
        cy = (int(M["m01"])/int(M["m00"]))      
        return cx, cy
    except:
        return 0, 0

def main():
    st.title("Symbol Contours")

    # Replace with the path to your image
    img_path = "processing_dataset/uploadedCropped.jpg"
    
    # Find contours, images with thin contours, and the image with IDs
    img_contours, img_with_ids, contour_images = find_symbol_contours(img_path)

    img1, img2 = st.columns(2)
    with img1:
        # Display the image with thin contours
        st.image(img_contours, caption="Thin Contours", use_column_width=True, channels="BGR")
    with img2:
        # Display the image with IDs
        st.image(img_with_ids, caption="Contours with IDs", use_column_width=True, channels="BGR")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        num_points = st.slider("Number of Points", min_value=0, max_value=300, value=(10, 120))
    with col2:
        # Add a filter on the contour area
        area_size = st.slider("Area Size", min_value=0, max_value=1000, value=(38, 201))
    with col3:
        # Add a filter on the aspect ratio
        aspect_ratio_range = st.slider("Aspect Ratio", min_value=0.1, max_value=2.0, value=(0.5, 1.5))
        # aspect ratio 
       
    # with col5:
    #     # COLOR PICKER TO SET R G AND B VALUES
    #     color = st.color_picker("Exlude a color", "#00f900")
    #     if color:
    #         R_ex = int(color[1:3], 16)
    #         G_ex = int(color[3:5], 16)
    #         B_ex = int(color[5:7], 16)
    #         st.write("Color excluded:", color, "R:", R_ex, "G:", G_ex, "B:", B_ex)
        

    # Filter the IDs to hide
    selected_ids = st.multiselect("Select IDs to hide", [idx + 1 for idx in range(len(contour_images))])

    # STREAMLIT LIST RESULTS
    st.subheader("Extraction Results")

    co1, co2, col3, col4, col5 = st.columns(5)
    with co1:
        # Add a button to save the filtered results
        if st.button("Save Filtered Results"):
            save_filtered_results(contour_images, "symbols", num_points, area_size, aspect_ratio_range, selected_ids)

    with co2:
        # Add a button to delete the "symbols" folder
        if st.button("Delete Symbols"):
            delete_symbols_folder()
            st.success("Symbols folder deleted.")    
           
    # Create five columns
    col1, col2, col3, col4, col5,col6, col7, col8, col9, col10 = st.columns(10)

    # Display information in each column
    with col1:
        st.write("ID")
    with col2:
        st.write("Points")
    with col3:  
        st.write("Area")    
    with col4:
        st.write("Image")
    with col5:
        st.write("Aspect Ratio")
    with col6:
        st.write("Cx")
    with col7:
        st.write("Cy")
    with col8:
        st.write("R")
    with col9:
        st.write("G")
    with col10:
        st.write("B")

    # Ordina le rilevazioni per riga e poi per colonna
    sorted_contours = sorted(contour_images, key=lambda x: (x[5], x[4]))

    # Raggruppa le rilevazioni per riga
    rows = {}
    for contour_info in sorted_contours:
        cy = contour_info[5]
        if cy not in rows:
            rows[cy] = []
        rows[cy].append(contour_info)

    # Itera attraverso le righe e le rilevazioni all'interno di ciascuna riga
    for cy, row_contours in sorted(rows.items()):
        # Ordina le rilevazioni all'interno di ciascuna riga da sinistra verso destra
        sorted_row_contours = sorted(row_contours, key=lambda x: x[4])

        # Itera attraverso le rilevazioni ordinate
        for idx, (contour, contour_id, area, aspect_ratio, cx, cy, R, G, B) in enumerate(sorted_row_contours):
            # Applica filtri
            if num_points[0] <= len(contour) <= num_points[1] and \
                    (contour_id not in selected_ids) and \
                    area_size[0] <= area <= area_size[1] and \
                    aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                # Display information in each column
                with col1:
                    st.write(contour_id)
                with col2:
                    st.write(len(contour))
                with col3:
                    st.write(area)
                with col4:
                    st.image(contour, width=25, channels="BGR")
                with col5:
                    st.write(round(aspect_ratio, 1))
                with col6:
                    st.write(round(cx, 1))
                with col7:
                    st.write(round(cy, 1))
                with col8:
                    st.write(round(R, 1))
                with col9:
                    st.write(round(G, 1))
                with col10:
                    st.write(round(B, 1))

def save_filtered_results(sorted_row_contours, output_folder, num_points, area_size, aspect_ratio_range, selected_ids):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over contour images and save only the filtered ones in the output folder
    for idx, (contour, contour_id, area, aspect_ratio,cx,cy,R,G,B) in enumerate(sorted_row_contours):
        # Apply filters
        if num_points[0] <= len(contour) <= num_points[1] and \
            (contour_id not in selected_ids) and \
            area_size[0] <= area <= area_size[1] and \
            aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            image_filename = f"{idx}_Contour_{contour_id}_Aspect_{aspect_ratio:.2f}_Area_{area:.2f}.png"
            image_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(image_path, cv2.cvtColor(contour, cv2.COLOR_BGR2RGB))


def delete_symbols_folder():
    # Delete the "symbols" folder and its contents
    shutil.rmtree("symbols", ignore_errors=True)
                
if __name__ == "__main__":
    main()
