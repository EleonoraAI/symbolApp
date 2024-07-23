import cv2
import numpy as np
import streamlit as st
import os
import shutil
import base64
from PIL import Image
from io import BytesIO
import pandas as pd

def find_symbol_contours(img_path):
    img = cv2.imread(img_path)
    if img is None:
        st.error("Error loading image. Please check the path and try again.")
        return None, None, []

    img_contours = img.copy()
    img_with_ids = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_images = []

    for idx, contour in enumerate(contours):
        color = np.random.randint(0, 256, size=3).tolist()
        cv2.drawContours(img_contours, [contour], -1, color, 1)
        cv2.drawContours(img_with_ids, [contour], -1, color, 1)
        x, y, _, _ = cv2.boundingRect(contour)
        number_position = (x + 10, y + 10)
        cv2.putText(img_with_ids, str(idx + 1), number_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = float(w) / h
        bbox_color = np.mean(img[y:y+h, x:x+w], axis=(0, 1)).astype(int)
        cx, cy = calculate_centroid(contour)
        contour_image = img[y:y+h, x:x+w]
        contour_images.append((contour_image, idx + 1, area, aspect_ratio, cx, cy, *bbox_color))

    return img_contours, img_with_ids, contour_images

def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return cx, cy

def save_filtered_results(contour_images, output_folder, num_points, area_size, aspect_ratio_range, selected_ids):
    os.makedirs(output_folder, exist_ok=True)
    for contour_image, contour_id, area, aspect_ratio, cx, cy, R, G, B in contour_images:
        if num_points[0] <= len(contour_image) <= num_points[1] and \
           (contour_id not in selected_ids) and \
           area_size[0] <= area <= area_size[1] and \
           aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            image_filename = f"{contour_id}_Contour_{contour_id}_Aspect_{aspect_ratio:.2f}_Area_{area:.2f}.png"
            image_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(image_path, cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))

def delete_symbols_folder():
    shutil.rmtree("symbols", ignore_errors=True)

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def main():
    st.title("Symbol Contours")

    img_path = "processing_dataset/uploadedCropped.jpg"
    if not os.path.exists(img_path):
        st.error("Image path does not exist.")
        return

    img_contours, img_with_ids, contour_images = find_symbol_contours(img_path)
    if img_contours is None:
        return

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_contours, caption="Thin Contours", use_column_width=True, channels="BGR")
    with col2:
        st.image(img_with_ids, caption="Contours with IDs", use_column_width=True, channels="BGR")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_points = st.slider("Number of Points", min_value=0, max_value=300, value=(10, 120))
    with col2:
        area_size = st.slider("Area Size", min_value=0, max_value=1000, value=(38, 201))
    with col3:
        aspect_ratio_range = st.slider("Aspect Ratio", min_value=0.1, max_value=2.0, value=(0.5, 1.5))

        

    with col1:
        selected_ids = st.multiselect("Select IDs to hide", [idx + 1 for idx in range(len(contour_images))])
        
    
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Filtered Results"):
                save_filtered_results(contour_images, "symbols", num_points, area_size, aspect_ratio_range, selected_ids)
                st.success("Filtered results saved successfully.")
        with col2:
            if st.button("Delete already saved symbols"):
                delete_symbols_folder()
                st.success("Symbols folder deleted.")

    st.subheader("Extraction Results")

    data = []
    for contour, contour_id, area, aspect_ratio, cx, cy, R, G, B in contour_images:
        if num_points[0] <= len(contour) <= num_points[1] and \
           (contour_id not in selected_ids) and \
           area_size[0] <= area <= area_size[1] and \
           aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            img_base64 = image_to_base64(contour)
            img_html = f'<img src="data:image/png;base64,{img_base64}" width="50">'
            data.append({
                "ID": contour_id,
                "Points": len(contour),
                "Area": area,
                "Image": img_html,
                "Aspect Ratio": round(aspect_ratio, 1),
                "Cx": round(cx, 1),
                "Cy": round(cy, 1),
                "R": round(R, 1),
                "G": round(G, 1),
                "B": round(B, 1)
            })

    df = pd.DataFrame(data)
    st.write(df.to_html(escape=False), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
