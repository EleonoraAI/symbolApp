import cv2
import numpy as np
import streamlit as st
import os
import shutil
import base64
from PIL import Image
from io import BytesIO
import pandas as pd
from collections import Counter

def find_symbol_contours(img_path):
    img = cv2.imread(img_path)
    if img is None:
        st.error("Error loading image. Please check the path and try again.")
        return None, None, [], []

    img_contours = img.copy()
    img_with_ids = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_images = []
    colors_in_symbols = []

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

        # Collect all colors in the bounding box
        for i in range(y, y + h):
            for j in range(x, x + w):
                colors_in_symbols.append(tuple(img[i, j]))

    return img_contours, img_with_ids, contour_images, colors_in_symbols

def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return cx, cy

def color_in_range(color, excluded_colors):
    # Check if the color is within ±5 range of any excluded color
    for excluded_color in excluded_colors:
        if all(abs(color[i] - excluded_color[i]) <= 5 for i in range(3)):
            return True
    return False

def symbol_contains_excluded_color(symbol_image, excluded_colors):
    # Check if the symbol image contains any excluded color
    for row in symbol_image:
        for pixel in row:
            if color_in_range(tuple(pixel), excluded_colors):
                return True
    return False

def save_filtered_results(contour_images, output_folder, num_points, area_size, aspect_ratio_range, excluded_colors, selected_ids):
    os.makedirs(output_folder, exist_ok=True)
    for contour_image, contour_id, area, aspect_ratio, cx, cy, R, G, B in contour_images:
        if num_points[0] <= len(contour_image) <= num_points[1] and \
           (contour_id not in selected_ids) and \
           area_size[0] <= area <= area_size[1] and \
           aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and \
           not symbol_contains_excluded_color(contour_image, excluded_colors):
            image_filename = f"{contour_id}_Contour_{contour_id}_Aspect_{aspect_ratio:.2f}_Area_{area:.2f}.png"
            image_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(image_path, cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))

def delete_symbols_folder():
    shutil.rmtree("symbols", ignore_errors=True)

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def group_colors_by_range(colors, range_val=25):
    grouped_colors = {}
    for color in colors:
        found = False
        for key in grouped_colors.keys():
            if all(abs(color[i] - key[i]) <= range_val for i in range(3)):
                grouped_colors[key] += 1
                found = True
                break
        if not found:
            grouped_colors[color] = 1
    return grouped_colors

def main():
    st.title("Symbol Contours")

    st.markdown("""
    In this step, a contour detection algorithm identifies symbols on the manuscript image and extracts them into a table. This phase is essential both for extracting the dataset of symbols to train the neural network, and for the recognition of symbols, i.e., to associate a class with the detected symbols on the manuscript.
    """)

    img_path = "processing_dataset/uploadedCropped.jpg"
    if not os.path.exists(img_path):
        st.error("Image path does not exist.")
        return

    img_contours, img_with_ids, contour_images, colors_in_symbols = find_symbol_contours(img_path)
    if img_contours is None:
        return

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_contours, caption="Thin Contours", use_column_width=True, channels="BGR")
    with col2:
        st.image(img_with_ids, caption="Contours with IDs", use_column_width=True, channels="BGR")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_points = st.slider("Number of points", min_value=0, max_value=300, value=(10, 120))
    with col2:
        area_size = st.slider("Area size", min_value=0, max_value=1000, value=(38, 201))
    with col3:
        aspect_ratio_range = st.slider("Aspect ratio (W/H)", min_value=0.1, max_value=2.0, value=(0.5, 1.5))

    excluded_colors = st.session_state.get('excluded_colors', [])

    def color_to_rgb(color):
        color = color.lstrip('#')
        return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

    with col1:
        add_color = st.color_picker("Pick a color to exclude")
        if st.button("Add Color to Exclusion List"):
            rgb_color = color_to_rgb(add_color)
            if rgb_color not in excluded_colors:
                excluded_colors.append(rgb_color)
                st.session_state.excluded_colors = excluded_colors
                st.success(f"Color {add_color} added to exclusion list")

    with col2:
        if excluded_colors:
            st.write("Excluded Colors:")
            for color in excluded_colors:
                col_color, col_button = st.columns([3, 1])
                with col_color:
                    st.markdown(f'<span style="color: rgb({color[0]}, {color[1]}, {color[2]});">&#9608;</span> {color}', unsafe_allow_html=True)
                with col_button:
                    if st.button(f"X", key=f"remove-{color}"):
                        excluded_colors.remove(color)
                        st.session_state.excluded_colors = excluded_colors
                        st.success(f"Color {color} removed from exclusion list")

    with col3:
        selected_ids = st.multiselect("Select IDs to hide", [idx + 1 for idx in range(len(contour_images))])

    st.subheader("Extraction Results")

    col1, col2 = st.columns([2, 2])

    with col1:
        data = []
        for contour, contour_id, area, aspect_ratio, cx, cy, R, G, B in contour_images:
            if num_points[0] <= len(contour) <= num_points[1] and \
               (contour_id not in selected_ids) and \
               area_size[0] <= area <= area_size[1] and \
               aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and \
               not symbol_contains_excluded_color(contour, excluded_colors):
                img_base64 = image_to_base64(contour)
                img_html = f'<img src="data:image/png;base64,{img_base64}" width="40">'
                data.append({
                    "ID": contour_id,
                    "Points": len(contour),
                    "Area": area,
                    "Image": img_html,
                    "W/H": round(aspect_ratio, 1),
                    "Cx": round(cx, 1),
                    "Cy": round(cy, 1)
                })

        df = pd.DataFrame(data)
        st.write(df.to_html(escape=False), unsafe_allow_html=True)

    with col2:
        # Verify if symbols folder exists
        if os.path.exists("symbols") and os.path.isdir("symbols"):
            if st.button("Delete already saved symbols"):
                delete_symbols_folder()
                st.success("Symbols folder deleted.")
            st.warning("The symbols folder already exists. If you want to save the new symbols, please delete the existing folder.")           
        else:
            st.success("No symbols folder found. You can save the filtered results.")
            if st.button("Save Filtered Results"):
                save_filtered_results(contour_images, "symbols", num_points, area_size, aspect_ratio_range, excluded_colors, selected_ids)
                st.success("Filtered results saved successfully.")

        st.markdown("### Data information")
        st.write("Number of filtered symbols:", len(df))
        st.write("Total symbols:", len(contour_images))

        # Display all colors found in the symbols
        st.write("Colors found in symbols")
        grouped_colors = group_colors_by_range(colors_in_symbols, range_val=25)
        total_colors = sum(grouped_colors.values())
        top_10_colors = sorted(grouped_colors.items(), key=lambda x: x[1], reverse=True)

        st.markdown("### Colors Found in Symbols")
        with st.expander("Show colors found in symbols"):
            for color, count in top_10_colors:
                percentage = (count / total_colors) * 100
                st.markdown(f'<span style="color: rgb({color[0]}, {color[1]}, {color[2]});">&#9608;</span> {color}: {count} ({percentage:.2f}%)', unsafe_allow_html=True)

        st.markdown("""
        ### Table columns explained:

        - **ID**: The unique identifier for each detected contour.
        - **Points**: The number of points that make up the contour.
        - **Area**: The area of the contour in pixels.
        - **Image**: A thumbnail image of the detected contour.
        - **W/H**: The width-to-height ratio (aspect ratio) of the contour.
        - **Cx**: The x-coordinate of the centroid of the contour.
        - **Cy**: The y-coordinate of the centroid of the contour.
        """)

if __name__ == "__main__":
    main()
