import streamlit as st
import function.clustering_detection as clustering
import function.start as start
import function.filtering as filtering
import function.contour_detection as contour
import function.extractarea as extractarea
import function.training as training_ai
import function.results as results_ai
import function.predict_ai as predict_ai
import function.dataset as dataset

st.set_page_config(
        page_title="Symbol Detection",
        page_icon="💻️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# crea un menu laterale con la selezione delle pagine
PAGES = {
    "Import image": start.main,
    "Image Filtering": filtering.main,
    "Area Extraction": extractarea.main,
    "Contour Detection": contour.main,
    "Clustering": clustering.main,
    "Dataset settings": dataset.main,
    "Training": training_ai.main,
    "Results": results_ai.main,
    "Test": predict_ai.main,
}

st.sidebar.title('Menu')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()

