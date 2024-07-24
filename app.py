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

icon = 'https://www.uniba.it/it/studenti/segreterie-studenti/amministrative/logounibacolorato.png/@@images/image.png'
bernasconilink='https://www.uniba.it/it/docenti/eleonora-bernasconi'
ferillilink='http://lacam.di.uniba.it/~ferilli/ufficiale/ferilli_ita.html'
citation_pdf = 'https://ceur-ws.org/Vol-3643/paper10.pdf'

st.set_page_config(
    page_title="Symbol Detection",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

citation = f'<a href="{bernasconilink}">Bernasconi</a>, E., & <a href="{ferillilink}">Ferilli</a>, S. (2024). A tool for empowering Symbol Detection through Technological Integration in Library Science. A case study on the Voynich manuscript. In IRCDL (pp. 94-107).'
# Centered welcome message with improved style
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write("")
with col2:
    st.markdown(f"<img src='{icon}' style='width: 80px; display: block; margin-left: auto; margin-right: auto;'><h1 style='text-align: center;'>Symbol Detection Platform</h1>", unsafe_allow_html=True)
    
    st.markdown(f"""<div style="text-align: justify">
        This application has been developed by the research team from the <a href="https://www.uniba.it/it/ricerca/dipartimenti/informatica" target='_blank'>Department of Computer Science at Aldo Moro University of Bari</a>.
        It is designed to facilitate the detection and analysis of symbols in images. 
        Use the sidebar menu to navigate through the different features and start your work.
        <p><b>Citation:</b> <span style="font-style: italic;">{citation}</span> <a href="{citation_pdf}" target='_blank'>Download</a></p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.write("")

# Create a sidebar with page selection
PAGES = {
    "Import image": start.main,
    "Image Filtering": filtering.main,
    "Area Extraction": extractarea.main,
    "Contour Detection": contour.main,
    "Clustering": clustering.main,
    "Dataset settings": dataset.main,
    "Training": training_ai.main,
    "Results": results_ai.main,
    "Analysis": predict_ai.main,
}

st.sidebar.title('Menu')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()
