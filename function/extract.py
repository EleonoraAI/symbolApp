# import PyPDF2
# from PIL import Image
# import io
# import cv2
# import streamlit as st

# def extract_images_from_pdf(pdf_file):
#     images = []
#     with open(pdf_file, "rb") as file:
#         pdf_reader = PyPDF2.PdfFileReader(file)
#         num_pages = pdf_reader.numPages
#         for page_number in range(num_pages):
#             page = pdf_reader.getPage(page_number)
#             if '/XObject' in page['/Resources']:
#                 x_objects = page['/Resources']['/XObject'].getObject()
#                 for obj in x_objects:
#                     if x_objects[obj]['/Subtype'] == '/Image':
#                         data = x_objects[obj]._data  # Get the image data
#                         image = Image.open(io.BytesIO(data))
#                         images.append(image)
#     return images
# if st.button("extract images"):
#     # Usage example
#     pdf_file_path = "data/VoynichManuscript.pdf"
#     extracted_images = extract_images_from_pdf(pdf_file_path)
#     # You can then process or save the extracted images as per your requirements
#     for i, image in enumerate(extracted_images):
#         image.save(f"extracted_image_{i+1}.png")