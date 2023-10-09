import streamlit as st
from PIL import Image


def ocr_image(image):
    import pytesseract
    return pytesseract.image_to_string(image)


st.title("OCR App using Streamlit")

st.write("Upload one or more images for OCR")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    images = [Image.open(file) for file in uploaded_files]
    file_names = [file.name for file in uploaded_files]

    selected_files = st.multiselect("Select files to process", file_names, default=file_names)

    for index, (file_name, image) in enumerate(zip(file_names, images)):
        if file_name in selected_files:
            st.image(image, caption=f"Uploaded Image: {file_name}", use_column_width=True)

    if st.button("Process"):
        for index, (file_name, image) in enumerate(zip(file_names, images)):
            if file_name in selected_files:
                extracted_text = ocr_image(image)
                st.subheader(f"Extracted Text from {file_name}")
                st.write(extracted_text)

    st.sidebar.title("About")
    st.sidebar.info(
        "This is an OCR application using Streamlit and pytesseract. Upload one or more images, and we'll extract the text for you!")
