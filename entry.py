"""
Nougat Streamlit App

This file contains the entry point for the Streamlit app used to convert scientific papers to markdown text.
"""

import logging
import os.path
import re
import tempfile
from functools import partial
from typing import List
from pathlib import Path
from nougat.postprocessing import markdown_compatible
from nougat import NougatModel, LazyDataset
from nougat.utils.device import move_to_device, default_batch_size
import logging
from pathlib import Path
import logging

import pypdf
import logging
import requests
import logging
import streamlit as st
import logging
import torch
import logging
from torch.utils.data import ConcatDataset
import logging
from tqdm import tqdm
import logging

from nougat import NougatModel
import logging
from nougat.postprocessing import markdown_compatible
import logging
from nougat.utils.dataset import LazyDataset
import logging
from nougat.utils.device import move_to_device, default_batch_size

BASE_URL = "https://github.com/facebookresearch/nougat/releases/download"
MODEL_TAG = "0.1.0-base"
CHECKPOINT_PATH = Path.cwd() / "checkpoints"


@st.cache_resource
def load_model():
    """Load the OCR model and set batch size.

    This function loads the OCR model from a remote location if not present locally, sets the batch size, and moves the model to the appropriate device (CPU/GPU). The function caches the model after the initial load.

    Returns:
        NougatModel: The loaded OCR model.
        int: The batch size for processing.
    """
    # Load model from a remote location if not present locally, set batch size and move to GPU
    # This will only run once and then be cached.
    # List of model files to be checked for existence or fetched
    files = [
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
        logging.info(f"Checkpoints folder created! Path - {CHECKPOINT_PATH}")
        # Check each file and fetch from remote if not present locally
    for file in files:
        if not os.path.exists(os.path.join(CHECKPOINT_PATH, file)):
            remote_path = f"{BASE_URL}/{MODEL_TAG}/{file}"
            binary_file = requests.get(f"{remote_path}").content
            logging.info(f"File fetched - {remote_path}")
            local_path = Path(CHECKPOINT_PATH) / file
            local_path.write_bytes(binary_file)
            logging.info(f"File stored - {local_path}")

    size = default_batch_size()
        # Initialize NougatModel with pre-trained weights
    m = NougatModel.from_pretrained(CHECKPOINT_PATH)
    # Move the model to the appropriate device (CPU/GPU)
    m = move_to_device(m, cuda=default_batch_size())
    m.eval()
    logging.info(f"Model Info - {m}")
    logging.info(f"Batch size set to {size}")
    return m, size


model, batch_size = load_model()


def convert(pdf_files):
    """Convert a list of PDF files to text using the OCR model.

    This function is responsible for converting a list of PDF files to text using the OCR model.

    Args:
        pdf_files (List[Path]): The list of PDF files to be converted.
    """
    """Convert a list of PDF files to text using the OCR model.

    This function is responsible for converting a list of PDF files to text using the OCR model.

    Args:
        pdf_files (List[Path]): The list of PDF files to be converted.
    """
    # This function is responsible for converting a list of PDF files to text using the OCR model.
    datasets = []
    for pdf in pdf_files:
        if not pdf.exists():
            continue
        logging.info(f"{pdf} exists!")
        try:
            # LazyDataset is used to load pdf files incrementally to save memory usage.
            dataset = LazyDataset(
                pdf,
                partial(model.encoder.prepare_input, random_padding=False),
            )
        except pypdf.errors.PdfStreamError:
            # Catch PdfStreamError if pdf file cannot be loaded due to corruption or incompatibility.
            logging.info(f"Could not load file {str(pdf)}.")
            continue
        datasets.append(dataset)

    if len(datasets) != 0:
        # Create a DataLoader to batch and efficiently load the dataset.
        dataloader = torch.utils.data.DataLoader(
            ConcatDataset(datasets),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate,
        )

        predictions = []
        file_index = 0
        page_num = 0
        # Process each batch of data from the DataLoader.
        for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):
            model_output = model.inference(
                image_tensors=sample, early_stopping=True
            )

            # check if model output is faulty
            for j, output in enumerate(model_output["predictions"]):
                if page_num == 0:
                    logging.info(
                        "Processing file %s with %i pages"
                        % (datasets[file_index].name, datasets[file_index].size)
                    )
                page_num += 1
                if output.strip() == "[MISSING_PAGE_POST]":
                    # uncaught repetitions -- most likely empty page
                    predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                elif model_output["repeats"][j] is not None:
                    if model_output["repeats"][j] > 0:
                        # If we end up here, it means the output is most likely not complete and was truncated.
                        logging.warning(f"Skipping page {page_num} due to repetitions.")
                        predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                    else:
                        # Handle markdown format when the model's output is not ideal due to training domain differences.
                        # This can happen e.g. for cover pages.
                        predictions.append(
                            f"\n\n[MISSING_PAGE_EMPTY:{i * batch_size + j + 1}]\n\n"
                        )
                else:
                    # Convert model output to a format compatible with markdown.
                    output = markdown_compatible(output)
                    predictions.append(output)
                if is_last_page[j]:
                    out = "".join(predictions).strip()
                    out = re.sub(r"\n{3,}", "\n\n", out).strip()
                    st.write(out, "\n\n")
                    predictions = []
                    page_num = 0
                    file_index += 1


def main():
    """Handle the Streamlit interface, file uploads, and the 'Process' button functionality.

    This function handles the Streamlit interface and allows users to upload PDF files for OCR processing. It also triggers the conversion of selected PDF files using the OCR model.
    """
    """Handle the Streamlit interface, file uploads, and the 'Process' button functionality.

    This function handles the Streamlit interface and allows users to upload PDF files for OCR processing. It also triggers the conversion of selected PDF files using the OCR model.
    """
    # Main function to handle the Streamlit interface, file uploads, and the 'Process' button functionality.
    st.title("OCR App using Streamlit for PDF files")

    st.write("Upload one or more PDF files for OCR")

    uploaded_files = st.file_uploader("Choose PDF files...", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        # Sidebar to provide information about the application and its usage.
        st.sidebar.title("About")
        st.sidebar.info(
            "This is an OCR application using Streamlit and pytesseract for PDF files. Upload one or more PDFs, "
            "and we'll extract the text for you!")

        # Convert UploadedFile objects to Path objects by saving them to temporary files
        # File uploader to allow users to select PDF files for OCR processing.
        pdf_names = set(file.name for file in uploaded_files)
        selected_files = st.multiselect("Select PDFs to process", pdf_names, default=pdf_names)

        # The 'Process' button triggers the conversion of selected PDF files using the OCR model.
        if st.button("Process"):
            pdf_paths = []
            for uploaded_file in uploaded_files:
                if uploaded_file.name in selected_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                        temp.write(uploaded_file.getvalue())
                        pdf_paths.append(Path(temp.name))

            with st.spinner('Processing...'):
                convert(pdf_paths)

            st.success('Processing Complete!')
            st.balloons()


if __name__ == "__main__":
    main()
