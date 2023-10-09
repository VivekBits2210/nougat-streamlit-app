import logging
import os.path
import re
import tempfile
from functools import partial
from pathlib import Path

import pypdf
import requests
import streamlit as st
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from nougat import NougatModel
from nougat.postprocessing import markdown_compatible
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device, default_batch_size

BASE_URL = "https://github.com/facebookresearch/nougat/releases/download"
MODEL_TAG = "0.1.0-base"
CHECKPOINT_PATH = "checkpoints"


@st.cache_resource
def load_resource():
    # This will only run once and then be cached.
    files = [
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    for file in files:
        if not os.path.exists(os.path.join(CHECKPOINT_PATH, file)):
            binary_file = requests.get(f"{BASE_URL}/{MODEL_TAG}/{file}").content
            (Path(CHECKPOINT_PATH) / file).write_bytes(binary_file)

    size = default_batch_size()
    m = NougatModel.from_pretrained(CHECKPOINT_PATH)
    m = move_to_device(m, cuda=default_batch_size())
    m.eval()

    return m, size


model, batch_size = load_resource()


def convert(pdf_files):
    datasets = []
    for pdf in pdf_files:
        if not pdf.exists():
            continue
        st.write(f"{pdf} exists!")
        try:
            dataset = LazyDataset(
                pdf,
                partial(model.encoder.prepare_input, random_padding=False),
            )
        except pypdf.errors.PdfStreamError:
            logging.info(f"Could not load file {str(pdf)}.")
            continue
        datasets.append(dataset)

    if len(datasets) != 0:
        dataloader = torch.utils.data.DataLoader(
            ConcatDataset(datasets),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate,
        )

        predictions = []
        file_index = 0
        page_num = 0
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
                        # If we end up here, it means the document page is too different from the training domain.
                        # This can happen e.g. for cover pages.
                        predictions.append(
                            f"\n\n[MISSING_PAGE_EMPTY:{i * batch_size + j + 1}]\n\n"
                        )
                else:
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
    st.title("OCR App using Streamlit for PDF files")

    st.write("Upload one or more PDF files for OCR")

    uploaded_files = st.file_uploader("Choose PDF files...", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        st.sidebar.title("About")
        st.sidebar.info(
            "This is an OCR application using Streamlit and pytesseract for PDF files. Upload one or more PDFs, "
            "and we'll extract the text for you!")

        # Convert UploadedFile objects to Path objects by saving them to temporary files
        pdf_names = set(file.name for file in uploaded_files)
        selected_files = st.multiselect("Select PDFs to process", pdf_names, default=pdf_names)

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
