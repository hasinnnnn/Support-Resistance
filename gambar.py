from pathlib import Path

import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image

try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


st.set_page_config(
    page_title="Text Extractor",
    page_icon="📝",
    layout="wide",
)


SUPPORTED_TYPES = [
    "pdf",
    "docx",
    "txt",
    "csv",
    "png",
    "jpg",
    "jpeg",
]


def extract_from_pdf(file_obj) -> str:
    reader = PdfReader(file_obj)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n--- Halaman {i} ---\n{text.strip()}\n")
    return "\n".join(pages).strip()


def extract_from_docx(file_obj) -> str:
    doc = Document(file_obj)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def extract_from_txt(file_obj) -> str:
    raw = file_obj.read()
    if isinstance(raw, bytes):
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def extract_from_csv(file_obj) -> str:
    df = pd.read_csv(file_obj)
    return df.to_string(index=False)


def extract_from_image(file_obj, language: str = "eng") -> str:
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "pytesseract belum tersedia. Install pytesseract dan Tesseract OCR terlebih dahulu."
        )
    image = Image.open(file_obj)
    return pytesseract.image_to_string(image, lang=language)


def extract_text(uploaded_file, ocr_language: str) -> str:
    suffix = Path(uploaded_file.name).suffix.lower().replace(".", "")

    if suffix == "pdf":
        return extract_from_pdf(uploaded_file)
    if suffix == "docx":
        return extract_from_docx(uploaded_file)
    if suffix == "txt":
        return extract_from_txt(uploaded_file)
    if suffix == "csv":
        return extract_from_csv(uploaded_file)
    if suffix in {"png", "jpg", "jpeg"}:
        return extract_from_image(uploaded_file, language=ocr_language)

    raise ValueError(f"Format file .{suffix} belum didukung.")


def make_download_name(original_name: str) -> str:
    stem = Path(original_name).stem
    return f"{stem}_extracted.txt"


st.title("📝 Text Extractor")
st.caption("Extract teks dari PDF, DOCX, TXT, CSV, dan gambar (OCR).")

with st.sidebar:
    st.header("Pengaturan")
    ocr_language = st.selectbox(
        "Bahasa OCR",
        options=["eng", "ind"],
        index=0,
        help="Untuk gambar. 'ind' membutuhkan data bahasa Indonesia di Tesseract.",
    )
    st.info(
        "OCR gambar memakai Tesseract. Kalau upload PNG/JPG/JPEG gagal, pastikan Tesseract OCR sudah terpasang di sistem."
    )

uploaded_files = st.file_uploader(
    "Upload satu atau beberapa file",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

if not uploaded_files:
    st.markdown(
        """
        ### Fitur
        - Extract teks dari **PDF**
        - Baca isi **DOCX**, **TXT**, dan **CSV**
        - OCR untuk **PNG/JPG/JPEG**
        - Preview hasil
        - Download hasil ekstraksi sebagai `.txt`
        """
    )
    st.stop()

for uploaded_file in uploaded_files:
    with st.container(border=True):
        st.subheader(f"📄 {uploaded_file.name}")
        st.write(f"Ukuran file: **{uploaded_file.size:,} bytes**")

        try:
            extracted_text = extract_text(uploaded_file, ocr_language=ocr_language)

            if not extracted_text.strip():
                st.warning("Tidak ada teks yang berhasil diekstrak dari file ini.")
                continue

            col1, col2 = st.columns([2, 1])

            with col1:
                st.text_area(
                    "Hasil ekstraksi",
                    value=extracted_text,
                    height=300,
                    key=f"preview_{uploaded_file.name}",
                )

            with col2:
                st.metric("Jumlah karakter", len(extracted_text))
                st.metric("Jumlah kata", len(extracted_text.split()))
                st.download_button(
                    label="⬇️ Download TXT",
                    data=extracted_text,
                    file_name=make_download_name(uploaded_file.name),
                    mime="text/plain",
                    key=f"download_{uploaded_file.name}",
                )

        except Exception as e:
            st.error(f"Gagal extract file: {e}")

st.divider()
st.caption("Dibuat dengan Python + Streamlit")
