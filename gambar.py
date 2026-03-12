from pathlib import Path
import io

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

try:
    from streamlit_paste_button import paste_image_button as pbutton
    PASTE_AVAILABLE = True
except Exception:
    PASTE_AVAILABLE = False


st.set_page_config(
    page_title="Text Extractor",
    page_icon="📝",
    layout="wide",
)

SUPPORTED_TYPES = ["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg"]


def configure_tesseract(custom_path: str | None = None) -> None:
    if OCR_AVAILABLE and custom_path and custom_path.strip():
        pytesseract.pytesseract.tesseract_cmd = custom_path.strip()


def tesseract_is_ready() -> tuple[bool, str]:
    if not OCR_AVAILABLE:
        return False, "Package pytesseract belum terinstall."

    try:
        _ = pytesseract.get_tesseract_version()
        return True, ""
    except Exception as e:
        return False, str(e)


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


def extract_from_pil_image(image: Image.Image, language: str = "eng") -> str:
    ok, err = tesseract_is_ready()
    if not ok:
        raise RuntimeError(
            "Tesseract belum siap. "
            "Install Tesseract OCR atau isi path Tesseract di sidebar.\n"
            f"Detail: {err}"
        )
    return pytesseract.image_to_string(image, lang=language)


def extract_from_image_upload(file_obj, language: str = "eng") -> str:
    image = Image.open(file_obj)
    return extract_from_pil_image(image, language=language)


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
        return extract_from_image_upload(uploaded_file, language=ocr_language)

    raise ValueError(f"Format file .{suffix} belum didukung.")


def make_download_name(original_name: str) -> str:
    stem = Path(original_name).stem
    return f"{stem}_extracted.txt"


def show_result_block(title: str, extracted_text: str, download_name: str, key_prefix: str):
    if not extracted_text.strip():
        st.warning("Tidak ada teks yang berhasil diekstrak.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.text_area(
            title,
            value=extracted_text,
            height=300,
            key=f"{key_prefix}_preview",
        )

    with col2:
        st.metric("Jumlah karakter", len(extracted_text))
        st.metric("Jumlah kata", len(extracted_text.split()))
        st.download_button(
            label="⬇️ Download TXT",
            data=extracted_text,
            file_name=download_name,
            mime="text/plain",
            key=f"{key_prefix}_download",
        )


st.title("📝 Text Extractor")
st.caption("Extract teks dari PDF, DOCX, TXT, CSV, dan gambar (OCR).")

with st.sidebar:
    st.header("Pengaturan")

    ocr_language = st.selectbox(
        "Bahasa OCR",
        options=["eng", "ind"],
        index=0,
        help="Untuk OCR gambar. 'ind' butuh bahasa Indonesia di Tesseract.",
    )

    tesseract_path = st.text_input(
        "Path Tesseract (opsional)",
        placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        help="Isi kalau Tesseract sudah terinstall tapi belum masuk PATH.",
    )

    configure_tesseract(tesseract_path)

    ok, err = tesseract_is_ready()
    if ok:
        st.success("Tesseract terdeteksi.")
    else:
        st.warning("Tesseract belum terdeteksi.")
        st.caption(err)

st.subheader("📋 Paste gambar langsung")
if PASTE_AVAILABLE:
    st.caption("Copy gambar ke clipboard, lalu klik tombol paste.")
    paste_result = pbutton(
        label="Paste gambar dari clipboard",
        key="paste_btn",
        errors="raise",
    )

    if paste_result.image_data is not None:
        st.image(paste_result.image_data, caption="Gambar dari clipboard", use_container_width=True)

        try:
            extracted_text = extract_from_pil_image(
                paste_result.image_data,
                language=ocr_language,
            )
            show_result_block(
                "Hasil OCR dari clipboard",
                extracted_text,
                "clipboard_image_extracted.txt",
                "clipboard",
            )
        except Exception as e:
            st.error(f"Gagal OCR dari clipboard: {e}")
else:
    st.info("Install package `streamlit-paste-button` untuk fitur paste clipboard.")

st.divider()

st.subheader("Upload file")
uploaded_files = st.file_uploader(
    "Upload satu atau beberapa file",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.container(border=True):
            st.subheader(f"📄 {uploaded_file.name}")
            st.write(f"Ukuran file: **{uploaded_file.size:,} bytes**")

            try:
                extracted_text = extract_text(uploaded_file, ocr_language=ocr_language)
                show_result_block(
                    "Hasil ekstraksi",
                    extracted_text,
                    make_download_name(uploaded_file.name),
                    uploaded_file.name,
                )
            except Exception as e:
                st.error(f"Gagal extract file: {e}")

st.divider()
st.caption("Dibuat dengan Python + Streamlit")
