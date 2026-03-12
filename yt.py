import streamlit as st
import yt_dlp
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="YouTube Title Exporter", layout="centered")
st.title("Export Judul Video YouTube ke TXT")

DEFAULT_CHANNEL = "https://www.youtube.com/@tvOneNews/videos"

channel_url = st.text_input("Channel URL", value=DEFAULT_CHANNEL)
days = st.slider("Ambil video dari berapa hari terakhir?", 1, 7, 2)
max_items = st.slider("Maksimal video terbaru yang dicek", 20, 300, 120, step=20)

def parse_upload_date(upload_date_str: str):
    if not upload_date_str:
        return None
    try:
        return datetime.strptime(upload_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def get_recent_titles(channel_url: str, days: int, max_items: int):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
        "extract_flat": False,     # penting: upload_date sering kosong kalau flat
        "playlistend": max_items,  # cek video terbaru saja
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    if not info:
        return []

    entries = info.get("entries", [])
    if not entries:
        return []

    results = []
    for entry in entries:
        if not entry:
            continue

        title = entry.get("title")
        upload_date = parse_upload_date(entry.get("upload_date"))

        if title and upload_date and upload_date >= cutoff:
            results.append({
                "title": title,
                "upload_date": upload_date.strftime("%Y-%m-%d"),
            })

    return results

if st.button("Ambil Judul Video"):
    try:
        data = get_recent_titles(channel_url, days, max_items)

        if not data:
            st.warning("Tidak ada video dalam rentang waktu itu, atau yt-dlp perlu di-update.")
        else:
            st.success(f"Ketemu {len(data)} video")

            for i, item in enumerate(data, start=1):
                st.write(f"{i}. {item['title']}")

            txt_content = "\n".join(item["title"] for item in data)

            st.download_button(
                label="Download TXT",
                data=txt_content,
                file_name="judul_video_tvonenews.txt",
                mime="text/plain",
            )

    except Exception as e:
        st.error(f"Gagal ambil data: {e}")
        st.info("Coba update yt-dlp lalu jalankan lagi: python -m pip install -U yt-dlp")
