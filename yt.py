import re
from datetime import datetime, date

import streamlit as st
import yt_dlp

st.set_page_config(page_title="YouTube Title Exporter", layout="centered")
st.title("Export Judul Video YouTube ke TXT")

DEFAULT_CHANNEL = "https://www.youtube.com/@tvOneNews/videos"

channel_url = st.text_input("Channel URL", value=DEFAULT_CHANNEL)
target_date = st.date_input("Pilih tanggal video", value=date.today())
max_items = st.slider("Maksimal video terbaru yang dicek", 20, 500, 200, step=20)


def parse_upload_date(upload_date_str):
    if not upload_date_str:
        return None
    try:
        return datetime.strptime(upload_date_str, "%Y%m%d").date()
    except Exception:
        return None


def build_video_url(entry):
    if entry.get("webpage_url"):
        return entry["webpage_url"]

    if entry.get("url"):
        url = entry["url"]
        if isinstance(url, str) and url.startswith("http"):
            return url
        return f"https://www.youtube.com/watch?v={url}"

    if entry.get("id"):
        return f"https://www.youtube.com/watch?v={entry['id']}"

    return None


def safe_channel_name(channel_url):
    match = re.search(r"youtube\.com/@([^/]+)", channel_url)
    if match:
        return match.group(1)
    return "channel"


def get_titles_by_date(channel_url, target_date, max_items):
    # Ambil daftar video dulu
    list_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
        "extract_flat": "in_playlist",
        "playlistend": max_items,
    }

    # Ambil detail tiap video supaya upload_date tersedia
    detail_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(list_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    if not info:
        return []

    entries = info.get("entries") or []
    if not entries:
        return []

    results = []

    with yt_dlp.YoutubeDL(detail_opts) as ydl:
        for entry in entries:
            if not entry:
                continue

            video_url = build_video_url(entry)
            if not video_url:
                continue

            detail = ydl.extract_info(video_url, download=False)
            if not detail:
                continue

            title = detail.get("title")
            upload_date = parse_upload_date(detail.get("upload_date"))

            if title and upload_date == target_date:
                results.append({
                    "title": title,
                    "upload_date": upload_date.strftime("%Y-%m-%d"),
                    "url": detail.get("webpage_url", video_url),
                })

    return results


if st.button("Ambil Judul Video"):
    try:
        data = get_titles_by_date(channel_url, target_date, max_items)

        if not data:
            st.warning(
                "Tidak ada video pada tanggal itu dalam rentang yang dicek. "
                "Kalau tanggalnya lebih lama, naikkan 'Maksimal video terbaru yang dicek'."
            )
        else:
            st.success(f"Ketemu {len(data)} video pada {target_date.strftime('%Y-%m-%d')}")

            for i, item in enumerate(data, start=1):
                st.write(f"{i}. {item['title']}")

            txt_content = "\n".join(item["title"] for item in data)
            channel_name = safe_channel_name(channel_url)

            st.download_button(
                label="Download TXT",
                data=txt_content,
                file_name=f"judul_video_{channel_name}_{target_date.isoformat()}.txt",
                mime="text/plain",
            )

    except Exception as e:
        st.error(f"Gagal ambil data: {e}")
        st.info("Coba update yt-dlp lalu jalankan lagi: python -m pip install -U yt-dlp")
