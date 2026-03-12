import streamlit as st
import requests
import json
import re
from datetime import datetime, timedelta

st.title("YouTube Video Title Scraper")

channel_url = "https://www.youtube.com/@tvOneNews/videos"

days = st.slider("Ambil video dari berapa hari terakhir", 1, 7, 2)

def get_video_titles(url):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, headers=headers)
    html = r.text

    json_text = re.search(r"var ytInitialData = ({.*?});", html)

    if not json_text:
        st.error("Data tidak ditemukan")
        return []

    data = json.loads(json_text.group(1))

    videos = []

    contents = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][1] \
        ["tabRenderer"]["content"]["richGridRenderer"]["contents"]

    for item in contents:
        if "richItemRenderer" in item:
            video = item["richItemRenderer"]["content"]["videoRenderer"]

            title = video["title"]["runs"][0]["text"]

            published = video.get("publishedTimeText", {}).get("simpleText", "")

            videos.append((title, published))

    return videos


if st.button("Ambil Data Video"):

    videos = get_video_titles(channel_url)

    filtered = []
    for title, published in videos:

        if "day" in published or "hari" in published:
            filtered.append(title)

    st.write("### Judul Video")
    for v in filtered:
        st.write(v)

    txt = "\n".join(filtered)

    st.download_button(
        "Download TXT",
        txt,
        file_name="judul_video_tvonenews.txt"
    )
