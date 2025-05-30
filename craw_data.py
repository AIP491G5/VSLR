#!/usr/bin/env python3
"""
Video crawler & downloader for qipedc.moet.gov.vn
================================================
* Selenium thu thập 219 trang (~4 362 video).
* ThreadPoolExecutor tải song song, tự động retry lỗi mạng.
* CSV 'Label.csv' luôn đồng bộ, không ghi file hỏng.
------------------------------------------------
Python ≥3.10  -  Dependencies:
pip install selenium webdriver-manager tqdm requests urllib3
"""

from __future__ import annotations

import os
import csv
import logging
import threading
from itertools import count
from functools import partial
from urllib.parse import urlparse

from concurrent.futures import ThreadPoolExecutor

import requests
import urllib3
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# --------------------------------------------------------------------------- #
# Configuration & global paths
# --------------------------------------------------------------------------- #
BASE_URL: str = "https://qipedc.moet.gov.vn"
DICTIONARY_URL: str = f"{BASE_URL}/dictionary"
OUTPUT_VIDEO_DIR: str = "Dataset/Video"
OUTPUT_TEXT_DIR: str = "Dataset/Text"
CSV_PATH: str = os.path.join(OUTPUT_TEXT_DIR, "Label.csv")

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)

csv_lock = threading.Lock()          # protects CSV + counter
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def init_csv() -> None:
    """Create Label.csv with header if it does not exist."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["STT", "VIDEO", "TEXT"])
        logging.info("Created Label.csv with header.")


def append_row(stt: int, video_name: str, label: str) -> None:
    """Thread-safe append of one row."""
    with csv_lock, open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([stt, video_name, label])


# --------------------------------------------------------------------------- #
# Selenium crawl
# --------------------------------------------------------------------------- #
def scrape_page(driver: webdriver.Chrome) -> list[dict]:
    """Return list[{label, url}] from the current dictionary page."""
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located(
            (
                By.CSS_SELECTOR,
                "section:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(1)",
            )
        )
    )

    videos: list[dict] = []
    anchors = driver.find_elements(
        By.CSS_SELECTOR,
        "section:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(1) a",
    )

    for a in anchors:
        label = a.find_element(By.CSS_SELECTOR, "p").text.strip()
        thumb_src = a.find_element(By.CSS_SELECTOR, "img").get_attribute("src")
        video_id = os.path.splitext(os.path.basename(thumb_src))[0]
        videos.append({"label": label, "url": f"{BASE_URL}/videos/{video_id}.mp4"})
    return videos


def crawl_all_pages() -> list[dict]:
    """Crawl entire dictionary; return consolidated list of video entries."""
    logging.info("Starting crawl …")

    opts = Options()
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--ignore-certificate-errors")
    opts.set_capability("acceptInsecureCerts", True)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    collected: list[dict] = []

    try:
        driver.get(DICTIONARY_URL)
        logging.info("Connected to dictionary page.")

        collected.extend(scrape_page(driver))

        # Pages 2-4 (index offset in HTML)
        for i in range(2, 5):
            driver.find_element(By.CSS_SELECTOR, f"button:nth-of-type({i if i == 2 else i + 1})").click()
            collected.extend(scrape_page(driver))

        # Pages 5-217 (repeat 'Next' button is 6th)
        for _ in range(5, 218):
            driver.find_element(By.CSS_SELECTOR, "button:nth-of-type(6)").click()
            collected.extend(scrape_page(driver))

        # Pages 218-219 (index shift on last page)
        for i in range(218, 220):
            driver.find_element(By.CSS_SELECTOR, f"button:nth-of-type({6 if i == 218 else 7})").click()
            collected.extend(scrape_page(driver))

    except Exception as exc:
        logging.error("Crawl error: %s", exc)
    finally:
        driver.quit()

    logging.info("Finished crawling. Total videos found: %d", len(collected))
    return collected


# --------------------------------------------------------------------------- #
# HTTP session & download
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    """Return a Session with retry + disabled SSL validation."""
    sess = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.verify = False  # site uses expired certificate
    return sess


def download_video(video: dict, session: requests.Session, counter: count) -> None:
    """Download one video; update CSV when done."""
    url, label = video["url"], video["label"]
    filename = os.path.basename(urlparse(url).path)
    final_path = os.path.join(OUTPUT_VIDEO_DIR, filename)
    tmp_path = final_path + ".part"

    if os.path.exists(final_path):
        logging.info("Skip %s (exists)", filename)
        return

    for attempt in range(3):
        try:
            with session.get(url, stream=True, timeout=(5, 10)) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))

                with open(tmp_path, "wb") as f, tqdm(
                    total=total,
                    desc=filename,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    ncols=90,
                    leave=False,
                ) as bar:
                    for chunk in r.iter_content(8192):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bar.update(len(chunk))

            os.rename(tmp_path, final_path)
            append_row(next(counter), filename, label)
            return  # success
        except Exception as exc:
            logging.warning("Retry %s %d/3: %s", filename, attempt + 1, exc)

    logging.error("FAILED %s after 3 attempts", filename)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.info("=== VIDEO CRAWLER STARTED ===")
    init_csv()

    videos = crawl_all_pages()
    if not videos:
        logging.error("No videos found — exiting.")
        return

    start_index = sum(1 for _ in open(CSV_PATH, encoding="utf-8"))
    counter = count(start_index)

    workers = min(16, os.cpu_count() * 2)
    logging.info("Starting parallel download (%d videos, %d workers).", len(videos), workers)

    session = make_session()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(partial(download_video, session=session, counter=counter), videos)

    logging.info("All tasks completed. Videos saved to %s", OUTPUT_VIDEO_DIR)


if __name__ == "__main__":
    main()
