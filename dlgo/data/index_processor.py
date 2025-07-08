import os
import time
import requests
from urllib.parse import urljoin
from html.parser import HTMLParser


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.current_tag = None
        self.current_attrs = {}

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)

        if tag == "a" and "href" in self.current_attrs:
            self.links.append(
                {
                    "href": self.current_attrs["href"],
                    "text": "",  # will be filled by data
                }
            )

    def handle_data(self, data):
        if self.current_tag == "a" and self.links:
            # Add text content to the last link found
            self.links[-1]["text"] += data.strip()

    def handle_endtag(self, tag):
        self.current_tag = None
        self.current_attrs = {}


def extract_download_links(html_content: str, base_url: str):
    parser = LinkParser()
    parser.feed(html_content)

    download_links = []
    for link in parser.links:
        href = link["href"]
        text = link["text"]

        if "playerdb" in href and href.endswith(".json.gz"):
            # convert relative urls to absolute urls
            absolute_url = urljoin(base_url, href)

            # extract filename
            filename = href.split("/")[-1]

            download_links.append(
                {"filename": filename, "url": absolute_url, "text": text}
            )

    return download_links


def download_file(url, filename, save_dir="go_data"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    if os.path.exists(filepath):
        print(f"File {filename} already exists. Skipping.")
        return filepath
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded {filename}")
        return filepath

    except requests.RequestException as e:
        print(f"Failed to download {filename}: {e}")
        return None


def main():
    main_url = "https://db.u-go.net/playerdb-archive/"
    response = requests.get(main_url)
    if response.status_code == 200:  # sucessful request
        html_content = response.text
        print("Successfully fetched page content")

        links = extract_download_links(html_content, main_url)
        for link in links:
            print(f"File: {link['filename']}")
            print(f"URL: {link['url']}")
            print(f"Text: {link['text']}")
            print("-" * 50)

    if links:
        print(f"\nStarting download of {len(links)} files...")
        downloaded = 0

        for file_info in links:
            filepath = download_file(file_info["url"], file_info["filename"])
            if filepath:
                downloaded += 1

            time.sleep(1)
    print(f"\nDownload complete! Downloaded {downloaded} files.")


if __name__ == "__main__":
    main()
