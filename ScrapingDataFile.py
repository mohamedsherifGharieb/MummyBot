import time
import re
import requests
from bs4 import BeautifulSoup

PHARAOHS = {
    "Dynasty 4 (Pyramid Builders ~2575–2465 BC)": [
        ("Snefru",      "https://en.wikipedia.org/wiki/Sneferu"),
        ("Khufu",       "https://en.wikipedia.org/wiki/Khufu"),
        ("Djedefre",    "https://en.wikipedia.org/wiki/Djedefre"),
        ("Khafre",      "https://en.wikipedia.org/wiki/Khafre"),
        ("Menkaure",    "https://en.wikipedia.org/wiki/Menkaure"),
        ("Shepseskaf",  "https://en.wikipedia.org/wiki/Shepseskaf"),
    ],
    "Dynasty 18 (New Kingdom Golden Age ~1550–1295 BC)": [
        ("Ahmose I",        "https://en.wikipedia.org/wiki/Ahmose_I"),
        ("Hatshepsut",      "https://en.wikipedia.org/wiki/Hatshepsut"),
        ("Thutmose III",    "https://en.wikipedia.org/wiki/Thutmose_III"),
        ("Amenhotep II",    "https://en.wikipedia.org/wiki/Amenhotep_II"),
        ("Thutmose IV",     "https://en.wikipedia.org/wiki/Thutmose_IV"),
        ("Amenhotep III",   "https://en.wikipedia.org/wiki/Amenhotep_III"),
        ("Akhenaten",       "https://en.wikipedia.org/wiki/Akhenaten"),
        ("Nefertiti",       "https://en.wikipedia.org/wiki/Nefertiti"),
        ("Tutankhamun",     "https://en.wikipedia.org/wiki/Tutankhamun"),
        ("Ay",              "https://en.wikipedia.org/wiki/Ay"),
        ("Horemheb",        "https://en.wikipedia.org/wiki/Horemheb"),
    ],
    "Dynasty 19 (Ramesside Period ~1295–1186 BC)": [
        ("Ramesses I",   "https://en.wikipedia.org/wiki/Ramesses_I"),
        ("Seti I",       "https://en.wikipedia.org/wiki/Seti_I"),
        ("Ramesses II",  "https://en.wikipedia.org/wiki/Ramesses_II"),
        ("Merneptah",    "https://en.wikipedia.org/wiki/Merneptah"),
        ("Seti II",      "https://en.wikipedia.org/wiki/Seti_II"),
        ("Tawosret",     "https://en.wikipedia.org/wiki/Twosret"),
    ],
    "Dynasty 25 (Nubian Pharaohs ~712–664 BC)": [
        ("Piye",      "https://en.wikipedia.org/wiki/Piye"),
        ("Shabaka",   "https://en.wikipedia.org/wiki/Shabaka"),
        ("Shebitku",  "https://en.wikipedia.org/wiki/Shebitku"),
        ("Taharqa",   "https://en.wikipedia.org/wiki/Taharqa"),
        ("Tantamani", "https://en.wikipedia.org/wiki/Tantamani"),
    ],
    "Ptolemaic Period (~304–30 BC)": [
        ("Ptolemy I",    "https://en.wikipedia.org/wiki/Ptolemy_I_Soter"),
        ("Ptolemy II",   "https://en.wikipedia.org/wiki/Ptolemy_II_Philadelphus"),
        ("Ptolemy III",  "https://en.wikipedia.org/wiki/Ptolemy_III_Euergetes"),
        ("Cleopatra VII","https://en.wikipedia.org/wiki/Cleopatra"),
        ("Arsinoe II",   "https://en.wikipedia.org/wiki/Arsinoe_II"),
    ],
}

def clean_text(text: str) -> str:
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[note \d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EgyptScraper/1.0)"}


def extract_wikipedia_content(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    content_div = soup.select_one('#mw-content-text .mw-parser-output')
    paragraphs = content_div.select('p') if content_div else soup.select('p')

    chunks = []
    for p in paragraphs:
        text = p.get_text().strip()
        if len(text) < 40:
            continue
        chunks.append(clean_text(text))
        if len(chunks) >= 8:
            break

    return '\n\n'.join(chunks)



def scrape_all(output_path: str = "egypt_raw_data.md"):
    lines = [
        "# Ancient Egypt Pharaohs — Raw Scraped Data\n",
        "---\n",
    ]

    total = sum(len(v) for v in PHARAOHS.values())
    done = 0

    for dynasty, pharaohs in PHARAOHS.items():
        lines.append(f"\n## {dynasty}\n")
        lines.append("---\n")

        for name, url in pharaohs:
            done += 1
            print(f"[{done}/{total}] Scraping: {name} ...")

            try:
                content = extract_wikipedia_content(url)

                lines.append(f"\n### {name}\n")
                lines.append(f"**Source:** {url}\n\n")
                lines.append(content)
                lines.append("\n\n---\n")

            except Exception as e:
                print(f"  ⚠  Failed: {e}")
                lines.append(f"\n### {name}\n")
                lines.append(f"**Source:** {url}\n\n")
                lines.append(f"_Scrape failed: {e}_\n\n---\n")

            time.sleep(1.5)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(lines))

    print(f"\nDone! Saved to: {output_path}")
    print(f"Kings scraped: {done}")


if __name__ == "__main__":
    scrape_all("egypt_raw_data.md")