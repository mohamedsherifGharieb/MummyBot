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
        ("Ptolemy I",     "https://en.wikipedia.org/wiki/Ptolemy_I_Soter"),
        ("Ptolemy II",    "https://en.wikipedia.org/wiki/Ptolemy_II_Philadelphus"),
        ("Ptolemy III",   "https://en.wikipedia.org/wiki/Ptolemy_III_Euergetes"),
        ("Cleopatra VII", "https://en.wikipedia.org/wiki/Cleopatra"),
        ("Arsinoe II",    "https://en.wikipedia.org/wiki/Arsinoe_II"),
    ],

    "Dynasty 1–2 (Early Dynastic ~3100–2686 BC)": [
        ("Narmer",      "https://en.wikipedia.org/wiki/Narmer"),
        ("Hor-Aha",     "https://en.wikipedia.org/wiki/Hor-Aha"),
        ("Djer",        "https://en.wikipedia.org/wiki/Djer"),
        ("Den",         "https://en.wikipedia.org/wiki/Den_(pharaoh)"),
        ("Khasekhemwy", "https://en.wikipedia.org/wiki/Khasekhemwy"),
    ],

    "Dynasty 3 (Old Kingdom ~2686–2613 BC)": [
        ("Djoser",     "https://en.wikipedia.org/wiki/Djoser"),
        ("Sekhemkhet", "https://en.wikipedia.org/wiki/Sekhemkhet"),
        ("Huni",       "https://en.wikipedia.org/wiki/Huni"),
    ],
    "Dynasty 5 (Old Kingdom ~2494–2345 BC)": [
        ("Userkaf",  "https://en.wikipedia.org/wiki/Userkaf"),
        ("Sahure",   "https://en.wikipedia.org/wiki/Sahure"),
        ("Niuserre", "https://en.wikipedia.org/wiki/Niuserre"),
        ("Unas",     "https://en.wikipedia.org/wiki/Unas"),
    ],
    "Dynasty 6 (Old Kingdom ~2345–2181 BC)": [
        ("Teti",     "https://en.wikipedia.org/wiki/Teti"),
        ("Pepi I",   "https://en.wikipedia.org/wiki/Pepi_I_Meryre"),
        ("Pepi II",  "https://en.wikipedia.org/wiki/Pepi_II_Neferkare"),
        ("Nitocris", "https://en.wikipedia.org/wiki/Nitocris_(pharaoh)"),
    ],

    "Dynasty 11 (Middle Kingdom ~2055–1985 BC)": [
        ("Mentuhotep II",  "https://en.wikipedia.org/wiki/Mentuhotep_II"),
        ("Mentuhotep III", "https://en.wikipedia.org/wiki/Mentuhotep_III"),
        ("Mentuhotep IV",  "https://en.wikipedia.org/wiki/Mentuhotep_IV"),
    ],
    "Dynasty 12 (Middle Kingdom ~1985–1773 BC)": [
        ("Amenemhat I",   "https://en.wikipedia.org/wiki/Amenemhat_I"),
        ("Senusret I",    "https://en.wikipedia.org/wiki/Senusret_I"),
        ("Senusret II",   "https://en.wikipedia.org/wiki/Senusret_II"),
        ("Senusret III",  "https://en.wikipedia.org/wiki/Senusret_III"),
        ("Amenemhat III", "https://en.wikipedia.org/wiki/Amenemhat_III"),
        ("Sobekneferu",   "https://en.wikipedia.org/wiki/Sobekneferu"),
    ],

    "Dynasty 15 (Hyksos ~1650–1550 BC)": [
        ("Salitis", "https://en.wikipedia.org/wiki/Salitis"),
        ("Apep",    "https://en.wikipedia.org/wiki/Apepi_(pharaoh)"),
        ("Khamudi", "https://en.wikipedia.org/wiki/Khamudi"),
    ],
    "Dynasty 17 (Theban ~1580–1550 BC)": [
        ("Seqenenre Tao", "https://en.wikipedia.org/wiki/Seqenenre_Tao"),
        ("Kamose",        "https://en.wikipedia.org/wiki/Kamose"),
    ],

    "Dynasty 20 (Late Ramesside ~1186–1069 BC)": [
        ("Ramesses III", "https://en.wikipedia.org/wiki/Ramesses_III"),
        ("Ramesses IV",  "https://en.wikipedia.org/wiki/Ramesses_IV"),
        ("Ramesses IX",  "https://en.wikipedia.org/wiki/Ramesses_IX"),
        ("Ramesses XI",  "https://en.wikipedia.org/wiki/Ramesses_XI"),
    ],

    "Dynasty 21 (Third Intermediate ~1069–945 BC)": [
        ("Smendes I",   "https://en.wikipedia.org/wiki/Smendes_I"),
        ("Pinedjem I",  "https://en.wikipedia.org/wiki/Pinedjem_I"),
        ("Psusennes I", "https://en.wikipedia.org/wiki/Psusennes_I"),
    ],
    "Dynasty 22 (Libyan ~945–720 BC)": [
        ("Shoshenq I",   "https://en.wikipedia.org/wiki/Shoshenq_I"),
        ("Osorkon II",   "https://en.wikipedia.org/wiki/Osorkon_II"),
        ("Shoshenq III", "https://en.wikipedia.org/wiki/Shoshenq_III"),
    ],
    "Dynasty 26 (Saite Period ~664–525 BC)": [
        ("Psamtik I",   "https://en.wikipedia.org/wiki/Psamtik_I"),
        ("Necho II",    "https://en.wikipedia.org/wiki/Necho_II"),
        ("Psamtik II",  "https://en.wikipedia.org/wiki/Psamtik_II"),
        ("Apries",      "https://en.wikipedia.org/wiki/Apries"),
        ("Ahmose II",   "https://en.wikipedia.org/wiki/Ahmose_II"),
        ("Psamtik III", "https://en.wikipedia.org/wiki/Psamtik_III"),
    ],
    "Dynasty 27 (First Persian ~525–404 BC)": [
        ("Cambyses II",  "https://en.wikipedia.org/wiki/Cambyses_II"),
        ("Darius I",     "https://en.wikipedia.org/wiki/Darius_I"),
        ("Xerxes I",     "https://en.wikipedia.org/wiki/Xerxes_I"),
        ("Artaxerxes I", "https://en.wikipedia.org/wiki/Artaxerxes_I"),
    ],
    "Dynasty 30 (Last Native ~380–343 BC)": [
        ("Nectanebo I",  "https://en.wikipedia.org/wiki/Nectanebo_I"),
        ("Teos",         "https://en.wikipedia.org/wiki/Teos_of_Egypt"),
        ("Nectanebo II", "https://en.wikipedia.org/wiki/Nectanebo_II"),
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
    paragraphs  = content_div.select('p') if content_div else soup.select('p')

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
    done  = 0

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
    print(f"Total pharaohs scraped: {done}")


if __name__ == "__main__":
    scrape_all("egypt_raw_data.md") 