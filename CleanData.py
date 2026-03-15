"""
clean_data.py

Reads egypt_raw_data.md → cleans each pharaoh's text → writes data_clean/*.txt
One file per dynasty, one section per pharaoh.
No field extraction — keeps clean readable Wikipedia paragraphs.

Run: python clean_data.py
"""

import re
from pathlib import Path

INPUT_FILE = Path("egypt_raw_data.md")
OUTPUT_DIR = Path("data_clean")


def clean_text(text: str) -> str:
    # Remove citation brackets [1], [2], [note 1]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[note \d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[a\]|\[b\]|\[c\]|\[d\]|\[e\]|\[f\]', '', text)

    # Remove pronunciation guides: (/hɑːtˈʃɛpsʊt/ ...)
    text = re.sub(r'\(/[^)]+/[^)]*\)', '', text)
    text = re.sub(r'listenⓘ', '', text)

    # Remove Ancient Egyptian hieroglyph sequences
    text = re.sub(r'𓀀-𔁕|[\U00013000-\U0001342F]+', '', text)

    # Remove romanization labels: (Ancient Egyptian: ...) keep just the name
    text = re.sub(r'\(Ancient Egyptian:[^)]+\)', '', text)
    text = re.sub(r'\(Koine Greek:[^)]+\)', '', text)
    text = re.sub(r'\(Ancient Greek:[^)]+\)', '', text)
    text = re.sub(r'\(Neo-Assyrian Akkadian:[^)]+\)', '', text)
    text = re.sub(r'\(Old Persian:[^)]+\)', '', text)

    # Remove transliteration lines like "ḥr-m-ḥb" that appear alone
    text = re.sub(r'\b[ḥḫꜣꜥṯḏṭṣḳꜢ][^\s,\.]+', '', text)

    # Remove markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)

    # Remove wiki table/list artifacts
    text = re.sub(r'^\|.*$', '', text, flags=re.MULTILINE)

    # Collapse multiple spaces and blank lines
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def parse_raw_md(filepath: Path) -> dict:
    """Returns {dynasty_name: [{name, source, paragraphs}]}"""
    content = filepath.read_text(encoding="utf-8")
    result  = {}

    sections = re.split(r'\n## ', content)

    for section in sections[1:]:
        lines   = section.strip().split("\n")
        dynasty = lines[0].strip()
        result[dynasty] = []

        pharaoh_blocks = re.split(r'\n### ', section)

        for block in pharaoh_blocks[1:]:
            block_lines = block.strip().split("\n")
            name        = block_lines[0].strip()

            src_match = re.search(r'\*\*Source:\*\*\s*(https?://\S+)', block)
            source    = src_match.group(1).strip() if src_match else ""

            # Extract raw text — everything after source line
            raw = re.sub(r'\*\*Source:\*\*.*?\n', '', block, flags=re.DOTALL)
            raw = re.sub(r'---+', '', raw)
            raw = clean_text(raw)

            # Skip scrape failures
            if "scrape failed" in raw.lower() or len(raw) < 50:
                continue

            # Keep only meaningful paragraphs (>60 chars)
            paragraphs = [
                p.strip() for p in raw.split('\n\n')
                if len(p.strip()) > 60
            ]

            if not paragraphs:
                continue

            result[dynasty].append({
                "name":       name,
                "source":     source,
                "paragraphs": paragraphs[:8],  # max 8 paragraphs per pharaoh
            })

    return result


def to_filename(dynasty: str) -> str:
    name = dynasty.lower()
    name = re.sub(r'[~()]', '', name)
    name = re.sub(r'\s*[–\-]\s*', '_', name)   # date ranges → underscore
    name = re.sub(r'[^\w]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_') + ".txt"


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: '{INPUT_FILE}' not found. Run egypt_scraper.py first.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Reading: {INPUT_FILE}\n")

    dynasties = parse_raw_md(INPUT_FILE)
    total     = 0
    DIVIDER   = "=" * 60

    for dynasty, pharaohs in dynasties.items():
        if not pharaohs:
            continue

        filename    = to_filename(dynasty)
        output_file = OUTPUT_DIR / filename
        out_lines   = [f"DYNASTY: {dynasty}", DIVIDER, ""]

        print(f"[{dynasty}] — {len(pharaohs)} pharaohs")

        for p in pharaohs:
            total += 1
            body = "\n\n".join(p["paragraphs"])

            out_lines += [
                f"PHARAOH: {p['name']}",
                f"SOURCE:  {p['source']}",
                "-" * 40,
                "",
                body,
                "",
                DIVIDER,
                "",
            ]

        output_file.write_text("\n".join(out_lines), encoding="utf-8")
        print(f"  → {output_file}\n")

    print(f"✅ Done. {total} pharaohs written to {OUTPUT_DIR}/")
    print("Next: run DataBaseChunking.py")


if __name__ == "__main__":
    main()