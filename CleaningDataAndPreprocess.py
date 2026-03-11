import re
from pathlib import Path

INPUT_FILE = "egypt_raw_data.md"
OUTPUT_DIR = Path("data")


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", "_", text.strip())
    return re.sub(r"_+", "_", text).strip("_")


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[note \d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Ancient (Egyptian|Greek)[^)]*\)", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_raw_md(filepath: str) -> dict:
    dynasties = {}
    current_dynasty = None
    current_king = None
    buffer = []

    def flush():
        if current_dynasty and current_king:
            cleaned = [clean_text(p) for p in buffer if clean_text(p)]
            dynasties[current_dynasty].append({
                "name":   current_king["name"],
                "source": current_king["source"],
                "text":   cleaned,
            })

    with open(filepath, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if re.match(r"^## .+", line):
                flush()
                current_king = None
                buffer = []
                current_dynasty = re.sub(r"^## ", "", line).strip()
                dynasties[current_dynasty] = []
                continue

            if re.match(r"^### .+", line):
                flush()
                buffer = []
                current_king = {"name": re.sub(r"^### ", "", line).strip(), "source": ""}
                continue

            m = re.match(r"^\*\*Source:\*\*\s*(.+)", line)
            if m and current_king:
                current_king["source"] = m.group(1).strip()
                continue

            if re.match(r"^(---|#[^#]|>)", line):
                continue

            if current_king and line.strip():
                buffer.append(line.strip())

    flush()
    return dynasties


def main():
    if not Path(INPUT_FILE).exists():
        print(f"ERROR: '{INPUT_FILE}' not found. Run egypt_scraper.py first.")
        return

    dynasties = parse_raw_md(INPUT_FILE)
    OUTPUT_DIR.mkdir(exist_ok=True)

    DIVIDER = "=" * 60

    for dynasty_title, kings in dynasties.items():
        if not kings:
            continue

        filename = OUTPUT_DIR / f"{slugify(dynasty_title)}.txt"
        lines = [
            f"DYNASTY: {dynasty_title}",
            DIVIDER,
            "",
        ]

        for king in kings:
            lines += [
                f"PHARAOH: {king['name']}",
                f"SOURCE:  {king['source']}",
                "-" * 40,
                "",
            ]
            lines += king["text"]
            lines += ["", DIVIDER, ""]

        filename.write_text("\n".join(lines), encoding="utf-8")
        print(f"✅ {filename}  ({len(kings)} kings)")

    print(f"\nDone - {len(dynasties)} files saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()