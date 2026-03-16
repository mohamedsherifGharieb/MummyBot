"""
clean_data.py

Reads egypt_raw_data.md → cleans + classifies into topic sections → writes data_clean/*.txt

Run: python clean_data.py
"""

import re
from pathlib import Path

INPUT_FILE = Path("egypt_raw_data.md")
OUTPUT_DIR = Path("data_clean")

TOPIC_RULES = {
    "IDENTITY": [
        "was a pharaoh", "was an ancient egyptian", "was the first pharaoh",
        "was the second pharaoh", "was the third pharaoh", "was the fourth",
        "was the fifth", "was the sixth", "was the seventh", "was the eighth",
        "was the last pharaoh", "was the last king", "was the founder",
        "was a king", "was a queen", "was the last ruler", "was a kushite",
        "was a nubian", "was a macedonian", "reigned", "ruled egypt",
        "pharaoh of the", "king of the", "dynasty of egypt",
        "succeeded", "ascended the throne", "came to power", "took the throne",
        "his reign", "her reign", "his rule", "her rule",
    ],
    "REIGN": [
        "campaign", "military", "expedition", "conquered", "defeated",
        "invaded", "battle", "war", "army", "troops", "victory",
        "reformed", "reorganized", "established", "introduced", "policy",
        "trade", "diplomatic", "alliance", "treaty", "negotiated",
        "expelled", "restored", "reunited", "unified", "rebellion",
    ],
    "MONUMENTS": [
        "built", "constructed", "commissioned", "erected", "pyramid",
        "temple", "obelisk", "sphinx", "mortuary", "tomb", "monument",
        "palace", "canal", "stele", "chapel", "sanctuary", "complex",
        "karnak", "luxor", "giza", "saqqara", "abydos", "thebes",
        "building project", "construction", "architecture",
    ],
    "FAMILY": [
        "son of", "daughter of", "father of", "mother of", "wife of",
        "husband of", "married", "children", "successor", "heir",
        "brother", "sister", "uncle", "aunt", "grandfather", "grandmother",
        "queen", "prince", "princess", "royal family", "consort",
        "parentage", "lineage", "descent",
    ],
    "DEATH": [
        "died", "death", "suicide", "assassinated", "murdered", "killed",
        "poison", "strangled", "execution", "fell in battle", "cause of death",
        "buried", "burial", "tomb", "mummy", "mummified", "sarcophagus",
        "valley of the kings", "necropolis", "funeral", "afterlife",
        "passed away", "end of his reign", "end of her reign",
    ],
    "LEGACY": [
        "legacy", "remembered", "known for", "famous for", "considered",
        "regarded as", "one of the greatest", "one of the most",
        "influence", "impact", "significance", "historically",
        "later period", "subsequent", "modern", "archaeolog",
        "discovered", "excavated", "museum", "artifact",
    ],
}


def clean_text(text: str) -> str:
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[note \d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[[a-f]\]', '', text)
    text = re.sub(r'\(/[^)]+/[^)]*\)', '', text)
    text = re.sub(r'listenⓘ', '', text)
    text = re.sub(r'[\U00013000-\U0001342F]+', '', text)
    text = re.sub(r'\(Ancient Egyptian:[^)]+\)', '', text)
    text = re.sub(r'\(Koine Greek:[^)]+\)', '', text)
    text = re.sub(r'\(Ancient Greek:[^)]+\)', '', text)
    text = re.sub(r'\(Neo-Assyrian Akkadian:[^)]+\)', '', text)
    text = re.sub(r'\(Old Persian:[^)]+\)', '', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'^\|.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_sentences(text: str) -> list:
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if len(s.strip()) > 30]


def classify_sentence(sentence: str) -> str:
    lower  = sentence.lower()
    scores = {topic: 0 for topic in TOPIC_RULES}
    for topic, keywords in TOPIC_RULES.items():
        for kw in keywords:
            if kw in lower:
                scores[topic] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "IDENTITY"


def build_topic_sections(paragraphs: list) -> dict:
    sections = {topic: [] for topic in TOPIC_RULES}
    for para in paragraphs:
        for sent in split_sentences(para):
            topic = classify_sentence(sent)
            sections[topic].append(sent)
    return sections


def parse_raw_md(filepath: Path) -> dict:
    content  = filepath.read_text(encoding="utf-8")
    result   = {}
    sections = re.split(r'\n## ', content)
    for section in sections[1:]:
        lines   = section.strip().split("\n")
        dynasty = lines[0].strip()
        result[dynasty] = []
        pharaoh_blocks  = re.split(r'\n### ', section)
        for block in pharaoh_blocks[1:]:
            name      = block.strip().split("\n")[0].strip()
            src_match = re.search(r'\*\*Source:\*\*\s*(https?://\S+)', block)
            source    = src_match.group(1).strip() if src_match else ""
            raw = re.sub(r'\*\*Source:\*\*.*?\n', '', block, flags=re.DOTALL)
            raw = re.sub(r'---+', '', raw)
            raw = clean_text(raw)
            if "scrape failed" in raw.lower() or len(raw) < 50:
                continue
            paragraphs = [p.strip() for p in raw.split('\n\n') if len(p.strip()) > 40]
            if paragraphs:
                result[dynasty].append({"name": name, "source": source, "paragraphs": paragraphs})
    return result


def to_filename(dynasty: str) -> str:
    name = dynasty.lower()
    name = re.sub(r'[~()]', '', name)
    name = re.sub(r'\s*[–\-]\s*', '_', name)
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
    DIVIDER   = "=" * 60
    total     = 0

    for dynasty, pharaohs in dynasties.items():
        if not pharaohs:
            continue
        output_file = OUTPUT_DIR / to_filename(dynasty)
        out_lines   = [f"DYNASTY: {dynasty}", DIVIDER, ""]
        print(f"[{dynasty}] — {len(pharaohs)} pharaohs")
        for p in pharaohs:
            total += 1
            sections = build_topic_sections(p["paragraphs"])
            out_lines += [f"PHARAOH: {p['name']}", f"SOURCE:  {p['source']}", "-" * 40, ""]
            for topic, sentences in sections.items():
                if not sentences:
                    continue
                body = " ".join(sentences)
                if len(body) < 40:
                    continue
                out_lines += [f"[{topic}]", body, ""]
            out_lines += [DIVIDER, ""]
        output_file.write_text("\n".join(out_lines), encoding="utf-8")
        print(f"  → {output_file}\n")

    print(f"✅ Done. {total} pharaohs written to {OUTPUT_DIR}/")
    print("Next: run DataBaseChunking.py")


if __name__ == "__main__":
    main()