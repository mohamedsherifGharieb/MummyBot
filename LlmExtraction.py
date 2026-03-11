"""
extract_fields.py

Two-step LLM extraction with anti-hallucination:
  Step 1 — Extract 6 fields using strict few-shot prompt
  Step 2 — Verify each field is grounded in source text

Install:  pip install ollama
Run:      python extract_fields.py
"""

import re
import time
import ollama
from pathlib import Path

DATA_DIR = Path("data_clean")
MODEL    = "llama3.2"
DELAY    = 1.5

EXTRACT_PROMPT = """
You are extracting facts about an ancient Egyptian ruler for a tourist chatbot.

RULES — follow exactly:
1. Use ONLY information that is explicitly stated in the SOURCE TEXT below.
2. Do NOT use your own knowledge. Do NOT guess or infer.
3. If a fact is not in the text, write exactly the word: Unknown
4. Keep language simple. A tourist with no history background will read this.
5. Output ONLY the 6 fields. No intro sentence. No explanation. No extra text.

EXAMPLE INPUT:
Khufu was the second pharaoh of the Fourth Dynasty of Egypt. He commissioned the
Great Pyramid of Giza. His only confirmed portrait is a small ivory figurine found
at Abydos. The cause of his death is not documented.

EXAMPLE OUTPUT:
WHO: Second pharaoh of the Fourth Dynasty of Egypt.
WHAT_THEY_DID: Commissioned the construction of the Great Pyramid of Giza.
AGE: Unknown
HOW_THEY_DIED: Unknown
ACHIEVEMENTS:
- Commissioned the Great Pyramid of Giza
- Only confirmed portrait is a small ivory figurine found at Abydos
EGYPT_ERA: Unknown

NOW EXTRACT FOR THIS RULER:

SOURCE TEXT:
{raw_text}
"""

VERIFY_PROMPT = """
You are a fact-checker. Compare the EXTRACTED FIELDS below against the SOURCE TEXT.

For each field, check: is this information actually stated in the source text?
- If YES → keep it exactly as is.
- If NO or UNCERTAIN → replace the field value with the single word: Unknown

RULES:
- Do NOT add new information.
- Do NOT rephrase or improve anything.
- Only replace with Unknown if the info is NOT in the source text.
- Output the same 6-field format. Nothing else.

SOURCE TEXT:
{raw_text}

EXTRACTED FIELDS:
{extracted}
"""

REQUIRED = ["WHO:", "WHAT_THEY_DID:", "AGE:", "HOW_THEY_DIED:", "ACHIEVEMENTS:", "EGYPT_ERA:"]


def parse_dynasty_file(filepath: Path) -> list[dict]:
    content = filepath.read_text(encoding="utf-8")
    kings   = []
    blocks  = re.split(r"\nPHARAOH:", content)
    for block in blocks[1:]:
        lines  = block.strip().split("\n")
        name   = lines[0].strip()
        src_m  = re.search(r"SOURCE:\s*(.+)", block)
        source = src_m.group(1).strip() if src_m else ""
        raw    = re.sub(r"^.*?SOURCE:.*?\n", "", block, flags=re.DOTALL).strip()
        raw    = re.sub(r"-{3,}|={3,}", "", raw).strip()
        kings.append({"name": name, "source": source, "raw_text": raw})
    return kings


def llm(prompt: str) -> str:
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip()


def extract(king: dict) -> str:
    if len(king["raw_text"]) < 30:
        return "\n".join([
            "WHO: Unknown", "WHAT_THEY_DID: No source data scraped for this ruler.",
            "AGE: Unknown", "HOW_THEY_DIED: Unknown",
            "ACHIEVEMENTS:\n- No data available", "EGYPT_ERA: Unknown"
        ])
    return llm(EXTRACT_PROMPT.format(raw_text=king["raw_text"][:4000]))


def verify(raw_text: str, extracted: str) -> str:
    return llm(VERIFY_PROMPT.format(raw_text=raw_text[:4000], extracted=extracted))


def validate(fields: str) -> list[str]:
    return [f for f in REQUIRED if f not in fields]


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: '{DATA_DIR}/' not found. Run clean_data.py first.")
        return

    files   = sorted(DATA_DIR.glob("*.txt"))
    failed  = []
    DIVIDER = "=" * 60

    for dynasty_file in files:
        print(f"\n[FILE] {dynasty_file.name}")
        kings      = parse_dynasty_file(dynasty_file)
        first_line = dynasty_file.read_text(encoding="utf-8").split("\n")[0]
        out        = [first_line, DIVIDER, ""]

        for i, king in enumerate(kings, 1):
            print(f"  [{i}/{len(kings)}] {king['name']}")

            print(f"    → extracting...", end=" ", flush=True)
            extracted = extract(king)
            time.sleep(DELAY)

            print(f"verifying...", end=" ", flush=True)
            verified = verify(king["raw_text"], extracted)
            time.sleep(DELAY)

            missing = validate(verified)
            if missing:
                print(f"⚠ missing: {missing}")
                failed.append(king["name"])
            else:
                print("✅")

            out += [
                f"PHARAOH: {king['name']}",
                f"SOURCE:  {king['source']}",
                "-" * 40, "",
                verified, "",
                DIVIDER, "",
            ]

        dynasty_file.write_text("\n".join(out), encoding="utf-8")
        print(f"  → Saved: {dynasty_file.name}")

    print(f"\n✅ Done.")
    if failed:
        print(f"⚠  Manual review needed: {', '.join(failed)}")
    print("Next: run vectorize.py")


if __name__ == "__main__":
    main()