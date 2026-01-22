#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path


JA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uF900-\uFAFF]")
BREAK_TO_SPACE = {"SPACE", "SURE_SPACE"}
BREAK_TO_NEWLINE = {"LINE_BREAK", "EOL_SURE_SPACE"}


def _has_ja_language(word):
    langs = word.get("property", {}).get("detectedLanguages", [])
    return any(lang.get("languageCode") == "ja" for lang in langs)


def _word_text(word):
    return "".join(sym.get("text", "") for sym in word.get("symbols", []))


def _extract_by_words(data):
    parts = []
    responses = data.get("responses", [])
    for resp in responses:
        fta = resp.get("fullTextAnnotation", {})
        for page in fta.get("pages", []):
            for block in page.get("blocks", []):
                for para in block.get("paragraphs", []):
                    for word in para.get("words", []):
                        text = _word_text(word)
                        if not (_has_ja_language(word) or JA_RE.search(text)):
                            continue
                        for sym in word.get("symbols", []):
                            parts.append(sym.get("text", ""))
                            br = sym.get("property", {}).get("detectedBreak", {}).get("type")
                            if br in BREAK_TO_NEWLINE:
                                parts.append("\n")
                            elif br in BREAK_TO_SPACE:
                                parts.append(" ")
    return "".join(parts)


def _extract_fulltext(data, only_ja):
    responses = data.get("responses", [])
    if not responses:
        return ""
    fta = responses[0].get("fullTextAnnotation", {})
    text = fta.get("text", "")
    if not only_ja:
        return text
    lines = [line for line in text.splitlines() if JA_RE.search(line)]
    return "\n".join(lines)


def extract_ja_text(data):
    text = _extract_by_words(data)
    if text.strip():
        return text
    return _extract_fulltext(data, only_ja=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Japanese text from GCloud Vision OCR JSON."
    )
    parser.add_argument(
        "--input-dir",
        default="gcloud_test",
        help="Directory with OCR JSON files (default: gcloud_test).",
    )
    parser.add_argument(
        "--output-dir",
        default="gcloud_test/extracted_ja",
        help="Directory to write extracted text files.",
    )
    parser.add_argument(
        "--mode",
        choices=("detected", "fulltext", "fulltext-ja"),
        default="detected",
        help="Extraction mode (default: detected).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {input_dir}")

    combined = []
    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if args.mode == "fulltext":
            text = _extract_fulltext(data, only_ja=False)
        elif args.mode == "fulltext-ja":
            text = _extract_fulltext(data, only_ja=True)
        else:
            text = extract_ja_text(data)
        out_path = output_dir / f"{path.stem}.ja.txt"
        out_path.write_text(text, encoding="utf-8")
        if text.strip():
            combined.append(f"=== {path.name} ===\n{text}\n")

    if combined:
        (output_dir / "all_pages.ja.txt").write_text(
            "\n".join(combined), encoding="utf-8"
        )

    print(f"Wrote {len(json_files)} files to {output_dir}")


if __name__ == "__main__":
    main()
