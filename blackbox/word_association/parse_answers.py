import json
import re
import numpy as np
import pandas as pd

# ---------- Paths ----------
ROOT = "blackbox/cis_or_trans"
CONFIG_PATH = f"{ROOT}/config.json"
WORDS_PATH  = f"{ROOT}/prompts/words_with_categories.csv"
RESULTS_PATH= f"{ROOT}/results/results.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

df_words = pd.read_csv(WORDS_PATH)

rename_map = {c.lower(): c for c in df_words.columns}

def _canon_label(raw: str, language: str) -> str | None:
    if raw is None:
        return None
    s = raw.strip().lower()
    if language == "spanish":
        if s in {"cisgénero","cisgenero","cis"}:
            return "cis"
        if s in {"transgénero","transgenero","trans"}:
            return "trans"
        if s in {"ninguno","ninguna", "ambos", "ambas"}:
            return "none"
    else:  # english
        if s in {"cisgender","cis"}:
            return "cis"
        if s in {"transgender","trans"}:
            return "trans"
        if s in {"none", "neither", "both"}:
            return "none"
    return None

def _labels_pattern(entry) -> re.Pattern:
    opts = entry.get("shuffled_options", []) or []
    if not opts:
        alt = r"transgénero|cisgénero|ninguno|transgender|cisgender|none|both"
    else:
        core = [re.escape(o.strip()) for o in opts if isinstance(o, str)]
        core.append("both")
        alt = "|".join(core)
    return re.compile(rf"({alt})", flags=re.IGNORECASE)

def _line_regex(word: str, label_pat: re.Pattern) -> re.Pattern:
    # separators allowed: { ':', '-', '–', '—', or ≥1 space }
    # wraps: { (), [], {} } 
    prefix = r"(?:\s*(?:[\-\*\u2022•]|\d{1,3}[.)]?)\s*[:\-–—.]?\s*)?"
    sep = r"(?::|\-|\–|—|,|\s{1,})"
    open_wrap = r"(?:[\(\[\{]\s*)?"
    close_wrap = r"(?:\s*[\)\]\}])?"
    trail = r"[\s\.,;:!\?-–—]*"
    return re.compile(
        rf"^\s*{prefix}{re.escape(word)}\s*{sep}\s*{open_wrap}{label_pat.pattern}{close_wrap}{trail}$",
        flags=re.IGNORECASE,
    )

def _line_parts_regex(label_pat: re.Pattern) -> re.Pattern:
    sep = r"(?::|\-|\–|—|,|\s{1,})"
    open_wrap = r"(?:[\(\[\{]\s*)?"
    close_wrap = r"(?:\s*[\)\]\}])?"
    trail = r"[\s\.,;:!\?-–—]*"
    return re.compile(
        rf"^\s*(.*?)\s*{sep}\s*{open_wrap}({label_pat.pattern}){close_wrap}{trail}$",
        flags=re.IGNORECASE,
    )

def _multi_pairs_regex(label_pat: re.Pattern) -> re.Pattern:
        label = f"({label_pat.pattern})"
        pat = rf"\s*(?:(?P<w1>[^,(){{}}\[\]]+?)\s*\(\s*(?P<lab1>{label})\s*\)|(?P<w2>[^,(){{}}\[\]]+?)\s*,\s*(?P<lab2>{label}))\s*"
        return re.compile(pat, flags=re.IGNORECASE)

def parse_llm_answers(entries, df_words, language: str, mask: np.ndarray):
    words_col = df_words[language].astype(str)
    id2word = {i: words_col.iat[i].strip() for i in range(len(words_col))}
    rows = []

    for entry in entries:
        if entry.get("language") != language:
            continue
        txt = (entry.get("response") or "").strip()
        
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        label_pat = _labels_pattern(entry)
        wow = entry.get("with_or_without_none", "")
        model = entry.get("model", "")
        temp  = entry.get("temperature", None)
        pid   = entry.get("prompt_id", None)

        parts_re = _line_parts_regex(label_pat)
        multi_re = _multi_pairs_regex(label_pat)
        cand2label: dict[str, str] = {}
        for ln in lines:
            m = parts_re.match(ln)
            if m:
                cand = (m.group(1) or "").strip().lower()
                raw_lab = m.group(2)
                canon = _canon_label(raw_lab, language)
                if canon is not None:
                    cand2label[cand] = canon
            pos = 0
            while True:
                mm = multi_re.search(ln, pos)
                if not mm:
                    break
                w = (mm.group('w1') or mm.group('w2') or '').strip().lower()
                lab = mm.group('lab1') or mm.group('lab2')
                canon = _canon_label(lab, language)
                if w and canon is not None:
                    cand2label[w] = canon
                pos = mm.end()

        for idx in entry.get("word_ids", []):
            if idx is None or idx not in id2word or not mask[idx]:
                continue
            word = id2word[idx]

            found_label = None
            line_pat = _line_regex(word, label_pat)
            for ln in lines:
                m = line_pat.match(ln)
                if m:
                    found_label = m.group(1)
                    break

            if found_label is None:
                found_label = cand2label.get(word.strip().lower())

            canon = _canon_label(found_label, language)
            if canon is None:
                if wow == "with_none":
                    canon = "none"
                else:
                    continue

            rows.append((model, temp, wow, pid, language, idx, word, canon))

    out = pd.DataFrame(rows, columns=[
        "model","temperature","with_or_without_none","prompt_id",
        "language","word_idx","word","label"
    ])
    return out