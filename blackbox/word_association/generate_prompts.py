import csv
import json
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv
from blackbox.cis_or_trans.utils import read_prompt

load_dotenv()

with open('blackbox/word_association/config.json', 'r') as f:
    config = json.load(f)

OPTION_WORDS_DICT = {
    "english": {
        "with_none": ["cisgender", "transgender", "none"],
        "without_none": ["cisgender", "transgender"],
    },
    "spanish": {
        "with_none": ["cisgénero", "transgénero", "ninguno"],
        "without_none": ["cisgénero", "transgénero"],
    }
}

def join_options(options: list[str], lang: str) -> str:
    if len(options) == 2:
        return f"{options[0]} o {options[1]}" if lang == "spanish" else f"{options[0]} or {options[1]}"
    else:
        if lang == "spanish":
            return f"{options[0]}, {options[1]} o {options[2]}"
        else:
            return f"{options[0]}, {options[1]} or {options[2]}"

def read_words(csv_path: Path):
    per_category_pn = dict()
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            english, spanish, sentiment = row[0].strip(), row[1].strip(), row[2].strip().lower()
            category = row[-1].strip().lower()
            item = {"english": english, "spanish": spanish, "sentiment": sentiment}

            if category not in per_category_pn.keys():
                per_category_pn[category] = {"positive": [], "negative": []}
            if sentiment == "positive":
                per_category_pn[category]["positive"].append(item)
            elif sentiment == "negative":
                per_category_pn[category]["negative"].append(item)
    return per_category_pn

categories = [
    "appearance",
    "danger", 
    "emotions", 
    "fakeness", 
    "knowledge", 
    "legality and crime", 
    "life outcomes",
    "morality"
]

def main():
    ap = argparse.ArgumentParser(description="Sample prompts from CSV: 5 positive + 5 negative words, template id 1..3, repeated N times.")
    ap.add_argument("--csv", type=Path, default=Path("blackbox/word_association/prompts/words_with_categories.csv"), help="Input CSV path (default: words.csv)")
    ap.add_argument("--out", type=Path, default=Path("samples.json"), help="Output JSON path (default: samples.json)")
    ap.add_argument("--n", type=int, default=50, help="Number of prompts to generate (default: 50)")
    ap.add_argument("--k_per_sentiment", type=int, default=5, help="Words per sentiment (default: 5)")
    args = ap.parse_args()
    random.seed(123)

    per_category_pn = read_words(args.csv)

    results = []
    for i in range(1, args.n + 1):
        for lang in config["languages"]:
            for model_name in config["models"]:
                for with_or_without in config["with_or_without_none"]:
                    for category in per_category_pn.keys():
                        prompt_id = random.choice([1, 2, 3])
                        pos_sample = random.sample(per_category_pn[category]["positive"], args.k_per_sentiment)
                        neg_sample = random.sample(per_category_pn[category]["negative"], args.k_per_sentiment)

                        words = pos_sample + neg_sample
                        random.shuffle(words)

                        words_seq = [word[lang] for word in words]

                        opts = OPTION_WORDS_DICT[lang][with_or_without]
                        shuffled_opts = random.sample(opts, k=len(opts))
                        joined_opts = join_options(shuffled_opts, lang)

                        template_path = f"blackbox/cis_or_trans/prompts/{lang}/{prompt_id}.txt"
                        prompt_template = read_prompt(template_path)

                        formatted = prompt_template.format(
                            word_list=", ".join(words_seq),
                            len=len(words_seq),
                            len_labels=len(shuffled_opts),
                            option_words=joined_opts,
                        )
                        entry = {
                            "id": i,
                            "prompt_id": prompt_id,
                            "words": words_seq,
                            "shuffled_options": shuffled_opts,
                            "formatted_prompt": formatted,
                            "temperature": 0,
                            "model": model_name,
                            "language": lang,
                            "with_or_without_none": with_or_without,
                        }
                        results.append(entry)

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} prompts to {args.out}")

if __name__ == "__main__":
    main()
