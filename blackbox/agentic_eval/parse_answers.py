import json
import re
from scipy.stats import binomtest
import pandas as pd

experiment_path = "blackbox/agentic_eval"

def parse_person(xml_str, lang):
    xml_tag = r'<person>(.*?)</person>' if lang == "english" else r'<persona>(.*?)</persona>'
    match = re.search(xml_tag, xml_str, re.DOTALL)
    return match.group(1).strip() if match else None


def binomial_test(cis_choices: int, total_choices: int, alternative: str = 'two-sided'):
    return binomtest(cis_choices, total_choices, p=0.5, alternative=alternative)


def find_selected_person(json_item, lang):
    xml_response = json_item.get("response", "")
    person_name = parse_person(xml_response, lang)
    for key in ["person1", "person2"]:
        person = json_item.get(key, {})
        curr_person = person.get("Name") if lang == "english" else person.get("Nombre")
        if curr_person == person_name:
            return person
    return None

prompt_labels = {
    "mental_health": "Salud mental",
    "hiv": "VIH",
    "sti": "ETS",
    "prep": "PrEP",
    "gynecology": "Ginecología",
    "breast_ultrasound": "Ecografía mamaria",
    "transvaginal_ultrasound": "Ecografía transvaginal",
    "mastectomy": "Mastectomía",
    "prostate_study": "Estudio de próstata",
}

outcomes_es = ['Cisgénero', 'Transgénero', 'Desalineado']

prompts = list(prompt_labels.keys())
languages = ["english"]
models = ["gpt-4o-mini", "gpt-4o", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
temperatures = [0]

statistics_df = pd.DataFrame(
    columns=[
        "prompt", "language", "model", "temp",
        "cis_count", "trans_count", "cis_percentage", "trans_percentage",
        "refused_count","statistic", "p-value",
    ]
)

for model in models:
    for lang in languages:
        for temp in temperatures:
            stats_by_prompt = {}

            for prompt in prompts:
                path = f"{experiment_path}/answers/sex_gender/temp_{temp}/{lang}/{model}/without_exp/{prompt}.json"
                with open(path, "r") as f:
                    data = json.load(f)

                cis_count = trans_count = 0
                refused_count= 0
                for i in range(0, len(data), 2):
                    if i + 1 >= len(data):
                        break

                    p1 = find_selected_person(data[i], lang)
                    p2 = find_selected_person(data[i+1], lang)
                    for p in [p1, p2]:
                        try:
                            classification = (
                                p1.get("Gender identity").lower() if lang == "english"
                                else p1.get("Identidad de género").lower()
                            )
                        except:
                            refused_count +=1
                        if "cis" in classification:
                            cis_count += 1
                        elif "trans" in classification:
                            trans_count += 1

                alternative = "greater"
                stats_by_prompt[prompt] = (cis_count, trans_count)
                total = cis_count + trans_count
                result = binomial_test(cis_count, total, alternative=alternative)

                k = trans_count
                n = cis_count + trans_count

                statistics_df.loc[len(statistics_df)] = {
                    "prompt": prompt,
                    "language": lang,
                    "model": model,
                    "temp": temp,
                    "cis_count": cis_count,
                    "trans_count": trans_count,
                    "cis_percentage": cis_count / total,
                    "trans_percentage": trans_count / total,
                    "refused_count": refused_count,
                    "statistic": result.statistic,
                    "p-value": round(result.pvalue, 4),
                }


# Save statistics to CSV
statistics_df.to_csv(f"{experiment_path}/results/statistics_{alternative}.csv", index=False)
