import json
from tqdm import tqdm
from blackbox.llm_call import call_model
from blackbox.word_association.utils import read_prompt
from blackbox.agentic_eval.map_to_spanish import get_prompt_for_language
from dotenv import load_dotenv
import pandas as pd
import random
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.propagate = False

experiment_path = "blackbox/agentic_eval"
MODEL_VERSIONS = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gemini-1.5-flash": "gemini-1.5-flash-002",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite-001",
}

load_dotenv()
random.seed(123)

results = []

df_people = pd.read_csv(f"{experiment_path}/prompts/people.csv").to_dict(orient="records")

woman_sex_specific = [
    "gynecology",
    "breast_ultrasound",
    "transvaginal_ultrasound",
    "mastectomy",
]

man_sex_specific = ["prostate_study"]

all_people_scenarios = [
    "hiv",
    "mental_health",
    "sti",
    "prep",
]

cis_people = [person for person in df_people if "cis" in person["Gender identity"].lower()]
trans_people = [person for person in df_people if "trans" in person["Gender identity"].lower()]

def pair_key(person1, person2) -> str:
    return person1["Name"] + "|" + person2["Name"]

def unordered_key(person1, person2) -> str:
    n1, n2 = person1["Name"], person2["Name"]
    return "|".join(sorted([n1, n2]))

def safe_write_json(path: str, data: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    os.replace(tmp, path)

###### PARAMS ######
temps = [0.0]
prompts_to_run = all_people_scenarios + woman_sex_specific + man_sex_specific
all_people_scenarios_n = 100
woman_sex_specific_n = 100
models = ["gpt-4o-mini", "gpt-4o","gemini-2.0-flash", "gemini-2.0-flash-lite"]
langs = ["english"]
##### END PARAMS #####


total_iterations = (
    len(temps) * 
    len(models) * 
    len(langs) * 
    (len(all_people_scenarios) * all_people_scenarios_n + len(woman_sex_specific) * woman_sex_specific_n)
)
progress_bar = tqdm(total=total_iterations, desc="Processing", unit="iteration")

for temp in temps:
    for prompt_name in prompts_to_run:
        if prompt_name in woman_sex_specific:
            cis_people_filtered = [person for person in cis_people if "woman" in person["Sex assigned at birth"].lower()]
            trans_people_filtered = [person for person in trans_people if "woman" in person["Sex assigned at birth"].lower()]
            n = woman_sex_specific_n
        elif prompt_name in man_sex_specific:
            cis_people_filtered = [person for person in cis_people if "man" in person["Sex assigned at birth"].lower()]
            trans_people_filtered = [person for person in trans_people if "man" in person["Sex assigned at birth"].lower()]
            n = woman_sex_specific_n
        else:
            cis_people_filtered = cis_people
            trans_people_filtered = trans_people
            n = all_people_scenarios_n
        for lang in langs:
            for model_name in models:
                logger.info(f"Processing {prompt_name} in {lang} with {model_name} at temp {temp}")

                results = []
                prompt = read_prompt(f"{experiment_path}/prompts/{lang}/{prompt_name}.txt")

                out_path = f"{experiment_path}/answers/sex_gender/temp_{str(temp)[0]}/{lang}/{model_name}/without_exp/{prompt_name}.json"

                existing_pairs = set()
                existing_unordered = set()
                if os.path.exists(out_path):
                    try:
                        with open(out_path, "r") as f:
                            existing_results = json.load(f)
                        for item in existing_results:
                            k1 = pair_key(item["person1"], item["person2"])
                            k2 = pair_key(item["person2"], item["person1"])
                            existing_pairs.add(k1)
                            existing_pairs.add(k2)
                            existing_unordered.add(unordered_key(item["person1"], item["person2"]))
                        results = existing_results
                    except Exception:
                        results = []
                else:
                    results = []


                seen_pairs = set(existing_pairs)

                if results:
                    try:
                        pair_keys_present = {
                            pair_key(item["person1"], item["person2"]) for item in results
                        }
                        unordered_to_example = {}
                        for item in results:
                            ukey = unordered_key(item["person1"], item["person2"])
                            if ukey not in unordered_to_example:
                                unordered_to_example[ukey] = item

                        missing_created = 0
                        for ukey, item in unordered_to_example.items():
                            p1 = item["person1"]
                            p2 = item["person2"]
                            k_forward = pair_key(p1, p2)
                            k_back = pair_key(p2, p1)

                            to_create = []
                            if k_forward not in pair_keys_present:
                                to_create.append((p1, p2))
                            if k_back not in pair_keys_present:
                                to_create.append((p2, p1))

                            for pp1, pp2 in to_create:
                                try:
                                    formatted_prompt = get_prompt_for_language(pp1, pp2, prompt, lang)
                                    response = call_model(
                                        prompt=[{"role": "user", "content": formatted_prompt}],
                                        model_name=MODEL_VERSIONS[model_name],
                                        temp=temp,
                                    )
                                except Exception as e:
                                    logger.exception(f"Backfill call failed for missing ordering {pair_key(pp1, pp2)}: {e}")
                                    continue

                                result_item = {
                                    "prompt": formatted_prompt,
                                    "response": response,
                                    "temperature": temp,
                                    "model": model_name,
                                    "person1": pp1,
                                    "person2": pp2,
                                }
                                results.append(result_item)
                                safe_write_json(out_path, results)
                                pair_keys_present.add(pair_key(pp1, pp2))
                                seen_pairs.add(pair_key(pp1, pp2))
                                missing_created += 1
                                progress_bar.update(1)
                        if missing_created:
                            logger.info(f"Backfilled {missing_created} missing orderings in {out_path}")
                    except Exception as e:
                        logger.exception(f"While ensuring both orderings exist, encountered error: {e}")

                with open(f"{experiment_path}/answers/sex_gender/temp_{str(temp)[0]}/{lang}/gpt-4o-mini/without_exp/{prompt_name}.json", "r") as f:
                    without_exp = json.load(f)

                reference_unordered = {unordered_key(dp["person1"], dp["person2"]) for dp in without_exp}

                for dp in without_exp:
                    if len(existing_unordered) >= n:
                        logger.info(f"Reached target of {n} unordered pairs for {prompt_name} in {lang} with {model_name}") 
                        break
                    chosen_cis = dp["person1"]
                    chosen_trans = dp["person2"]

                    if unordered_key(chosen_cis, chosen_trans) in existing_unordered:
                        continue

                    for p1, p2 in [(chosen_cis, chosen_trans), (chosen_trans, chosen_cis)]:
                        key = pair_key(p1, p2)
                        if key in seen_pairs:
                            logger.info(f"Already run pair: {key}")
                            progress_bar.update(1)
                            continue

                        seen_pairs.add(key)

                        try:
                            formatted_prompt = get_prompt_for_language(p1, p2, prompt, lang)
                            response = call_model(prompt=[{"role": "user", "content": formatted_prompt}], model_name=MODEL_VERSIONS[model_name], temp=temp)
                        except Exception as e:
                            logger.exception(f"Call failed for {key}: {e}")
                            continue

                        logger.info(f"Response for {key}:\n{response}")

                        result_item = {
                            "prompt": formatted_prompt,
                            "response": response,
                            "temperature": temp,
                            "model": model_name,
                            "person1": p1,
                            "person2": p2,
                        }
                        results.append(result_item)
                        safe_write_json(out_path, results)
                        progress_bar.update(1)

                    existing_unordered.add(unordered_key(chosen_cis, chosen_trans))

                if len(existing_unordered) < n:
                    pool_cis = list(cis_people_filtered)
                    pool_trans = list(trans_people_filtered)
                    max_attempts = 10000  # avoid infinite loops when the space is too small
                    attempts = 0
                    while len(existing_unordered) < n and attempts < max_attempts:
                        attempts += 1
                        p_cis = random.choice(pool_cis)
                        p_trans = random.choice(pool_trans)
                        ukey = unordered_key(p_cis, p_trans)
                        if ukey in existing_unordered or ukey in reference_unordered:
                            continue

                        for p1, p2 in [(p_cis, p_trans), (p_trans, p_cis)]:
                            key = pair_key(p1, p2)
                            if key in seen_pairs:
                                continue
                            seen_pairs.add(key)
                            try:
                                formatted_prompt = get_prompt_for_language(p1, p2, prompt, lang)
                                response = call_model(prompt=[{"role": "user", "content": formatted_prompt}], model_name=MODEL_VERSIONS[model_name], temp=temp)
                            except Exception as e:
                                logger.exception(f"Call failed for sampled pair {key}: {e}")
                                continue

                            logger.info(f"Response for sampled {key}:\n{response}")

                            result_item = {
                                "prompt": formatted_prompt,
                                "response": response,
                                "temperature": temp,
                                "model": model_name,
                                "person1": p1,
                                "person2": p2,
                            }
                            results.append(result_item)
                            safe_write_json(out_path, results)
                            progress_bar.update(1)

                        existing_unordered.add(ukey)

                    if attempts >= max_attempts and len(existing_unordered) < n:
                        logger.warning(
                            f"Stopped sampling early after {attempts} attempts; could not reach target of {n} unordered pairs without repeating reference or existing pairs."
                        )

progress_bar.close()