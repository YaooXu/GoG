import re
import sys

from networkx import topological_generations

sys.path.append("./src")
from freebase_func import id2entity_name_or_type

from freebase_interface import convert_id_to_label

import json
from utils.utils import format_prompt, run_llm

with open("./data/cwq/train_dev.json", "r") as f:
    samples = json.load(f)

prompt_path = "prompts/task_primitives/extract_entity"
with open(prompt_path, "r") as f:
    prompt = f.read()
    prompt = format_prompt(prompt)

for sample in samples:
    input = prompt + f"Q: {sample['question']}\nA: "
    response = run_llm(input, engine="gpt-3.5-turbo")

    ids = re.findall(r"ns:(m\.[\w.:]+)", sample["sparql"])
    ids = sorted(list(set(ids)))

    entities = []
    label_to_id = {}
    for id in ids:
        label = convert_id_to_label(id)
        entities.append(label)

        label_to_id[label] = id

    if len(entities) == 1:
        conj_ents = entities
    else:
        llm_ents = eval(response)
        conj_ents = set()
        for ent in llm_ents:
            for true_ent in entities:
                if ent.lower() in true_ent.lower() or true_ent.lower() in ent.lower():
                    conj_ents.add(true_ent)

        if not len(topic_entity):
            print(llm_ents)
            print(entities)
            print()

    topic_entity = {label_to_id[label]: label for label in conj_ents}
    sample["topic_entity"] = topic_entity
    print(topic_entity)


with open("./data/cwq/train_dev.json", "w") as f:
    json.dump(samples, f, indent=2)
