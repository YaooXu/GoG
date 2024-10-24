from pathlib import Path
import requests
from prompt_list import *
import json
import time
import openai
import re
import sys

sys.path.append("src")
# from sentence_transformers import util
# from sentence_transformers import SentenceTransformer
import os
from datasets import load_dataset


def set_environment_variable(func):
    def wrapper(*args, **kwargs):
        if "custom_proxy" in os.environ:
            os.environ["http_proxy"] = os.environ["custom_proxy"]
            os.environ["https_proxy"] = os.environ["custom_proxy"]

            result = func(*args, **kwargs)

            del os.environ["http_proxy"]
            del os.environ["https_proxy"]
        else:
            result = func(*args, **kwargs)

        return result

    return wrapper


@set_environment_variable
def proxy_load_dataset(*args, **kwargs):
    return load_dataset(*args, **kwargs)


def remove_ns(entity_id):
    if entity_id.startswith("ns:"):
        entity_id = entity_id[3:]
    return entity_id


def add_ns(entity_id):
    if not entity_id.startswith("ns:"):
        entity_id = "ns:" + entity_id
    return entity_id


def convert_triples_to_str(triples, sep=", "):
    for i, triple in enumerate(triples):
        triple = [remove_ns(item) for item in triple]
        triples[i] = sep.join(triple)
    return "\n".join(triples)


def shorten_relation(rel):
    rel = remove_ns(rel)
    return ".".join(rel.split(".")[-2:])


def read_file(filepath: str):
    with open(filepath, "r") as f:
        content = f.read()
    return content


def load_json(filepath: str):
    with open(filepath, "r") as f:
        content = json.load(f)
    return content


def load_jsonl(filepath: str):
    datas = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            content = json.loads(line)
            datas.append(content)
    return datas


def format_prompt(prompt: str, n_new_line=2):
    return prompt.rstrip("\n") + "\n" * n_new_line


def extract_task(output):
    pattern = r"\[(.*?)\]"
    result = re.findall(pattern, output)
    if result:
        return result[0]
    else:
        return None


def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations = []
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ";" in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append(
                {"entity": entity_id, "relation": relation, "score": score, "head": True}
            )
        else:
            relations.append(
                {"entity": entity_id, "relation": relation, "score": score, "head": False}
            )
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1 / len(topn_scores))] * len(topn_scores)
    i = 0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append(
                {"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True}
            )
        else:
            relations.append(
                {"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False}
            )
        i += 1
    return True, relations


def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates) == 1 and entity_candidates[0] == "UnName_Entity":
        return entity_candidates
    entity_candidates = [
        candidate for candidate in entity_candidates if candidate != "UnName_Entity"
    ]
    return entity_candidates


def clean_scores(string, entity_candidates):
    scores = re.findall(r"\d+\.\d+", string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1 / len(entity_candidates)] * len(entity_candidates)


def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
    dict = {"question": question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    with open("ToG_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")


def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index + 1 : end_index].strip()
    else:
        return ""


def if_true(prompt):
    if prompt.lower().strip().replace(" ", "") == "yes":
        return True
    return False


def generate_without_explored_paths(question, args):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(
        prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type
    )
    return response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def prepare_dataset(dataset_name):
    if dataset_name.startswith("cwq"):
        with open(f"./data/{dataset_name}.json", encoding="utf-8") as f:
            datas = json.load(f)
        question_string = "question"
    elif dataset_name.startswith("webqsp"):
        with open(f"./data/{dataset_name}.json", encoding="utf-8") as f:
            datas = json.load(f)
        question_string = "RawQuestion"
    elif dataset_name == "grailqa":
        with open("./data/grailqa.json", encoding="utf-8") as f:
            datas = json.load(f)
        question_string = "question"
    elif dataset_name == "simpleqa":
        with open("./data/SimpleQA.json", encoding="utf-8") as f:
            datas = json.load(f)
        question_string = "question"
    elif dataset_name == "qald":
        with open("./data/qald_10-en.json", encoding="utf-8") as f:
            datas = json.load(f)
        question_string = "question"
    elif dataset_name == "webquestions":
        with open("./data/WebQuestions.json", encoding="utf-8") as f:
            datas = json.load(f)
        question_string = "question"
    elif dataset_name == "trex":
        with open("./data/T-REX.json", encoding="utf-8") as f:
            datas = json.load(f)
        question_string = "input"
    elif dataset_name == "zeroshotre":
        with open("./data/Zero_Shot_RE.json", encoding="utf-8") as f:
            datas = json.load(f)
        question_string = "input"
    elif dataset_name == "creak":
        with open("./data/creak.json", encoding="utf-8") as f:
            datas = json.load(f)
        question_string = "sentence"
    else:
        print(
            "dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}."
        )
        exit(-1)
    return datas, question_string


def convert_list_to_str(topic_entity_names, sep=" | "):
    return "[" + sep.join(f"{w}" for w in topic_entity_names) + "]"


def parse_llm_output_to_list(output, sep="|"):
    match = re.search("\[(.*)\]", output)
    if match:
        answers = match.group(1)
        answers = [item.strip() for item in answers.split(sep)]
        return answers
    else:
        # may be truncated
        if "[" in output:
            answers = output[output.index("[") + 1 :]
            answers = [item.strip() for item in answers.split(sep)]
            return answers
        else:
            return None

def convert_jsonl_to_json(jsonl_filepath):
    with open(jsonl_filepath, "r") as f:
        output_datas = [json.loads(line) for line in f.readlines()]

    output_datas = sorted(output_datas, key=lambda x: x["index"])

    jsonl_filepath = Path(jsonl_filepath)
    json_filepath = jsonl_filepath.parent / f"{jsonl_filepath.stem}.json"

    with open(json_filepath, "w") as f:
        json.dump(output_datas, f, indent=2)

    return str(json_filepath)

if __name__ == "__main__":
    print(parse_llm_output_to_list("[Cody Linley, Joe Jonas, Nicholas Braun, Abli"))
