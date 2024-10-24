from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from collections import defaultdict, deque
from copy import deepcopy
from importlib.metadata import entry_points
import json
import os
from pathlib import Path
import random
import re
from typing import Dict, List
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np

from kb_interface.freebase_func import (
    convert_id_to_name,
)

import sys
from llms.interface import run_llm

from utils import format_prompt, proxy_load_dataset, read_file, load_json
from datasets import load_dataset

sys.path.append("src")

SPARQLPATH = os.environ['SPARQLPATH']
sparql = SPARQLWrapper(SPARQLPATH)
sparql.setReturnFormat(JSON)
ns_prefix = "http://rdf.freebase.com/ns/"


def remove_ns_prefix(id):
    if id.startswith("ns:"):
        return id[3:]
    else:
        return id


def _random_select_crucial_triple(triples, k=1) -> List:
    ent_triples = []
    for triple in triples:
        if triple[0].startswith("ns:") and triple[-1].startswith("ns:"):
            ent_triples.append(triple)
    return random.sample(ent_triples, k=k)


def convert_path_to_edge_set(path):
    edges = []
    for i in range(len(path) - 1):
        edges.append((path[i], path[i + 1]))
    return set(edges)

def convert_path_to_edge_list(path):
    edges = []
    for i in range(len(path) - 1):
        edges.append((path[i], path[i + 1]))
    return edges

def find_all_paths_with_relations(triples, start, end):
    def find_path(graph, current_entity, target_entity, current_path=None):
        if current_path is None:
            current_path = []

        current_path = current_path + [(current_entity, graph[current_entity])]

        if current_entity == target_entity:
            return [current_path]

        if current_entity not in graph:
            return []

        paths = []
        for neighbor, relation in graph[current_entity]:
            if neighbor not in [node[0] for node in current_path]:
                new_paths = find_path(graph, neighbor, target_entity, current_path)
                for p in new_paths:
                    paths.append(p)

        return paths

    graph = {}
    for triple in triples:
        start_entity, relation, end_entity = triple
        if start_entity not in graph:
            graph[start_entity] = []
        graph[start_entity].append((end_entity, relation))

    all_paths = find_path(graph, start, end)

    return all_paths


def get_topic_entities(sample: dict):
    ids = re.findall(r"ns:([m|g]\.[\w.:]+)", sample["sparql"])
    ids = sorted(list(set(ids)))

    entities = []
    label_to_id = {}
    for id in ids:
        label = convert_id_to_name(id)
        entities.append(label)

        label_to_id[label] = id

    prompt_path = "prompts_v2/primitive_tasks/extract_entity"
    prompt = read_file(prompt_path)
    prompt = format_prompt(prompt)

    input = prompt + f"Q: {sample['question']}\nA: "
    response = run_llm(input, engine="gpt-3.5-turbo-0613")
    llm_ents = eval(response)
    conj_ents = set()
    for ent in llm_ents:
        for true_ent in entities:
            if ent.lower() in true_ent.lower() or true_ent.lower() in ent.lower():
                conj_ents.add(true_ent)

    topic_entities = {label_to_id[label]: label for label in conj_ents}
    return topic_entities


def get_crucial_edges_in_sparql(
    data_path="./data/cwq_dev.json", n_considered_source_nodes=1, drop_prob=0.1
):
    with open(data_path, "r") as f:
        samples = json.load(f)
    print(len(samples))

    for sample in tqdm(samples):
        topic_mids = [mid for mid, label in sample["topic_entity"].items()]

        if len(topic_mids) == 0:
            print(sample)

        if "sparql" in sample:
            sparql_query = sample["sparql"]
        else:
            sparql_query = sample["Parses"][0]["Sparql"]
        lines = sparql_query.split("\n")

        triples = re.findall(
            r"(\?\w+|ns:[\w.:]+)\s+(ns:[\w.:]+)\s+(\?\w+|ns:[\w.:]+)", sparql_query
        )
        triples = [list(triple) for triple in triples]

        variables = re.findall(r"(\?\w+)", sparql_query)
        variables = sorted(list(set(variables)))

        topic_ents = set()
        all_rels = set()
        for triple in triples:
            all_rels.add(triple[1])
            if triple[0].startswith('ns:'):
                topic_ents.add(triple[0])
            if triple[-1].startswith('ns:'):
                topic_ents.add(triple[0])
        sample['topic_ents'] = list(topic_ents)
        sample['all_rels'] = list(all_rels)
           
        for i, line in enumerate(lines):
            if line.startswith("SELECT DISTINCT"):
                parts = line.split()
                parts = parts[:2] + variables
                lines[i] = " ".join(parts)
            # elif "LIMIT 1" in line:
            #     lines[i] = ""

        sparql_query = "\n".join(lines)

        sparql.setQuery(sparql_query)
        results = sparql.query().convert()

        try:
            all_bound_triples = []
            all_topic_node_to_path = []
            for binding in results["results"]["bindings"]:
                ans = binding["x"]["value"].replace(ns_prefix, "")
                bound_triples = []
                for triple in triples:
                    bound_triple = triple.copy()

                    if triple[0].startswith("?") and triple[0][1:] in binding:
                        bound_triple[0] = binding[triple[0][1:]]["value"].replace(ns_prefix, "ns:")

                    if triple[-1].startswith("?") and triple[-1][1:] in binding:
                        bound_triple[-1] = binding[triple[-1][1:]]["value"].replace(
                            ns_prefix, "ns:"
                        )

                    # remove all ns prefix:
                    for i in range(3):
                        bound_triple[i] = remove_ns_prefix(bound_triple[i])

                    if bound_triple[0].startswith("?") or bound_triple[-1].startswith("?"):
                        # ignore unbound triple without bound var
                        continue

                    bound_triples.append(bound_triple)

                all_bound_triples.extend(bound_triples)
            
            crucial_edges = set()
            for triple in all_bound_triples:
                h, t = triple[0], triple[-1]
                if h[:2] in ('m.', 'g.') and t[:2] in ('m.', 'g.'):
                    if random.random() < drop_prob:
                        edge = sorted([h, t])
                        crucial_edges.add(tuple(edge))
            crucial_edges = list(crucial_edges)
            
        except Exception as e:
            # in some samples, answer entity and topic entity are the same
            crucial_edges = []
            print(e)
            print(all_topic_node_to_path)
            
        sample["crucial_edges"] = crucial_edges

    return samples


def generate_crucial_triples(filepath, n_considered_source_nodes=1, drop_prob=0.1):
    samples = get_crucial_edges_in_sparql(
        filepath, n_considered_source_nodes=n_considered_source_nodes, drop_prob=drop_prob
    )

    samples_with_crucial_edges = []
    for sample in samples:
        crucial_edges = sample.pop("crucial_edges")
        # print(crucial_edges)
        crucial_triples = []

        # also consider their inversion ralation
        for edge in crucial_edges:
            triples = get_triples_by_edges(edge[0], edge[-1])
            crucial_triples.extend(triples)

        sample["mid_crucial_triples"] = crucial_triples
        samples_with_crucial_edges.append(sample)

    return samples_with_crucial_edges


def convert_triples_to_str(triples):
    str_triples = []
    for i, triple in enumerate(triples):
        triple[0] = "ns:" + triple[0]
        triple[1] = "ns:" + triple[1]
        triple[2] = "ns:" + triple[2]
        str_triples.append(" ".join(triple) + " .")
    return "\n".join(str_triples)


def delete_triples_from_kb(triples):
    print(len(triples))
    chunk_size = 100
    for i in range(0, len(triples), chunk_size):
        batch_triples = triples[i : i + chunk_size]
        __delete_triples_from_kb(batch_triples)


def __delete_triples_from_kb(triples):
    str_triples = convert_triples_to_str(triples)
    query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    DELETE DATA
    {{
        GRAPH  <http://freebase.com>
        {{
            {str_triples}
        }}
    }}
    """.format(
        str_triples=str_triples
    )
    sparql.setQuery(query)
    results = sparql.query().convert()
    print(results["results"]["bindings"])


def insert_triples_into_kb(triples):
    print(len(triples))
    chunk_size = 100
    for i in range(0, len(triples), chunk_size):
        batch_triples = triples[i : i + chunk_size]
        __insert_triples_into_kb(batch_triples)


def __insert_triples_into_kb(triples):
    str_triples = convert_triples_to_str(triples)
    query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    INSERT DATA
    {{
        GRAPH  <http://freebase.com>
        {{
            {str_triples}
        }}
    }}
    """.format(
        str_triples=str_triples
    )
    sparql.setQuery(query)
    results = sparql.query().convert()
    print(results["results"]["bindings"])


def get_triples_by_edges(head_id, tail_id):
    triples = set()

    query1 = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?r1 WHERE {{
        ns:{head} ?r1 ns:{tail} .
        FILTER regex(?r1, "http://rdf.freebase.com/ns/")
    }}
    """.format(
        head=head_id, tail=tail_id
    )

    sparql.setQuery(query1)
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        r1 = result["r1"]["value"].replace(ns_prefix, "")

        triples.add((head_id, r1, tail_id))

    query2 = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?r1 WHERE {{
        ns:{tail} ?r1 ns:{head} .
        FILTER regex(?r1, "http://rdf.freebase.com/ns/")
    }}
    """.format(
        head=head_id, tail=tail_id
    )

    sparql.setQuery(query2)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        r1 = result["r1"]["value"].replace(ns_prefix, "")

        triples.add((tail_id, r1, head_id))

    return [list(triple) for triple in triples]


def sort_samples(filepath="data/webqsp/webqsp.json"):
    samples = load_json(filepath)
    samples = sorted(samples, key=lambda x: int(x["QuestionId"][9:]))

    with open(filepath, "w") as f:
        json.dump(samples, f, indent=2)

def convert_id_to_name_in_triples(triples, return_map=False):
    id_to_label = {}
    for triple in triples:
        for i in [0, -1]:
            ent_id = triple[i]
            if ent_id[:2] in ["m.", "g."]:
                if ent_id not in id_to_label:
                    id_to_label[ent_id] = convert_id_to_name(ent_id)
                triple[i] = id_to_label[ent_id]

    if return_map:
        return triples, id_to_label
    else:
        return triples

if __name__ == "__main__":
    k = 1000
    # dataset = "cwq"
    dataset = "webqsp"

    if dataset == "cwq":
        original_filepath = "data/cwq/cwq.json"
        sample_filepath = f"data/cwq/cwq_{k}.json"
        processed_samples = proxy_load_dataset("rmanluo/RoG-cwq", split="test")
    elif dataset == "webqsp":
        original_filepath = "data/webqsp/webqsp.json"
        sample_filepath = f"data/webqsp/webqsp_{k}.json"
        processed_samples = proxy_load_dataset("rmanluo/RoG-webqsp", split="test")

    question_to_idx = {}
    for i, sample in enumerate(processed_samples):
        question_to_idx[sample["question"]] = i
    print("processed_samples", len(question_to_idx))

    if os.path.exists(sample_filepath):
        with open(sample_filepath, "r") as f:
            samples = json.load(f)
    else:
        with open(original_filepath, "r") as f:
            samples = json.load(f)

        sample_idxes = list(range(k))
        samples = [samples[i] for i in sample_idxes]

        with open(sample_filepath, "w") as f:
            json.dump(samples, f, indent=2)

    for n_considered_source_nodes in [-1]:
        drop_probs = [0, 0.2, 0.4, 0.6, 0.8]

        for drop_prob in drop_probs:
            samples_with_crucial_triples = generate_crucial_triples(
                sample_filepath, n_considered_source_nodes, drop_prob=drop_prob
            )
            print(len(samples_with_crucial_triples))

            # update answers and assign new idx from 0
            for i, sample in enumerate(samples_with_crucial_triples):
                if "question" in sample:
                    question = sample["question"]
                else:
                    question = sample["ProcessedQuestion"]

                idx = question_to_idx.get(question, -1)
                if idx == -1:
                    print("not in processed_samples, continue")
                    continue

                sample["answer"] = processed_samples[idx]["answer"]
                sample["idx_in_processed_samples"] = idx
                sample["index"] = i

            # filter samples that not in processed_samples
            samples_with_crucial_triples = [
                sample
                for sample in samples_with_crucial_triples
                if "idx_in_processed_samples" in sample
            ]

            mid_crucial_triples = []

            # convert id to label
            for sample in samples_with_crucial_triples:
                mid_crucial_triples.extend(deepcopy(sample["mid_crucial_triples"]))
                sample["crucial_triples"] = convert_id_to_name_in_triples(
                    deepcopy(sample["mid_crucial_triples"])
                )
            print(len(mid_crucial_triples) / len(samples_with_crucial_triples))

            output_filepath = (
                Path(sample_filepath).parent
                / f"data_with_ct_{drop_prob}.json"
            )
            
            with open(output_filepath, "w") as f:
                json.dump(samples_with_crucial_triples, f, indent=1)

    