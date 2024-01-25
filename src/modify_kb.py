from collections import defaultdict, deque
from copy import deepcopy
from importlib.metadata import entry_points
import json
from pathlib import Path
import random
import re
from typing import Dict, List
from SPARQLWrapper import SPARQLWrapper, JSON

from kb_interface.freebase_interface import (
    convert_id_to_name_in_triples,
)


SPARQLPATH = "http://210.75.240.139:18890/sparql"
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


def get_crucial_edges(all_topic_node_to_path: List[Dict[str, List]], k=1, n_topic=1) -> List:
    """_summary_

    Args:
        all_topic_node_to_path (List[Dict[str, List]]): _description_
        k (int, optional): num of edges to drop in each path. Defaults to 1.
        n_topic (int, optional): _description_. Defaults to 1.

    Returns:
        List: _description_
    """

    crucial_edges = []

    if n_topic == -1:
        n_topic = len(all_topic_node_to_path[0])

    for topic_node in random.sample(all_topic_node_to_path[0].keys(), n_topic):
        all_path = [
            topic_node_to_path[topic_node]
            for topic_node_to_path in all_topic_node_to_path
            if topic_node in topic_node_to_path
        ]
        if not len(all_path):
            continue

        all_edge_sets = [convert_path_to_edge_set(path) for path in all_path]

        edge_to_cnt = defaultdict(int)
        for edge_set in all_edge_sets:
            for edge in edge_set:
                edge_to_cnt[edge] += 1
        # overlap between multiple grounding paths
        edge_to_cnt = list(sorted(edge_to_cnt.items(), key=lambda x: x[1], reverse=True))

        max_cnt = edge_to_cnt[0][1]
        candidate_edges = [edge for edge, cnt in edge_to_cnt if cnt == max_cnt]
        crucial_edges.extend(random.sample(candidate_edges, min(k, len(candidate_edges))))

        # candidate_edges = [edge for edge, cnt in edge_to_cnt]
        # crucial_edges.extend(candidate_edges)

    return crucial_edges


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


def bfs_shortest_paths(triples, source, destinations):
    graph = defaultdict(list)
    for triple in triples:
        start_entity, relation, end_entity = triple
        if start_entity.startswith("m.") and end_entity.startswith("m."):
            graph[start_entity].append(end_entity)
            graph[end_entity].append(start_entity)

    # Mark the source node as visited
    visited = {source}
    queue = deque([[source]])

    node_to_path = {}

    while queue:
        current_path = queue.popleft()
        current_node = current_path[-1]

        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = current_path + [neighbor]
                queue.append(new_path)

                if neighbor in destinations:
                    node_to_path[neighbor] = new_path

    return node_to_path


def get_crucial_edges_in_sparql(data_path="./data/cwq_dev.json"):
    with open(data_path, "r") as f:
        samples = json.load(f)
    print(len(samples))

    for sample in samples:
        topic_mids = [mid for mid, label in sample["topic_entity"].items()]
        print(topic_mids)

        if len(topic_mids) == 0:
            print(sample)

        sparql_query = sample["sparql"]
        lines = sparql_query.split("\n")

        triples = re.findall(
            r"(\?\w+|ns:[\w.:]+)\s+(ns:[\w.:]+)\s+(\?\w+|ns:[\w.:]+)", sparql_query
        )
        triples = [list(triple) for triple in triples]

        variables = re.findall(r"(\?\w+)", sparql_query)
        variables = sorted(list(set(variables)))

        for i, line in enumerate(lines):
            if line.startswith("SELECT DISTINCT"):
                parts = line.split()
                parts = parts[:2] + variables
                lines[i] = " ".join(parts)
            elif "LIMIT 1" in line:
                lines[i] = ""

        sparql_query = "\n".join(lines)

        sparql.setQuery(sparql_query)
        results = sparql.query().convert()

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
                    bound_triple[-1] = binding[triple[-1][1:]]["value"].replace(ns_prefix, "ns:")

                # remove all ns prefix:
                for i in range(3):
                    bound_triple[i] = remove_ns_prefix(bound_triple[i])

                if bound_triple[0].startswith("?") or bound_triple[-1].startswith("?"):
                    # ignore unbound triple without bound var
                    continue

                bound_triples.append(bound_triple)

            topic_node_to_path = bfs_shortest_paths(bound_triples, ans, topic_mids)

            if len(topic_node_to_path):
                all_topic_node_to_path.append(topic_node_to_path)
                all_bound_triples.append(bound_triples)
        try:
            crucial_edges = get_crucial_edges(all_topic_node_to_path)
        except Exception as e:
            crucial_edges = []
            print(e)
        sample["crucial_edges"] = crucial_edges

    return samples


def generate_crucial_triples(filepath):
    samples = get_crucial_edges_in_sparql(filepath)

    for sample in samples:
        crucial_edges = sample.pop("crucial_edges")
        crucial_triples = []

        # also consider their inversion ralation

        for edge in crucial_edges:
            triples = get_triples_by_edges(edge[0], edge[-1])
            crucial_triples.extend(triples)

        sample["crucial_triples"] = crucial_triples

    return samples


def convert_triples_to_str(triples):
    str_triples = []
    for i, triple in enumerate(triples):
        triple[0] = "ns:" + triple[0]
        triple[1] = "ns:" + triple[1]
        triple[2] = "ns:" + triple[2]
        str_triples.append(" ".join(triple) + " .")
    return "\n".join(str_triples)


def delete_triples_from_kb(triples):
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


if __name__ == "__main__":
    # with open("./data/cwq/ComplexWebQuestions_train.json", "r") as f:
    #     samples = json.load(f)
    # samples = samples[:100]
    # with open("./data/cwq/train_dev.json", "w") as f:
    #     json.dump(samples, f, indent=2)

    # filepath = "./data/cwq_dev.json"
    # samples_with_crucial_triples = generate_crucial_triples(filepath)

    # mid_crucial_triples = []

    # # convert id to label
    # for sample in samples_with_crucial_triples:
    #     mid_crucial_triples.extend(deepcopy(sample["crucial_triples"]))
    #     sample["crucial_triples"] = convert_id_to_label_in_triples(sample["crucial_triples"])

    # output_filepath = "./data/cwq_dev_mid_crucial_triples.json"
    # with open(output_filepath, "w") as f:
    #     json.dump(mid_crucial_triples, f, indent=1)

    # output_filepath = "./data/cwq_dev_samples_with_crucial_triples.json"
    # with open(output_filepath, "w") as f:
    #     json.dump(samples_with_crucial_triples, f, indent=1)

    output_filepath = "./data/cwq_dev_mid_crucial_triples.json"
    with open(output_filepath, "r") as f:
        mid_crucial_triples = json.load(f)

    # delete_triples_from_kb(mid_crucial_triples)
    insert_triples_into_kb(mid_crucial_triples)

    # triples, _ = convert_id_to_label_in_triples(crucial_triples)
    # print(triples_to_str(triples))
