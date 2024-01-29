from typing import List, Tuple, Union
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import urllib
from pathlib import Path
from numpy import tri
from tqdm import tqdm

from utils.utils import add_ns, remove_ns


sparql = SPARQLWrapper("http://210.75.240.139:18890/sparql")
sparql.setReturnFormat(JSON)
ns_prefix = "http://rdf.freebase.com/ns/"


def execurte_sparql(sparql_query):
    sparql.setQuery(sparql_query)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    return [
        relation["relation"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for relation in relations
    ]


def replace_entities_prefix(entities):
    return [
        entity["entity"]["value"].replace("http://rdf.freebase.com/ns/", "") for entity in entities
    ]


def get_tail_entity(entity_id, relation):
    sparql_pattern = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?entity
    WHERE {{
        ns:{head} ns:{relation} ?entity .
    }}
    """
    sparql_text = sparql_pattern.format(head=entity_id, relation=relation)
    entities = execurte_sparql(sparql_text)

    entities = replace_entities_prefix(entities)
    entity_ids = [entity for entity in entities if entity.startswith("m.")]

    return entity_ids


def get_head_entity(entity_id, relation):
    sparql_pattern = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?entity
    WHERE {{
        ?entity ns:{relation} ns:{tail}  .
    }}
    """
    sparql_text = sparql_pattern.format(tail=entity_id, relation=relation)
    entities = execurte_sparql(sparql_text)

    entities = replace_entities_prefix(entities)
    entity_ids = [entity for entity in entities if entity.startswith("m.")]

    return entity_ids


def get_out_relations(entity_id):
    sparql_pattern = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?relation
    WHERE {{
        ns:{head} ?relation ?x .
    }}
    """
    sparql_text = sparql_pattern.format(head=entity_id)
    relations = execurte_sparql(sparql_text)

    relations = replace_relation_prefix(relations)

    return relations


def get_in_relations(entity_id):
    sparql_pattern = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?relation
    WHERE {{
        ?x ?relation ns:{tail} .
    }}"""
    sparql_text = sparql_pattern.format(tail=entity_id)
    relations = execurte_sparql(sparql_text)

    relations = replace_relation_prefix(relations)

    return relations


def get_ent_triples_by_rel(entity_id, filtered_relations):
    out_relations = get_out_relations(entity_id)
    out_relations = list(set(out_relations))

    in_relations = get_in_relations(entity_id)
    in_relations = list(set(in_relations))

    triples = []
    for relation in filtered_relations:
        if relation in out_relations:
            tail_entity_ids = get_tail_entity(entity_id, relation)
            triples.extend(
                [[entity_id, relation, tail_entity_id] for tail_entity_id in tail_entity_ids]
            )
        elif relation in in_relations:
            head_entity_ids = get_head_entity(entity_id, relation)
            triples.extend(
                [[head_entity_id, relation, entity_id] for head_entity_id in head_entity_ids]
            )
        else:
            continue

    id_to_lable = {}
    for triple in triples:
        for i in [0, -1]:
            ent_id = triple[i]
            if ent_id not in id_to_lable:
                id_to_lable[ent_id] = convert_id_to_name(ent_id)
            triple[i] = id_to_lable[ent_id]

    return triples, id_to_lable


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


def convert_id_to_name(entity_id):
    entity_id = remove_ns(entity_id)

    sparql_id = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?tailEntity
    WHERE {{
        {{
            ?entity ns:type.object.name ?tailEntity .
            FILTER(?entity = ns:%s)
        }}
        UNION
        {{
            ?entity ns:common.topic.alias ?tailEntity .
            FILTER(?entity = ns:%s)
        }}
    }}
    """
    sparql_query = sparql_id % (entity_id, entity_id)

    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if len(results["results"]["bindings"]) == 0:
        return entity_id
    else:
        return results["results"]["bindings"][0]["tailEntity"]["value"]


def convert_label_to_id(label):
    sparql_id = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?entity
    WHERE {{
        {{
            ?entity ns:type.object.name "%s"@en .
        }}
        UNION
        {{
            ?entity ns:common.topic.alias "%s"@en .
        }}
    }}
    """
    sparql_query = sparql_id % (label, label)

    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if len(results["results"]["bindings"]) == 0:
        return None
    else:
        return results["results"]["bindings"][0]["entity"]["value"].replace(ns_prefix, "")


__query_record = {}


def pre_filter_relations(relations: List[str]):
    ignored_relations = [
        "type.object.type",
        "type.object.name",
    ]
    filtered_relations = []
    for relation in relations:
        if (
            relation in ignored_relations
            or relation.startswith("freebase.")
            or relation.startswith("common.")
            or relation.startswith("kg.")
        ):
            continue
        else:
            filtered_relations.append(relation)
    return filtered_relations


def get_1hop_triples(entity_ids: Union[str, List]):
    if type(entity_ids) is str:
        entity_ids = [entity_ids]

    entity_ids = [add_ns(entity_id) for entity_id in entity_ids]

    triples = set()

    query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?mid ?subject ?predicate ?object WHERE {{
        VALUES ?mid {{ {entity_ids} }}
        {{ ?subject ?predicate ?mid }}
        UNION
        {{ ?mid ?predicate ?object }}
        FILTER regex(?predicate, "http://rdf.freebase.com/ns/")
    }}
    """.format(
        entity_ids=" ".join(entity_ids)
    )

    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)

    for result in results["results"]["bindings"]:
        mid = result["mid"]["value"].replace(ns_prefix, "")
        predicate = result["predicate"]["value"].replace(ns_prefix, "")
        if "subject" in result:
            subject = result["subject"]["value"].replace(ns_prefix, "")
            triples.add((subject, predicate, mid))
        elif "object" in result:
            object = result["object"]["value"].replace(ns_prefix, "")
            triples.add((mid, predicate, object))

    relations = list(set([triple[1] for triple in triples]))
    relations = pre_filter_relations(relations)

    triples = [list(triple) for triple in triples if triple[1] in relations]

    return triples, relations


def get_2hop_triples(entity_id: str, one_hop_relations=None):
    entity_id = remove_ns(entity_id)
    one_hop_relations = [remove_ns(rel) for rel in one_hop_relations]
    one_hop_relations = pre_filter_relations(one_hop_relations)

    if one_hop_relations:
        one_hop_relations = " ".join(["ns:" + relation for relation in one_hop_relations])
        one_hop_relations = f"VALUES ?r2 {{ {one_hop_relations} }} ."
    else:
        one_hop_relations = ""

    triples = set()
    query1 = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x1 ?r1 ?x2 ?r2 WHERE {{
        ?x1 ?r1 ?x2 .
        ?x2 ?r2 ns:{entity} .
        {one_hop_relations}
        FILTER regex(?r1, "http://rdf.freebase.com/ns/")
        FILTER regex(?r2, "http://rdf.freebase.com/ns/")
    }}
    """.format(
        entity=entity_id, one_hop_relations=one_hop_relations
    )

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results["results"]["bindings"]:
        x1 = result["x1"]["value"].replace(ns_prefix, "")
        r1 = result["r1"]["value"].replace(ns_prefix, "")
        x2 = result["x2"]["value"].replace(ns_prefix, "")
        r2 = result["r2"]["value"].replace(ns_prefix, "")

        triples.add((x1, r1, x2))
        triples.add((x2, r2, entity_id))

    # print(len(triples))

    query2 = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x1 ?r1 ?x2 ?r2 WHERE {{
        ?x2 ?r1 ?x1 .
        ?x2 ?r2 ns:{entity} .
        {one_hop_relations}
        FILTER regex(?r1, "http://rdf.freebase.com/ns/")
        FILTER regex(?r2, "http://rdf.freebase.com/ns/")
    }}
    """.format(
        entity=entity_id, one_hop_relations=one_hop_relations
    )

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results["results"]["bindings"]:
        x1 = result["x1"]["value"].replace(ns_prefix, "")
        r1 = result["r1"]["value"].replace(ns_prefix, "")
        x2 = result["x2"]["value"].replace(ns_prefix, "")
        r2 = result["r2"]["value"].replace(ns_prefix, "")

        triples.add((x2, r1, x1))
        triples.add((x2, r2, entity_id))

    # print(len(triples))

    query3 = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x1 ?r1 ?x2 ?r2 WHERE {{
        ?x1 ?r1 ?x2 .
        ns:{entity} ?r2 ?x2 .
        {one_hop_relations}
        FILTER regex(?r1, "http://rdf.freebase.com/ns/")
        FILTER regex(?r2, "http://rdf.freebase.com/ns/")
    }}
    """.format(
        entity=entity_id, one_hop_relations=one_hop_relations
    )

    sparql.setQuery(query3)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query3)
        exit(0)
    for result in results["results"]["bindings"]:
        x1 = result["x1"]["value"].replace(ns_prefix, "")
        r1 = result["r1"]["value"].replace(ns_prefix, "")
        x2 = result["x2"]["value"].replace(ns_prefix, "")
        r2 = result["r2"]["value"].replace(ns_prefix, "")

        triples.add((x1, r1, x2))
        triples.add((entity_id, r2, x2))

    # print(len(triples))

    query4 = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x1 ?r1 ?x2 ?r2 WHERE {{
        ?x2 ?r1 ?x1 .
        ns:{entity} ?r2 ?x2 .
        {one_hop_relations}
        FILTER regex(?r1, "http://rdf.freebase.com/ns/")
        FILTER regex(?r2, "http://rdf.freebase.com/ns/")
    }}
    """.format(
        entity=entity_id, one_hop_relations=one_hop_relations
    )

    sparql.setQuery(query4)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query4)
        exit(0)
    for result in results["results"]["bindings"]:
        x1 = result["x1"]["value"].replace(ns_prefix, "")
        r1 = result["r1"]["value"].replace(ns_prefix, "")
        x2 = result["x2"]["value"].replace(ns_prefix, "")
        r2 = result["r2"]["value"].replace(ns_prefix, "")

        triples.add((x2, r1, x1))
        triples.add((entity_id, r2, x2))

    relations = list(set([triple[1] for triple in triples]))
    relations = pre_filter_relations(relations)

    triples = [list(triple) for triple in triples if triple[1] in relations]

    return triples, relations


def get_types(entity_id: str) -> List[str]:
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX ns: <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {{
    SELECT DISTINCT ?x0  WHERE {{
        ns:{entity_id} ns:type.object.type ?x0 . 
    }}
    }}
    """.format(
        entity_id=entity_id
    )

    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results["results"]["bindings"]:
        rtn.append(result["value"]["value"].replace("http://rdf.freebase.com/ns/", ""))

    return rtn


if __name__ == "__main__":
    entity_id = ["m.0f8l9c", "m.05g2b"]
    # print(convert_id_to_label(entity_id))

    triples, relations = get_1hop_triples(entity_id)
    print(triples)

    # id = convert_label_to_id("Baltimore Ravens")
    # print(id)

    # relations = ['base.popstra.celebrity.dated',
    #              'base.popstra.celebrity.friendship']
    # triples, relations = get_2hop_triples(
    #     entity_id, one_hop_relations=relations)
    # print(len(triples))
    # print(len(relations))

    # relations = ['base.popstra.celebrity.friendship', 'base.popstra.celebrity.dated']
    # one_hop_relations = ' '.join(['ns:'+relation for relation in relations])
    # one_hop_relations = f'VALUES ?r2 {{ {one_hop_relations} }} .'
    # print(one_hop_relations)
