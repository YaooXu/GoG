from typing import List, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import urllib
from pathlib import Path
from numpy import tri
from tqdm import tqdm


sparql = SPARQLWrapper("http://127.0.0.1:18890/sparql")
sparql.setReturnFormat(JSON)
ns_prefix = 'http://rdf.freebase.com/ns/'


def execurte_sparql(sparql_query):
    sparql.setQuery(sparql_query)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/", "") for relation in relations]


def replace_entities_prefix(entities):
    return [entity['entity']['value'].replace("http://rdf.freebase.com/ns/", "") for entity in entities]


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
            triples.extend([[entity_id, relation, tail_entity_id]
                           for tail_entity_id in tail_entity_ids])
        elif relation in in_relations:
            head_entity_ids = get_head_entity(entity_id, relation)
            triples.extend([[head_entity_id, relation, entity_id]
                           for head_entity_id in head_entity_ids])
        else:
            continue

    id_to_lable = {}
    for triple in triples:
        for i in [0, -1]:
            ent_id = triple[i]
            if ent_id not in id_to_lable:
                id_to_lable[ent_id] = id_to_name(ent_id)
            triple[i] = id_to_lable[ent_id]

    return triples, id_to_lable


def convert_id_to_label_in_triples(triples):
    id_to_lable = {}
    for triple in triples:
        for i in [0, -1]:
            ent_id = triple[i]
            if ent_id not in id_to_lable:
                id_to_lable[ent_id] = id_to_name(ent_id)
            triple[i] = id_to_lable[ent_id]

    return triples


def triples_to_str(triples):
    for i, triple in enumerate(triples):
        triple.append('.')
        triples[i] = '  '.join(triple)
    return '\n'.join(triples)


def id_to_name(entity_id):
    if entity_id.startswith('ns:'):
        entity_id = entity_id[3:]
        
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
        return results["results"]["bindings"][0]['tailEntity']['value']


__query_record = {}


def pre_filter_relations(relations: List[str]):
    ignored_relations = [
        'common.topic.image',
        'common.topic.webpage'
    ]
    filtered_relations = []
    for relation in relations:
        if relation in ignored_relations or relation.startswith('freebase'):
            continue
        else:
            filtered_relations.append(relation)
    return filtered_relations


def get_1hop_triples(entity_id):
    triples = set()

    relation_to_new_entity = {}

    query1 = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x1 ?r1 WHERE {{
        ?x1 ?r1 ns:{entity} .
        FILTER regex(?r1, "http://rdf.freebase.com/ns/")
    }}
    """.format(entity=entity_id)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        x1 = result['x1']['value'].replace(ns_prefix, '')
        r1 = result['r1']['value'].replace(ns_prefix, '')
        relation_to_new_entity[r1] = x1

        triples.add((x1, r1, entity_id))

    query2 = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x1 ?r1 WHERE {{
        ns:{entity} ?r1 ?x1 .
        FILTER regex(?r1, "http://rdf.freebase.com/ns/")
    }}
    """.format(entity=entity_id)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        x1 = result['x1']['value'].replace(ns_prefix, '')
        r1 = result['r1']['value'].replace(ns_prefix, '')
        relation_to_new_entity[r1] = x1

        triples.add((entity_id, r1, x1))

    relations = list(set([triple[1] for triple in triples]))
    relations = pre_filter_relations(relations)

    triples = [list(triple) for triple in triples if triple[1] in relations]

    # relation_to_new_entity = {
    #     relation: entity for relation, entity in relation_to_new_entity.items()
    #     if relation in relations
    # }

    return triples, relations


def get_2hop_triples(entity_id: str, one_hop_relations=None):
    # if entity_id in __query_record:
    #     return __query_record[entity_id]
    if one_hop_relations:
        one_hop_relations = ' '.join(
            ['ns:'+relation for relation in one_hop_relations])
        one_hop_relations = f'VALUES ?r2 {{ {one_hop_relations} }} .'
    else:
        one_hop_relations = ''

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
    """.format(entity=entity_id, one_hop_relations=one_hop_relations)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        x1 = result['x1']['value'].replace(ns_prefix, '')
        r1 = result['r1']['value'].replace(ns_prefix, '')
        x2 = result['x2']['value'].replace(ns_prefix, '')
        r2 = result['r2']['value'].replace(ns_prefix, '')

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
    """.format(entity=entity_id, one_hop_relations=one_hop_relations)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        x1 = result['x1']['value'].replace(ns_prefix, '')
        r1 = result['r1']['value'].replace(ns_prefix, '')
        x2 = result['x2']['value'].replace(ns_prefix, '')
        r2 = result['r2']['value'].replace(ns_prefix, '')

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
    """.format(entity=entity_id, one_hop_relations=one_hop_relations)

    sparql.setQuery(query3)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query3)
        exit(0)
    for result in results['results']['bindings']:
        x1 = result['x1']['value'].replace(ns_prefix, '')
        r1 = result['r1']['value'].replace(ns_prefix, '')
        x2 = result['x2']['value'].replace(ns_prefix, '')
        r2 = result['r2']['value'].replace(ns_prefix, '')

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
    """.format(entity=entity_id, one_hop_relations=one_hop_relations)

    sparql.setQuery(query4)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query4)
        exit(0)
    for result in results['results']['bindings']:
        x1 = result['x1']['value'].replace(ns_prefix, '')
        r1 = result['r1']['value'].replace(ns_prefix, '')
        x2 = result['x2']['value'].replace(ns_prefix, '')
        r2 = result['r2']['value'].replace(ns_prefix, '')

        triples.add((x2, r1, x1))
        triples.add((entity_id, r2, x2))

    relations = list(set([triple[1] for triple in triples]))
    relations = pre_filter_relations(relations)

    triples = [list(triple) for triple in triples if triple[1] in relations]

    return triples, relations


if __name__ == "__main__":
    entity_id = 'm.03_dwn'
    triples, relations = get_1hop_triples(entity_id)
    print(len(triples))
    print(len(relations))

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
