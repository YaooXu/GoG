from collections import defaultdict
import random
import re
import spacy
from loguru import logger
from kb_interface.freebase_interface import (
    convert_id_to_name_in_triples,
    convert_label_to_id,
    get_1hop_triples,
    get_2hop_triples,
)
from utils.bm25 import BM25Retrieve, retrieve_ids_by_labels
from utils.utils import format_prompt, read_file, shorten_relation, convert_triples_to_str
from llms import run_llm


class FreeBaseEnv:
    def __init__(self, args, topic_entities) -> None:
        self.args = args
        self.topic_entities = topic_entities

        self.records = []

        self.triples = []
        self.abbr_rel_to_rel = {}

        self.id_to_label = {}
        self.name_to_id = {}

        self.update_id_to_name(topic_entities)

        self.doc_to_vec = spacy.load("en_core_web_lg")

    def update_name_to_id(self, name_to_id):
        name_to_id = {name.lower(): id for name, id in name_to_id.items()}
        self.name_to_id.update(name_to_id)
        self.id_to_label.update({id: label for label, id in name_to_id.items()})

    def update_id_to_name(self, id_to_name):
        id_to_name = {id: name.lower() for id, name in id_to_name.items()}
        self.id_to_label.update(id_to_name)
        self.name_to_id.update({label: id for id, label in id_to_name.items()})

    def convert_records_to_str(self):
        string = ""
        for record in self.records:
            string += "Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {observation}\n".format(
                i=record["i"],
                action=record["action"],
                thought=record["thought"],
                observation=record["observation"],
            )
        return string

    @property
    def last_thought(self):
        return self.records[-1]["thought"]

    @property
    def last_action(self):
        return self.records[-1]["action"]

    def step(self, action_str=None):
        logger.debug(action_str)

        pattern = r"(\w+)(\[.+\])"
        result = re.match(pattern, action_str)

        action = result.group(1)
        parameter = result.group(2)

        if action == "search":
            if parameter == "[ALL]":
                # repeat last search, but search two-hop
                self.records[-1]["thought"] = self.records[-2]["thought"]
                return self.search(str(self.records[-2]["new_entities"]))
            else:
                return self.search(parameter)
        elif action == "generate":
            return self.generate(parameter)

    def generate(self, thought):
        prompt_path = read_file("prompts2/primitive_tasks/generate_triples")
        prompt = format_prompt(prompt_path)

        sep = "\t"
        prompt = (
            prompt + f"Thought: {self.last_thought}\n"
            f"Known Triples: {convert_triples_to_str(sorted(self.triples), sep=sep)}\n"
            f"Generated Triples: "
        )
        response = run_llm(
            prompt,
            self.args.temperature_exploration,
            self.args.max_length,
            self.args.opeani_api_keys,
            self.args.LLM_type,
            stop=None,
        )

        generated_triples = []
        for line in response.split("\n"):
            try:
                h, r, t = [item.strip() for item in line.split(sep)]
                generated_triples.append([h, r, t])
            except Exception as e:
                logger.debug(e)
                logger.debug(line)

        generated_triples = sorted(generated_triples)

        self.records[-1]["generated_triples"] = generated_triples

        return convert_triples_to_str(generated_triples)

    def search(self, entity_names):
        entity_names = eval(entity_names)

        all_related_triples = []
        for entity_name in entity_names:
            entity_id = self.convert_name_to_id(entity_name)

            triples, relations = get_1hop_triples(entity_id)

            for i in range(len(relations)):
                # only remain the last two parts
                abbr_rel = shorten_relation(relations[i])
                self.abbr_rel_to_rel[abbr_rel] = relations[i]
                relations[i] = abbr_rel

            for i in range(len(triples)):
                abbr_rel = shorten_relation(triples[i][1])
                self.abbr_rel_to_rel[abbr_rel] = triples[i][1]
                triples[i][1] = abbr_rel

            relations = sorted(relations)

            filtered_relations = self.filter_relations(entity_name, relations, self.last_thought)
            logger.debug(f"filtered_relations: {filtered_relations}")

            related_triples = self.sample_triples_by_relation(triples, filtered_relations)

            related_triples, id_to_label = convert_id_to_name_in_triples(related_triples, True)
            self.update_id_to_name(id_to_label)

            all_related_triples.extend(related_triples)

        all_related_triples = sorted(all_related_triples)

        self.records[-1]["triples"] = all_related_triples
        self.records[-1]["entity_names"] = entity_names
        self.records[-1]["one_hop_relations"] = filtered_relations

        new_entities = set()
        for triple in all_related_triples:
            new_entities.add(triple[0])
            new_entities.add(triple[2])
        new_entities -= set(entity_names)
        self.records[-1]["new_entities"] = list(new_entities)

        return convert_triples_to_str(all_related_triples)

    def expand(self):
        # from one-hop to two hop
        entity_names = self.records[-2]["entity_names"]
        one_hop_relations = self.records[-2]["one_hop_relations"]

        all_triples, all_relations = [], []

        for entity_name in entity_names:
            id = self.convert_name_to_id(entity_name)
            triples, relations = get_2hop_triples(
                id, [self.abbr_rel_to_rel[rel] for rel in one_hop_relations]
            )

            all_triples.extend(triples)
            all_relations.extend(relations)

        for i in range(len(all_relations)):
            # only remain the last two parts
            abbr_rel = shorten_relation(all_relations[i])
            self.abbr_rel_to_rel[abbr_rel] = all_relations[i]
            all_relations[i] = abbr_rel

        # drop relations that have been considered
        relations = list(set(all_relations) - set(one_hop_relations))
        relations = sorted(relations)

        two_hop_relations = self.filter_relations(entity_names, relations, self.last_thought)
        self.records[-1]["two_hop_relations"] = two_hop_relations

        for i in range(len(all_triples)):
            abbr_rel = shorten_relation(all_triples[i][1])
            self.abbr_rel_to_rel[abbr_rel] = all_triples[i][1]
            all_triples[i][1] = abbr_rel

        related_triples = self.sample_triples_by_relation(
            all_triples, one_hop_relations + two_hop_relations
        )

        related_triples, id_to_label = convert_id_to_name_in_triples(
            related_triples, return_map=True
        )
        self.update_id_to_name(id_to_label)

        related_triples = sorted(related_triples)
        self.records[-1]["triples"] = related_triples

        return convert_triples_to_str(related_triples)

    def filter_relations(self, entity_name, relations, thought):
        # logger.debug(f"{thought}\n{entity_name}")

        prompt_path = read_file("prompts2/primitive_tasks/filter_relations")
        prompt = format_prompt(prompt_path)
        relations = sorted(relations)

        prompt = (
            prompt + f"Thought: {thought}\n"
            f"Entity: {entity_name}\n"
            f"Relation: {', '.join(relations)}\n"
            f"Answer: "
        )

        filtered_relations = run_llm(
            prompt,
            self.args.temperature_exploration,
            self.args.max_length,
            self.args.opeani_api_keys,
            self.args.LLM_type,
        )

        filtered_relations = [rel.strip() for rel in filtered_relations.split(",")]

        return filtered_relations

    def sample_triples_by_relation(self, triples, filtered_relations):
        # only remain related triples
        relation_to_triples = defaultdict(list)
        for triple in triples:
            relation = triple[1]
            if relation in filtered_relations:
                relation_to_triples[relation].append(triple)

        related_triples = []
        for rel, triples in relation_to_triples.items():
            if len(triples) >= 5:
                relation_to_triples[rel] = random.sample(triples, k=5)
            related_triples.extend(relation_to_triples[rel])
        return related_triples

    def convert_name_to_id(self, entity_name):
        entity_id = None
        if entity_name.startswith("m."):
            entity_id = entity_name
        else:
            if entity_name.lower() in self.name_to_id:
                logger.debug("explored entity")
                entity_id = self.name_to_id[entity_name.lower()]

        if not entity_id:
            if self.records[-1]["i"] == 1:
                logger.debug("first search, select most similar entities from topic entities")
                vec1 = self.doc_to_vec(entity_name)
                mid_to_sim = {}
                for name, id in self.name_to_id.items():
                    vec2 = self.doc_to_vec(name)
                    sim = vec1.similarity(vec2)
                    mid_to_sim[id] = sim
                mids = sorted(mid_to_sim.keys(), key=lambda x: mid_to_sim[x], reverse=True)
                logger.debug(f"{self.id_to_label[mids[0]]}")
                return mids[0]

            else:
                # not in explored graph, may be generated by LLM

                # TODO: default first one
                entity_id = retrieve_ids_by_labels([entity_name], self.last_thought)["ids"][0][0]
                logger.debug(f"generated entity: {entity_id}")

                self.update_name_to_id({entity_name: entity_id})

        return entity_id
