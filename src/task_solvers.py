import re
from kb_interface.freebase_interface import (
    convert_id_to_name_in_triples,
    convert_label_to_id,
    get_2hop_triples,
    get_ent_triples_by_rel,
    get_1hop_triples,
    convert_triples_to_str,
)
from utils.utils import extract_task, format_prompt, run_llm


class Graph:
    def __init__(self) -> None:
        self.triples = []
        self.topic_entities = []
        self.id_to_label = {}
        self.label_to_id = {}
        self.entities_to_explore = []
        self.explored_relations = []
        self.topic_entities = None

    def update_entity_to_id(self, entity_to_id):
        self.label_to_id.update(entity_to_id)
        self.id_to_label.update({id: label for label, id in entity_to_id.items()})

    def update_id_to_entity(self, id_to_entity):
        self.id_to_label.update(id_to_entity)
        self.label_to_id.update({label: id for id, label in id_to_entity.items()})

    def get_triples_str(self):
        return convert_triples_to_str(sorted(self.triples))

    @property
    def mids_to_explore(self):
        return [self.label_to_id[label] for label in self.latest_entities_to_explore]

    @property
    def latest_entities_to_explore(self):
        return self.entities_to_explore[-1]

    @property
    def latest_explored_relations(self):
        return self.explored_relations[-1]


class EntityExtract:
    def __init__(self, task_name, prompt_file, task_name_to_solver=None) -> None:
        self.task_name = task_name
        self.prompt_file = prompt_file
        with open(prompt_file, "r") as f:
            self.prompt = format_prompt(f.read())
        self.task_name_to_solver = task_name_to_solver

        self.prompt_pattern = self.prompt + "Q: {question}\nA: "

    def solve(self, question, data, args, depth, graph):
        records = {"task": self.task_name, "question": question, "history": [], "answer": None}

        prompt = self.prompt_pattern.format(prompt=self.prompt, question=question)

        response = run_llm(
            prompt,
            args.temperature_exploration,
            args.max_length,
            args.opeani_api_keys,
            args.LLM_type,
        )

        records["history"].append(response)
        records["answer"] = response

        return records


class RelationFilter:
    def __init__(self, task_name, prompt_file, task_name_to_solver=None) -> None:
        self.task_name = task_name
        self.prompt_file = prompt_file
        with open(prompt_file, "r") as f:
            self.prompt = format_prompt(f.read())
        self.task_name_to_solver = task_name_to_solver

        self.prompt_pattern = (
            self.prompt + "Q: {question}\nTopic Entity: {entities}\nRelations: {relations}\nA: "
        )

    def solve(self, question, data, args, depth, graph: Graph):
        records = {"task": self.task_name, "question": question, "history": [], "answer": None}

        entities = graph.latest_entities_to_explore
        str_entities = str(entities).replace("'", '"')

        relations = graph.latest_explored_relations
        str_relations = str(relations).replace("'", '"')

        prompt = self.prompt_pattern.format(
            question=question, entities=str_entities, relations=str_relations
        )

        filtered_relations = run_llm(
            prompt,
            args.temperature_exploration,
            args.max_length,
            args.opeani_api_keys,
            args.LLM_type,
        )
        filtered_relations = eval(filtered_relations)

        filtered_relations = filtered_relations[:3]

        records["history"].append(str(filtered_relations))
        records["answer"] = str(filtered_relations)

        return records


class TriplesGenerator:
    def __init__(self, task_name, prompt_file, task_name_to_solver=None) -> None:
        self.task_name = task_name
        self.prompt_file = prompt_file
        with open(prompt_file, "r") as f:
            self.prompt = format_prompt(f.read())
        self.task_name_to_solver = task_name_to_solver

        self.prompt_pattern = "Q: {question}\nKnown Triples:\n{triples}\nGenerated triples:\n"

    def solve(self, question, data, args, depth, graph: Graph):
        records = {"task": self.task_name, "question": question, "answer": None}

        triples = graph.get_triples_str()
        prompt = self.prompt + self.prompt_pattern.format(question=question, triples=triples)

        response = run_llm(
            prompt,
            args.temperature_exploration,
            args.max_length,
            args.opeani_api_keys,
            args.LLM_type,
            stop=None,
        )

        triples = []
        for line in response.split("\n"):
            h, r, t = line.split(",")
            (
                h,
                r,
                t,
            ) = (
                h.strip(),
                r.strip(),
                t.strip(),
            )
            triples.append([h, r, t])

        records["answer"] = triples

        return records


class AnswerFinder:
    def __init__(self, task_name, prompt_file, task_name_to_solver=None) -> None:
        self.task_name = task_name
        self.prompt_file = prompt_file
        with open(prompt_file, "r") as f:
            self.prompt = format_prompt(f.read())
        self.task_name_to_solver = task_name_to_solver

        self.prompt_pattern = "Q: {question}\n"

    def solve(self, question, data, args, depth, graph: Graph):
        records = {"task": self.task_name, "question": question, "answer": None}

        prompt = self.prompt + self.prompt_pattern.format(question=question)

        related_triples = graph.get_triples_str()

        prompt += f"{related_triples}\nA: "

        response = run_llm(
            prompt,
            args.temperature_exploration,
            args.max_length,
            args.opeani_api_keys,
            args.LLM_type,
        )

        records["answer"] = response

        return records


def get_one_more_hop_neighbors(entity_ids, considered_relations=None):
    all_triples, all_relations = [], []

    for id in entity_ids:
        if not considered_relations or not len(considered_relations):
            # get 1 hop triples
            triples, relations = get_1hop_triples(id)
        else:
            triples, relations = get_2hop_triples(id, considered_relations)

        all_triples.extend(triples)
        all_relations.extend(relations)

    return all_triples, all_relations


class OneHopQuerySolver:
    def __init__(self, task_name, prompt_file, task_name_to_solver=None) -> None:
        self.task_name = task_name
        self.prompt_file = prompt_file
        with open(prompt_file, "r") as f:
            self.prompt = format_prompt(f.read())
        self.task_name_to_solver = task_name_to_solver

        self.model_question_regex = re.compile("\[([^\]]+)\] (.*)")

        self.prompt_pattern = self.prompt + "QC: {question}\n"

    def is_end(self, response):
        if "[EOS]" in response:
            return True
        else:
            return False

    def solve(self, question, data, args, depth, graph: Graph):
        records = {
            "task": self.task_name,
            "question": question,
            "history": [],
            "sub_task_to_answer": [],
        }

        # extract_entity
        model = self.task_name_to_solver["extract_entity"]
        sub_record = model.solve(question, data, args, depth + 1, graph)
        records["history"].append(sub_record)
        response = sub_record["answer"]
        entities = eval(response)
        print(" " * depth, entities)

        entity_to_id = {}

        for entity in entities:
            # first consider topic entities
            ground_to_topic_entities = False
            for id, label in graph.topic_entities.items():
                if entity in label or label in entity:
                    entity_to_id[label] = id
                    ground_to_topic_entities = True
                    break

            if ground_to_topic_entities:
                continue
            else:
                id = convert_label_to_id(entity)
                entity_to_id[entity] = id

        graph.update_entity_to_id(entity_to_id)
        graph.entities_to_explore.append(entity_to_id.keys())

        find_answer = False
        considered_relations = []
        for _ in range(1):
            if find_answer:
                break

            triples, relations = get_one_more_hop_neighbors(
                graph.mids_to_explore, considered_relations
            )

            # drop relations that have been considered
            relations = list(set(relations) - set(considered_relations))
            relations = sorted(relations)

            graph.explored_relations.append(relations)

            model = self.task_name_to_solver["filter_relations"]
            sub_record = model.solve(question, data, args, depth + 1, graph)

            records["history"].append(sub_record)
            response = sub_record["answer"]
            considered_relations.extend(eval(response))
            print(" " * depth, "considered relations: ", considered_relations)

            # only remain related triples
            related_triples = [triple for triple in triples if triple[1] in considered_relations]
            related_triples, id_to_label = convert_id_to_name_in_triples(
                related_triples, return_map=True
            )

            graph.update_id_to_entity(id_to_label)
            graph.triples.extend(related_triples)

            for triple in related_triples:
                print(" " * depth, triple)

            # triples generate
            model = self.task_name_to_solver["generate_triples"]
            sub_record = model.solve(question, data, args, depth + 1, graph)
            generated_triples = sub_record["answer"]
            print("generated triples:")
            for i in generated_triples:
                print(i)
            graph.triples.extend(generated_triples)

            # answer find
            model = self.task_name_to_solver["find_answer"]
            sub_record = model.solve(question, data, args, depth + 1, graph)
            records["history"].append(sub_record)
            response = sub_record["answer"]
            print(" " * depth, response)

            answers = re.findall(r"Answer: (\[.*?\])", response)
            if len(answers) and not "m." in answers[0]:
                find_answer = True
                answers = eval(answers[0])
                records["answer"] = answers

        return records


class QuestionDecomposer:
    def __init__(self, task_name, prompt_file, task_name_to_solver=None) -> None:
        self.task_name = task_name
        self.prompt_file = prompt_file
        with open(prompt_file, "r") as f:
            self.prompt = format_prompt(f.read())
        self.task_name_to_solver = task_name_to_solver

        self.model_question_regex = re.compile("\[([^\]]+)\] (.*)")

        self.prompt_pattern = self.prompt + "QC: {question}\n"

    def is_end(self, response):
        if "[EOS]" in response:
            return True
        else:
            return False

    def solve(self, question, data, args=None, depth=0, graph=None):
        records = {
            "task": self.task_name,
            "question": question,
            "history": [],
            "sub_task_to_answer": [],
        }
        prompt = self.prompt_pattern.format(prompt=self.prompt, question=question)
        if graph is None:
            graph = Graph()

        print("  " * depth + f"QC: {question}")

        while True:
            prompt += "QS: "
            response = run_llm(
                prompt,
                args.temperature_exploration,
                args.max_length,
                args.opeani_api_keys,
                args.LLM_type,
            )

            print("  " * depth + f"QS: {response}")

            if self.is_end(response):
                break

            prompt += response + "\n"

            task_question_match = self.model_question_regex.match(response)

            if task_question_match:
                sub_task = task_question_match.group(1)
                sub_question = task_question_match.group(2)
                if sub_task and sub_task in self.task_name_to_solver:
                    sub_task_solver = self.task_name_to_solver[sub_task]
                    sub_record = sub_task_solver.solve(sub_question, data, args, depth + 1, graph)
                    ans = sub_record["answer"]

                    print("  " * depth + f"A: {ans}")
                    prompt += f"A: {ans}\n"

                    try:
                        ans = eval(ans)
                    except:
                        ans = ans

                    records["history"].append((sub_task, sub_question, ans, sub_record))

        # ans of last of sub question
        records["answer"] = ans
        return records
