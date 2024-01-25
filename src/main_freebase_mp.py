import os
import sys

from networkx import subgraph

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from pathlib import Path
from tqdm import tqdm
import argparse
from kb_interface.freebase_func import *
import random
from kb_interface.client import *
from multiprocessing import Pool
from threading import Lock
from datasets import load_dataset, load_from_disk


def save_2_jsonl(idx, question, answer, cluster_chain_of_entities, file_name, lock):
    dict = {"index": idx, "question": question, "results": answer, "reasoning_chains": cluster_chain_of_entities}

    with lock:
        with open("./src/ToG_{}.jsonl".format(file_name), "a") as outfile:
            json_str = json.dumps(dict)
            outfile.write(json_str + "\n")


def half_stop(idx, question, cluster_chain_of_entities, depth, args, lock):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(idx, question, answer, cluster_chain_of_entities, args.dataset, lock)


def convert_jsonl_to_json(jsonl_filepath):
    with open(jsonl_filepath, "r") as f:
        output_datas = [json.loads(line) for line in f.readlines()]

    output_datas = sorted(output_datas, key=lambda x: x["index"])

    jsonl_filepath = Path(jsonl_filepath)
    json_filepath = jsonl_filepath.parent / f"{jsonl_filepath.stem}.json"

    with open(json_filepath, "w") as f:
        json.dump(output_datas, f, indent=2)


@set_environment_variable
def get_preprocessed_dataset(filename="rmanluo/RoG-cwq", split="test"):
    dataset = load_dataset(filename, split=split)
    dataset = dataset.select(range(100))
    return dataset


def __find_answer(worker_idx, st, ed):
    print(worker_idx, st, ed)

    t1 = time.time()
    for n, idx in enumerate(range(st, ed)):
        if (n + 1) % 10 == 0:
            t2 = time.time()
            print(f"{worker_idx}: {n / (ed - st)}", t2 - t1)
            t1 = t2

        data = datas[idx]
        # preprocessed_data = preprocessed_dataset[idx]
        # subgraph = preprocessed_data["graph"]
        # for triple in subgraph:
        #     if triple[0] in preprocessed_data["q_entity"] or triple[-1] in preprocessed_data["q_entity"]:
        #         print(triple)
        # num_relations = len(set([relation for _, relation, _ in subgraph]))
        # print(num_relations)

        question = data[question_string]
        print(question)

        topic_entity = data["topic_entity"]
        idx = data["index"]

        try:
            cluster_chain_of_entities = []
            if len(topic_entity) == 0:
                results = generate_without_explored_paths(question, args)
                save_2_jsonl(idx, question, results, [], args.dataset, lock)
                continue
            pre_relations = []
            pre_heads = [-1] * len(topic_entity)
            flag_printed = False
            for depth in range(1, args.depth + 1):
                current_entity_relations_list = []
                i = 0
                for entity in topic_entity:
                    if entity != "[FINISH_ID]":
                        retrieve_relations_with_scores = relation_search_prune(
                            entity, topic_entity[entity], pre_relations, pre_heads[i], question, args
                        )  # best entity triplet, entitiy_id
                        current_entity_relations_list.extend(retrieve_relations_with_scores)
                    i += 1
                    # print(current_entity_relations_list)
                total_candidates = []
                total_scores = []
                total_relations = []
                total_entities_id = []
                total_topic_entities = []
                total_head = []

                for entity in current_entity_relations_list:
                    if entity["head"]:
                        entity_candidates_id = entity_search(entity["entity"], entity["relation"], True)
                    else:
                        entity_candidates_id = entity_search(entity["entity"], entity["relation"], False)

                    if args.prune_tools == "llm":
                        if len(entity_candidates_id) >= 20:
                            entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                    if len(entity_candidates_id) == 0:
                        continue
                    scores, entity_candidates, entity_candidates_id = entity_score(
                        question, entity_candidates_id, entity["score"], entity["relation"], args
                    )

                    (
                        total_candidates,
                        total_scores,
                        total_relations,
                        total_entities_id,
                        total_topic_entities,
                        total_head,
                    ) = update_history(
                        entity_candidates,
                        entity,
                        scores,
                        entity_candidates_id,
                        total_candidates,
                        total_scores,
                        total_relations,
                        total_entities_id,
                        total_topic_entities,
                        total_head,
                    )

                if len(total_candidates) == 0:
                    half_stop(idx, question, cluster_chain_of_entities, depth, args, lock)
                    flag_printed = True
                    break
                # print(total_candidates, total_scores)
                flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(
                    total_entities_id,
                    total_relations,
                    total_candidates,
                    total_topic_entities,
                    total_head,
                    total_scores,
                    args,
                )
                cluster_chain_of_entities.append(chain_of_entities)
                # print(cluster_chain_of_entities)
                if flag:
                    stop, results = reasoning(question, cluster_chain_of_entities, args)
                    if stop:
                        # print("ToG stoped at depth %d." % depth)
                        save_2_jsonl(idx, question, results, cluster_chain_of_entities, args.dataset, lock)
                        flag_printed = True
                        break
                    else:
                        # print("depth %d still not find the answer." % depth)
                        flag_finish, entities_id = if_finish_list(entities_id)
                        if flag_finish:
                            half_stop(idx, question, cluster_chain_of_entities, depth, args, lock)
                            flag_printed = True
                        else:
                            topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                            continue
                else:
                    half_stop(idx, question, cluster_chain_of_entities, depth, args, lock)
                    flag_printed = True

            if not flag_printed:
                results = generate_without_explored_paths(question, args)
                save_2_jsonl(idx, question, results, [], args.dataset, lock)
        except Exception as e:
            print(e)


lock = Lock()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq_train_dev", help="choose the dataset.")
    parser.add_argument("--max_length", type=int, default=256, help="the max length of LLMs output.")
    parser.add_argument(
        "--temperature_exploration", type=float, default=0.4, help="the temperature in exploration stage."
    )
    parser.add_argument("--temperature_reasoning", type=float, default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int, default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int, default=3, help="choose the search depth of ToG.")
    parser.add_argument(
        "--remove_unnecessary_rel", type=bool, default=True, help="whether removing unnecessary relations."
    )
    parser.add_argument("--LLM_type", type=str, default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument(
        "--opeani_api_keys",
        type=str,
        default=None,
        help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.",
    )
    parser.add_argument(
        "--num_retain_entity", type=int, default=5, help="Number of entities retained during entities search."
    )
    parser.add_argument(
        "--prune_tools",
        type=str,
        default="llm",
        help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n_process", type=int, default=1)
    args = parser.parse_args()

    datas, question_string = prepare_dataset(args.dataset)
    print("Start Running ToG on %s dataset." % args.dataset)

    # preprocessed_dataset = get_preprocessed_dataset()

    if "index" not in datas[0]:
        for i, data in enumerate(datas):
            data["index"] = i
        with open(f"./data/{args.dataset}.json", "w") as f:
            json.dump(datas, f, indent=2)

    if os.path.exists("./src/ToG_{}.jsonl".format(args.dataset)) and not args.force:
        with open("./src/ToG_{}.jsonl".format(args.dataset), "r") as f:
            output_datas = [json.loads(line) for line in f.readlines()]
        processed_idxes = set([data["index"] for data in output_datas])

        not_processed_idxes = set(range(len(datas))) - processed_idxes
        not_processed_idxes = sorted(list(not_processed_idxes))

        # datas = [data for data in datas if data["index"] not in processed_idxes]
        # dataset = dataset.filter(lambda x: x["index"] not in processed_idxes)
    else:
        not_processed_idxes = range(len(datas))
        with open("./src/ToG_{}.jsonl".format(args.dataset), "w") as f:
            pass

    num_samples = len(not_processed_idxes)
    print(num_samples)

    n_process = min(args.n_process, num_samples)
    print(n_process)

    with Pool(processes=n_process) as pool:
        num_samples_in_chunk = num_samples // n_process
        jobs = []
        st = 0
        for i in range(n_process):
            ed = st + num_samples_in_chunk
            ed = min(ed, num_samples)
            jobs.append([i, st, ed])

            st = ed

        results = pool.starmap(__find_answer, jobs)

    convert_jsonl_to_json("./src/ToG_{}.jsonl".format(args.dataset))
