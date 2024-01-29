from multiprocessing import Pool
import os
from pathlib import Path
import random
import re
import time
from tqdm import tqdm
import json
import argparse
from environment import FreeBaseEnv

from dotenv import load_dotenv
from llms import run_llm
from utils.bm25 import BM25Retrieve
from threading import Lock
from datasets import load_dataset
from utils.utils import format_prompt, read_file, proxy_load_dataset, convert_list_to_str
from loguru import logger

logger.add(sink="log", mode="w")
load_dotenv()

lock = Lock()


def convert_jsonl_to_json(jsonl_filepath):
    with open(jsonl_filepath, "r") as f:
        output_datas = [json.loads(line) for line in f.readlines()]

    output_datas = sorted(output_datas, key=lambda x: x["index"])

    jsonl_filepath = Path(jsonl_filepath)
    json_filepath = jsonl_filepath.parent / f"{jsonl_filepath.stem}.json"

    with open(json_filepath, "w") as f:
        json.dump(output_datas, f, indent=2)


def answer_question_without_kg(env: FreeBaseEnv, prompt):
    output = run_llm(
        prompt,
        args.temperature_exploration,
        512,
        args.opeani_api_keys,
        args.LLM_type,
        stop=None,
        stream=True,
    )
    match = re.search("Finish(\[.*\])", output)
    if match:
        answers = match.group(1)
        answers = eval(answers)
    else:
        answers = ["unknown"]
    env.llm_output = output
    return answers


def write_results(data, preprocessed_data, env: FreeBaseEnv, answer):
    with lock:
        with open("./predictions.jsonl", "a") as f:
            res = {
                "index": data["index"],
                "question": data["question"],
                "prediction": answer,
                "ground_truth": preprocessed_data["answer"],
                "records": env.records,
            }
            if env.llm_output:
                res["llm_output"] = env.llm_output

            f.write(json.dumps(res) + "\n")


def find_answer(process_idx, idxes_to_process):
    logger.debug(f"{process_idx}, {idxes_to_process[0]}")

    instruction = format_prompt(read_file("prompts2/instruction"))
    example = format_prompt(read_file("prompts2/example"))

    t1 = time.time()
    for n, idx in enumerate(idxes_to_process):
        if (n + 1) % 10 == 0:
            t2 = time.time()
            logger.debug(f"{process_idx}: {n / len(idxes_to_process)}, {t2 - t1}")
            t1 = t2

        try:
            data = datas[idx]
            preprocessed_data = dataset[data["index"]]

            topic_entity_names = sorted(data["topic_entity"].values())
            topic_entity_names_str = convert_list_to_str(topic_entity_names)

            base_prompt = (
                instruction
                + example
                + f'Question: {data["question"]}\nTopic Entity: {topic_entity_names_str}\n'
            )
            logger.debug(data["question"])

            env = FreeBaseEnv(args, data["topic_entity"])
            n_calls, n_badcalls, n_expand = 0, 0, 0
            done = False
            prompt = base_prompt

            if args.no_kb:
                answers = answer_question_without_kg(env, base_prompt)
                write_results(data, preprocessed_data, env, answers)
                continue

            for _ in range(6):
                i = len(env.records) + 1

                n_calls += 1
                thought_action = run_llm(
                    prompt + f"Thought {i}:",
                    args.temperature_exploration,
                    args.max_length,
                    args.opeani_api_keys,
                    args.LLM_type,
                    [f"\nObservation {i}:"],
                )
                try:
                    thought, action = thought_action.strip().split(f"\nAction {i}: ")
                except:
                    logger.debug(f"ohh... {thought_action}")
                    n_badcalls += 1
                    n_calls += 1
                    thought = thought_action.strip().split("\n")[0]
                    action = run_llm(
                        prompt + f"Thought {i}: {thought}\nAction {i}:",
                        args.temperature_exploration,
                        args.max_length,
                        args.opeani_api_keys,
                        args.LLM_type,
                        stop=[f"\n"],
                    ).strip()

                logger.debug(f"{thought}, {action}")
                match = re.search("Finish(\[.*\])", action)
                if match:
                    answers = match.group(1)
                    answers = eval(answers)
                    if answers[0][:2] in ["m.", "g."]:
                        # Finish["m.0h3d7qb"]
                        action = "Search" + action[6:]
                    elif answers[0].lower() == "unknown":
                        # Finish["unkown"]
                        # not enough information even after generation
                        # expand to 2-hop sub-graph
                        logger.debug("roll back and expand")
                        while "generate" in env.last_action.lower():
                            env.records.pop()
                        assert "search" in env.last_action.lower()
                        action = "Search[ALL]"
                        n_expand += 1
                        if n_expand >= 3:
                            logger.debug("max n_expand, break")
                            answers = answer_question_without_kg(env, base_prompt)
                            done = True
                    else:
                        done = True

                env.records.append({"i": i, "thought": thought, "action": action})
                if done:
                    write_results(data, preprocessed_data, env, answers)
                    break

                obs = env.step(action[0].lower() + action[1:])
                obs = obs.replace("\\n", "")
                env.records[-1]["observation"] = obs

                logger.debug(obs)

                records_str = env.convert_records_to_str()
                prompt = base_prompt + records_str

            logger.debug(f"ground truth: {preprocessed_data['answer']}")
        except Exception as e:
            logger.error(f'{e}')

    logger.info(f"{process_idx} finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="./data/cwq_dev.json", help="choose the dataset."
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="the max length of LLMs output."
    )
    parser.add_argument(
        "--temperature_exploration",
        type=float,
        default=0.3,
        help="the temperature in exploration stage.",
    )
    parser.add_argument(
        "--temperature_reasoning", type=float, default=0, help="the temperature in reasoning stage."
    )
    parser.add_argument("--width", type=int, default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int, default=3, help="choose the search depth of ToG.")
    parser.add_argument(
        "--remove_unnecessary_rel",
        type=bool,
        default=True,
        help="whether rem6oving unnecessary relations.",
    )
    parser.add_argument(
        "--LLM_type", type=str, default="gpt-3.5-turbo-0613", help="base LLM model."
    )
    parser.add_argument(
        "--opeani_api_keys",
        type=str,
        default=None,
        help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.",
    )
    parser.add_argument(
        "--num_retain_entity",
        type=int,
        default=5,
        help="Number of entities retained during entities search.",
    )
    parser.add_argument(
        "--prune_tools",
        type=str,
        default="llm",
        help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n_process", type=int, default=1)
    parser.add_argument("--no_kb", action="store_true")

    args = parser.parse_args()

    datas = json.load(open(args.dataset, "r"))
    dataset = proxy_load_dataset("rmanluo/RoG-cwq", split="test")

    datas = datas[:100]

    if "index" not in datas[0]:
        for i, data in enumerate(datas):
            data["index"] = i
        with open(args.dataset, "w") as f:
            json.dump(datas, f, indent=2)

    if os.path.exists("./predictions.jsonl") and not args.force:
        with open("./predictions.jsonl", "r") as f:
            output_datas = [json.loads(line) for line in f.readlines()]
        processed_idxes = set([data["index"] for data in output_datas])

        idxes_to_process = set(range(len(datas))) - processed_idxes
        idxes_to_process = sorted(list(idxes_to_process))
    else:
        idxes_to_process = list(range(len(datas)))
        with open("./predictions.jsonl".format(args.dataset), "w") as f:
            pass

    num_samples = len(idxes_to_process)
    logger.debug(num_samples)

    n_process = min(args.n_process, num_samples)
    logger.debug(n_process)

    # random.shuffle(idxes_to_process)
    # idxes_to_process = [52]

    if n_process > 1:
        with Pool(processes=n_process) as pool:
            num_samples_in_chunk = num_samples // n_process
            jobs = []
            st = 0
            for i in range(n_process):
                ed = st + num_samples_in_chunk
                ed = min(ed, num_samples)
                jobs.append([i, idxes_to_process[st:ed]])

                st = ed

            results = pool.starmap(find_answer, jobs)
    else:
        find_answer(0, idxes_to_process)

    convert_jsonl_to_json("./predictions.jsonl")
