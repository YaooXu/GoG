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
from evaluate import eval_results
from kb_interface.freebase_func import convert_name_to_id
from llms import run_llm
from threading import Lock
from datasets import load_dataset
from utils import (
    convert_jsonl_to_json,
    format_prompt,
    parse_llm_output_to_list,
    read_file,
    proxy_load_dataset,
    convert_list_to_str,
)
from loguru import logger

load_dotenv()

lock = Lock()


def answer_question_without_kg(env: FreeBaseEnv, prompt):
    output = run_llm(
        prompt,
        args.temperature,
        512,
        args.opeani_api_keys,
        args.LLM_type,
        stop=None,
        stream=True,
    )
    match = re.search("Finish(\[.*\])", output)
    if match:
        answers = match.group(1)
        answers = parse_llm_output_to_list(answers)
    else:
        answers = ["unknown"]
    env.llm_output = output
    return answers


def write_results(data, env: FreeBaseEnv, prediction):
    with lock:
        with open(output_file, "a") as f:
            res = {
                "index": data["index"],
                "question": data["question"],
                "prediction": prediction,
                "ground_truth": data["answer"],
                "records": env.records,
            }
            if env.llm_output:
                res["llm_output"] = env.llm_output

            f.write(json.dumps(res) + "\n")


def find_answer(process_idx, idxes_to_process):
    logger.debug(f"{process_idx}, {idxes_to_process[0]}")

    instruction = format_prompt(read_file("prompts2/instruction"))

    example = format_prompt(read_file("prompts2/examples"))

    t1 = time.time()
    for n, idx in enumerate(idxes_to_process):
        if (n + 1) % 10 == 0:
            t2 = time.time()
            logger.debug(f"{process_idx}: {n / len(idxes_to_process)}, {t2 - t1}")
            t1 = t2

        try:
            data = datas[idx]
            if "question" not in data:
                data["question"] = data["ProcessedQuestion"]

            topic_entity_names = sorted(data["topic_entity"].values())
            topic_entity_names_str = "[" + " | ".join(topic_entity_names) + "]"

            base_prompt = (
                instruction
                + example
                + f'Question: {data["question"]}\nTopic Entity: {topic_entity_names_str}\n'
            )
            logger.debug(data["question"])

            env = FreeBaseEnv(args, data["topic_entity"])
            env.mid_crucial_triples = data["mid_crucial_triples"]

            n_calls, n_badcalls, n_expand = 0, 0, 0
            done = False
            prompt = base_prompt

            if args.no_kb:
                prediction = answer_question_without_kg(env, base_prompt)
                write_results(data, env, prediction)
                continue

            for _ in range(6):
                i = len(env.records) + 1

                n_calls += 1
                thought_action = run_llm(
                    prompt + f"Thought {i}:",
                    args.temperature,
                    args.max_length,
                    args.opeani_api_keys,
                    args.LLM_type,
                    # [f"\nObservation {i}:"],
                    [f"Observation"],
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
                        args.temperature,
                        args.max_length,
                        args.opeani_api_keys,
                        args.LLM_type,
                        stop=[f"\n"],
                    ).strip()

                logger.debug(f"{thought}, {action}")
                match = re.search("Finish(\[.*\])", action)
                if match:
                    prediction = match.group(1)
                    prediction = parse_llm_output_to_list(prediction)
                    if prediction[0][:2] in ["m.", "g."]:
                        # Finish["m.0h3d7qb"]
                        action = "Search" + action[6:]
                    elif prediction[0].lower() == "unknown":
                        # Finish["unkown"]
                        # not enough information even after generation
                        # expand to 2-hop sub-graph
                        logger.debug("roll back and expand")
                        while "generate" in env.last_action.lower():
                            env.records.pop()
                        assert "search" in env.last_action.lower()

                        i = len(env.records) + 1

                        action = "Search[ALL]"
                        n_expand += 1
                        if n_expand >= args.max_n_expand:
                            logger.debug("max n_expand, break")
                            prediction = answer_question_without_kg(env, base_prompt)
                            done = True
                    else:
                        done = True

                env.records.append({"i": i, "thought": thought, "action": action})
                if done:
                    write_results(data, env, prediction)
                    break

                obs = env.step(action[0].lower() + action[1:])
                obs = obs.replace("\\n", "")
                env.records[-1]["observation"] = obs

                logger.debug(obs)

                records_str = env.convert_records_to_str()
                prompt = base_prompt + records_str

            if not done:
                prediction = answer_question_without_kg(env, base_prompt)
                write_results(data, env, prediction)

            logger.debug(f"ground truth: {data['answer']}")
        except Exception as e:
            logger.error(f"{e}, trying get answer without kg")
            prediction = answer_question_without_kg(env, base_prompt)
            write_results(data, env, prediction)

    logger.info(f"{process_idx} finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/cwq/samples_with_crucial_triples_100_0.json",
        help="choose the dataset.",
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="the max length of LLMs output."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="the temperature in exploration stage.",
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
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_n_expand", default=3)
    parser.add_argument("--n_related_triples", type=int, default=10)

    args = parser.parse_args()
    logger.debug(f"{args}")

    datas = json.load(open(args.dataset, "r"))

    dataset_name = args.dataset.split("/")[1]
    output_file = (
        Path(f"./{args.output_dir}/{dataset_name}/{args.LLM_type}")
        / f"{args.n_related_triples}_{args.max_n_expand}_{args.temperature}_{Path(args.dataset).stem}_predictions.jsonl"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        sink=output_file.parent / f"{args.temperature}_{Path(args.dataset).stem}.log",
        mode="w",
    )

    print(output_file)

    if os.path.exists(output_file) and not args.force:
        with open(output_file, "r") as f:
            output_datas = [json.loads(line) for line in f.readlines()]
        processed_idxes = set([data["index"] for data in output_datas])

        idxes_to_process = set(range(len(datas))) - processed_idxes
        idxes_to_process = sorted(list(idxes_to_process))
        print(idxes_to_process)
    else:
        idxes_to_process = range(len(datas))
        with open(output_file, "w") as f:
            pass

    num_samples = len(idxes_to_process)
    logger.debug(num_samples)

    n_process = min(args.n_process, num_samples)
    logger.debug(n_process)

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
    elif n_process == 1:
        find_answer(0, idxes_to_process)

    json_filepath = convert_jsonl_to_json(output_file)
    eval_results(json_filepath)
