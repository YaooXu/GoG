from tqdm import tqdm
import json
import argparse

from task_solvers import (
    AnswerFinder,
    OneHopQuerySolver,
    EntityExtract,
    RelationFilter,
    QuestionDecomposer,
    Graph,
    TriplesGenerator,
)

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="./data/cwq_train_dev.json", help="choose the dataset."
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="the max length of LLMs output."
    )
    parser.add_argument(
        "--temperature_exploration",
        type=float,
        default=0.4,
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
        help="whether removing unnecessary relations.",
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
    args = parser.parse_args()

    datas = json.load(open(args.dataset, "r"))
    datas = datas[:10]

    config = {
        "decomposed_qa": (QuestionDecomposer, "./prompts/decomposed_qa"),
        "extract_entity": (EntityExtract, "./prompts/primitive_tasks/extract_entity"),
        "filter_relations": (RelationFilter, "./prompts/primitive_tasks/filter_relations"),
        "find_answer": (AnswerFinder, "./prompts/primitive_tasks/find_answer"),
        "one_hop_qa": (OneHopQuerySolver, "./prompts/one_hop_qa"),
        "generate_triples": (TriplesGenerator, "./prompts/primitive_tasks/generate_triples")
        # 'retrieve': (QuestionDecomposer, './prompts/retrieve'),
    }

    task_name_to_solver = {}
    for task_name, (MODEL, prompt_file) in config.items():
        task_name_to_solver[task_name] = MODEL(task_name, prompt_file, task_name_to_solver)

    for data in tqdm(datas[:]):
        decomposer = task_name_to_solver["decomposed_qa"]

        graph = Graph()
        graph.topic_entities = data["topic_entity"]

        record = decomposer.solve(data["question"], data, args, 0, graph)

        # try:
        #     record = decomposer.solve(data["question"], data, args)
        #     print("Answer: ", record["answer"])
        # except Exception as e:
        #     print(e)
        #     pass
