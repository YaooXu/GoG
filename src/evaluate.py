import json
import re
import sys

import numpy as np

sys.path.append("src")

from utils import convert_jsonl_to_json


import argparse


def exact_match(response, answers):
    clean_result = response.strip().lower()
    for answer in answers:
        clean_answer = answer.strip().lower()
        if (
            clean_result == clean_answer
            or clean_result in clean_answer
            or clean_answer in clean_result
        ):
            return True
    return False


def eval_results(prediction_pathfile):
    if prediction_pathfile.endswith("jsonl"):
        prediction_pathfile = convert_jsonl_to_json(prediction_pathfile)
        prediction_pathfile = prediction_pathfile

    with open(prediction_pathfile, encoding="utf-8") as f:
        output_datas = json.load(f)

    print(prediction_pathfile)

    # parts = prediction_pathfile.split('/')
    # data_file = 'data/' + parts[-1]
    # with open(data_file, encoding="utf-8") as f:
    #     ori_datas = json.load(f)
    #     idx_to_n_ct = {sample['index']: len(sample['crucial_triples']) for sample in ori_datas}
    #     idx_to_n_answer = {sample['index']: len(sample['answer']) for sample in ori_datas}

    # output_datas = output_datas[::2]
    # output_datas = [data for data in output_datas if idx_to_n_ct[data['index']] > 0]
    # output_datas = [data for data in output_datas if len(data['records']) > 0]
    
    print(len(output_datas))
    
    num_right = 0
    num_error = 0
    wrong = []
    empty_rc = 0
    empty_rc_correct = 0
    num_gen = 0
    num_gen_correct = 0
    gen_repeate_res = []
    gen_no_repeate_res = []

    for i, data in enumerate(output_datas):
        answers = list(set(data["ground_truth"]))
        response = " ".join(data["prediction"])

        if "llm_output" in data:
            empty_rc += 1

        correct = None
        if exact_match(response, answers):
            correct = 1
            num_right += 1
            if "llm_output" in data:
                empty_rc_correct += 1
        else:
            correct = 0
            num_error += 1
            # wrong.append(i)
            if "llm_output" not in data:
                wrong.append(i)

        record = data["records"]
        for step in record:
            if "llm_output" in data:
                break
            if step["action"].startswith("Generate"):
                pattern = r"\[(.*?)\]"
                result = re.findall(pattern, step["action"])[0]
                num_gen += 1
                if correct:
                    num_gen_correct += 1
                else:
                    pass
                    # print(i, data["question"])
                s1 = set(result.lower().split())
                s2 = set(data["question"].replace("?", "").lower().split())
                if len(s1.intersection(s2)) / len(s2) > 0.6:
                    # print(data["question"])
                    # print(step["action"])
                    gen_repeate_res.append(correct)
                else:
                    gen_no_repeate_res.append(correct)

    # print(wrong)
    # print(prediction_pathfile)
    print("Exact Match: {}".format(float(num_right / len(output_datas))))
    print(f"Empty reasoning_chain: {empty_rc}, {empty_rc / len(output_datas)}")
    print(
        f"Empty reasoning_chain correct: {empty_rc_correct}, {empty_rc_correct / num_right}, {empty_rc_correct / empty_rc}"
    )
    print("right: {}, error: {}".format(num_right, num_error))
    print(num_gen, num_gen_correct / num_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default="prob_results/cwq/gpt-3.5-turbo-0613/10_3_0.7_data_with_ct_0.2_predictions.json",
        help="the output file name.",
    )
    args = parser.parse_args()

    eval_results(args.output_file)
