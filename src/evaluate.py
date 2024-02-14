import json
import sys

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

    num_right = 0
    num_error = 0
    wrong = []
    empty_rc = 0
    empty_rc_correct = 0
    
    for i, data in enumerate(output_datas):
        answers = list(set(data["ground_truth"]))
        response = " ".join(data["prediction"])
        
        if 'llm_output' in data:
            empty_rc += 1
            
        if exact_match(response, answers):
            num_right += 1
            if 'llm_output' in data:
                empty_rc_correct += 1
        else:
            num_error += 1
            wrong.append(i)

    print(wrong)
    print(prediction_pathfile)
    print("Exact Match: {}".format(float(num_right / len(output_datas))))
    print(f'Empty reasoning_chain: {empty_rc}, {empty_rc / len(output_datas)}')
    print(f'Empty reasoning_chain correct: {empty_rc_correct}, {empty_rc_correct / num_right}, {empty_rc_correct / empty_rc}')
    print("right: {}, error: {}".format(num_right, num_error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default='/data/yaoxu/KBQA/ToG-xy/results/cwq/gpt-3.5-turbo-0613/0.7_data_with_ct_100_-1_1_predictions.jsonl',
        help="the output file name.",
    )
    args = parser.parse_args()

    eval_results(args.output_file)
