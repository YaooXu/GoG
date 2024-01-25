from utils.bm25 import BM25Retrieve

from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route("/label2id", methods=["POST"])
def convert_label_to_id():
    try:
        data = request.json  # 获取 JSON 数据
        if "labels" in data:
            labels = data["labels"]
            question = data["question"]
            ids = bm25_retrieve.retrieve_ids_by_names(labels, question)
            result = {"status": "success", "ids": ids}
        else:
            result = {"status": "error", "message": 'Key "labels" not found in JSON data.'}
    except Exception as e:
        result = {"status": "error", "message": str(e)}

    return jsonify(result)


if __name__ == "__main__":
    bm25_retrieve = BM25Retrieve.load()

    app.run(host="127.0.0.1")
