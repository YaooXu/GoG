from dotenv import load_dotenv
load_dotenv()

import bz2
from collections import defaultdict
import re
from statistics import median_grouped
import time
import os
import multiprocessing as mp
import pickle
from typing import List
from SPARQLWrapper import SPARQLWrapper, JSON
import requests

from tqdm import tqdm
from rank_bm25 import BM25Okapi
import urllib
from flask import Flask, request, jsonify

import sys

sys.path.append("src")

app = Flask(__name__)


sparql = SPARQLWrapper(os.environ['SPARQLPATH'])
sparql.setReturnFormat(JSON)
ns_prefix = "http://rdf.freebase.com/ns/"


def get_types(entity: str) -> List[str]:
    query = (
        """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX ns: <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
        "ns:" + entity + " ns:type.object.type ?x0 . "
        """
    }
    }
    """
    )
    # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results["results"]["bindings"]:
        rtn.append(result["value"]["value"].replace("http://rdf.freebase.com/ns/", ""))

    return rtn


def chunkify_file(filepath, num_chunks, skiplines=-1):
    """
    function to divide a large text file into num_chunks chunks and the chunks are line aligned

    Params :
        fname : path to the file to be chunked
        num_chunks : number of chunks
        skiplines : number of lines in the begining to skip, -1 means don't skip any lines
    Returns :
        start and end position of chunks in Bytes
    """
    chunks = []
    file_end = os.path.getsize(filepath)
    print(f"file size : {file_end}")
    size = file_end // num_chunks

    with open(filepath, "rb") as f:
        if skiplines > 0:
            for _ in range(skiplines):
                f.readline()

        chunk_end = f.tell()
        count = 0
        while True:
            chunk_start = chunk_end

            f.seek(f.tell() + size, os.SEEK_SET)
            f.readline()  # make this chunk line aligned
            chunk_end = f.tell()
            chunks.append((chunk_start, chunk_end - chunk_start, filepath))
            count += 1

            if chunk_end >= file_end:
                break

    assert len(chunks) == num_chunks

    return chunks


def process_chunk(chunk_data):
    """
    function to apply a function to each line in a chunk

    Params :
        chunk_data : the data for this chunk
    Returns :
        list of the non-None results for this chunk
    """
    chunk_start, chunk_size, file_path, process_class = chunk_data[:4]
    func_args = chunk_data[4:]

    processer = process_class()

    print(f"start {chunk_start}")

    i = 0
    st = time.time()

    with open(file_path, "rb") as f:
        f.seek(chunk_start)

        while True:
            i += 1

            line = f.readline().decode(encoding="utf-8")

            if line == "":
                # the last chunk of file ends with ''
                break

            processer.process_line(line, *func_args)

            if i % 1_000_000 == 0:
                ed = time.time()
                print(
                    chunk_start,
                    ed - st,
                    f.tell() - chunk_start,
                    "/",
                    chunk_size,
                    (f.tell() - chunk_start) / chunk_size,
                )
                st = ed

            if f.tell() - chunk_start >= chunk_size:
                break

    return processer.return_result()


def parallel_process_file(input_file_path, process_class, num_procs, skiplines=0):
    """
    function to apply a supplied function line by line in parallel

    Params :
        input_file_path : path to input file
        num_procs : number of parallel processes to spawn, max used is num of available cores - 1
        process_class : a function which expects a line and outputs None for lines we don't want processed
        skiplines : number of top lines to skip while processing
        fout : do we want to output the processed lines to a file
        merge_func : merge function dealing with outputs of chunks
    Returns :
        merged output
    """
    num_parallel = num_procs
    print(f"num parallel: {num_procs}")

    jobs = chunkify_file(input_file_path, num_procs, skiplines)

    jobs = [list(x) + [process_class] for x in jobs]

    print("Starting the parallel pool for {} jobs ".format(len(jobs)))

    # maxtaskperchild - if not supplied some weird happend and memory blows as the processes keep on lingering
    with mp.Pool(num_parallel, maxtasksperchild=1000) as pool:
        outputs = []

        t1 = time.time()
        outputs = pool.map(process_chunk, jobs)

    output = process_class.merge_results(outputs)

    print("All Done in time ", time.time() - t1)

    return output


class ConstructName2id:
    def __init__(self) -> None:
        self.name_to_id_dict = defaultdict(set)

    def process_line(self, line):
        s, p, name, _ = line.strip().split("\t")
        mid = s[s.rfind("/") + 1 : -1]
        if p == "<http://rdf.freebase.com/ns/type.object.name>" and mid[:2] in ("m.", "g."):
            name = eval(name[:-3]).lower()
            self.name_to_id_dict[name].add(mid)

    def return_result(self):
        return self.name_to_id_dict

    @staticmethod
    def merge_results(list_of_results):
        merged_dict = defaultdict(set)

        for dictionary in list_of_results:
            for key, value in dictionary.items():
                merged_dict[key].update(value)

        for k, v in merged_dict.items():
            merged_dict[k] = list(v)
        return merged_dict


class BM25Retrieve:
    def __init__(self, filepath="Freebase/virtuoso-opensource/database/FilterFreebase") -> None:
        self.name_to_ids = parallel_process_file(
            filepath,
            ConstructName2id,
            10,
        )

        self.all_fns = list(self.name_to_ids.keys())
        self.tokenized_all_fns = [fn.split() for fn in self.all_fns]
        print(self.tokenized_all_fns)
        self.bm25_all_fns = BM25Okapi(self.tokenized_all_fns)

    @staticmethod
    def load(
        filepath="Freebase/virtuoso-opensource/database/FilterFreebase", re_process=False
    ) -> "BM25Retrieve":
        pkl_path = "Freebase/bm25.pkl"
        if os.path.exists(pkl_path) and not re_process:
            print(f"loading bm25 from {pkl_path}")
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        else:
            retrieve = BM25Retrieve(filepath)
            with open(pkl_path, "wb") as f:
                pickle.dump(retrieve, f)
            return retrieve

    def retrieve_id2types_by_names(self, names):
        all_mid_to_types = []
        for fn_org in names:
            if fn_org.lower() not in self.name_to_ids:
                print("fn_org: {}".format(fn_org.lower()))
                tokenized_query = fn_org.lower().split()
                fn = self.bm25_all_fns.get_top_n(tokenized_query, self.all_fns, n=1)[0]
                print("sub fn: {}".format(fn))
            else:
                fn = fn_org
            if fn.lower() in self.name_to_ids:
                mids = self.name_to_ids[fn.lower()]

            mid_to_types = {}
            for mid in mids:
                types = get_types(mid)
                types = [
                    type
                    for type in types
                    if not type.startswith("common.") and not type.startswith("base.")
                ]
                if not len(types):
                    continue
                mid_to_types[mid] = sorted(types)

            mid_to_types = dict(
                sorted(mid_to_types.items(), key=lambda x: len(x[-1]), reverse=True)
            )
            all_mid_to_types.append(mid_to_types)

        return all_mid_to_types


@app.route("/name2ids", methods=["POST"])
def convert_name_to_id():
    try:
        data = request.json  # 获取 JSON 数据
        if "names" in data:
            names = data["names"]
            all_mid_to_types = bm25.retrieve_id2types_by_names(names)
            result = {"status": "success", "all_mid_to_types": all_mid_to_types}
        else:
            result = {"status": "error", "message": 'Key "names" not found in JSON data.'}
    except Exception as e:
        result = {"status": "error", "message": str(e)}

    return jsonify(result)


def retrieve_id2types_by_name(name):
    url = "http://localhost:18891/name2ids"
    data = {
        "names": [name],
    }
    response = requests.post(url, json=data)

    return response.json()['all_mid_to_types'][0]


if __name__ == "__main__":
    bm25 = BM25Retrieve.load()

    app.run(port=18891, host="0.0.0.0")
