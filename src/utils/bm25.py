import bz2
from collections import defaultdict
import re
import time
import os
import multiprocessing as mp
import pickle
import requests

from tqdm import tqdm
from rank_bm25 import BM25Okapi


from kb_interface.freebase_interface import get_types


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

    # print(f"start {chunk_start}")

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
                # print(
                #     ed - st,
                #     f.tell() - chunk_start,
                #     "/",
                #     chunk_size,
                #     (f.tell() - chunk_start) / chunk_size,
                # )
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

    # print("Starting the parallel pool for {} jobs ".format(len(jobs)))

    # maxtaskperchild - if not supplied some weird happend and memory blows as the processes keep on lingering
    with mp.Pool(num_parallel, maxtasksperchild=1000) as pool:
        outputs = []

        t1 = time.time()
        outputs = pool.map(process_chunk, jobs)

    output = process_class.merge_results(outputs)

    print("All Done in time ", time.time() - t1)

    return output


class Construct_label_to_id:
    def __init__(self) -> None:
        self.name_to_id_dict = {}

    def process_line(self, line):
        info = line.split("\t")
        name = info[0]
        score = float(info[1])
        mid = info[2].strip()
        if name in self.name_to_id_dict:
            self.name_to_id_dict[name][mid] = score
        else:
            self.name_to_id_dict[name] = {}
            self.name_to_id_dict[name][mid] = score

    def return_result(self):
        return self.name_to_id_dict

    @staticmethod
    def merge_results(list_of_results):
        merged_dict = {}

        for dictionary in list_of_results:
            for key, value in dictionary.items():
                if key in merged_dict:
                    merged_dict[key].update(value)
                else:
                    merged_dict[key] = value

        return merged_dict


class BM25Retrieve:
    def __init__(self, filepath="Freebase/surface_map_file_freebase_complete_all_mention") -> None:
        self.label_to_id = parallel_process_file(
            filepath,
            Construct_label_to_id,
            10,
        )

        self.all_fns = list(self.label_to_id.keys())
        self.tokenized_all_fns = [fn.split() for fn in self.all_fns]

        self.bm25_all_fns = BM25Okapi(self.tokenized_all_fns)

    @staticmethod
    def load(
        filepath="Freebase/surface_map_file_freebase_complete_all_mention", re_process=False
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

    def retrieve_ids_by_names(self, names, question):
        return_mid_list = []
        for fn_org in names:
            drop_dot = fn_org.split()
            drop_dot = [seg.strip(".") for seg in drop_dot]
            drop_dot = " ".join(drop_dot)
            if fn_org.lower() not in question and drop_dot.lower() in question:
                fn_org = drop_dot
            if fn_org.lower() not in self.label_to_id:
                print("fn_org: {}".format(fn_org.lower()))
                tokenized_query = fn_org.lower().split()
                fn = self.bm25_all_fns.get_top_n(tokenized_query, self.all_fns, n=1)[0]
                print("sub fn: {}".format(fn))
            else:
                fn = fn_org
            if fn.lower() in self.label_to_id:
                id_dict = self.label_to_id[fn.lower()]
            if len(id_dict) > 15:
                mids = self.get_right_mid_set(fn.lower(), id_dict, question)
            else:
                mids = sorted(id_dict.items(), key=lambda x: x[1], reverse=True)
                mids = [mid[0] for mid in mids]
            return_mid_list.append(mids)
        return return_mid_list

    def get_right_mid_set(self, fn, id_dict, question):
        type_to_mid_dict = {}
        type_list = []
        for mid in id_dict:
            types = get_types(mid)
            for cur_type in types:
                if not cur_type.startswith("common.") and not cur_type.startswith("base."):
                    if cur_type not in type_to_mid_dict:
                        type_to_mid_dict[cur_type] = {}
                        type_to_mid_dict[cur_type][mid] = id_dict[mid]
                    else:
                        type_to_mid_dict[cur_type][mid] = id_dict[mid]
                    type_list.append(cur_type)
        tokenized_type_list = [re.split("\.|_", doc) for doc in type_list]
        #     tokenized_question = tokenizer.tokenize(question)
        tokenized_question = question.split()
        bm25 = BM25Okapi(tokenized_type_list)
        top10_types = bm25.get_top_n(tokenized_question, type_list, n=10)
        selected_types = top10_types[:3]
        selected_mids = []
        for any_type in selected_types:
            # logger.info("any_type: {}".format(any_type))
            # logger.info("type_to_mid_dict[any_type]: {}".format(type_to_mid_dict[any_type]))
            selected_mids += list(type_to_mid_dict[any_type].keys())
        return selected_mids


def retrieve_ids_by_labels(labels, question):
    url = "http://210.75.240.139:18891/label2id"
    data = {
        "labels": labels,
        "question": question,
    }
    response = requests.post(url, json=data)

    return response.json()


if __name__ == "__main__":
    # with open("Freebase/surface_map_file_freebase_complete_all_mention") as f:
    #     lines = f.readlines()
    # name_to_id_dict = {}
    # for line in tqdm(lines):
    #     info = line.split("\t")
    #     name = info[0]
    #     score = float(info[1])
    #     mid = info[2].strip()
    #     if name in name_to_id_dict:
    #         name_to_id_dict[name][mid] = score
    #     else:
    #         name_to_id_dict[name] = {}
    #         name_to_id_dict[name][mid] = score

    bm25 = BM25Retrieve()
