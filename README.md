# Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering

This repository contains official implementation for paper `Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering`.

In this documentation, we detail how to construct Incomplete KGs and run GoG.

## Freebase Setup

See ./Freebase/README.md

## Selecting Crucial Triples

This step can be skipped, as all processed data are contained in the /data.
```bash
python src/generate_samples_with_crucial_edges.py
```

## Starting name_to_id service

Downloading the pickle file from [Google Drive](https://drive.google.com/file/d/1PIUDBwbiuhUHJ4FdujfXfuOUWkb40NDJ/view?usp=drive_link), and put it in Freebase/bm25.pkl.

Start the service with this command, and the default port is 18891.
```bash
python src/bm25_name2ids.py
```

## Runing GoG

```bash
python src/GoG.py --n_process=4 --dataset data/cwq/data_with_ct_1000_-1_1.json
```

## Proxy

If you want to access openai and huggingface with proxy, please uncomment lines and change the proxy address in `run_llm` in `src/llms/interface.py` and `set_environment_variable` in `src/utils.py`.