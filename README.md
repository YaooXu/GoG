# Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering

This repository contains official implementation for paper `Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering`.

In this documentation, we detail how to construct Incomplete KGs and run GoG.

## Freebase Setup

See ./Freebase/README.md

## Setting .env file

You should set a .env file to config your service port, api_key and proxy if necessary.
```bash
# this proxy is used to access openai and huggingface, not necessary.
# custom_proxy=socks5://127.0.0.1:11300

SPARQLPATH=http://127.0.0.1:18890/sparql

# change the base_url and opeani_api_keys if you use vllm to deploy your local service
base_url=https://api.openai.com/v1
opeani_api_keys=
```

## Selecting Crucial Triples

**This step can be skipped, as all processed data are contained in the /data.**
The results on WebQSP could be slightly different, as we overwrite the original files mistakenly.

```bash
python src/generate_samples_with_crucial_edges_by_prob.py
```



## Starting name_to_id service

Downloading the pickle file from [Google Drive](https://drive.google.com/file/d/1PIUDBwbiuhUHJ4FdujfXfuOUWkb40NDJ/view?usp=drive_link), and put it in Freebase/bm25.pkl.

Start the service with this command, and the default port is 18891.
```bash
python src/bm25_name2ids.py
```

## Runing GoG

```bash
python src/GoG.py --n_process=1 --dataset data/cwq/data_with_ct_0.2.json
```

`data_with_ct_0.2.json` represents IKG-20% in the paper.

The results could be different from that of the original paper, as the gpt-3.5-turbo-0613 we use is not available. We suggest using Qwen-1.5-72b-chat.