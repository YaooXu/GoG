# Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering

This repository contains official implementation for paper `Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering`.

In this documentation, we detail how to construct Incomplete KGs and run GoG.

## Freebase Setup

See ./Freebase/README.md

## Selecting Crucial Triples

```bash
python src/generate_samples_with_crucial_edges.py
```

## Runing GoG

```bash
python src/GoG.py --n_process=4 --dataset data/cwq/data_with_ct_1000_-1_1.json --LLM_type gpt-3.5-turbo
```