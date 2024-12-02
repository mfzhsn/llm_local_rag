# Understanding Generative AI
# Author: Mohammd Zaman

There are couple of ways how to feed users data to existing LLMs Models. 

There are 2 major methods when it comems to train LLM Models with users data.

1. Fine-Tuning
2. Retrival Augmented Generation (RAG)


## Fine Tuning

Fine tuning involves a specific dataset which needs to be trained on a pretrained LLM Model, It is kind of re-training with users specific data. Fine-tuning is supervised learning where the model and new data both are structured, labeled and organized.

Fine-tuning is out of scope for this repo.


## Retrival Agumented Generation (RAG)

RAG system provides a contineous method of live feeding data to LLM which is reliable and factual which is not already present in the LLM. 

Generally RAG involves 3 steps:

1. Generating Embedding for the users Data
2. Retrival using the Query
3. Augmneted Generation

### Semnatic Search

Also called as Similarity Search 


## Installation

Makes you have a `LM Studio` or also you can use `ollama` which can run LLM models.
In this repo I am using LM Studio with the following details:

| Items    | Links |
| -------- | ------- |
| Embedding Model  |  nomic    |
| Embedding API  |  http://10.0.1.223:1234/v1/embeddings    |
| LLM Model  | llama3.2 8B     |
| LLM API  |   http://10.0.1.223:1234/v1/chat/completions   |


### Required Packages

```
pip install requirements_lm_studio.txt
```

#### Step-1: Generation of documents embeddings

```
step1_generate_embeddings_chromadn.py
```

#### Step-2: Retrival Process

```
step2_with_distances.py
```

#### Step-3: Augmented Generation

```
step3-Generation.py
```


