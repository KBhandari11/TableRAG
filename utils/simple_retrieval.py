from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


import torch

def min_max_normalize(values):
    min_val = min(values)
    max_val = max(values)
    return [(x - min_val) / (max_val - min_val) for x in values]

#Use the BM25 algorithm to find the most relevant table based on the query tokens and the corpus.
def bm25_retrieve_relevant_table(query, corpus,list_tables_info,index_list_tables_info,actual_table_info,top_k_list):
    query_tokens = word_tokenize(query)
    corpus_tokenize = [word_tokenize(table) for table in corpus]

    bm25 = BM25Okapi(corpus_tokenize)
    scores = bm25.get_scores(query_tokens)

    # Get indices of top-k most relevant tables
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(top_k_list)]
    top_k_tables = [list_tables_info[i] for i in top_k_indices]
    top_k_index_list_tables = [index_list_tables_info[i] for i in top_k_indices] 

    # Check if actual_table_info is in any of the top-k lists
    results = {k: actual_table_info in top_k_tables[:k] for k in top_k_list}
    results_index = {k: top_k_index_list_tables[:k] for k in top_k_list}
    results_values = {k: top_k_tables[:k] for k in top_k_list}
    return results, results_index, results_values,top_k_indices, min_max_normalize(scores)


def sentence_embedding_retrieve_relevant_table(model, query, corpus,list_tables_info,index_list_tables_info,actual_table_info,top_k_list):
    device = model.device
    query_tokens = query
    with torch.no_grad():
        query_embedding = model.encode(query_tokens, convert_to_tensor=True)
        table_embeddings = model.encode(corpus, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = torch.nn.functional.cosine_similarity(query_embedding, table_embeddings, dim=-1)

    # Get indices of top-k most relevant tables
    top_k_indices = torch.topk(cosine_scores, max(top_k_list)).indices.tolist()
    top_k_tables = [list_tables_info[i] for i in top_k_indices]
    top_k_index_list_tables = [index_list_tables_info[i] for i in top_k_indices] 

    # Check if actual_table_info is in any of the top-k lists
    results = {k: actual_table_info in top_k_tables[:k] for k in top_k_list}
    results_index = {k: top_k_index_list_tables[:k] for k in top_k_list}
    results_values = {k: top_k_tables[:k] for k in top_k_list}
    return results, results_index, results_values,top_k_indices,min_max_normalize(cosine_scores.tolist())

def dpr_retrieve_relevant_table(models_w_tokenizer,query, corpus, list_tables_info,index_list_tables_info, actual_table_info, top_k_list):
    # Load DPR models and tokenizers
    ((question_tokenizer,context_tokenizer),(question_encoder,context_encoder))= models_w_tokenizer
    device_1 = question_encoder.device
    device_2 = context_encoder.device

    query_inputs = question_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        query_embedding = question_encoder(**query_inputs.to(device_1)).pooler_output 
    
    # Encode the corpus of tables
    table_embeddings = []
    for table in corpus:
        table_inputs = context_tokenizer(table, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            table_embedding = context_encoder(**table_inputs.to(device_2)).pooler_output
        table_embeddings.append(table_embedding)

    # Stack all table embeddings into a single tensor
    table_embeddings_tensor = torch.cat(table_embeddings, dim=0) 

    similarities = torch.nn.functional.cosine_similarity(query_embedding.to("cpu"), table_embeddings_tensor.to("cpu"))

    top_k_indices = torch.argsort(similarities, descending=True)[:max(top_k_list)]
    top_k_tables = [list_tables_info[i] for i in top_k_indices]
    top_k_index_list_tables = [index_list_tables_info[i] for i in top_k_indices] 

    # Check if actual_table_info is in any of the top-k lists
    results = {k: actual_table_info in top_k_tables[:k] for k in top_k_list}
    results_index = {k: top_k_index_list_tables[:k] for k in top_k_list}
    results_values = {k: top_k_tables[:k] for k in top_k_list}
    return results, results_index, results_values, top_k_indices,min_max_normalize(similarities.tolist())