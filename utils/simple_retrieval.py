from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import torch

#Use the BM25 algorithm to find the most relevant table based on the query tokens and the corpus.
def bm25_retrieve_relevant_table(query, corpus,list_tables_info,actual_table_info,top_k_list):
    query_tokens = word_tokenize(query)
    corpus_tokenize = [word_tokenize(table) for table in corpus]

    bm25 = BM25Okapi(corpus_tokenize)
    scores = bm25.get_scores(query_tokens)

    # Get indices of top-k most relevant tables
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(top_k_list)]
    top_k_tables = [list_tables_info[i] for i in top_k_indices]

    # Check if actual_table_info is in any of the top-k lists
    results = {k: actual_table_info in top_k_tables[:k] for k in top_k_list}
    return results


def dpr_retrieve_relevant_table(query, corpus,list_tables_info,actual_table_info,top_k_list):
    # Load a pretrained Sentence Transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_tokens = query
    query_embedding = model.encode(query_tokens, convert_to_tensor=True)
    table_embeddings = model.encode(corpus, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = torch.nn.functional.cosine_similarity(query_embedding, table_embeddings, dim=-1)

    # Get indices of top-k most relevant tables
    top_k_indices = torch.topk(cosine_scores, max(top_k_list)).indices.tolist()
    top_k_tables = [list_tables_info[i] for i in top_k_indices]

    # Check if actual_table_info is in any of the top-k lists
    results = {k: actual_table_info in top_k_tables[:k] for k in top_k_list}
    return results
    