import json
import sys
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import torch
from ast import literal_eval
import os



from sentence_transformers import SentenceTransformer
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from utils.simple_retrieval import bm25_retrieve_relevant_table, dpr_retrieve_relevant_table,sentence_embedding_retrieve_relevant_table


# Download NLTK tokenizer data if not already available
nltk.download('punkt', download_dir='/gpfs/u/home/LLMG/LLMGbhnd/scratch/')
nltk.download('punkt_tab', download_dir='/gpfs/u/home/LLMG/LLMGbhnd/scratch/')
nltk.data.path.append('/gpfs/u/home/LLMG/LLMGbhnd/scratch/')


def get_table_list():
    table_list = pd.read_csv("/gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/wtq/table_list.csv", delimiter=',')
    return table_list.reset_index()

def get_table_info_sample(table_list,table_info_name, actual_index,random_sample_index_list):
    table_info_name_list = []
    for rand_sampl in random_sample_index_list:
        rand_sampl_info = table_list[table_list["index"] == rand_sampl][table_info_name].iloc[0] 
        table_info_name_list.append(rand_sampl_info)

    #table_info_name_list = table_list[table_list["index"].isin(random_sample_index_list)][table_info_name].tolist() 
    actual = table_list[table_list["index"] == actual_index]
    return actual[table_info_name].iloc[0] , table_info_name_list

def preprocess(text):
    """Tokenize and preprocess text."""
    return word_tokenize(text.lower())

def table_to_text(table):
    """
    Converts a table (list of lists) to a string.
    Each row is concatenated with spaces, and rows are joined with newlines.
    """
    return " ".join([" ".join(map(str, row)) for row in table])


if __name__ == "__main__":
    data_path =f"/gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data"
    #for k in [50,100,200,500]:
    n = sys.argv[1]
    if n != "all":
        n = int(n)
    result = []
    
    wtq_retrieval_dataset_path = f"{data_path}/wtq/retrieval/wtq_retrieval_{n}.json" 
    top_k_list = [1, 5, 10, 20, 30, 50, 100, 200, 250, 300]
    top_k_list_short = [1, 5, 10, 20, 30, 50]
    # Open and read the JSON file
    #with open(wtq_dataset_path, 'r') as file:
    #    data = json.load(file)

    data = []
    with open(wtq_retrieval_dataset_path, 'r') as file:
        data = json.load(file)
    print("total tables: ", len(data))
    table_list = get_table_list()

    # Load a pretrained Sentence Transformer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"torch_dtype": "float16"}).to("cuda:0" ) 
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to("cuda:1" )
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to("cuda:2")

    dpr_model = ((question_tokenizer,context_tokenizer),(question_encoder,context_encoder))
    

    if os.path.isfile( f'{data_path}/results/wtq/results_retieval_algo_independent_question_dpr_{n}.csv'):
        d = pd.read_csv(f'{data_path}/results/wtq/results_retieval_algo_independent_question_dpr_{n}.csv')
        result = d.to_dict('records') 
        df_exists = len(d["idx"].unique().tolist())
        print(df_exists, " already exists.",flush=True)
    else:
        df_exists = 0
        result = []
    for idx, data_point in enumerate(data):
        if df_exists > idx:
            continue
        # Prepare the query text
        query_text = f"{data_point['question']}".lower()
        if idx % 50 == 0:
            print(f"{idx}", flush=True)
        
        ### N0(ALL) ------> N1(250) ###
        N1 = 250
        N_retrieval_index = data_point[f"random_retrieval_sample"][1] #ALL table
        N_idx_table_info, N_table_info = get_table_info_sample(table_list, "title_col_table_text", data_point[f"table_index"], N_retrieval_index) 
        N_corpus = [table.lower() for table in N_table_info]
        bm25_top_found_it, bm25_retrieved_table_index, bm25_retrieved_values,bm25_retrieved_indices, bm25_retrieved_scores = bm25_retrieve_relevant_table(query_text, N_corpus, N_table_info,N_retrieval_index, N_idx_table_info,top_k_list) 
        
        ### N1(250) ------> N2(50) ### 
        N2 = 50#0.2*N1
        if not(bm25_top_found_it[N1]):
            N1_retrieval_index = data_point[f"random_retrieval_sample"][1] #ALL table
            N1_idx_table_info, N1_table_info = get_table_info_sample(table_list, "title_col_info", data_point[f"table_index"], N1_retrieval_index) 
            N1_corpus = [table.lower() for table in N1_table_info]
            #sentence_top_found_it, sentence_retrieved_table_index, sentence_retrieved_values,sentence_retrieved_indices,sentence_retrieved_scores = sentence_embedding_retrieve_relevant_table(sentence_model,query_text, N1_corpus,N1_table_info,N1_retrieval_index,N1_idx_table_info,top_k_list_short) 
            sentence_top_found_it, sentence_retrieved_table_index, sentence_retrieved_values,sentence_retrieved_indices,sentence_retrieved_scores = dpr_retrieve_relevant_table(dpr_model,query_text, N1_corpus,N1_table_info,N1_retrieval_index,N1_idx_table_info,top_k_list_short) 
            rerank_bm_score = bm25_retrieved_scores
        else:
            N1_retrieval_index = bm25_retrieved_table_index[N1] #250 table
            N1_idx_table_info, N1_table_info = get_table_info_sample(table_list, "title_col_info", data_point[f"table_index"], N1_retrieval_index) 
            N1_corpus = [table.lower() for table in N1_table_info]
            #sentence_top_found_it, sentence_retrieved_table_index, sentence_retrieved_values,sentence_retrieved_indices, sentence_retrieved_scores = sentence_embedding_retrieve_relevant_table(sentence_model,query_text, N1_corpus,N1_table_info,N1_retrieval_index,N1_idx_table_info,top_k_list_short) 
            sentence_top_found_it, sentence_retrieved_table_index, sentence_retrieved_values,sentence_retrieved_indices, sentence_retrieved_scores = dpr_retrieve_relevant_table(dpr_model,query_text, N1_corpus,N1_table_info,N1_retrieval_index,N1_idx_table_info,top_k_list_short) 
            rerank_bm_score = [bm25_retrieved_scores[top_index] for top_index in bm25_retrieved_indices[:N1]]


        for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            beta = round(1 - alpha, 1)
            #print("Score: ",[(alpha*bm_score,beta*sentence_score) for bm_score,sentence_score in zip(shorten_bm_score,sentence_retrieved_scores)] ) 
            combined_score = [alpha*bm_score+beta*sentence_score for bm_score,sentence_score in zip(rerank_bm_score,sentence_retrieved_scores)] 
            final_top_k_indices = sorted(range(len(combined_score)), key=lambda i: combined_score[i], reverse=True)[:max(top_k_list_short)]
            
            top_k_tables = [N1_table_info[i] for i in final_top_k_indices]
            top_k_index_list_tables = [N1_retrieval_index[i] for i in final_top_k_indices] 

            final_results = {k: N1_idx_table_info in top_k_tables[:k] for k in top_k_list_short}
            final_results_index = {k: top_k_index_list_tables[:k] for k in top_k_list_short}
            final_results_values = {k: top_k_tables[:k] for k in top_k_list_short}


            result.append({"idx":idx,
                        "first_retrieval": bm25_top_found_it[N1],
                        "first_retrieval_info": bm25_retrieved_table_index, 
                        "second_retrieval": sentence_top_found_it[N2],
                        "second_retrieval_info": sentence_retrieved_table_index,  
                        "alpha":alpha,
                        "beta":beta,
                        "final_retrieval": final_results,
                        "final_retrieval_info": final_results_index,
                        "actual_table": data_point[f"table_index"],
                        "question":data_point[f"question"],
                        "answer":data_point[f"answer"],
                        }) 
        if idx % 50 == 0:
            print(idx,data_point[f"table_index"],bm25_top_found_it[N1],sentence_top_found_it[N2],final_results_index)
            df = pd.DataFrame(result)
            df.to_csv(f'{data_path}/results/wtq/results_retieval_algo_independent_question_dpr_{n}.csv',index=False)
            del df
    print("Completed")
    df = pd.DataFrame(result)
    df.to_csv(f'{data_path}/results/wtq/results_retieval_algo_independent_question_dpr_{n}.csv',index=False)