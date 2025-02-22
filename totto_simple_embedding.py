import json
import sys
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import torch
import os


from sentence_transformers import SentenceTransformer
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from utils.simple_retrieval import bm25_retrieve_relevant_table, dpr_retrieve_relevant_table,sentence_embedding_retrieve_relevant_table


# Download NLTK tokenizer data if not already available
nltk.download('punkt', download_dir='/gpfs/u/home/LLMG/LLMGbhnd/scratch/')
nltk.download('punkt_tab', download_dir='/gpfs/u/home/LLMG/LLMGbhnd/scratch/')
nltk.data.path.append('/gpfs/u/home/LLMG/LLMGbhnd/scratch/')


def get_table_list():
    table_list = pd.read_csv("/gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/totto/table_list.csv", delimiter=',')
    return table_list.reset_index()

def get_table_info_sample(table_list,table_info_name, actual_index,random_sample_index_list):
    table_info_name_list = table_list[table_info_name].tolist()
    actual = table_list[table_list["index"] == int(actual_index)]
    return actual[table_info_name].iloc[0] ,[table_info_name_list[int(idx)] for idx in random_sample_index_list]


def cascading_retrieval(table_list,retrieval_data,query_text,actual_index,method,model=None):
    if method == "bm25":
        if not(retrieval_data["title_table-description"][0]):
            return  "title_table-description", 200, False
        first_retrieval_index = retrieval_data["title_table-description"][1][200]
        actual_table_info, first_retrieval_list_tables_info = get_table_info_sample(table_list,"title_col_info", actual_index,first_retrieval_index)
        first_retrieval_corpus = [table.lower() for table in first_retrieval_list_tables_info]
        bm25_top_k_found_it, bm25_retrieved_index, bm25_retrieved_values = bm25_retrieve_relevant_table(query_text, first_retrieval_corpus,first_retrieval_list_tables_info,first_retrieval_index,actual_table_info,[1,50])  
        if bm25_top_k_found_it[50]: 
            second_retrieval_index = bm25_retrieved_index[50]
            actual_table_info, second_retrieval_list_tables_info = get_table_info_sample(table_list,"title_col_table_text", actual_index,second_retrieval_index)
            second_retrieval_corpus = [table.lower() for table in second_retrieval_list_tables_info] 
            bm25_top_k_found_it, bm25_retrieved_index, bm25_retrieved_values = bm25_retrieve_relevant_table(query_text, second_retrieval_corpus,second_retrieval_list_tables_info,second_retrieval_index,actual_table_info,[1, 5, 10, 20, 30])  
            for k in  [1, 5, 10, 20, 30]:
               if bm25_top_k_found_it[k]:
                    return "title_col_table_text", k, True  
            return "title_col_table_text", k, False 
        else:
            return "title_col_info", 50, False 

    if method == "sentence_embed":
        if not(retrieval_data["title_table-description"][0]):
            return  "title_table-description", 200, False 
        first_retrieval_index = retrieval_data["title_table-description"][1][200]
        actual_table_info, first_retrieval_list_tables_info = get_table_info_sample(table_list,"title_col_info", actual_index,first_retrieval_index)
        first_retrieval_corpus = [table.lower() for table in first_retrieval_list_tables_info] 
        sentence_top_k_found_it, sentence_retrieved_index, sentence_retrieved_values = sentence_embedding_retrieve_relevant_table(model,query_text, first_retrieval_corpus,first_retrieval_list_tables_info,first_retrieval_index,actual_table_info,[1,50])  
        if sentence_top_k_found_it[50]: 
            second_retrieval_index = sentence_retrieved_index[50]
            actual_table_info, second_retrieval_list_tables_info = get_table_info_sample(table_list,"title_col_table_text", actual_index,second_retrieval_index)
            second_retrieval_corpus = [table.lower() for table in second_retrieval_list_tables_info] 
            sentence_top_k_found_it, sentence_retrieved_index, sentence_retrieved_values = sentence_embedding_retrieve_relevant_table(model,query_text, second_retrieval_corpus,second_retrieval_list_tables_info,second_retrieval_index,actual_table_info,[1, 5, 10, 20, 30])  
            for k in  [1, 5, 10, 20, 30]:
               if sentence_top_k_found_it[k]:
                    return "title_col_table_text", k, True  
            return "title_col_table_text", k, False 
        else:
            return "title_col_info", 50, False 


def preprocess(text):
    """Tokenize and preprocess text."""
    return word_tokenize(text.lower())

def table_to_text(table):
    """
    Converts a table (list of lists) to a string.
    Each row is concatenated with spaces, and rows are joined with newlines.
    """
    return " ".join([" ".join(map(str, row)) for row in table])

def compare_retrievals(results, k_values):
    comparisons = {}
    contexts = list(results.keys())
    for i, ctx1 in enumerate(contexts):
        for ctx2 in contexts[i+1:]:
            comparisons[f"{ctx1} vs {ctx2}"] = {}
            for k in k_values:
                set1, set2 = set(results[ctx1][1][k]), set(results[ctx2][1][k])
                intersection = set1 & set2
                union = set1 | set2
                difference = set1 - set2
                jaccard = len(intersection) / len(union) if union else 0
                comparisons[f"{ctx1} vs {ctx2}"][k] = {
                    "intersection": len(intersection),
                    "union": len(union),
                    "difference": len(difference),
                    "jaccard_similarity": jaccard,
                }
    return comparisons


if __name__ == "__main__":
    data_path =f"/gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data"
    #for k in [50,100,200,500]:
    n = sys.argv[1]
    if n != "all":
        n = int(n)
    
    totto_retrieval_dataset_path = f"{data_path}/totto/retrieval/totto_retrieval_{n}.json" 
    top_k_list = [1, 5, 10, 20, 30, 50, 100, 200, 250, 300]

    result = []
    
    data = []
    with open(totto_retrieval_dataset_path, 'r') as file:
        data = json.load(file)
    print("total tables: ", len(data))

    table_list = get_table_list()

    # Load a pretrained Sentence Transformer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"torch_dtype": "float16"}).to("cuda:0" ) 
    '''question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to("cuda:1" )
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to("cuda:2")

    dpr_model = ((question_tokenizer,context_tokenizer),(question_encoder,context_encoder))'''
    if os.path.isfile( f'{data_path}/results/totto/results_independent_answer_{n}.csv'):
        d = pd.read_csv(f'{data_path}/results/totto/results_independent_answer_{n}.csv')
        result = d.to_dict('records') 
        df_exists = len(d["idx"].unique().tolist())
        print(df_exists, " already exists.",flush=True)
    else:
        df_exists = 0
        result = []
    for idx, data_point in enumerate(data):
        # Prepare the query text
        if df_exists > idx:
            continue
        query_text = f"{data_point['summary']}".lower()
        
        if idx % 50 == 0:
            print(f"{idx}", flush=True)
        retrieved_data = {"bm25":{}, "sentence_embed": {}}
        for table_info in ["title_table-description","title_col_info", "title_col_table_text"]:
            actual_table_info, list_all_table_info = get_table_info_sample(table_list,table_info, data_point[f"index"],data_point[f"random_retrieval_sample"][1]) 
            corpus = [table.lower() for table in list_all_table_info]
            bm25_top_k_found_it, bm25_retrieved_index, bm25_retrieved_values = bm25_retrieve_relevant_table(query_text, corpus,list_all_table_info,data_point[f"random_retrieval_sample"][1],actual_table_info,top_k_list) 
            sentence_top_k_found_it, sentence_retrieved_index, sentence_retrieved_values = sentence_embedding_retrieve_relevant_table(sentence_model,query_text, corpus,list_all_table_info,data_point[f"random_retrieval_sample"][1],actual_table_info,top_k_list) 
            #dpr_top_k_found_it = dpr_retrieve_relevant_table(dpr_model,query_text, corpus,list_all_table_info,actual_table_info,top_k_list)  

            retrieved_data["bm25"][table_info] = (bm25_top_k_found_it[200], bm25_retrieved_index, bm25_retrieved_values) 
            retrieved_data["sentence_embed"][table_info] = (sentence_top_k_found_it[200], sentence_retrieved_index, sentence_retrieved_values) 

            for top_k in top_k_list: 
                result.append({"idx":idx,
                            "top_k":top_k,
                            "type":"independent",
                            "table_info":table_info , 
                            "bm_25":bm25_top_k_found_it[top_k],
                            "sentence_embed":sentence_top_k_found_it[top_k],
                            #"dpr":dpr_top_k_found_it[top_k]
                            })
        bm25_compare_retrievals = compare_retrievals(retrieved_data["bm25"], top_k_list)
        sentence_compare_retrievals = compare_retrievals(retrieved_data["sentence_embed"], top_k_list)
        for comparison in bm25_compare_retrievals:
            for top_k in top_k_list: 
                result.append({"idx":idx,
                        "top_k":top_k,
                        "type":"compare_retrievals",
                        "table_info": comparison, 
                        "bm_25": bm25_compare_retrievals[comparison][top_k],
                        "sentence_embed":sentence_compare_retrievals[comparison][top_k],
                        })    
                 
        # Cascading Results 
        bm_cascade = cascading_retrieval(table_list,retrieved_data["bm25"],query_text,data_point[f"index"],"bm25") 
        sentence_cascade = cascading_retrieval(table_list,retrieved_data["sentence_embed"],query_text,data_point[f"index"],"sentence_embed",model=sentence_model) 

        result.append({"idx":idx,
                "top_k":None,
                "type":"cascading_retrieval",
                "table_info": None, 
                "bm_25":list(bm_cascade),
                "sentence_embed":list(sentence_cascade),
                #"dpr":dpr_top_k_found_it[top_k]
                })   
        
        if idx % 50 == 0:
            print(idx)
            df = pd.DataFrame(result)
            df.to_csv(f'{data_path}/results/totto/results_independent_answer_{n}.csv',index=False)
            del df 
    print("Completed")
    df = pd.DataFrame(result)
    df.to_csv(f'{data_path}/results/totto/results_independent_answer_{n}.csv',index=False)