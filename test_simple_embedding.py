import json
import sys
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

from utils.simple_retrieval import bm25_retrieve_relevant_table, dpr_retrieve_relevant_table


# Download NLTK tokenizer data if not already available
nltk.download('punkt')
nltk.download('punkt_tab')
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
    result = []
    for k in [50,100,200,500]:
        totto_retrieval_dataset_path = f"./data/retrieval_data/totto_retrieval_{k}.json" 
        top_k_list = [1, 5, 10, 15, 20]
        # Open and read the JSON file
        #with open(wtq_dataset_path, 'r') as file:
        #    data = json.load(file)

        
        data = []
        with open(totto_retrieval_dataset_path, 'r') as file:
            data = json.load(file)
        print("total tables: ", len(data))
        for idx, data_point in enumerate(data):
            # Prepare the query text
            query_text = f"{data_point['summary']}".lower()
            if idx % 50 == 0:
                print(f"{idx}", flush=True)
                
            for table_info in ["title_tab-description","title_column_header", "title_col_table","exact_row"]:
                actual_table_info, list_all_table_info = data_point[f"list_{table_info}_retrieval"] 
                corpus = [table.lower() for table in list_all_table_info]
                bm25_top_k_found_it = bm25_retrieve_relevant_table(query_text, corpus,list_all_table_info,actual_table_info,top_k_list) 
                dpr_top_k_found_it = dpr_retrieve_relevant_table(query_text, corpus,list_all_table_info,actual_table_info,top_k_list)  
                
                for top_k in top_k_list: 
                    result.append({"idx":idx,
                                "top_k":top_k,
                                "table_info":table_info , 
                                "bm_25":bm25_top_k_found_it[top_k],
                                "dpr":dpr_top_k_found_it[top_k]
                                })
                if idx % 50 == 0:
                    print(f"\t{table_info}: BM: {bm25_top_k_found_it}, DPR: {dpr_top_k_found_it}", flush=True)
                    if table_info == "exact_row": 
                        df = pd.DataFrame(result)
                        print(df["bm_25"].mean(),df["dpr"].mean())
                        df.to_csv(f'./results/results_{k}.csv',index=False)
                        del df 
        df = pd.DataFrame(result)
        print(df["bm_25"].mean(),df["dpr"].mean())
        df.to_csv(f'./results/results_{k}.csv',index=False)