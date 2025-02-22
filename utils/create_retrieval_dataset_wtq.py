import random
import json
import copy
import pandas as pd
from ast import literal_eval
from datasets import load_dataset
from get_gemini_table_title_description import parse_table_summary, parse_table_paths, get_gemini_title_description_data 



# Using json.JSONEncoder for customization
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tuple):
            return list(obj)
        return super().default(obj)
    
def table_to_text(table):
    """
    Converts a table (list of lists) to a string.
    Each row is concatenated with spaces, and rows are joined with newlines.
    """
    return " ".join([" ".join(map(str, row)) for row in table])


def create_table_retrieval_dataset(list_tables, data, num_table_retrieval=50):
    '''
    Creates a new dataset that selects random other num_table_retrieval number of table information for each datapoint. 
    Objective: Use the "summary" to extract the relevant table among the list of num_table_retrieval tables 
    '''
    new_dataset = []
    for idx, data_point in enumerate(data):
        new_data_point = {}
        sampled_keys = random.sample([key for key in list_tables.keys() if key !=  data_point["table"]["name"].split(".")[0]], num_table_retrieval-1)
        sampled_keys.append(data_point["table"]["name"].split(".")[0])
        random.shuffle(sampled_keys) 
        new_data_point["index"] =  data_point["id"]
        new_data_point["table_index"] = data_point["table"]["name"].split(".")[0]
        new_data_point["actual_table_info"] = list_tables[data_point["table"]["name"].split(".")[0]]  
        #new_data_point["list_title_tab-description_retrieval"] = (list_tables[idx]["title_table-description"], [list_tables[key]["title_table-description"] for key in sampled_keys])
        #new_data_point["list_title_column_header_retrieval"] = (list_tables[idx]["title_col_info"],[list_tables[key]["title_col_info"] for key in sampled_keys])
        #new_data_point["list_title_col_table_retrieval"] = (list_tables[idx]["title_col_table_text"],[list_tables[key]["title_col_table_text"] for key in sampled_keys])
        new_data_point["random_retrieval_sample"] = (data_point["table"]["name"].split(".")[0], [key for key in sampled_keys])
        new_data_point["question"] = data_point["question"]
        new_data_point["answer"] = data_point["answers"][0] 
        new_data_point["table"] = [data_point["table"]["header"]]+data_point["table"]["rows"]
        new_dataset.append(new_data_point)
    return new_dataset


    
def get_table_list(data,gemini_table_path_idx_dict, gemini_title_description_dict ):
    table_list = {}
    for idx, data_point in enumerate(data["validation"]):
        if data_point["id"] in table_list:
            continue 
        
        table = [data_point["table"]["header"]]+data_point["table"]["rows"]
        header_info = table_to_text([data_point["table"]["header"]])
        title_description = get_gemini_title_description_data(data_point["table"]["name"].split(".")[0]+".csv",gemini_title_description_dict,gemini_table_path_idx_dict)
        assert gemini_title_description != None 

        table_list[data_point["table"]["name"].split(".")[0]] = {"index":data_point["table"]["name"].split(".")[0], 
                           "title_table-description": f"{title_description}",
                           "column_header_info": header_info,
                           "title_col_info": f"{title_description} {header_info}",
                           "table_text": table_to_text(table),
                           "title_col_table_text": f"{title_description} {table_to_text(table)}", 
                           }
    for idx, data_point in enumerate(data["train"]):
        if data_point["table"]["name"].split(".")[0] in table_list:
            continue 
        table = [data_point["table"]["header"]]+data_point["table"]["rows"]
        header_info = table_to_text([data_point["table"]["header"]])
        title_description = get_gemini_title_description_data(data_point["table"]["name"].split(".")[0]+".csv",gemini_title_description_dict,gemini_table_path_idx_dict)
        assert gemini_title_description != None 

        table_list[data_point["table"]["name"].split(".")[0]] = {"index":data_point["table"]["name"].split(".")[0], 
                           "title_table-description": f"{title_description}",
                           "column_header_info": header_info,
                           "title_col_info": f"{title_description} {header_info}",
                           "table_text": table_to_text(table),
                           "title_col_table_text": f"{title_description} {table_to_text(table)}", 
                           } 
    for idx, data_point in enumerate(data["test"]):
        if data_point["table"]["name"].split(".")[0] in table_list:
            continue 
        table = [data_point["table"]["header"]]+data_point["table"]["rows"]
        header_info = table_to_text([data_point["table"]["header"]])
        title_description = get_gemini_title_description_data(data_point["table"]["name"].split(".")[0]+".csv",gemini_title_description_dict,gemini_table_path_idx_dict)
        assert gemini_title_description != None 

        table_list[data_point["table"]["name"].split(".")[0]] = {"index":data_point["table"]["name"].split(".")[0], 
                           "title_table-description": f"{title_description}",
                           "column_header_info": header_info,
                           "title_col_info": f"{title_description} {header_info}",
                           "table_text": table_to_text(table),
                           "title_col_table_text": f"{title_description} {table_to_text(table)}", 
                           }   
    return table_list

if __name__ == "__main__":
    gemini_data_path = "/gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/wtq"
    wtq_retrieval_path = "/gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/wtq/retrieval"

    gemini_table_path_idx, gemini_title_description =  parse_table_paths(f"{gemini_data_path}/tableId_to_path.json"),  parse_table_summary(f"{gemini_data_path}/table_summary_cleaned.csv") 
    data = load_dataset("Stanford/wikitablequestions")
    table_list = get_table_list(data,gemini_table_path_idx, gemini_title_description)

    # save the table_list as a csv file to index the data
    table_df = pd.DataFrame([table_list[idx] for idx in table_list])
    table_df.to_csv('/gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/wtq/table_list.csv', index=False)


    for n_val in [50, 100, 200, 500,2000, "all"]: #[50, 100, 200, 500, 
        if n_val == "all":
            n = len(table_list)
        else:
            n = n_val
        new_dataset = create_table_retrieval_dataset(table_list, data["validation"], num_table_retrieval=n)
        with open(f"{wtq_retrieval_path}/wtq_retrieval_{n_val}.json", 'w') as f:
            json.dump(new_dataset, f, cls=CustomEncoder, indent=4)  # `indent=4` for pretty formatting
        print(f"Data successfully saved to data/retrieval_data/wtq_retrieval_{n_val}.json",flush=True)