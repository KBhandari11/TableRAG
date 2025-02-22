import sys
import os
import torch
import pandas as pd
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from get_gemini_table_title_description import parse_table_summary, parse_table_paths, get_gemini_title_description_data 

def format_table_for_prompt(table_data):
    """
    Formats the table data into the required structured prompt format.
    """
    formatted_table = "<BOC> " + " <SOC> ".join(table_data["header"]) + " <EOC>\n"
    if len(table_data["rows"]) > 5: 
        for row in table_data["rows"][:3]: 
            formatted_row = "<BOR> " + " <SOR> ".join([str(cell) if cell != "" else "-" for cell in row]) + " <EOR>\n"
            formatted_table += formatted_row
        formatted_table += "...\n" 
        for row in table_data["rows"][-2:]: 
            formatted_row = "<BOR> " + " <SOR> ".join([str(cell) if cell != "" else "-" for cell in row]) + " <EOR>\n"
            formatted_table += formatted_row
    else:
        for row in table_data["rows"]:  
            formatted_row = "<BOR> " + " <SOR> ".join([str(cell) if cell != "" else "-" for cell in row]) + " <EOR>\n"
            formatted_table += formatted_row 
    return formatted_table

def get_prompt(table):
    formatted_table = format_table_for_prompt(table)

    # Construct the final prompt
    prompt = f"""
    You are an advanced AI highly skilled in analyzing and understanding structured data within tables. 
    Your task is to generate a meaningful title and description that comprehensively summarizes the provided table's content and context in detail. 
    The table below contains rows of data where intermediate rows are omitted for brevity and represented by "...". 
    Empty cells are denoted with a "-" symbol. 
    Please analyze the data and focus only on cells with meaningful values while predicting the title and description. 

    Your output should follow the format: {{Title: [Predicted Table Title], Description: [Predicted Description]}} 

    Here is the table: 
    {formatted_table}

    Output:

    """
    return formatted_table, prompt


def generate(generator, prompt):
    """
    Generates metadata (title and description) for a given table string.
    """
    output = generator(prompt,max_new_tokens=200)#, do_sample=True, temperature=0.7, top_p=0.9,penalty_alpha=0.4)#
    generated_text = output[0]['generated_text']
    return generated_text.replace(prompt,"")

if __name__ == "__main__":
    model_id = sys.argv[1]
    wtq_table_title_dataset_path = sys.argv[2]
    #gemini_data_path = "/gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/wtq"
    #gemini_table_path_idx, gemini_title_description =  parse_table_paths(f"{gemini_data_path}/tableId_to_path.json"),  parse_table_summary(f"{gemini_data_path}/table_summary_cleaned.csv")
    dataset = load_dataset("Stanford/wikitablequestions")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto"
    )
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    if os.path.isfile(f'{wtq_table_title_dataset_path}/wtq_table_title_description.csv'):
        d = pd.read_csv(f'{wtq_table_title_dataset_path}/wtq_table_title_description.csv')
        data = d.to_dict('records') 
        print(len(data), " already exists.",flush=True)
    else:
        data = []
    for idx, example in enumerate(dataset['validation']): 
        if len(data)> idx:
            continue
        question = example["question"]
        answer = example["answers"] 
        table = example["table"]
        formatted_table, prompt = get_prompt(table)
        #gemini_title_description = get_gemini_title_description_data(example["table"]["name"].split(".")[0]+".csv",gemini_title_description,gemini_table_path_idx)
        llama_title_description = generate(generator, prompt)
        #print("llama",llama_title_description, flush=True)
        data.append({
            "id":example["id"], 
            "question":question,
            "answer":answer,
            "formatted_table":formatted_table,
            "name":example["table"]["name"],
            "table":table,
            "llama3_title_description":llama_title_description,
            #"gemini_title_description":gemini_title_description
        })
        if idx % 50 == 0:
            print("\t DATA:",data[-1],flush=True)
            df = pd.DataFrame(data)
            df.to_csv(f'{wtq_table_title_dataset_path}/wtq_table_title_description.csv',index=False)
            del df 
    df = pd.DataFrame(data)
    df.to_csv(f'{wtq_table_title_dataset_path}/wtq_table_title_description.csv',index=False)
    del df 