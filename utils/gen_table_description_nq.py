import pandas as pd
from transformers import pipeline
import torch
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

import time
import sys
import os
'''
def get_model_gemini(api_key="GEMINI_API_KEY"):
    genai.configure(api_key=api_key)
    
    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    chat_session = model.start_chat(
        history=[]
    )
    return model, chat_session 
'''

def get_model(model_id):
    generator = pipeline("text-generation", model=model_id,torch_dtype=torch.float16, device_map="auto")
    return generator


def get_data(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip')
    table_index = {}
    for (index,table_text) in df.iterrows():
        table_index[index] = table_text
    return table_index

def format_output(response):
    response = str(response).strip().replace("\n","").replace("\"","")
    response = response.replace("*","")
    if "</think>" in response:
        response = response.split("</think>")[-1]
    elif "</thinking>" in response:
        response = response.split("</thinking>")[-1]
    elif "Table Title:" in response:
       response = "Table Title:"+response.split("Table Title:")[-1] 
    return response
        


def create_dataset(table_index,existing_index):
    prompt_list = []
    idx_list = []
    for idx, (id,table) in enumerate(table_index.items()):
        if id in existing_index:
            continue
        #if id < 2:
        #    continue
        #f.Ensure the description answers potential questions about the data, such as 'What key information is this table providing?' or 'How can this table be used in answering specific questions?'
        prompt = """You are an advanced AI specialized in analyzing and summarizing structured data within tables. 
        Your task is to generate a meaningful title and description for the provided table. 
        The title should be concise and clearly reflect the table's primary content, while the description should summarize the context and provide insights into the key data. 
        The description must be tailored for query ranking, meaning it should help identify the table's relevance to possible questions.
        Guidelines: 
        1. Table Title:    
        a.Provide a concise, clear title that reflects the content and context of the table.   
        b.Be mindful of all shorthand terms in the table and expand them in the title and description. 
        2.Table Description: 
        a.Summarize the main purpose of the table, the key columns, and the types of data it contains.   
        b.Must include abbreviations or terms that might be unclear, expand them in the description.   
        c.Identify and explain any patterns or notable trends. For example, if certain data points frequently appear together or if there are gaps, mention them.   
        d.Point out the significance of key relationships in the data (e.g., correlations between columns or rows).   
        e.If there are columns with categorical data (e.g., 'Region', 'Category'), mention their distribution and any specific categories of interest.  
        
        
        The output should be of format and nothing else.
        "Table Title":"", "TableDescription":""
        Here is the Table: """
        prompt = prompt + table['table_text']
        prompt_list.append(prompt)
        idx_list.append(idx)
    return Dataset.from_dict({"prompt": prompt_list}),idx_list,prompt_list 

if __name__ == "__main__":
    #test => Index 2063
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    print("Cuda Version: ",torch.version.cuda, file=sys.stderr)
    file_path = sys.argv[1]
    output_file =  sys.argv[2] #"/scratch/asing725/IP/Multi-Table-RAG/NQ_Tables/nq_table_summary.json"
    table_index = get_data(file_path)
    gemini = False
    #Get Model
    if gemini:
        api_key ="AIzaSyCcKnMjTmH6NOTyrNT5v-4r1gujGxZpDk8"
        model, chat_session = get_model(api_key=api_key) 
    else:
        model_pipeline = get_model(model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")#"deepseek-ai/DeepSeek-R1-Distill-Llama-70B" 
            
    flag = 0
    print("Starting...",flush=True)
    generated_description_dict = []
    if os.path.isfile(output_file):
        df = pd.read_csv(output_file)
        generated_description_dict = df.to_dict('records')
        existing_index = df["index"].tolist()
        print("Found existing file with title and description generated for:", existing_index, flush=True)
    else:
        existing_index = []
    prompt_dataset,idx_list, prompt_list = create_dataset(table_index,existing_index)

    if gemini:
        for idx, prompt in zip(idx_list, prompt_list): 
            response = str(chat_session.send_message(prompt).text) 
            generated_description_dict.append({"index": id, "Generated": str(response).replace("\n","").replace("\"","")})
            if idx%50==0:
                df = pd.DataFrame(generated_description_dict)
                df.to_csv(output_file,index=False)
    else:
        for idx, prompt, response in zip(idx_list,prompt_list, model_pipeline(KeyDataset(prompt_dataset, "prompt"), batch_size=2,temperature= 0.95,max_new_tokens=1000)):
            generated_response = response[0]["generated_text"]
            generated_response = generated_response.replace(prompt,"")
            generated_response = format_output(generated_response)
            generated_description_dict.append({"index": idx, "Generated": generated_response})
            if idx%10==0: 
               print(idx,str(generated_response), flush=True) 
            if idx%50==0:
                df = pd.DataFrame(generated_description_dict)
                df.to_csv(output_file,index=False)
        
   
    
        
    df = pd.DataFrame(generated_description_dict)
    df.to_csv(output_file,index=False)
#python ./utils/gen_table_description_nq.py ./data/nqtables/nq_tables_linearized.csv ./data/nqtables/nq_table_summary.csv > result-nq.txt