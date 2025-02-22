import re
import json 
import pandas as pd


def parse_table_summary(csv_file):
    df = pd.read_csv(csv_file)
    table_info = {
        str(row["Table Index"]): f"{row['Table Title']} {row['Table Description']}"    
        for _, row in df.iterrows()
    }
    return table_info


# Function to parse table ID to path mapping
def parse_table_paths(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    table_paths = {}
    for entry in data:
        for key, value in entry.items():
            table_paths[str(value)] = str(key)  # Reverse mapping: file path -> table ID
    
    return table_paths


def get_gemini_title_description_data(file_name, table_info, table_paths):
    if file_name in table_paths:
        table_id = str(table_paths[file_name])
        if table_id in table_info:
            return table_info[table_id]
    return None