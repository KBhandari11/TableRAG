import json
from TableRAG.src.ColBERT.scripts.retrieve_results_colbert import load_table_index
from pathlib import Path

# File paths
script_dir = Path(__file__).resolve().parent.parent
wtq_table_index_file = script_dir / "/wtq_table_index.tsv"
wtq_question_index_file = script_dir / "/wtq_question_index.tsv"
random_split_train_file = "/scratch/asing725/IP/Multi-Table-RAG/WikiTableQuestions/data/random-split-1-train.tsv"
wtq_full_table_ranking_file = script_dir / "/experiments/msmarco/tableRetriever/2025-01/08/19.48.25/wtq_full_table.ranking.tsv"
output_file = script_dir / "data/retrievalResult_.json"
triples_file = "triples.jsonl"

def generate_triples_file():
    with open("retrievalResult_.json", "r") as file:
        data = json.load(file)
    
    # Open the output file for writing
    with open(triples_file, "r") as f:
        for entry in data:
            question_id = entry.get("question-id", "N/A")
            gold_table = entry.get("Gold Table", "N/A")
            # negative_table= entry.get("retrieved tables")
            negative_table = entry.get("retrieved tables", [])
        
            # Initialize variables to store the table with the least rank that is not the gold table
            least_rank_table_id = None
            
            # Iterate through the retrieved tables
            for table in negative_table:
                table_id = table.get("TableID")                
                # Ensure rank is numeric and the table is not the gold table
                if table_id != gold_table:
                    least_rank_table_id = table_id
                    break
                    
                  # Ensure we're writing negative examples
            triple = [question_id, gold_table, least_rank_table_id]
            f.write(json.dumps(triple) + "\n")
                
    
    print(f"Data has been successfully written to {triples_file}")