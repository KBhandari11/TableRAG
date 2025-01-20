from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import os
import csv
import argparse

def create_queries_file(directory, output_file,id_counter):
    # Get a sorted list of all .tsv files in the directory
    # tsv_files = sorted([f for f in os.listdir(directory) if re.match(r'random-split-\d+-train\.tsv$', f)])
    tsv_file = "/scratch/asing725/IP/Multi-Table-RAG/WikiTableQuestions/data/random-split-1-train.tsv"
    tsv_path = os.path.join(directory, tsv_file)
    id_counter = 1
    with open(tsv_path, 'r', encoding='utf-8') as infile, open(output_file, 'a', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        try:
            reader = csv.DictReader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')

            for row in enumerate(reader):
            # row =  (11309, {'id': 'nt-14145', 'utterance': 'which competition has the highest meters listed?', 'context': 'csv/203-csv/436.csv', 'targetValue': 'World Junior Championships'})
                (number,queries_dict) = row
                question = queries_dict['utterance']  # Extract the 'utterance' column
                writer.writerow([id_counter, question])  # Write the new id and question
                id_counter = id_counter + 1

        except Exception as e:
            print(f"Error reading file {infile}: {e}")

def index_wtq_queries():
    output_file = "/scratch/asing725/IP/Multi-Table-RAG/robust-tableqa/src/ColBERT/wtq_question_index.tsv"
    directory = "/scratch/asing725/IP/Multi-Table-RAG/WikiTableQuestions/data"      
    create_queries_file(directory, output_file, 1)
    
if __name__ == '__main__':
    index_wtq_queries()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Search WikiTableQuestions queries using ColBERT.")

    parser.add_argument("--root", type=str, required=True,
                        help="Path to the ColBERT experiments root directory.",default= "/scratch/asing725/IP/Multi-Table-RAG/robust-tableqa/src/ColBERT/experiments")
    parser.add_argument("--index_name", type=str, required=True,
                        help="Name of the index to use for searching/Samme as in Indexing",default="wtq_full_table.nbits=2")
    parser.add_argument("--queries_path", type=str, required=True,
                        help="Path to the queries file.",default="/scratch/asing725/IP/Multi-Table-RAG/robust-tableqa/src/ColBERT/wtq_question_index.tsv")
    parser.add_argument("--ranking_output_name", type=str, required=True,
                        help="Path/Name to save this ranking output.",default="wtq_full_table.ranking.tsv")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use for searching.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of top results to retrieve for each query.")

    args = parser.parse_args()

    # Set up ColBERT searching
    with Run().context(RunConfig(nranks=args.gpus, experiment="wtq")):
        config = ColBERTConfig(root=args.root)

        searcher = Searcher(index=args.index, config=config)
        queries = Queries(args.queries_path)
        ranking = searcher.search_all(queries, k=args.k)
        
        # Save the ranking results
        ranking.save(args.ranking_output_path)

    print(f"Ranking saved to {args.ranking_output_path}")

# print(f"Retrieval JSON file '{output_file}s' generated successfully.")
# def generate_triples_file(gold_table_path):
#     table_index = load_table_index(wtq_table_index_file)
#     with open("retrievalResult_.json", "r") as file:
#         data = json.load(file)
    
#     # Open the output file for writing
#     with open(triples_file, "w") as f:
#         for entry in data:
#             question_id = entry.get("question-id", "N/A")
#             gold_table = entry.get("Gold Table", "N/A")
#             # negative_table= entry.get("retrieved tables")
#             negative_table = entry.get("retrieved tables", [])
        
#             # Initialize variables to store the table with the least rank that is not the gold table
#             min_rank = float('inf')  # Set to infinity initially
#             least_rank_table_id = None
            
#             # Iterate through the retrieved tables
#             for table in negative_table:
#                 table_id = table.get("TableID")                
#                 # Ensure rank is numeric and the table is not the gold table
#                 if table_id != gold_table:
#                     least_rank_table_id = table_id
#                     break
                    
#                   # Ensure we're writing negative examples
#             triple = [question_id, gold_table, least_rank_table_id]
#             f.write(json.dumps(triple) + "\n")
                
    
#     print(f"Data has been successfully written to {triples_file}")
# generate_triples_file(gold_table_path)

