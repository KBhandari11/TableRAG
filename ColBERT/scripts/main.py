from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import os
import csv
import json
import argparse

def retrieveTable(gpus,root,table_index_name,queries_path,k,ranking_output_name):
    # Set up ColBERT searching
    with Run().context(RunConfig(nranks=gpus, experiment="wtq-test")):
        config = ColBERTConfig(root=root)

        searcher = Searcher(index=table_index_name, config=config)
        queries = Queries(queries_path)
        ranking = searcher.search_all(queries, k)
        
        # Save the ranking results
        ranking.save(ranking_output_name)
    print(f"Ranking saved to {ranking_output_name}") 
    
def linearize_table(file_path):
    retriever_sequence = []
    reader_sequence = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        headers = rows[0]  # First row is assumed to be the header
        data_rows = rows[1:]  # Remaining rows are data

        # Data Cleaning 
        headers = [header.replace("\n", " ") for header in headers]
        data_rows = [[cell.replace("\n", " ") for cell in row] for row in data_rows]
        
        # Linearize for the retriever component
        # Adarsh view HTML dataset and add title here

        retriever_seq = "<SOT> [table title] <EOT> <BOC> "
        retriever_seq += " <SOC> ".join(headers) + " <EOC>"
        retriever_headers = retriever_seq
        for row in data_rows:
            retriever_seq += " <BOR>" + " <SOR> ".join(row) + " <EOR>"
        retriever_sequence.append(retriever_seq)

        # Linearize for the reader component
        reader_seq = "[HEAD] " + " | ".join(headers)
        for i, row in enumerate(data_rows, start=1):
            reader_seq += f" [ROW] {i} : " + " | ".join(row)
        reader_sequence.append(reader_seq)
    
    return retriever_headers, retriever_sequence[0], reader_sequence[0]

def create_queries_file(queries_file_path):
    # Get a sorted list of all .tsv files in the directory
    # tsv_files = sorted([f for f in os.listdir(directory) if re.match(r'random-split-\d+-train\.tsv$', f)])
    tsv_path = "/scratch/asing725/IP/Multi-Table-RAG/WikiTableQuestions/data/pristine-unseen-tables.tsv"
    # tsv_path = "/scratch/asing725/IP/Multi-Table-RAG/WikiTableQuestions/data/pristine-unseen-tables.tsv"
    with open(tsv_path, 'r', encoding='utf-8') as infile, open(queries_file_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        try:
            reader = csv.DictReader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')
            id_counter = 0
            for row in enumerate(reader):
            # row =  (11309, {'id': 'nt-14145', 'utterance': 'which competition has the highest meters listed?', 'context': 'csv/203-csv/436.csv', 'targetValue': 'World Junior Championships'})
                (number,queries_dict) = row
                question = queries_dict['utterance']  # Extract the 'utterance' column
                writer.writerow([id_counter, question])  # Write the new id and question
                id_counter = id_counter + 1

        except Exception as e:
            print(f"Error reading file {infile}: {e}")

def merge_all_table_files(directory, output_file, onlyHeaders, csv_folder,tableId_to_path, id_counter = 1):
    # Get a sorted list of all .tsv files in the directory
    all_tables = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])
    total_number_of_tables = 0
    with open(output_file, 'a', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        # Initialize ID counter
        for table in all_tables:
            table_path = os.path.join(directory, str(table))
            # try:
            retriever_headers, retriever_linearized, reader_linearized = linearize_table(table_path)
            if onlyHeaders:
                # Write API Call here Adarsh
                writer.writerow([id_counter, retriever_headers])
            else:
                writer.writerow([id_counter, retriever_linearized])
            tableId_to_path.append({id_counter:f"csv/{csv_folder}/{table}"})
            id_counter += 1
            total_number_of_tables = id_counter
            # except Exception as e:
            #     print(f"Error reading file {table}: {e}")

    return total_number_of_tables, tableId_to_path

def check_and_delete_file(file_path):
    # Check if the path exists and is a file
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            os.remove(file_path)
            print(f"File '{file_path}' has been deleted.")
        except Exception as e:
            print(f"An error occurred while deleting the file: {e}")
    else:
        print(f"The file '{file_path}' does not exist.")

def get_table_data(onlyHeaders,table_index_file):
    id_counter = 0 
    tableId_to_path_file = '/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/tableId_to_path.json'
    check_and_delete_file(tableId_to_path_file)
    check_and_delete_file(table_index_file)
    tableId_to_path_data = []
    for i in range(5):    
        directory =   f"/scratch/asing725/IP/Multi-Table-RAG/WikiTableQuestions/csv/20{i}-csv"      
        id_counter, tableId_to_path_data = merge_all_table_files(directory, table_index_file, onlyHeaders,f"20{i}-csv",tableId_to_path_data, id_counter)

        print(f"Parsing 20{i}-csv folder for Tables,  Total {id_counter} tables parsed") 
        
    with open(tableId_to_path_file, 'w') as out_file:
        json.dump(tableId_to_path_data, out_file, indent=4)

def index_table(gpus, table_index_name, onlyHeaders):
    # Set up ColBERT indexing
    table_index_path = ""
    if onlyHeaders:
        table_index_path = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv"
    else:
        table_index_path = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv"
        
    checkpoint_root="/scratch/asing725/IP/Multi-Table-RAG/ColBERT_pretrained/colbertv2.0"
    # print("Indexing Start")
    with Run().context(RunConfig(nranks=gpus, experiment="wtq-test")):
        config = ColBERTConfig(
            nbits=2,
            root=checkpoint_root,
        )
        # print("Loading Indexer")
        indexer = Indexer(checkpoint=checkpoint_root, config=config)
        # print("Indexing Tables ...")
        indexer.index(name=table_index_name, collection=table_index_path, overwrite=True)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Index WikiTableQuestions dataset with ColBERT.")
    parser.add_argument("--k", type=int, default=5,required=True, help="Number of top results to retrieve for each query.")
    parser.add_argument("--full_table_index", type=str, required=True, help="Path to the full table index file.",default="/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv")
    parser.add_argument("--only_header_index", type=str, required=True, help="Path to the only header index file.",default="/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_map.tsv")
    parser.add_argument("--table_index_name", type=str, required=True, help="Name for the index to be created.",default="wtq_full_table.nbits=2")
    parser.add_argument("--ranking_output_name", type=str, required=True,help="Path/Name to save this ranking output.",default="wtq_full_table.ranking.tsv")
    parser.add_argument("--gpus", type=int, default=1,required=True,help="Number of GPUs")
    parser.add_argument("--only_headers", type=int, help="Use only headers for indexing if set.",default=False)

    args = parser.parse_args()

    # Determine which index file to use based on the --only_headers flag
    table_index_path = args.only_header_index if args.only_headers==1 else args.full_table_index
    print("Table Index Path ========================== ",table_index_path)
    # Preprocess the dataset
    # Generates [id, linear_table/only Header]=wtq_table_index.tsv , [id,table_path]=tableId_to_path.json
    get_table_data(args.only_headers, table_index_path)

    index_table(args.gpus,args.table_index_name,args.only_headers)

    # Ranking Similarity between Table and Queries
    # Generates
    queries_file_path = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_question_index.tsv"
    # print("Load Queries file")
    create_queries_file(queries_file_path)
    # print("Generating Retriving Results")
    root = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments"
    retrieveTable(args.gpus,root,args.table_index_name,queries_file_path,args.k,args.ranking_output_name)
