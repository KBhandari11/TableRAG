import json
import csv
import os
import traceback
from tqdm import tqdm  # Progress bar library
from datetime import datetime

# File paths
wtq_table_index_file = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv"
# wtq_question_index_file = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_question_index.tsv"
random_split_train_file = "/scratch/asing725/IP/Multi-Table-RAG/WikiTableQuestions/data/pristine-unseen-tables.tsv"
triples_file = "triples.jsonl"

# Load wtq_table_index into a dictionary {tid: FullTable}
def load_table_index(file_path):
    table_index = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            table_id, full_table = row
            table_index[table_id] = full_table
    return table_index

# Load wtq_question_index into a dictionary {qid: question}
def load_question_index(file_path):
    question_index = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            qid, question, gold_table_path , answer = row
            question_index[qid] = [question,gold_table_path,answer]
    return question_index

# Load random-split-1-train.tsv into a dictionary {qid: answer}
def load_answers(file_path):
    answers = {}
    gold_table_paths = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        flag = True
        for row in reader:
            if flag:
                flag = False
                continue
            qid, question, gold_table_path, answer = row
            answers[qid] = answer
            gold_table_paths[qid] = gold_table_path
    return answers, gold_table_paths

def find_key_in_contexts(search_value):
    file_path = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/tableId_to_path.json"
    try:
        with open(file_path, 'r') as file:
            contexts = json.load(file)
            for entry in contexts:
                for key, value in entry.items():
                    if str(value) == str(search_value):
                        return key
            return None
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Generate the JSON file
def generate_result_file(k, wtq_full_table_ranking_file, view_results, full_table_index_file = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv"):
    # print("Load Question Index")
    question_index = load_question_index(random_split_train_file)
    # print("Load Gold Tables")
    # answers, gold_table_path = load_answers(random_split_train_file)
    table_index = load_table_index(full_table_index_file)
    results = []
    count_retrieved_gold_table = 0
    total_queries = 0
    with open(wtq_full_table_ranking_file, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        current_qid = None
        current_object = None
        # print("Reading from rankings started")s
        for row in tqdm(reader, desc=f"Calculating results for Recall@{k}"):
            qid, tid, rank, retrieval_score = row
            [question,gold_table_path,answer] = question_index["nu-"+qid]
            gold_table_id = find_key_in_contexts(gold_table_path)

            if gold_table_id == tid:
                count_retrieved_gold_table += 1

            if qid != current_qid:
                if current_object:
                    total_queries += 1
                    results.append(current_object)

                current_qid = qid
                current_object = {
                    "question-id": qid,
                    "question": question,
                    "answer": answer,
                    "Gold Table": str(gold_table_id),
                    "k": k,
                    "retrieved tables": []
                }

            current_object["retrieved tables"].append({
                "rank": rank,
                "RetScore": retrieval_score,
                "TableID": tid,
                "Full Table": table_index[tid]
            })

        if current_object:
            results.append(current_object)

    # Write results to the JSON file
    recall = count_retrieved_gold_table / total_queries
    
    with tqdm(total=len(results), desc="Writing results to JSON file") as pbar:
        with open(view_results, 'w') as out_file:
            for result in results:
                json.dump(result, out_file, indent=4)
                out_file.write("\n")
                pbar.update(1)
    return view_results, recall

# File mapping and output logs
# table_ranking_file = {
#     "100":["data/results_full_table_colbert_k_100_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/19.27.23/wtq_full_table_k_100.ranking.tsv"],
#     "50":["data/results_full_table_colbert_k_50_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/19.26.14/wtq_full_table_k_50.ranking.tsv"],
#     "30":["data/results_full_table_colbert_k_30_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/19.25.02/wtq_full_table_k_30.ranking.tsv"],
#     "10":["data/results_full_table_colbert_k_10_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/19.23.50/wtq_full_table_k_10.ranking.tsv"],
#     "5":["data/results_full_table_colbert_k_5_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/19.22.42/wtq_full_table_k_5.ranking.tsv"],
#     "1":["data/results_full_table_colbert_k_1_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/19.21.28/wtq_full_table_k_1.ranking.tsv"]
#     }
table_ranking_file = {
    "100":["data/results_only_headers_colbert_k_100_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/20.14.33/wtq_only_headers_table_k_100.ranking.tsv"],
    "50":["data/results_only_headers_colbert_k_50_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/20.13.13/wtq_only_headers_table_k_50.ranking.tsv"],
    "30":["data/results_only_headers_colbert_k_30_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/20.12.11/wtq_only_headers_table_k_30.ranking.tsv"],
    "10":["data/results_only_headers_colbert_k_10_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/20.11.05/wtq_only_headers_table_k_10.ranking.tsv"],
    "5":["data/results_only_headers_colbert_k_5_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/20.10.03/wtq_only_headers_table_k_5.ranking.tsv"],
    "1":["data/results_only_headers_colbert_k_1_test.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq-test/scripts.main/2025-01/16/20.08.13/wtq_only_headers_table_k_1.ranking.tsv"]
    }
output_log_file = "scripts/result_logs.txt"

with open(output_log_file, "a") as log_file:
    log_file.write("---------------------------------------------------------------------")
    log_file.write('\n')
    
    now = datetime.now()
    formatted_time = now.strftime("%I:%M %p, %d %B %Y")
    log_file.write("Date: "+formatted_time)
    log_file.write('\n')    

    for k, [view_results, generated_ranking_path] in table_ranking_file.items():
        print(f"Started for Recall@{k}")
        try:
            file_base_name = os.path.basename(generated_ranking_path)
            log_file.write(f"Calculating Recall@{k} and storing in {file_base_name}\n")
            view_results, recall = generate_result_file(int(k), generated_ranking_path, view_results)
            log_file.write(f"Result File Generated at {view_results} and Recall is {recall}\n")
        except Exception as e:
            log_file.write(f"An error occurred while processing k={k}: {str(e)}\n")
            log_file.write("Stack Trace:\n")
            log_file.write(traceback.format_exc())
            log_file.write("\n")
