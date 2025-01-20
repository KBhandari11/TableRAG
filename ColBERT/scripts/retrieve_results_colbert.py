import json
import csv
import os
import os
import traceback
# File paths
wtq_table_index_file = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv"

wtq_question_index_file = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_question_index.tsv"

random_split_train_file = "/scratch/asing725/IP/Multi-Table-RAG/WikiTableQuestions/data/training.tsv"

table_ranking_file = ["/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq/scripts.main/2025-01/15/01.22.43/wtq_full_table_k_1.ranking.tsv","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq/scripts.main/2025-01/15/00.57.35/wtq_full_table_k_5.ranking.tsv", "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq/scripts.main/2025-01/15/01.26.06/wtq_full_table_k_30.ranking.tsv", "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq/scripts.main/2025-01/15/01.28.59/wtq_full_table_k_50.ranking.tsv", "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq/scripts.main/2025-01/15/01.31.44/wtq_full_table_k_100.ranking.tsv"]

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
            qid, question = row
            question_index[qid] = question
    return question_index

# Load random-split-1-train.tsv into a dictionary {qid: answer}
def load_answers(file_path):
    answers = {}
    gold_table_paths ={}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        flag = True
        for row in reader:
            if(flag):
                flag = False
                continue
            qid, question, gold_table_path, answer = row
            answers[qid] = answer
            gold_table_paths[qid] = gold_table_path
    return answers,gold_table_paths

def find_key_in_contexts(search_value):
    # Define the path to the JSON file
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
def generate_result_file(k, wtq_full_table_ranking_file, view_results):
    print("Load Question Index")
    question_index = load_question_index(wtq_question_index_file)
    print("Load Gold Tables")
    answers,gold_table_path = load_answers(random_split_train_file)
    # print("Answers :",answers)
    # print("gold Tables: ", gold_table_path)
    results = []
    # results.append({"answers":str(answers)})
    count_retrieved_gold_table = 0
    total_queries = 0
    with open(wtq_full_table_ranking_file, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        current_qid = None
        current_object = None
        print_flag = True
        print("Reading from rankings started")
        for row in reader:
            qid, tid, rank, retrieval_score = row

            if print_flag:
                print("Finding Gold Table for this query")
            

            gold_table_id = find_key_in_contexts(gold_table_path.get("nt-"+str(int(qid-1))))
            if print_flag:
                print(f"Got Gold Table for 1st Question it is = {gold_table_id}")
            print_flag = False
            if gold_table_id == tid:
                count_retrieved_gold_table = count_retrieved_gold_table + 1

            if qid != current_qid:
                # Save the previous object to results
                if current_object:
                    total_queries = total_queries + 1
                    results.append(current_object)

                # Start a new object Contexts : 0 path,1 path etc...
                current_qid = qid
                current_object = {
                    "question-id": qid,
                    "question": question_index.get(qid, ""),
                    "answer": answers.get("nt-"+qid,""),
                    "Gold Table": str(gold_table_id),
                    "k": k,
                    "retrieved tables": []
                }

            # Add table details to the current object
            current_object["retrieved tables"].append({
                "rank": rank,
                "RetScore": retrieval_score,
                "TableID": tid
                # "Full Table": table_index.get(tid, "")
            })

        # Add the last object to results
        if current_object:
            results.append(current_object)
    
    # Write results to the JSON file
    recall = count_retrieved_gold_table / total_queries
    if k==30:
        with open(view_results, 'w') as out_file:
            json.dump(results, out_file, indent=4)
    return view_results,recall


table_ranking_file = {"100":["data/results_full_table_colBERT_k_100.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq/scripts.main/2025-01/16/04.19.08/wtq_full_table_k_100.ranking.tsv"],
                      "50":["data/results_full_table_colBERT_k_50.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq/scripts.main/2025-01/16/04.17.04/wtq_full_table_k_50.ranking.tsv"],
                      "30":["data/results_full_table_colBERT_k_30.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq/scripts.main/2025-01/16/04.12.05/wtq_full_table_k_30.ranking.tsv"],
                      "10":["data/results_full_table_colBERT_k_10.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq/scripts.main/2025-01/16/04.09.51/wtq_full_table_k_10.ranking.tsv"],
                      "5":["data/results_full_table_colBERT_k_5.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq/scripts.main/2025-01/16/04.05.48/wtq_full_table_k_5.ranking.tsv"],
                      "1":["data/results_full_table_colBERT_k_1.json","/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments/wtq/scripts.main/2025-01/16/04.02.33/wtq_full_table_k_1.ranking.tsv"],
                      }
# Output log files
output_log_file = "scripts/result_logs.txt"

with open(output_log_file, "w") as log_file: 
    for k, [view_results, generated_ranking_path] in table_ranking_file.items():
        print(f"Started for Recall@{k}")
        try:
            file_base_name = os.path.basename(generated_ranking_path)
            log_file.write(f"Calculating Recall@{k} and storing in {file_base_name}\n")
            
            # Assume generate_result_file() is defined and returns these values
            view_results, recall = generate_result_file(k, generated_ranking_path, view_results)
            
            log_file.write(f"Result File Generated at {view_results} and Recall is {recall}\n")
        except Exception as e:
            # Log the error message and stack trace to the file
            log_file.write(f"An error occurred while processing k={k}: {str(e)}\n")
            log_file.write("Stack Trace:\n")
            log_file.write(traceback.format_exc())
            log_file.write("\n")


