

import sys
import os
import pandas as pd
import numpy as np
import random
import copy
import argparse 
from tqdm import tqdm
import pathlib

import torch
from transformers import pipeline, AutoTokenizer,StoppingCriteria, StoppingCriteriaList
from accelerate import Accelerator
from evaluate import load

from utils.dataset import *



def random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = pipeline(
            "text-generation",
            model=args.model_id,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto")
    #model.tokenizer.pad_token_id = model.model.config.eos_token_id
    #model.tokenizer.padding_side = "left"
    return model, tokenizer


def format_answer_evaluation(prediction):
    answer = prediction.split("\n")[0]
    answer = answer.replace(",","")
    answer = answer.replace(".","")
    return answer.split(" ")

def checkpoint(result, args):
    result = pd.DataFrame(result)
    directory = "/".join(args.results_filename.split("/")[:-1])
    filename = args.results_filename
    if args.use_fewshot:
        directory = directory +f"/fewshot_{args.few_shot}"    
        filename = directory+"/"+filename.split("/")[-1]

    pathlib.Path(directory).mkdir(parents=True, exist_ok=True) 
    result.to_csv(filename,index=False)

def run(model, dataloader, tokenizer, args):
    exact_accuracy_list = []
    if args.from_scratch:
        result={"question":[],"table":[],"reference":[],"generated":[],"score":[]}
    else:
        try:
            prior = pd.read_csv(args.results_filename)
            result =prior.to_dict(orient='list')
        except: 
            result={"question":[],"table":[],"reference":[],"generated":[],"score":[]}

    terminators = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    stop_token_ids = tokenizer.convert_tokens_to_ids(["<|eot_id|>"])
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False
    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    for batch_idx,data in enumerate(dataloader):
        sequences = model(
            data["prompt"],
            max_new_tokens=256,
            eos_token_id=terminators,
            stopping_criteria=stopping_criteria,
            pad_token_id=50256,
            temp= 0.6,
            #do_sample=True,
            #top_k=10,
            return_full_text = False,
        )
        for idx, seq in enumerate(sequences):
            exact_accuracy = evaluate(reference=data["answers"][idx],prediction=seq[0]['generated_text'], args=args)
            exact_accuracy_list.append(exact_accuracy)
            result["table"].append(data["table"][idx])
            result["question"].append(data["questions"][idx])
            result["reference"].append(data["answers"][idx])
            result["generated"].append(seq[0]['generated_text'])
            result["score"].append(exact_accuracy)

        if len(exact_accuracy_list)%args.checkpoint == 0 and batch_idx != 0:
            print(f"Checkpoint: {len(exact_accuracy_list)}\t | Total Number of data: {len(result['score'])} | \tMean = {sum(exact_accuracy_list)/len(exact_accuracy_list)}", flush=True)
            checkpoint(result, args)
    return result

    
def evaluate(reference, prediction, args):
    if not isinstance(reference,list):
        reference = [str(reference)]
    c = 0
    t = 0
    for correct in reference:
        if correct in prediction:
            c+=1
        t+=1
    return c/t

if __name__ == "__main__":
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        parser = argparse.ArgumentParser()
        #model
        parser.add_argument('--model_id', type=str)
        parser.add_argument('--evaluation_metric', type=str, default="exact_match")
        parser.add_argument('--dataset_filename', type=str)
        parser.add_argument('--num_evaluation', type=int, default=None)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--num_process', type=str, default=8)
        parser.add_argument('--results_filename', type=str)
        parser.add_argument('--from_scratch', action='store_true')
        parser.add_argument('--use_fewshot', action='store_true')
        parser.add_argument('--few_shot', type=int)
        parser.add_argument('--no_table', action='store_true')
        parser.add_argument('--checkpoint', type=int, default=100)


        args = parser.parse_args()
        random_seed(args.seed)
        model, tokenizer = load_model(args)
        dataset = create_dataloader(tokenizer, args)

        result = run(model,dataset, tokenizer, args)

        print("Mean Exact Accuracy", sum(result["score"])/len(result["score"]))
        checkpoint(result,args)