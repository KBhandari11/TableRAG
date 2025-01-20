from TableRAG.src.ColBERT.colbert import Run
from TableRAG.src.ColBERT.colbert.infra.config import ColBERTConfig, RunConfig
from TableRAG.src.ColBERT.colbert import Trainer


def train():
    # use 4 gpus (e.g. four A100s, but you can use fewer by changing nway,accumsteps,bsize).
    with Run().context(RunConfig(nranks=4)):
        triples = '/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/triples.jsonl'  # `wget https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/resolve/main/examples.json?download=true` (26GB)
        queries = '/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_question_index.tsv'
        collection = '/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_tables_index.json'

        config = ColBERTConfig(bsize=32, lr=1e-05, warmup=1000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False, nway=64, accumsteps=1, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        trainer.train(checkpoint='colbert-ir/colbertv1.9')  # or start from scratch, like `bert-base-uncased`

if __name__ == '__main__':
    train()