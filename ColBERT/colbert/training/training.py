import time
import torch
import random
import torch.nn as nn
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints



def train(config: ColBERTConfig, triples, queries=None, collection=None):
    config.checkpoint = config.checkpoint or 'bert-base-uncased'

    if config.rank < 1:
        config.help()

    # Set random seeds for reproducibility
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure batch size is divisible by the number of ranks
    assert config.bsize % config.nranks == 0, f"Batch size {config.bsize} must be divisible by number of ranks {config.nranks}"
    config.bsize //= config.nranks

    print(f"Using batch size {config.bsize} per process and accumulation steps {config.accumsteps}")

    # Initialize data reader
    if collection is None:
        raise ValueError("Collection cannot be None")

    reader_class = RerankBatcher if config.reranker else LazyBatcher
    reader = reader_class(config, triples, queries, collection, rank=(0 if config.rank == -1 else config.rank), nranks=config.nranks)

    # Initialize model
    if config.reranker:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)
    else:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)

    colbert = colbert.to(DEVICE)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(
        colbert, device_ids=[config.rank], output_device=config.rank, find_unused_parameters=True
    )

    # Optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    scheduler = (
        get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup, num_training_steps=config.maxsteps)
        if config.warmup is not None
        else None
    )
    optimizer.zero_grad()

    # Set BERT gradient if applicable
    if config.warmup_bert is not None:
        set_bert_grad(colbert, False)

    # Mixed precision
    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss, train_loss_mu = None, 0.999
    start_batch_idx = 0

    # Training loop
    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        if config.warmup_bert and config.warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            config.warmup_bert = None

        this_batch_loss = 0.0

        for batch in BatchSteps:
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                except ValueError:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                scores = colbert(*encoding)

                # Handle IB negatives
                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores *= config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                else:
                    labels = labels[:scores.size(0)]  # Ensure label size matches scores
                    loss = nn.CrossEntropyLoss()(scores, labels)

                if config.use_ib_negatives:
                    loss += ib_loss

                loss /= config.accumsteps

            amp.backward(loss)
            this_batch_loss += loss.item()

        train_loss = train_loss if train_loss is not None else this_batch_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        amp.step(colbert, optimizer, scheduler)

        if config.rank < 1:
            print_message(f"Batch {batch_idx}/{config.maxsteps}, Train Loss: {train_loss:.4f}")
            manage_checkpoints(config, colbert, optimizer, batch_idx + 1)

    if config.rank < 1:
        print_message("Training complete. Saving final checkpoint.")
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx + 1, consumed_all_triples=True)
        return ckpt_path


def set_bert_grad(colbert, value):
    """Set requires_grad for BERT parameters."""
    for p in colbert.bert.parameters():
        p.requires_grad = value

