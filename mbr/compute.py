import os
from parser import get_mbr_parser

import numpy as np

# import torch
from datasets import load_metric
from tqdm import tqdm
from utils import load_samples, sample_dir, score_dir


def get_metric(metric):
    if metric == "bleu":
        sacrebleu = load_metric("sacrebleu")

        def compute_bleu(hyp, ref):
            # TODO: Batch the computation
            return sacrebleu.compute(predictions=[hyp], references=[[ref]])["score"]

        return compute_bleu
    elif metric == "bleurt":
        bleurt = load_metric("bleurt", checkpoint="BLEURT-tiny")

        def compute_bleurt(hyp, ref):
            # Input is a batch
            return np.array(bleurt.compute(predictions=hyp, references=ref)["scores"])

        return compute_bleurt
    else:
        assert False


def compute_mbr_score(samples, score_function, seq_scores=None, include_self=False):
    n_samples = len(samples)

    scores = []
    for i in range(n_samples):
        if include_self:
            score = score_function(hyp=np.array([samples[i]] * n_samples), ref=samples)
        elif seq_scores is None:
            score = score_function(
                hyp=np.array([samples[i]] * (n_samples - 1)),
                ref=np.concatenate((samples[:i], samples[i + 1 :])),
            )
        else:
            score = score_function(
                hyp=np.array([samples[i]] * (n_samples - 1)),
                ref=np.concatenate((samples[:i], samples[i + 1 :])),
            ) * np.concatenate((seq_scores[:i], seq_scores[i + 1 :]))
        scores.append(score)
    return np.array(scores)


if __name__ == "__main__":
    # torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = get_mbr_parser("compute")
    args = parser.parse_args()

    dataset = args.dataset
    n_lines = args.n_lines
    eps = args.eps
    metric = args.metric

    distance_func = get_metric(metric)

    os.makedirs(os.path.join(score_dir, dataset), exist_ok=True)

    for line_id in tqdm(range(n_lines)):
        line = load_samples(dataset, line_id, eps)
        score_matrix = compute_mbr_score(line, distance_func, include_self=True)

        score_path = os.path.join(
            score_dir, dataset, "{:04d}_eps-{:.2f}_{}".format(line_id, eps, metric)
        )

        np.savetxt(score_path, score_matrix)
