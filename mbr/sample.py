import os
from parser import get_mbr_parser

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import set_seed
from utils import (
    StoppingCriteriaSub,
    load_dataset,
    load_kwargs,
    load_model,
    prompt_dir,
    sample_dir,
)

# Length penalty has no effect for sampling in this codebase.


def compute_probability_s2s(sample_output):
    """
    This compute_prob function is compatible with seq2seq models.
    Doesn't work on language models.
    """
    bsz = sample_output.sequences.shape[0]
    probs = np.array([1.0] * bsz)
    # terms = [False] * bsz
    for i in range(len(sample_output.scores)):
        p = np.array([1.0] * bsz)
        for b in range(bsz):
            if hasattr(tokenizer, "pad_token_id"):
                if sample_output.sequences[b][i + 1] == tokenizer.pad_token_id:
                    continue
            log_probs = torch.nn.functional.log_softmax(
                sample_output.scores[i][b], dim=-1
            )
            p[b] = torch.exp(log_probs[sample_output.sequences[b][i + 1]])
        probs *= p
        # print('p=', p)
    return probs


def get_texts(tokenizer, outputs, input_length):
    """
    This function is only compatible with langauge models. not for seq2seq
    """
    bsz = outputs.sequences.shape[0]
    output_texts = []
    for b in range(bsz):
        output_text = tokenizer.decode(
            outputs.sequences[b][input_length:], skip_special_tokens=True
        )
        output_texts.append(output_text)
    return output_texts


def sample(
    dataset,
    tokenizer,
    model,
    src_lines,
    torch_device,
    n_lines,
    start_iter,
    n_samples,
    bsz,
    eps,
    topk,
    topp,
    model_n,
):
    n_batches = n_samples // bsz

    os.makedirs(os.path.join(sample_dir, dataset, model_n), exist_ok=True)

    model_kwargs = load_kwargs(dataset)

    for sample_id in tqdm(range(start_iter, n_lines)):
        if sample_id > len(src_lines):
            break

        input_source = src_lines[sample_id]
        model_inputs = tokenizer(input_source, return_tensors="pt", truncation=True).to(
            torch_device
        )
        stopping_criteria = None

        set_seed(42)

        rows = []

        for i in range(n_batches):

            sample_output = model.generate(
                **model_inputs,
                **model_kwargs,
                do_sample=True,
                epsilon_cutoff=eps,
                top_k=topk,
                top_p=topp,
                num_beams=1,
                num_return_sequences=bsz,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
                output_scores=True,
                forced_bos_token_id=model.config.forced_bos_token_id,
            )

            probs = compute_probability_s2s(sample_output)

            for j in range(bsz):
                sample_text = tokenizer.decode(
                    sample_output.sequences[j], skip_special_tokens=True
                )
                rows.append((sample_text, probs[j]))

        filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}".format(
            sample_id, eps, topk, topp
        )

        outfilepath = os.path.join(sample_dir, dataset, model_n, filename)

        df = pd.DataFrame(rows, columns=["text", "probability"])
        df.to_csv(outfilepath, index=False)


if __name__ == "__main__":
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model
    prompt_path = args.prompt
    n_lines = args.n_lines
    start_iter = args.start_iter

    n_samples = args.n_samples
    bsz = args.bsz
    eps = args.eps
    topk = args.topk
    topp = args.topp

    src_lines = load_dataset(dataset)
    tokenizer, model, model_name, stop_tokens = load_model(
        dataset, torch_device, model_name
    )

    if prompt_path == "None":
        prompt = "None"
    else:
        with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()

    sample(
        dataset,
        tokenizer,
        model,
        src_lines,
        torch_device,
        n_lines,
        start_iter,
        n_samples,
        bsz,
        eps,
        topk,
        topp,
        model_n=os.path.basename(model_name),
    )
