import os
import argparse
from tqdm import tqdm
import json

import numpy as np
import pandas as pd

import boto3
from evaluate import load
from comet import download_model, load_from_checkpoint

from utility_func import *
from utils import load_dataset, load_matrix, load_samples_from_file, result_dir, matrix_dir #, approx_dir, diverse_dir
from parser import get_mbr_parser

from policy.mbr import compute_score_matrix, compute_mbr, compute_kmbr, compute_nbys_mbr, compute_c2f_mbr
from policy.approx_mbr import compute_ambr, compute_pmbr


def compute_score(df, d_best, trg, compute_evaluate, src=None):
    d_hyp = df.iloc[d_best]['text']
    d_score = compute_evaluate(d_hyp, trg, src)
    return d_score


if __name__ == "__main__":
    """
    This script is the "main function" of the experiment.
    """
    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model

    sample_dir = args.sample_dir
    
    n_lines = args.n_lines
    n_samples = args.n_samples

    epsilon = args.eps
    topk = args.topk
    topp = args.topp

    sim = args.sim
    eval_func = args.eval

    # Algorithm config
    algorithm = args.algorithm
    recompute_matrix = args.recompute_matrix
    approx_iters = args.approx_iters
    r_0 = args.r_0
    r_increase = args.r_increase
    pruning_alpha = args.pruning_alpha
    

    if args.approx_budgets < 0:
        exact = n_samples * (n_samples - 1)
        # minimum = int(np.floor(n_samples * np.log(n_samples)))
        # exact = int(n_samples * (n_samples - 1) / 2)
        # minimum = int(np.floor(n_samples * np.log(n_samples)))
        # T_budgets = list(np.arange(minimum, exact, int(n_samples * np.log(n_samples))))        
        
        T_budgets = [exact // n for n in [32, 16, 8, 4, 2]]
        # for n in range(1, 10):
        #     bud = exact / (2 ** n)
        #     if bud >= n_samples * np.log(n_samples):
        #         T_budgets.append(int(np.floor(bud)))
        #     else:
        #         break
    else:
        T_budgets = [args.approx_budgets]
        
    if (len(T_budgets) == 0) and (algorithm in ["approx", "c2f", "nbys"]):
        print('T_budgets is empty!')
        assert False


    compute_similarity, similarity = load_similarity(sim)
    compute_distance = load_distance(sim, compute_similarity)
    compute_evaluate, evaluator = load_evaluate(eval_func, sim, similarity)

    if algorithm in ['dbs', 'diverse', 'diversesample']:
        compute_pairwise, _ = load_evaluate(pairwise_eval, sim, similarity)

    # Load dataset
    src_lines = load_dataset(dataset) # src is used only by comet and clip.
    trg_lines = load_dataset(dataset, ref=True)
    
    model_n = os.path.basename(model_name)

    os.makedirs(os.path.join(matrix_dir, dataset, model_n), exist_ok=True)


    files = sorted(os.listdir(sample_dir))

    filtered_files = load_samples_from_file(files, epsilon, topk, topp, do_sample, diverse_k, diversity_penalty)
    
    assert len(filtered_files) > 0

    print('first 10 files=', filtered_files[:10])

    rows = []

    if algorithm == 'c2f':
        compute_coarse_sim, _ = load_similarity("sacrebleu")
    elif algorithm == 'c2ff1':
        compute_coarse_sim, _ = load_similarity("unigramf1")

    for filename in filtered_files:

        sample_id = int(filename.split('_')[0])
        assert "{:04}".format(sample_id) in filename

        if sample_id >= n_lines:
            break


        src_input = src_lines[sample_id]
        trg = trg_lines[sample_id]


        df = pd.read_csv(os.path.join(sample_dir, filename))

        df = df[:n_samples]
        df.fillna("", inplace=True)
        hyp = df.iloc[:]['text']
 
        if not recompute_matrix:
            # This makes loading a matrix of size larger
            matrix = load_matrix(os.path.join(matrix_dir, dataset, model_n), filename, sim, n_samples)
            # if not (matrix is None):
            #     print('matrix loaded and reshaped to', matrix.shape)
        else:
            matrix = None
        if matrix is None:
            matrix_filename = filename + "_" + sim + "_" + str(n_samples)
            matrix_path = os.path.join(matrix_dir, dataset, model_n, matrix_filename)

            matrix = compute_score_matrix(hyp, compute_similarity, [src_input] * len(hyp))
            np.savetxt(matrix_path, matrix)

                    
        if algorithm == 'incremental':
            ed_bests = compute_mbr(matrix=matrix, incremental=True)
            cache = {}
            ed_scores = []
            for ed_best in ed_bests:
                if ed_best in cache.keys():
                    ed_score = cache[ed_best]
                else:
                    ed_score = compute_score(df, ed_best, trg, compute_evaluate, src=src_input)
                    cache[ed_best] = ed_score
                ed_scores.append(ed_score)
            # ed_scores = [compute_score(df, ed_best, trg, compute_evaluate, src=src_input) for ed_best in ed_bests]
            row = [[sample_id, ed_scores[i], ed_bests[i]] for i in range(len(ed_bests))]
        elif algorithm == "approx":
            row = [sample_id]
            cache = {}
            for T_budget in T_budgets:
                approx_bests = []
                approx_scores = []
                for itr in range(approx_iters):
                    approx_best, _ = compute_ambr(matrix=matrix, T_budget=T_budget)
                    if approx_best in cache.keys():
                        approx_score = cache[approx_best]
                    else:
                        approx_score = compute_score(df, approx_best, trg, compute_evaluate, src=src_input)
                        cache[approx_best] = approx_score
                    approx_bests.append(approx_best)
                    approx_scores.append(approx_score)
                row.append(approx_scores)
                row.append(approx_bests)
        elif algorithm == "pruning":
            row = [sample_id]
            cache = {}
            r_schedule = [r_0 * (r_increase ** i) for i in range(20)]
            for T_budget in T_budgets:
                approx_bests = []
                approx_scores = []
                for itr in range(approx_iters):
                    approx_best = compute_pmbr(matrix=matrix, T_budget=T_budget, 
                        r_schedule=r_schedule, pruning_alpha=pruning_alpha)
                    if approx_best in cache.keys():
                        approx_score = cache[approx_best]
                    else:
                        approx_score = compute_score(df, approx_best, trg, compute_evaluate, src=src_input)
                        cache[approx_best] = approx_score
                    approx_bests.append(approx_best)
                    approx_scores.append(approx_score)
                row.append(approx_scores)
                row.append(approx_bests)
        elif "c2f" in algorithm:
            row = [sample_id]
            
            if algorithm == "c2ff1":
                coarse_sim = "unigramf1"
            else:
                coarse_sim = "sacrebleu"
            
            if not recompute_matrix:
                coarse_matrix = load_matrix(os.path.join(matrix_dir, dataset, model_n), 
                                            filename, coarse_sim, n_samples)
                if not (coarse_matrix is None):
                    print('coarse_matrix loaded and reshaped to', coarse_matrix.shape)
            else:
                coarse_matrix = None
            if coarse_matrix is None:
                coarse_matrix = compute_score_matrix(hyp, compute_coarse_sim, [src_input] * len(hyp))
                
                coarse_matrix_filename = filename + "_" + coarse_sim + "_" + str(n_samples)
                coarse_matrix_path = os.path.join(matrix_dir, dataset, model_n, coarse_matrix_filename)
                np.savetxt(coarse_matrix_path, coarse_matrix)

            
            if (algorithm == "c2f") or (algorithm == "c2ff1"):
                for T_budget in T_budgets:
                    c2f_best = compute_c2f_mbr(hyp=hyp, matrix=matrix, src=src_input, T_budget=T_budget, 
                                            compute_coarse=compute_coarse_sim, coarse_matrix=coarse_matrix)
                    c2f_score = compute_score(df, c2f_best, trg, compute_evaluate, src=src_input)
                    row.append(c2f_score)
                    row.append(c2f_best)
            elif algorithm == "c2fa":
                print('Not implemented yet')
                assert False
            else:
                assert False
        elif algorithm == "nbys":
            row = [sample_id]
            for T_budget in T_budgets:
                nbys_best = compute_nbys_mbr(matrix=matrix, T_budget=T_budget)
                nbys_score = compute_score(df, nbys_best, trg, compute_evaluate, src=src_input)
                row.append(nbys_score)
                row.append(nbys_best)
        else:
            assert False
        rows.append(row)
    
    if algorithm == 'incremental':
        # TODO: Add other algorithms if needed.
        columns = ['sample_id', "ed_score", "ed_best"]
        postfix = ""
    elif algorithm == 'approx':
        columns = ['sample_id']
        for T_budget in T_budgets:
            columns.append("approx-{:05d}_score".format(T_budget))
            columns.append("approx-{:05d}_best".format(T_budget))
        postfix = "_approx"
    elif algorithm == 'pruning':
        columns = ['sample_id']
        for T_budget in T_budgets:
            columns.append("pruning-{:05d}_score".format(T_budget))
            columns.append("pruning-{:05d}_best".format(T_budget))
        postfix = "_pruning_r{:02d}_alpha{:.2f}".format(r_0, pruning_alpha)
    elif algorithm == "c2f":
        columns = ['sample_id']
        for T_budget in T_budgets:
            columns.append("c2f-{:05d}_score".format(T_budget))
            columns.append("c2f-{:05d}_best".format(T_budget))
        postfix = "_c2f"
    elif algorithm == "c2ff1":
        columns = ['sample_id']
        for T_budget in T_budgets:
            columns.append("c2ff1-{:05d}_score".format(T_budget))
            columns.append("c2ff1-{:05d}_best".format(T_budget))
        postfix = "_c2ff1"
    elif algorithm == "nbys":
        columns = ['sample_id']
        for T_budget in T_budgets:
            columns.append("nbys-{:05d}_score".format(T_budget))
            columns.append("nbys-{:05d}_best".format(T_budget))
        postfix = "_nbys"
    else:
        assert False
    
    if args.approx_budgets >= 0:
        postfix += "_{:05d}".format(args.approx_budgets)

    if algorithm != 'incremental':
        df = pd.DataFrame(rows, columns=columns)
        
        if algorithm == 'dbs':
            filename = "{}_{}_{}{}.csv".format(dataset, model_n, eval_func, postfix)
        else:
            filename = "{}_{}_{:03}_{:.2f}_{:02d}_{:.2f}_{}_{}{}.csv".format(dataset, model_n, n_samples, 
                                                                        epsilon, topk, topp, sim, eval_func, postfix)

        df_path = os.path.join(result_dir, filename)
        df.to_csv(df_path, index=False)

    else:
        for i, r in enumerate(rows):
            assert len(r) == n_samples
            
        for i_n_samples in range(0, n_samples):
            i_rows = [r[i_n_samples] for r in rows]
            df = pd.DataFrame(i_rows, columns=columns)
            filename = "{}_{}_{:03}_{:.2f}_{:02d}_{:.2f}_{}_{}{}.csv".format(dataset, model_n, i_n_samples+1, 
                                                                        epsilon, topk, topp, sim, eval_func, postfix)

            df_path = os.path.join(result_dir, filename)
            df.to_csv(df_path, index=False)
