import numpy as np
import pandas as pd
from evaluate import load


def pull_arm(
    n_samples,
    arms,
    num_pulls_per_arm,
    dist_func,
    cache_matrix=None,
    has_evaluated=None,
    src_input=None,
):
    # TODO: Can optimize assuming num_pulls_per_arm == 1

    # sample_n = len(samples)

    # TODO: What happen if we sample with weight propotional to the Pmodel here?
    Tmean = np.zeros(arms.shape[0], dtype=float)

    remaining_references = [
        sample for sample in range(n_samples) if has_evaluated[sample] == 0
    ]
    # print('remaining_references: {}'.format(remaining_references))
    if len(remaining_references) > 0:
        tmp_pos = np.array(
            np.random.choice(
                remaining_references, size=num_pulls_per_arm, replace=False
            ),
            dtype="int",
        )
        # print('tmp_pos: {}'.format(tmp_pos))

        for i, a in enumerate(arms):
            for t in tmp_pos:
                has_evaluated[t] = 1

                if a == t:
                    # Assume the distance between the same sequence is 0.
                    Tmean[i] += 0.0
                else:
                    # If the function is symmetric, we can save half of the computation.
                    # up = max(a, t)
                    # bottom = min(a, t)
                    cache = cache_matrix[a, t]

                    if np.isnan(cache):
                        # TODO: Compute it in a batch.
                        value = dist_func(a, t, src_input)
                        # value = dist_func([samples[a]], [samples[t]], src_input)
                        cache_matrix[a, t] = value[0]
                        Tmean[i] += value[0]
                    else:
                        Tmean[i] += cache

            Tmean[i] = Tmean[i] / tmp_pos.shape[0]
    else:
        # If no references are left for evaluation, then we do not pull any arms.
        pass

    return Tmean, cache_matrix, has_evaluated


def compute_ambr(
    hyp=None, score_function=None, matrix=None, weights=None, src=None, T_budget=None
):
    # TODO: Currently the algorithm does not use the fact that
    #       the score_function is symmetric. This reduces the computation a bit (up to a factor of 2).

    if matrix is None:
        # TODO: Make it batched for speed.
        assert False
        distance_function = lambda a, b, src_input: score_function(
            [hyp[a]], [hyp[b]], src_input
        )
    else:
        # Because the matrix is a similarity matrix, we need to convert it to a distance matrix.
        distance_function = lambda a, b, src_input: [1.0 - matrix[a, b]]

    n = matrix.shape[0]
    S = np.arange(n)
    T = np.zeros(n, dtype=int)
    estimate = np.zeros(n)

    summary = []
    summary_pulls = []
    nqueries = []
    # incum_bests = [-1]

    round = 0

    cache_matrix = np.empty((n, n))
    cache_matrix[:] = np.nan
    # 0: not evaluated, 1: evaluated. This is to keep track of the evaluated references to make the sampling without replacement.
    has_evaluated = np.zeros(n, dtype=int)

    while (
        (len(S) > 1)
        and ((~np.isnan(cache_matrix)).sum() < T_budget)
        and not (has_evaluated.all())
    ):
        tr = int(min(max(1, np.floor(T_budget / S.shape[0] / np.ceil(np.log(n)))), n))
        # print("Round {}: #hypothesis={}, #pulls={}".format(round, S.shape[0], tr))

        # We pull arms one by one for the sake of recoding.
        # It does not change the algorithm at all.
        for _ in range(tr):
            if ((~np.isnan(cache_matrix)).sum() + len(S)) >= T_budget:
                # If we are about to exceed the budget, we stop.
                # The original algorithm ignores a few pulls of exceeds but in our implementation we take it strict.
                break

            Tmean, updated_cache_matrix, updated_has_evaluated = pull_arm(
                n, S, 1, distance_function, cache_matrix, has_evaluated, src
            )
            cache_matrix = updated_cache_matrix
            has_evaluated = updated_has_evaluated

            # Take a moving average over the previous mean (estimate[arms]) and current mean (Tmean)
            estimate[S] = (estimate[S] * T[S] + Tmean) / (T[S] + 1.0)
            # estimate[arms] = Tmean # theoretically don't use past
            # if 1 == n:
            #     estimate[S] = Tmean
            T[S] = T[S] + 1

            # This lst computation is not necessary for the algorithm but for the sake of recording.
            lst = estimate[S]
            summary.append(S[np.argmin(lst)])
            summary_pulls.append(T.sum())
            nqueries.append((~np.isnan(cache_matrix)).sum())

        # Recompute lst so that it is independent to the recording procedure above.
        lst = estimate[S]

        med = np.median(lst)
        locs = np.where(lst <= med)[0]

        # incum_bests.append(estimate.argmin())

        if tr == n:  # calculated exactly
            S = [S[np.argmin(lst)]]
            break
        S = S[locs]

        if (len(S) == 1) or (len(locs) == len(lst)):
            break
        round += 1

    assert T_budget >= nqueries[-1]

    # if T_budget < nqueries[-1]:
    #     print('Over budget!')
    #     print('T_budget = {}'.format(T_budget))
    #     print('nqueries = {}'.format(nqueries[-1]))
    #     print('summary_pulls = {}'.format(summary_pulls[-1]))

    df = pd.DataFrame(
        {
            # "iters": list(range(len(summary))),
            "hyp_index": summary,
            "nevals": summary_pulls,
            "nqueries": nqueries,
        }
    )

    best_solution = df.iloc[-1]["hyp_index"]
    return best_solution, df

    # return S[0], df


def compute_pmbr(
    hyp=None,
    score_function=None,
    matrix=None,
    weights=None,
    src=None,
    T_budget=None,
    r_schedule=None,
    pruning_alpha=0.9,
    n_bootstraps=500,
):

    if matrix is None:
        # TODO: Make it batched for speed.
        assert False
        distance_function = lambda a, b, src_input: score_function(
            [hyp[a]], [hyp[b]], src_input
        )
    else:
        # Because the matrix is a similarity matrix, we need to convert it to a distance matrix.
        distance_function = lambda a, b, src_input: [1.0 - matrix[a, b]]

    if r_schedule is None:
        r_schedule = [4 * (2**i) for i in range(10)]
    # print("r_schedule=", r_schedule)

    n = matrix.shape[0]
    S = np.arange(n)
    # T = np.zeros(n, dtype=int)
    # estimate = np.zeros(n)

    # summary = []
    # summary_pulls = []
    # nqueries = []

    iteration = 0

    reference_pool = []

    r_i = 0
    used_budget = 0

    # print("T_budget=", T_budget)

    while (len(S) > 1) and (used_budget < T_budget):
        # Compute the required budget
        using_budget = len(S) * (r_schedule[iteration] - r_i)

        # print("used_budget=", used_budget)
        # print("using_budget=", using_budget)
        # print("S=", S)

        if used_budget + using_budget > T_budget:
            residual = (T_budget - used_budget) // len(S)
            if residual > 0:
                r_i = r_i + residual
            else:
                break
        else:
            r_i = r_schedule[iteration]

        used_budget += using_budget

        # print("r_i=", r_i)
        if r_i > n:
            break

        while len(reference_pool) < r_i:
            remaining_references = [
                sample for sample in range(n) if sample not in reference_pool
            ]
            tmp_pos = np.random.choice(
                remaining_references, size=1, replace=False
            ).item()
            reference_pool.append(tmp_pos)

        # print("reference_pool=", reference_pool)

        ########################
        # Pruning
        ########################
        reduce_matrix = matrix[:, reference_pool]
        ybarbar = reduce_matrix.sum(axis=1).argmax()
        ybarbar_score = reduce_matrix.sum(axis=1).max()

        # print("ybarbar=", ybarbar)
        # print("ybarbar_score=", ybarbar_score)

        R_is = []
        y_barbar_bs_score = []
        for i in range(n_bootstraps):
            R_i = np.random.choice(remaining_references, size=r_i, replace=True)
            R_is.append(R_i)
            y_barbar_bs_score.append(matrix[ybarbar, R_i].sum())

        S_next = []
        for y in S:
            wins = 0
            for i in range(n_bootstraps):
                y_score = matrix[y, R_is[i]].sum()
                # print('y_score= {}, ybarbar_score= {}'.format(y_score, y_barbar_bs_score[i]))
                if y_score >= y_barbar_bs_score[i]:
                    wins += 1
            win_ratio = wins / n_bootstraps
            # print("y: {}, win_ratio: {}".format(y, win_ratio))
            if win_ratio > 1.0 - pruning_alpha:
                S_next.append(y)

        # print('S_next=', S_next)

        S = S_next
        iteration += 1

    best_solution = matrix[:, reference_pool].sum(axis=1).argmax()

    return best_solution


# def compute_c2f_ambr(hyp=None, compute_similatiy=None, matrix=None, weights=None, src=None, T_budget=None,
#                     compute_coarse=None, coarse_matrix=None):
#     assert hyp is not None
#     assert (compute_coarse is not None) or (coarse_matrix is not None)

#     n_samples = matrix.shape[0]

#     # Compute the number of samples to use for coarse to fine.
#     nk = n_samples
#     s_budget = nk * (nk - 1) / 2 + nk * (n_samples - nk)
#     while s_budget > T_budget:
#         nk -= 1
#         s_budget = nk * (nk - 1) / 2 + nk * (n_samples - nk)

#     # Compute coarse measure and pick the best k as a candidate.
#     if coarse_matrix is None:
#         coarse_matrix = compute_score_matrix(hyp, compute_coarse, [src] * len(hyp))
#     coarse_bests = compute_kmbr(matrix=coarse_matrix, k=nk)

#     if matrix is None:
#         matrix = compute_score_matrix(hyp, compute_similarity, [src] * len(hyp))

#     # Pick the best from the coarse set.
#     # The candidates are limited but the references are set the same.
#     compressed_matrix = matrix[coarse_bests]


#     best_in_orig_ind = coarse_bests[best_in_comp_ind]


if __name__ == "__main__":
    # import sys
    # sys.path.append("../")
    # from . import mbr

    n = 64
    T_budget = np.ceil(n * np.log(n) * 3)

    similarity = load("bertscore")

    def compute_similarity(hyp, ref, src):
        return similarity.compute(predictions=hyp, references=ref, lang="en")["f1"]

    def compute_distance(hyp, ref, src):
        return [1.0 - sim for sim in compute_similarity(hyp, ref, src)]

    df = pd.read_csv(
        "./samples/wmt19.de-en/wmt19-de-en/0000_eps-0.02_topk-00_topp-1.00"
    )
    samples = df["text"]
    samples = samples[:n]  # Limit the number of sequences for debugging
    # samples = [
    #     "Hello, my name is David.",
    #     "Hello, I am David.",
    #     "Hello, I'm David,",
    #     "This is it."
    # ]
    matrix = np.loadtxt(
        "./matrix/wmt19.de-en/wmt19-de-en/0000_eps-0.02_topk-00_topp-1.00_bertscore_127"
    )
    matrix = matrix[:n, :n]

    print("Optimal=", matrix.sum(axis=1).argmax())

    # _, df = compute_ambr(samples, compute_distance, matrix=matrix, T_budget=T_budget)

    # print("------------")
    # for i in range(len(df)):
    #     print("Round {}: Best arm = {}, #evaluations = {}, #queries = {}".format(i, df['hyp_index'][i], df['nevals'][i], df['nqueries'][i]))
    # print("------------")

    # em = mbr.compute_score_matrix(samples, compute_similarity)
    # em_scores = em.sum(axis=1)
    # exact_arm = em_scores.argmax()
    # exact_score = em_scores.max()

    # approx_arm = df['hyp_index'].iloc[-1]
    # approx_score = em_scores[df['hyp_index'].iloc[-1]]

    # print("Exact:       Best arm = {}, score = {}, with {} total evaluations".format(exact_arm, exact_score, int(n * (n-1)/2)))
    # print("Approximate: Best arm = {}, score = {}, with {} total evaluations".format(approx_arm, approx_score, df['nevals'].iloc[-1]))

    print("##################")
    print("##################")
    T_budget = 2000
    pruning_result = compute_pmbr(
        samples, compute_distance, matrix=matrix, T_budget=T_budget
    )
    print("T_budget = {}, Pruning result: {}".format(T_budget, pruning_result))
    T_budget = 1000
    pruning_result = compute_pmbr(
        samples, compute_distance, matrix=matrix, T_budget=T_budget
    )
    print("T_budget = {}, Pruning result: {}".format(T_budget, pruning_result))
    T_budget = 500
    pruning_result = compute_pmbr(
        samples, compute_distance, matrix=matrix, T_budget=T_budget
    )
    print("T_budget = {}, Pruning result: {}".format(T_budget, pruning_result))

    # dist_func = np.array([
    #     [0, 1, 0.5, 0, 0.3],
    #     [1, 0, 0.2, 0, 0.2],
    #     [0.5, 0.2, 0, 0, 0.1],
    #     [0, 0, 0, 0, 0.1],
    #     [0.3, 0.2, 0.1, 0.1, 0],
    # ])

    # n = len(dist_func)
    # S = np.arange(n)
    # T = np.zeros(n, dtype=int)
    # estimate = np.zeros(n)

    # summary = [0]
    # summary_pulls = [0]

    # # for r in range(0, np.log(n) - 1):
    # while len(S) > 1:
    #     round = 0

    #     tr = int(min(max(1, np.floor(float(T_budget) / S.shape[0] / np.ceil(np.log(n)))), n))
    #     # print(tr)

    #     tmp_pos = np.array(np.random.choice(n, size=tr, replace=False), dtype=int)

    #     Tmean = pull_arm(S, tr, dist_func, n)

    #     # Take a moving average over the previous mean (estimate[arms]) and current mean (Tmean)
    #     estimate[S]   = (estimate[S]*T[S] + Tmean*tr)/( T[S] + tr + 0.0 )
    #     # estimate[arms] = Tmean # theoretically don't use past
    #     if tr == n:
    #         estimate[S] = Tmean
    #     T[S] = T[S]+tr

    #     lst = estimate[S]
    #     summary.append(S[np.argmin(lst)])
    #     summary_pulls.append(T.mean())

    #     med = np.median(lst)
    #     locs = np.where(lst<=med)[0]

    #     if tr == n: # calculated exactly
    #         S = [S[np.argmin(lst)]]
    #         break
    #     S = S[locs]

    #     if len(S)==1:
    #         break
    #     round += 1
