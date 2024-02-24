# Adaptive Minimum Bayes Risk Decoding

This repository contains the code for the experiments in [Hyperparameter-Free Approach for Faster Minimum Bayes Risk Decoding](https://arxiv.org/abs/2401.02749) by Yuu Jinnai and Ariu Kaito.

The code is provided mostly as is with little effort on refactoring.

## Installation

```
git clone git@github.com/jinnaiyuu/adaptive-mbr
cd adaptive-mbr
pip install requirements.txt
```

## Usage

The code runs in two steps.
1. `sample.sh` samples candidates.
2. `run_mbr.sh ` computes the MBR candidate from the candidates sampled.

### Sampling candidates

```
./experiments/sample.sh -d [DATASET] -s [NUMBER OF SAMPLES] 
```

### Computing MBR

```
./experiments/run_mbr.sh -d [DATASET] -s [NUMBER OF SAMPLES] -a [ALGORITHM]
```

### Example: WMT'21 En-De

```
./experiments/sample.sh -d wmt21.en-de
```

Running adaptive MBR

```
./experiments/run_mbr.sh -d wmt21.en-de -a approx
```

Running confidence based pruning (CBP)

```
./experiments/run_mbr.sh -d wmt21.en-de -a pruning
```


