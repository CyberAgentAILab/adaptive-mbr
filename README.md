# Adaptive Minimum Bayes Risk Decoding

This repository contains the code for the experiments in [Hyperparameter-Free Approach for Faster Minimum Bayes Risk Decoding](https://arxiv.org/abs/2401.02749) by Yuu Jinnai and Ariu Kaito.

The code is tested on Ubuntu 20.04 using Python 3.8 and CUDA 11.0 (Docker image nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04).
The code is provided mostly as is with little effort on refactoring. 

## Installation

```
git clone git@github.com:CyberAgentAILab/adaptive-mbr
cd adaptive-mbr
pip install -r requirements.txt
```

## Usage

The code runs in two steps.
1. `sample.sh` samples candidates.
2. `run_mbr.sh` computes the MBR candidate from the candidates sampled.

### Sampling candidates

```
./experiments/sample.sh -d [DATASET] -s [NUMBER OF SAMPLES]
```

### Computing MBR

```
./experiments/run_mbr.sh -d [DATASET] -s [NUMBER OF SAMPLES] -a [ALGORITHM]
```

### Example: WMT'21 En-De

1. Use [sacrebleu](https://github.com/mjpost/sacrebleu) to prepare the benchmark dataset.
```
mkdir -p ./dataset/wmt21
sacrebleu -t wmt21 -l en-de --echo src > ./dataset/wmt21/wmt21.en-de.en
sacrebleu -t wmt21 -l en-de --echo ref > ./dataset/wmt21/wmt21.en-de.de
```

2. Sample candidates
```
./experiments/sample.sh -d wmt21.en-de
```

3. Run adaptive MBR

```
./experiments/run_mbr.sh -d wmt21.en-de -a approx
```

4. Run confidence based pruning (CBP)

```
./experiments/run_mbr.sh -d wmt21.en-de -a pruning
```

## Reference

[Jinnai, Y., & Ariu, K. (2024). Hyperparameter-Free Approach for Faster Minimum Bayes Risk Decoding. arXiv preprint arXiv:2401.02749.](https://arxiv.org/abs/2401.02749)

Bibtex:
```
@article{jinnai2024hyperparameterfree,
      title={Hyperparameter-Free Approach for Faster Minimum Bayes Risk Decoding},
      author={Yuu Jinnai and Kaito Ariu},
      year={2024},
      journal={arXiv preprint arXiv:2401.02749}
}
```

## Contact
For any questions, feel free to raise an issue or contact me at jinnai_yu@cyberagent.co.jp.

## Acknowledgements

[MS COCO dataset](https://cocodataset.org/#home) is licensed under a [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/).
