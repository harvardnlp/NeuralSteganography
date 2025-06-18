# STEGASURAS
STEGanography via Arithmetic coding and Strong neURAl modelS

This repository contains implementations of the steganography algorithms from ["Neural Linguistic Steganography," Zachary M. Ziegler*, Yuntian Deng*, Alexander M. Rush](https://arxiv.org/abs/1909.01496).

## Update (June 2025)

Thanks to [Sabrina Ning](https://github.com/sabrina-ning) and [Wei-Chiu Ma](https://www.cs.cornell.edu/~weichiu/) for updating this repository to be compatible with the latest versions of PyTorch and Hugging Face Transformers.

## Online Demo

Our online demo can be found at [https://steganography.live/](https://steganography.live/).

## Language model

Experiments in the paper use the medium (345M parameter) GPT model via [pytorch_transformers](https://github.com/huggingface/pytorch-transformers). For compute reasons the default in this code base is the small version but the medium or large versions can be used by changing the `model_name` parameter of `get_model`.

## Algorithms

The steganography algorithms implemented are:
1. Our proposed arithmetic coding-based algorithm
2. The Huffman algorithm from [RNN-Stega: Linguistic Steganography Based on Recurrent Neural Networks](https://ieeexplore.ieee.org/document/8470163)
3. The binning algorithm from [Generating Steganographic Text with LSTMs](https://arxiv.org/abs/1705.10742)

An example of encoding and decoding a message is in `run_single.py`. The algorithm used is determined by the `mode` parameter.


## Citation

```
@inproceedings{ziegler-etal-2019-neural,
    title = "Neural Linguistic Steganography",
    author = "Ziegler, Zachary  and
      Deng, Yuntian  and
      Rush, Alexander",
    editor = "Inui, Kentaro  and
      Jiang, Jing  and
      Ng, Vincent  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1115/",
    doi = "10.18653/v1/D19-1115",
    pages = "1210--1215",
    abstract = "Whereas traditional cryptography encrypts a secret message into an unintelligible form, steganography conceals that communication is taking place by encoding a secret message into a cover signal. Language is a particularly pragmatic cover signal due to its benign occurrence and independence from any one medium. Traditionally, linguistic steganography systems encode secret messages in existing text via synonym substitution or word order rearrangements. Advances in neural language models enable previously impractical generation-based techniques. We propose a steganography technique based on arithmetic coding with large-scale neural language models. We find that our approach can generate realistic looking cover sentences as evaluated by humans, while at the same time preserving security by matching the cover message distribution with the language model distribution."
}
```
