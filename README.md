# STEGASURAS
STEGanography via Arithmetic coding and Strong neURAl modelS

Fork of the original implementations of the steganography algorithms from ["Neural Linguistic Steganography," Zachary M. Ziegler*, Yuntian Deng*, Alexander M. Rush](https://arxiv.org/abs/1909.01496).

This fork is maintained and updated by Sabrina Ning.

## Updates in this Fork

Legacy libraries replaced:
- `pytorch_transformers` &rarr; `transformers`
- Updated other deprecated APIs

All dependencies are listed in `requirements.txt`, including CUDA 12 packages for GPU support.

**NOTE:** This fork only updates the arithmetic coding-based algorithm for compatibility with modern Python and PyTorch environments. The Huffman and binning algorithms remain in their original state and may not work out of the box.

## Online Demo

Our online demo can be found at [https://steganography.live/](https://steganography.live/).

## Language Model

Experiments in the paper use the medium (345M parameter) GPT model via [pytorch_transformers](https://github.com/huggingface/pytorch-transformers). For compute reasons the default in this code base is the small version but the medium or large versions can be used by changing the `model_name` parameter of `get_model`.

## Algorithms

The steganography algorithms implemented are:
1. Our proposed arithmetic coding-based algorithm
2. The Huffman algorithm from [RNN-Stega: Linguistic Steganography Based on Recurrent Neural Networks](https://ieeexplore.ieee.org/document/8470163)
3. The binning algorithm from [Generating Steganographic Text with LSTMs](https://arxiv.org/abs/1705.10742)

An example of encoding and decoding a message is in `run_single.py`. The algorithm used is determined by the `mode` parameter.

An example of encoding and decoding a message using the arithmetic coding-based algorithm is in `run_arithmetic.py`. To change the model, modify the `model_name` parameter of `run_single`.

Currently supported models:
- `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`