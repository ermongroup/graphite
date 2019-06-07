# Graphite: Iterative Generative Modeling of Graphs
Source code for our paper ["Graphite: Iterative Generative Modeling of Graphs"](https://arxiv.org/abs/1803.10459).

If you find it helpful, please consider citing our paper.

    @article{grover2018iterative,
      title={Graphite: Iterative Generative Modeling of Graphs},
      author={Grover, Aditya and Zweig, Aaron and Ermon, Stefano},
      journal={arXiv preprint arXiv:1803.10459},
      year={2018}
    }

## Requirements
1. python 2.7
2. Tensorflow
3. Networkx 1.11

## Training

```
python train.py --epochs 500 --model feedback --edge_dropout 0.5 --learning_rate 0.01 --autoregressive_scalar 0.5
```

This code was built on top of a graph autoencoder (GAE) implementation available [here](https://github.com/tkipf/gae). 

If you have any questions, feel free to contact <adityag@cs.stanford.edu> or <azweig@cs.stanford.edu>.
