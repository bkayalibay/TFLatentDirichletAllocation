## A tensorflow implementation of Latent Dirichlet Allocation with Stochastic Variational Inference
### Introduction
This repository contains a tensorflow implementation of LDA as described in [1]. I used the numpy
implementation by Matthew Hoffman as a reference. You can find it at: https://github.com/blei-lab/onlineldavb.
### TODOs

- [ ] add citations, reference to onlineldavb
- [ ] add documentation
- [ ] provide evaluation metrics like preplexity, log-likelihood, elbo
- [x] add `LDA.fit()`, which will wrap around `LDA.variational_em()` and handle word-count matrix inputs
- [x] use a tf loop for local steps, early stopping in case of convergence
### References
1. Hoffman et al.; [Stochastic Variational Inference](http://jmlr.org/papers/v14/hoffman13a.html); Journal of Machine Learning Research; pages 1303-1347; 2013
