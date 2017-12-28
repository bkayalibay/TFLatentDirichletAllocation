## A tensorflow implementation of Latent Dirichlet Allocation with Stochastic Variational Inference
This repository contains a tensorflow implementation of LDA as described in [cite]. I used the numpy
implementation by Matthew Hoffman as a reference. You can find it at: https://github.com/blei-lab/onlineldavb.
### TODOs

- [ ] add citations, reference to onlineldavb
- [ ] add documentation
- [x] add `LDA.fit()`, which will wrap around `LDA.variational_em()` and handle word-count matrix inputs
- [x] use a tf loop for local steps, early stopping in case of convergence
