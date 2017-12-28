## A tensorflow implementation of Latent Dirichlet Allocation with Stochastic Variational Inference

### TODOs

- [] add citations, reference to onlineldavb
- [] add `LDA.fit()`, which will wrap around `LDA.variational_em()` and handle word-count matrix inputs
- [x] use a tf loop for local steps, early stopping in case of convergence
