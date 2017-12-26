"""
This is a tensorflow implementation of Latent Dirichlet Allocation[1] with
stochastic variational inference[2]. In LDA, we have a probabilistic model
of documents based on topics. Each document is a collection of words from
a vocabulary and each topic is a Dirichlet distribution over this vocabulary.
We also have a Dirichlet distribution over the set of topics for each document.
A word is assigned to a topic by first sampling from the document Dirichlet and
then sampling from a multinomial distribution with that sample. Then, a sample
from that topic is used to parameterize a multinomial distribution over the
vocabulary and a sample from that distribution produces the word itself. This is
the generative model of the set of documents.

We would like to do posterior inference of:
   * Local variables:
       Probability of each topic in every document
   * Global variables:
       Probability of each word in every topic

To accomplish this, we will construct an approximate variational posterior
and use stochastic variational inference.
Alternatively, we could also use MCMC with collapsed Gibbs sampling[3].

[1] Blei et al.; Latent Dirichlet Allocation; JMLR 2003
[2] Hoffman et al.; Stochastic Variational Inference; JMLR 2013
[3] Griffiths and Steyvers; Finding Scientific Topics; PNAS 2004
"""
import tqdm
import numpy as np
import tensorflow as tf


class LDA:

    def __init__(self, n_docs, n_topics, n_words, tau0=1.0, kappa=0.9):
        self._D = n_docs
        self._K = n_topics
        self._V = n_words
        self._tau0 = tau0
        self._kappa = kappa
        self._alloc_vars()
        self._batch_size = None

    @property
    def D(self):
        return self._D

    @property
    def K(self):
        return self._K

    @property
    def V(self):
        return self._V

    @property
    def tau0(self):
        return self._tau0

    @property
    def kappa(self):
        return self._kappa

    @property
    def alpha(self):
        return self._alpha

    @property
    def eta(self):
        return self._eta

    @property
    def gammas(self):
        return self._gammas

    @property
    def lambdas(self):
        return self._lambdas

    @property
    def sess(self):
        return self._sess

    def _alloc_vars(self):
        D = self.D
        K = self.K
        V = self.V

        # Prior
        self._alpha = tf.ones([D, K]) * 0.1
        self._eta = tf.ones([K, V]) * 0.01

        # Variational parameters:
        self._gammas = tf.Variable(tf.ones([D, K]) * 0.1)
        self._lambdas = tf.Variable(
            tf.convert_to_tensor(
                np.random.exponential(
                    D * 100. / (K*V), (K, V)).astype('float32')))

        self._sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _prepare_update(self, batch_size, local_steps):
        if self._batch_size != batch_size:
            if self._batch_size is not None:
                tf.reset_default_graph()
                self._alloc_vars()
            self._batch_size = batch_size
            self._batch_data = [tf.placeholder(
                                    dtype='int32',
                                    name='batch_data_{}'.format(i))
                                for i in range(batch_size)]
            self._batch_indices = tf.placeholder(
                dtype='int32', name='batch_indices')
            self._rho = tf.placeholder(dtype='float32', name='rho')
            gammas, phi = self._e_step(local_steps)
            lambdas = self._m_step(phi)
            self._update = tf.assign(
                self.lambdas,
                self.lambdas * (1 - self._rho) + self._rho * lambdas)

    def _e_step(self, local_steps):
        """Do the E-step of variational EM.

        The E-step of LDA is:
            \phi_{dn}^k = \exp\{E[\log \theta_{dk}] + E[\log \beta_{k, w_{dn}}]\}
            \gamma_d = \alpha + \sum_n \phi_{dn}
        """
        init_gamma = np.random.gamma(
            100., 1./100, (len(self._batch_data), self.K))
        gamma = tf.convert_to_tensor(init_gamma.astype('float32'))
        alpha = tf.gather(self.alpha, self._batch_indices)
        new_gamma = []
        phi = []
        exp_E_log_beta = \
            tf.exp(tf.digamma(self.lambdas) -
                   tf.digamma(tf.reduce_sum(self.lambdas,
                                            axis=-1,
                                            keep_dims=True)))
        exp_E_log_beta = tf.transpose(exp_E_log_beta, [1, 0])
        for i, d in enumerate(self._batch_data):
            g = gamma[i]
            a = alpha[i]
            exp_E_log_beta_d = tf.gather(exp_E_log_beta, d)  # N x K
            for _ in range(local_steps):
                phi_d = tf.ones([tf.shape(d)[0], self.K])
                exp_E_log_theta_d = tf.exp(tf.digamma(g))  # K
                phi_d *= exp_E_log_theta_d
                phi_d *= exp_E_log_beta_d
                phinorm = tf.matmul(
                    exp_E_log_beta_d,
                    tf.expand_dims(exp_E_log_theta_d, axis=-1)) + 1e-6
                phi_d /= phinorm
                g = a + tf.reduce_sum(phi_d, axis=0)
            new_gamma.append(g)
            phi.append(phi_d)
        gamma = tf.stack(new_gamma)
        return gamma, phi

    def _m_step(self, phi):
        """Do the M-step of variational EM.

        The M-step of LDA is:
            \lambda_k = \eta + D\sum_n \phi_{dn}^k w_{dn}
        """
        update = tf.zeros([self.K, self.V])
        bs = float(len(self._batch_data))
        for i, d in enumerate(self._batch_data):
            phi_d = tf.transpose(phi[i], [1, 0])
            ws = tf.one_hot(d, depth=self.V, axis=-1)
            update += tf.matmul(phi_d, ws)
        return self.eta + (self.D/bs) * update

    def variational_em(self, data, n_update, batch_size,
                       local_steps, use_tqdm=False):
        """Run the variational EM algorithm by alternating e_step and m_step."""
        if not use_tqdm:
            loop = range(n_update)
        else:
            loop = tqdm.trange(n_update)

        # Prepare update operation if necessary
        self._prepare_update(batch_size, local_steps)

        for t in loop:
            rho = np.power(t + self.tau0, -self.kappa)
            indices = np.random.permutation(len(data))[:batch_size]
            batch_data = [data[i] for i in indices]
            feed_dict = {
                bd: bd_in for bd, bd_in in zip(self._batch_data, batch_data)}
            feed_dict.update({self._batch_indices: indices, self._rho: rho})
            self.sess.run(self._update, feed_dict=feed_dict)

    def list_topics(self, vocabulary, top_N=10):
        lambda_np = self.sess.run(self.lambdas)
        print('')
        for k, lambda_k in enumerate(lambda_np):
            print('Topic {}'.format(k+1))
            print('-' * 15)
            lambda_k_plus_indices = np.stack(
                [np.arange(0, self.V), lambda_k], axis=1)
            sorted_lambda_k = sorted(
                lambda_k_plus_indices,
                key=lambda x: x[1])
            top_N_elems = sorted_lambda_k[-top_N:]
            top_N_ix = [tup[0] for tup in top_N_elems]
            for i in top_N_ix:
                print(vocabulary[int(i)])
            print('')
