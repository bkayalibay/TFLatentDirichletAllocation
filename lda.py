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

[1] Blei et al.; Latent Dirichlet Allocation; JMLR 2003
[2] Hoffman et al.; Stochastic Variational Inference; JMLR 2013
"""
import six
import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as ds

import utils

from collections import OrderedDict


class LDA:

    def __init__(self, n_topics, vocab_size=None):
        self._K = n_topics
        self._V = vocab_size
        self._D = None
        self._batch_size = None
        self._eta = None
        self._alpha = None
        self._lambdas = None
        self._gammas = None
        self._sess = None

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

    def _compute_elbo(self, gammas, phi, data):
        self._log_lik = 0.0
        kl_thetas = []
        kl_zs = []
        for i, (p, d) in enumerate(zip(phi, data)):
            g = gammas[i]

            # Data log-likelihood:
            lambdas = tf.nn.softmax(self.lambdas)
            lambdas = tf.clip_by_value(lambdas, 1e-2, 1-1e-2)
            word_proportions = tf.gather(
                tf.transpose(lambdas, [1, 0]), d)
            word_proportions = tf.expand_dims(word_proportions, -1)
            p = tf.expand_dims(p, 1)
            log_lik = tf.matmul(p, tf.log(word_proportions))[:, 0, 0]
            log_lik = tf.reduce_sum(log_lik, axis=0)
            self._log_lik += log_lik

            # KL[q(z|phi) || p(z|theta)]
            E_log_theta = tf.digamma(g) - tf.digamma(tf.reduce_sum(g))
            p = tf.clip_by_value(p, 1e-3, 1 - 1e-3)
            kl_z = tf.reduce_sum((tf.log(p) - E_log_theta) * p)
            kl_zs.append(kl_z)

            # KL[q(theta|gamma) || q(theta|alpha)]
            a = self.alpha[i]
            kl_theta_d = tf.lgamma(tf.reduce_sum(g))
            kl_theta_d -= tf.reduce_sum(tf.lgamma(g))
            kl_theta_d -= tf.lgamma(tf.reduce_sum(a))
            kl_theta_d += tf.reduce_sum(tf.lgamma(a))
            kl_theta_d += tf.reduce_sum((g - a) * E_log_theta)
            kl_thetas.append(kl_theta_d)

        # KL[q(beta|lambda) || p(beta|eta)]
        E_log_beta = tf.digamma(self.lambdas)
        E_log_beta -= tf.digamma(tf.reduce_sum(
            self.lambdas, axis=1, keep_dims=True))
        kl_beta = tf.lgamma(tf.reduce_sum(self.lambdas, axis=1))
        kl_beta -= tf.reduce_sum(tf.lgamma(self.lambdas), axis=1)
        kl_beta -= tf.lgamma(tf.reduce_sum(self.eta, axis=1))
        kl_beta += tf.reduce_sum(tf.lgamma(self.eta), axis=1)
        kl_beta += tf.reduce_sum((self.lambdas-self.eta)*E_log_beta, axis=1)
        kl_beta = tf.reduce_sum(kl_beta, axis=0)

        self._kl_terms = OrderedDict(
            kl_z=tf.reduce_sum(kl_zs, axis=0),
            kl_beta=kl_beta,
            kl_theta=tf.reduce_sum(kl_thetas, axis=0),
        )
        kl_list = list(six.itervalues(self._kl_terms))
        self._elbo = self._log_lik - tf.reduce_sum(kl_list)

    def _prepare_update(self, batch_size, local_steps,
                        local_threshold):
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
            gammas, phi = self._e_step(local_steps, local_threshold)

            self._compute_elbo(gammas, phi, self._batch_data)

            lambdas = self._m_step(phi)
            self._update = tf.assign(
                self.lambdas,
                self.lambdas * (1 - self._rho) + self._rho * lambdas)

    def _e_step(self, max_local_steps, local_threshold):
        """Do the E-step of variational EM.

        The E-step of LDA is:
            \phi_{dn}^k = \exp\{E[\log \theta_{dk}] + E[\log \beta_{k, w_{dn}}]\}
            \gamma_d = \alpha + \sum_n \phi_{dn}
        """
        gamma_shape = (len(self._batch_data), self.K)
        gamma_concentration = np.zeros(gamma_shape) + 100.
        gamma_rate = np.zeros(gamma_shape) + 1./100
        gamma = ds.Gamma(
            gamma_concentration.astype('float32'),
            gamma_rate.astype('float32')).sample()
        alpha = tf.gather(self.alpha, self._batch_indices)
        new_gamma = []
        phi = []
        exp_E_log_beta = \
            tf.exp(tf.digamma(self.lambdas) -
                   tf.digamma(tf.reduce_sum(self.lambdas,
                                            axis=-1,
                                            keep_dims=True)))
        exp_E_log_beta = tf.transpose(exp_E_log_beta, [1, 0])

        max_steps = tf.constant(max_local_steps)

        def proceed(gtm1, gt, t, _):
            mean_change = tf.reduce_mean(tf.abs(gt - gtm1))
            not_converged = tf.greater(mean_change, local_threshold)
            not_done = tf.less(t, max_steps)
            return tf.logical_and(not_converged, not_done)

        for i, d in enumerate(self._batch_data):
            g = gamma[i]
            a = alpha[i]
            exp_E_log_beta_d = tf.gather(exp_E_log_beta, d)  # N x K
            phi_d_shape = [tf.shape(d)[0], self.K]

            def body(gtm1, gt, t, phi_dtm1):
                exp_E_log_theta_d = tf.exp(
                    tf.digamma(gt) - tf.digamma(tf.reduce_sum(gt)))
                phi_dt = tf.ones(phi_d_shape) * exp_E_log_theta_d
                phi_dt *= exp_E_log_beta_d
                phinorm = tf.matmul(
                    exp_E_log_beta_d,
                    tf.expand_dims(exp_E_log_theta_d, axis=-1)) + 1e-6
                phi_dt /= phinorm
                gtp1 = a + tf.reduce_sum(phi_dt, axis=0)
                gtp1.set_shape([self.K])
                phi_dt.set_shape([None, self.K])
                return gt, gtp1, t + 1, phi_dt

            zero = tf.constant(0)
            _, g, _, phi_d = tf.while_loop(
                cond=proceed, body=body,
                loop_vars=[tf.zeros_like(g), g, zero,
                           tf.ones(phi_d_shape)])

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
                       max_local_steps=100, use_tqdm=False,
                       tau0=1., kappa=0.9, local_threshold=1e-3,
                       report_every=5):
        """Run the variational EM algorithm by alternating e_step and m_step."""
        if not use_tqdm:
            loop = range(n_update)
        else:
            loop = tqdm.trange(n_update)

        # Prepare update operation if necessary
        self._prepare_update(
            batch_size, max_local_steps,
            local_threshold)

        losses = []

        for t in loop:
            rho = np.power(t + tau0, -kappa)
            indices = np.random.permutation(len(data))[:batch_size]
            batch_data = [data[i] for i in indices]
            feed_dict = {
                bd: bd_in for bd, bd_in in zip(self._batch_data, batch_data)}
            feed_dict.update({self._batch_indices: indices, self._rho: rho})
            if t % report_every == 0 and t > 0:
                kl_list = list(six.itervalues(self._kl_terms))
                _, elbo, ll, kl_z, kl_beta, kl_theta = self.sess.run(
                    [self._update, self._elbo, self._log_lik] + kl_list,
                    feed_dict=feed_dict)
                losses.append([elbo, ll, kl_z, kl_beta, kl_theta])
                if use_tqdm:
                    loop.set_description('log p(x) >= {:.4f}'.format(elbo))
            else:
                self.sess.run(self._update, feed_dict=feed_dict)

        return losses

    def _assert_data_compatibility(self, data, word_count_input):
        if word_count_input:
            vocab_size = data.shape[1]
            data = utils.expand_docs(data)
        else:
            vocab_size = np.max([d.max() for d in data]) + 1

        if self.V is None:
            self._V = vocab_size
        elif self.V != vocab_size:
            raise ValueError(
                "Data vocabulary size does not match" +
                " model's vocabulary size: {} != {}".format(self.V,
                                                            vocab_size)
            )
        if self.D is None:
            self._D = len(data)
        elif self.D != len(data):
            raise ValueError(
                "Number of documents does not match" +
                " model's number of documents: {} != {}".format(self.D,
                                                                len(data))
            )

        return data

    def fit(self, data, n_update, batch_size, max_local_steps=100,
            tau0=1., kappa=0.9, local_threshold=1e-3,
            use_tqdm=False, word_count_input=True,
            report_every=5):
        data = self._assert_data_compatibility(data, word_count_input)
        return self.variational_em(
            data, n_update, batch_size,
            max_local_steps, use_tqdm,
            tau0, kappa, local_threshold,
            report_every)

    def list_topics(self, vocabulary, top_N=10):
        lambda_np = self.sess.run(self.lambdas)
        print('')
        for k, lambda_k in enumerate(lambda_np):
            lambda_k_plus_indices = np.stack(
                [np.arange(0, self.V), lambda_k], axis=1)
            sorted_lambda_k = sorted(
                lambda_k_plus_indices,
                key=lambda x: x[1])
            top_N_elems = sorted_lambda_k[-top_N:]
            top_N_ix = [tup[0] for tup in top_N_elems]
            print('Topic {}: '.format(k+1) +
                  ' '.join([vocabulary[int(i)] for i in top_N_ix[::-1]]))
