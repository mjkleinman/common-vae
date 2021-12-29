import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from dataclasses import dataclass
# from ignite.engine import create_supervised_trainer, \
#     create_supervised_evaluator
# from ignite.metrics import Accuracy
from torch.utils.data import DataLoader
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn.metrics import log_loss

# from .testing import infer


EPS = 1e-12


class HigginsScore:
    def __init__(self, data, n_factors, n_batches=100,
                 batch_size=50, infer=True, lr=0.01):
        self.data = data
        self.n_factors = n_factors
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.infer = infer
        self.lr = lr

    def _factoR_coeffsampler(self):
        zs, classes = self.data

        for i in range(self.n_batches):
            genfact = np.random.choice(self.n_factors)

            n_vals = classes[:, genfact].max()
            gf_val = np.random.choice(n_vals, size=self.batch_size)

            inputs, targets = [], []

            for gf_i in gf_val:
                idx = (classes[:, genfact] == gf_i).nonzero()
                idx = np.random.choice(idx, size=2)

                inputs.append((zs[idx[0]] - zs[idx[0]]).abs())
                targets.append(gf_i)

            yield (torch.as_tensor(inputs, dtype=torch.float32),
                   torch.as_tensor(targets, dtype=torch.long))

    def compute_score(self, model):
        if self.infer:
            training_data = infer(model, self.data)

        classifier = nn.Linear(model.latent_size, self.n_factors)
        loss = nn.CrossEntropyLoss()

        trainer = create_supervised_trainer(classifier,
                                            torch.optim.Adagrad(self.lr), loss)
        trainer.run(training_data)

        evaluator = create_supervised_evaluator(classifier,
                                                {'acc': Accuracy()})
        score = evaluator.run(training_data)['acc']

        return score


@dataclass(frozen=True)
class DCIResults:
    R_coeff: np.ndarray
    disentanglement_scores: np.ndarray
    overall_disentanglment: np.ndarray
    completness_scores: np.ndarray
    log_losses: np.ndarray
    # info_scores: np.ndarray

    def get_scores(self):
        return (self.overall_disentanglment,
                self.disentanglement_scores,
                self.completness_scores,
                self.log_losses)


class DCIMetrics:
    def __init__(self, data, n_factors, regressor='lasso',
                 regressoR_coeffkwargs=None, infer=True):

        kwargs = {'cv': 5, 'selection': 'random',
                  'alphas': [0.02]}

        if regressoR_coeffkwargs is not None:
            kwargs.update(regressoR_coeffkwargs)

        if regressor == 'lasso':
            regressor = LassoCV(**kwargs)
        elif regressor == 'random-forest':
            regressor = RandomForestRegressor(**kwargs)
        elif regressor == 'ensemble':
            regressor = ensemble.GradientBoostingClassifier()
        else:
            raise ValueError()

        self.data = data
        self.n_factors = n_factors
        self.regressor = regressor

    def _get_regressoR_coeffscores(self, X, y, Xtest, ytest):
        """
        Compute R_coeff{dk} for each code dimension D and each
        generative factor K
        """
        R = []
        train_errors = []
        test_errors = []
        log_losses = []

        for k in range(self.n_factors):
            y_k = y[:, k]
            y_test = ytest[:, k]
            if len(np.unique(y_k)) > 1:
                self.regressor.fit(X, y_k)
                train_errors.append(sum(self.regressor.predict(X) == y_k.numpy()) / len(y_k))
                test_errors.append(sum(self.regressor.predict(Xtest) == y_test.numpy()) / len(y_test))
                log_losses.append(log_loss(y_true=y_test.numpy(), y_pred=self.regressor.predict_proba(Xtest)))
                # err = np.sum((yp - y_k.numpy()) ** 2) / len(yp)
                # R.append(np.abs(self.regressor.coef_))
                R.append(np.abs(self.regressor.feature_importances_))
            else:
                R.append(np.zeros(7))

        return np.stack(R).T, np.stack(train_errors).T, np.stack(test_errors).T, np.stack(log_losses).T

    def _disentanglement(self, R_coeff):
        """
        Disentanglement score as in Eastwood et al., 2018.
        """

        # Normalizing factors wrt to generative factors for each latent var
        sums_k = R_coeff.sum(axis=1, keepdims=True) + EPS
        weights = (sums_k / sums_k.sum()).squeeze()

        # Compute probabilities and entropy
        probs = R_coeff / sums_k
        log_probs = np.log(probs + EPS) / np.log(self.n_factors)
        entropy = - (probs * log_probs).sum(axis=1)

        # Compute scores
        di = (1 - entropy)
        total_di = (di * weights).sum()

        return di, total_di

    def _completness(self, R_coeff):
        """
        Completness score as in Eastwood et al., 2018
        """

        # Normalizing factors along each latent for each generative factor
        sums_d = R_coeff.sum(axis=0) + EPS

        # Probabilities and entropy
        probs = R_coeff / sums_d
        log_probs = np.log(probs + EPS) / np.log(R_coeff.shape[0])
        entropy = - (probs * log_probs).sum(axis=0)

        # return completness scores
        return 1 - entropy

    def _informativeness(self, z_p, z):
        if isinstance(self.regressor, LassoCV):
            regressor = MultiTaskLassoCV(cv=self.regressor.cv,
                                         max_iter=2000,
                                         selection='random')

        regressor.fit(z_p, z)
        return self.regressor.score(z_p)

    def compute_score(self, model, model_zs=None):
        if model_zs is None:
            X, y = infer(model, self.data)
            X, y = X.numpy(), y.numpy()
        else:
            X, y = model_zs

        X = (X - X.mean(axis=0))  # / (X.std(axis=0) + EPS)
        # y = (y - y.mean(axis=0)) / (y.std(axis=0) + EPS)

        ntrain = int(0.8 * X.shape[0])
        R_coeff, err_tr, err_test, log_losses = self._get_regressoR_coeffscores(X[:ntrain, :], y[:ntrain], X[ntrain:, :], y[ntrain:])
        # print(R_coeff)

        # compute metrics
        d_scores, total_d_score = self._disentanglement(R_coeff)
        c_scores = self._completness(R_coeff)
        print(f'train acc is {err_tr}')
        print(f'test acc is {err_test}')
        print(f'log losses on test set is {log_losses}')
        # info_score = self._informativeness(X, y)

        return DCIResults(R_coeff, d_scores, total_d_score, c_scores, log_losses)

    def __call__(self, model, model_zs=None):
        return self.compute_score(model, model_zs)


def compute_dci_metrics(models, model_names, factors, data):
    """
    Convenience function to compute the DCI metrics for a set of models
    in a given dataset.
    """
    n_factors = data.n_gen_factors

    loader = DataLoader(data, batch_size=64, num_workers=4, pin_memory=True)

    eastwood = DCIMetrics(loader, n_factors=n_factors)

    results = [eastwood(vae) for vae in models]

    return results


def dci2df(dci_results, model_names, factors):
    """
    Transforms a set of DCI metrics results to a coherent dataframe
    for easy plotting of the results.
    """
    scores = [r.get_scores() for r in dci_results]
    total_disent_score, disent_scores, completness_scores = zip(*scores)

    idx = pd.MultiIndex.from_product([model_names,
                                      range(len(disent_scores[0]))],
                                     names=['models', 'latent'])

    disent_scores = pd.Series(np.concatenate(disent_scores), index=idx)
    disent_scores.name = 'disentanglement'

    idx = pd.MultiIndex.from_product([model_names, factors],
                                     names=['models', 'factor'])
    completness_scores = np.concatenate(completness_scores)
    completness_scores.name = pd.Series(completness_scores,
                                        index=idx,
                                        name='completeness')

    idx = pd.MultiIndex.from_product([model_names], names=['models'])
    overall_disent_score = pd.Series(total_disent_score, index=idx)
    overall_disent_score.name = 'overall disentanglement'

    return overall_disent_score, disent_scores, completness_scores
