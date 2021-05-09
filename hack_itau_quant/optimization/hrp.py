# Based on https://medium.com/turing-talks/otimiza%C3%A7%C3%A3o-de-investimentos-com-intelig%C3%AAncia-artificial-548cf34dad4d

import seaborn as sns
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np


class HRP:

    def __init__(self, cov_matrix: pd.DataFrame):

        self._cov_matriz = cov_matrix
        self._columns = cov_matrix.columns.to_list()

    def optimize(self):

        seriation_columns = self._matrix_seriation()

        weights = self._get_weights(seriation_columns)

        return weights

    def _matrix_seriation(self):

        dendogram = sns.clustermap(
            self._cov_matriz, method='ward', metric='euclidean')

        seriation_columns = dendogram.dendrogram_col.reordered_ind

        seriation_columns = [self._columns[index]
                             for index in seriation_columns]

        return seriation_columns

    def _get_weights(self, seriation_columns):
        # Inicialização de weights

        weights = pd.Series(1, index=seriation_columns)
        parities = [seriation_columns]

        while len(parities) > 0:
            parities = [cluster[start:end]
                        for cluster in parities
                        for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                        if len(cluster) > 1]

            for subcluster in range(0, len(parities), 2):

                left_cluster = parities[subcluster]
                right_cluster = parities[subcluster + 1]

                left_cov_matrix = self._cov_matriz[left_cluster].loc[left_cluster]
                inversa_diagonal = 1 / np.diag(left_cov_matrix.values)
                weights_left_cluster = inversa_diagonal / \
                    np.sum(inversa_diagonal)
                vol_left_cluster = np.dot(weights_left_cluster, np.dot(
                    left_cov_matrix, weights_left_cluster))

                right_cov_matrix = self._cov_matriz[right_cluster].loc[right_cluster]
                inversa_diagonal = 1 / np.diag(right_cov_matrix.values)
                weights_right_cluster = inversa_diagonal / \
                    np.sum(inversa_diagonal)
                vol_right_cluster = np.dot(weights_right_cluster, np.dot(
                    right_cov_matrix, weights_right_cluster))

                alocation_factor = 1 - vol_left_cluster / \
                    (vol_left_cluster + vol_right_cluster)

                weights[left_cluster] *= alocation_factor
                weights[right_cluster] *= 1 - alocation_factor

        weights = weights[self._columns].to_numpy()

        return weights