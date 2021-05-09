# Biblioteca do Turing USP implementada por nós!
from turingquant.optimizers import Markowitz
import pandas as pd
import numpy as np
from .hrp import HRP


class Backtesting:

    def __init__(self, prices, rebalance_frequency, initial_investment, investment_on_rebalance):

        self._prices = prices

        returns = prices.pct_change()[1:]

        self._returns = returns
        self._n_assets = returns.shape[1]
        self._n_periods = returns.shape[0]

        self._rebalance_frequency = rebalance_frequency
        self._initial_investment = initial_investment
        self._investment_on_rebalance = investment_on_rebalance

    def run(self):
        cumulative_equal_weights = list()
        cumulative_markowitz = list()
        cumulative_hrp = list()

        initial_value = self._initial_investment

        for t in range(self._rebalance_frequency + 2, self._n_periods, self._rebalance_frequency):

            cov_matrix = self._returns[:t].cov()
            returns = self._returns[:t]
            prices = self._prices[:t]

            cumulative_equal_weights.append(
                self._get_daily_returns(weights=Backtesting.get_equal_weights(self._n_assets),
                                        date_index=t, initial_value=initial_value,
                                        freq=self._rebalance_frequency)
            )
            cumulative_markowitz.append(
                self._get_daily_returns(weights=Backtesting.get_markowitz_weights(prices),
                                        date_index=t, initial_value=initial_value,
                                        freq=self._rebalance_frequency)
            )

            cumulative_hrp.append(
                self._get_daily_returns(weights=Backtesting.get_hrp_weights(cov_matrix),
                                        date_index=t, initial_value=initial_value,
                                        freq=self._rebalance_frequency)
            )

            initial_value_equal_weight = cumulative_equal_weights[-1][-1] + \
                self._investment_on_rebalance
            initial_value_markowitz = cumulative_markowitz[-1][-1] + \
                self._investment_on_rebalance
            initial_value_hrp = cumulative_hrp[-1][-1] + \
                self._investment_on_rebalance

        prices_left = self._n_periods % self._rebalance_frequency
        has_prices_left = prices_left <= self._rebalance_frequency
        if has_prices_left:

            t = self._n_periods - prices_left

            cumulative_equal_weights.append(
                self._get_daily_returns(weights=Backtesting.get_equal_weights(self._n_assets),
                                        date_index=t, initial_value=initial_value,
                                        freq=self._rebalance_frequency)
            )
            cumulative_markowitz.append(
                self._get_daily_returns(weights=Backtesting.get_markowitz_weights(prices),
                                        date_index=t, initial_value=initial_value,
                                        freq=self._rebalance_frequency)
            )

            cumulative_hrp.append(
                self._get_daily_returns(weights=Backtesting.get_hrp_weights(cov_matrix),
                                        date_index=t, initial_value=initial_value,
                                        freq=self._rebalance_frequency)
            )

        cumulative_equal_weights = Backtesting.to_numpy_array(cumulative_equal_weights)
        cumulative_markowitz = Backtesting.to_numpy_array(cumulative_markowitz)
        cumulative_hrp = Backtesting.to_numpy_array(cumulative_hrp)

        df = pd.DataFrame()

        df['Equal Weight'] = cumulative_equal_weights
        df['Markowitz'] = cumulative_markowitz
        df['HRP'] = cumulative_hrp

        df.index = self._prices.index[-df.shape[0]:]

        return df

    def _get_daily_returns(self, weights, date_index, initial_value, freq):

        initial_values = np.multiply(initial_value, weights)

        cumulative_returns = self._returns.copy()

        for value, stock in zip(initial_values, range(len(self._returns.columns))):
            cumulative_returns.iloc[:, stock] = value * (
                1 + self._returns.iloc[date_index - freq: date_index, stock]).cumprod()

        returns = np.sum(
            cumulative_returns.iloc[date_index - freq: date_index], axis=1)

        return list(returns)

    @staticmethod
    def get_markowitz_weights(prices):

        markowitz = Markowitz(prices)
        weights = markowitz.best_portfolio('volatility')

        return weights

    @staticmethod
    def get_equal_weights(n_assets):

        return np.ones(n_assets) / n_assets

    @staticmethod
    def get_hrp_weights(cov_matrix):
        hrp = HRP(cov_matrix)
        weights = hrp.optimize()

        return weights

    @staticmethod
    def to_numpy_array(lists):

        final_list = []

        for l in lists:
            for item in l:
                final_list.append(item)

        return np.array(final_list)
