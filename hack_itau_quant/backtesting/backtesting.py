from hack_itau_quant import EfficientFrontier
import numpy as np
import pandas as pd

def to_numpy_array(lists):

    final_list = []

    for l in lists:
        for item in l:
              final_list.append(item)

    return np.array(final_list)

class Backtesting:
    
    def __init__(self, returns, rebalance_frequency, initial_investment, investment_on_rebalance):
        
        self._returns = returns
        self._n_assets = returns.shape[1]
        self._n_periods = returns.shape[0]
        
        self._rebalance_frequency = rebalance_frequency
        self._initial_investment = initial_investment
        self._investment_on_rebalance = investment_on_rebalance
        
    def run(self):
        cumulative_daily = list()
        
        initial_value = self._initial_investment

        for t in range(self._rebalance_frequency + 2, self._n_periods, self._rebalance_frequency):

            cumulative_daily.append(
                self._get_daily_returns(date_index = t, initial_value = initial_value, freq = self._rebalance_frequency)
            )

            initial_value = cumulative_daily[-1][-1] + self._investment_on_rebalance

        prices_left = self._n_periods % self._rebalance_frequency
        has_prices_left = prices_left <= self._rebalance_frequency
        if has_prices_left:
                
            t = self._n_periods - prices_left
            
            cumulative_daily.append(
                self._get_daily_returns(t, initial_value = initial_value, freq = prices_left)
            )

        rts = to_numpy_array(cumulative_daily)
        df = pd.DataFrame(rts)
        df.index = returns.index
        df.columns = ['Patrimônio']

        return df

    def _get_daily_returns(self, date_index, initial_value, freq):
        
        weights = Backtesting.get_markowitz_weights(self._returns.iloc[:date_index, :])

        initial_values = np.multiply(initial_value , weights)

        cumulative_returns = self._returns.copy()

        for value, stock in zip(initial_values, range(len(self._returns.columns))):
            cumulative_returns.iloc[:, stock] = value * (1 + self._returns.iloc[date_index -freq: date_index, stock]).cumprod()

        returns = np.sum(cumulative_returns.iloc[date_index - freq: date_index], axis=1)

        return list(returns)
    
    @staticmethod
    def get_markowitz_weights(returns):
    
        expected_returns = returns.mean()
        cov_matrix = returns.cov()

        ef = EfficientFrontier(expected_returns, cov_matrix)
        weights = ef.min_risk()
        
        return weights
    
    
    @staticmethod
    def get_equal_weights(n_assets):
        
        return np.ones(n_assets) / n_assets
    
    @staticmethod
    def get_hrp_weights(cov_matriz):
        