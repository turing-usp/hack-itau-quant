import numpy as np
import matplotlib.pyplot as plt
from .optimization import Markowitz
import pandas as pd

import plotly.offline as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class EfficientFrontier:

    def __init__(self, expected_returns: np.array, cov_matrix: np.array):
        self._expected_returns = expected_returns
        self._cov_matrix = cov_matrix
        self._solver = Markowitz(
            expected_returns=self._expected_returns.to_numpy(),
            cov_matrix=self._cov_matrix.to_numpy())

    def efficient_risk(self, target_risk):
        return self._solver.optimal_risk(target_risk)

    def efficient_return(self, target_return):
        return self._solver.optimal_return(target_return)

    def max_loss(self, loss, period, z_alpha=-1.645):

        start_return = self._solver.get_start_return()

        for r in np.arange(start_return, 1, 0.00005):

            w = self.efficient_return(r).reshape(-1)
            rp = np.dot(w.T, self._expected_returns) * period
            sigma_p = np.sqrt(
                np.dot(w.T, np.dot(self._cov_matrix.to_numpy() * np.sqrt(period), w)))

            z = (loss - rp) / (sigma_p / np.sqrt(period))

            is_alternative_hypotesis = z <= z_alpha
            if is_alternative_hypotesis:
                return w

        return None

    def plot_efficient_frontier(self, n_points=100):

        returns, risks, weigths = self._solver.get_efficient_curve(n_points)
        self._plot(returns, risks, weigths)

    def _plot(self, returns, risks, weigths):

        cols_name = self._cov_matrix.columns

        df = pd.DataFrame(weigths, columns=cols_name)
        df = (df * 100).round(2).astype(float)

        df['Returns'] = returns
        df['Volatilies'] = risks
        df['Sharpe'] = (df['Returns']) / (df['Volatilies'])

        max_sharpe = df[df.Sharpe == df.Sharpe.max()]
        min_vol = df[df.Volatilies == df.Volatilies.min()]

        Efficient_Frontier = go.Scatter(x = df['Volatilies'],
                            y = df['Returns'],
                            text= df['Sharpe'],
                            customdata = np.stack(tuple(df[col] for col in cols_name), axis = -1),
                            hovertemplate="Volatilities: %{x:,.2f}<br>"+
                            "Returns: %{y:.5f}<br>"+
                            "Sharpe: %{text:.5f}<br>"+
                            "BPAC11.SA: %{customdata[0]}%<br>"+
                            "ITUB4.SA: %{customdata[1]}%<br>"+
                            "OIBR3.SA: %{customdata[2]}%<br>"+
                            "PETR4.SA: %{customdata[3]}%<br>",
                            mode = ' markers + lines',
                            showlegend = False)

        Max_Sharpe = go.Scatter(x = max_sharpe.Volatilies,
                            y = max_sharpe.Returns,
                            text=df[df['Volatilies']  == float(max_sharpe.Volatilies)]['Sharpe'],
                            customdata = np.stack((df[df['Volatilies']  == float(max_sharpe.Volatilies)]['BPAC11.SA'], 
                                                df[df['Volatilies']  == float(max_sharpe.Volatilies)]['ITUB4.SA'], 
                                                df[df['Volatilies']  == float(max_sharpe.Volatilies)]['OIBR3.SA'], 
                                                df[df['Volatilies']  == float(max_sharpe.Volatilies)]['PETR4.SA']), 
                                                axis = -1),
                            hovertemplate=
                            "Volatilities: %{x:,.2f}<br>"+
                            "Returns: %{y:.5f}<br>"+
                            "Sharpe: %{text:.4f}<br>"+
                            "BPAC11.SA: %{customdata[0]}%<br>"+
                            "ITUB4.SA: %{customdata[1]}%<br>"+
                            "OIBR3.SA: %{customdata[2]}%<br>"+
                            "PETR4.SA: %{customdata[3]}%<br>",
                            mode = ' markers + lines',
                            name = 'Maximum Sharpe')

        Min_Vol = go.Scatter(x = min_vol.Volatilies,
                            y = min_vol.Returns,
                            text=df[df['Volatilies']  == float(min_vol.Volatilies)]['Sharpe'],
                            customdata = np.stack((df[df['Volatilies']  == float(min_vol.Volatilies)]['BPAC11.SA'], 
                                                df[df['Volatilies']  == float(min_vol.Volatilies)]['ITUB4.SA'], 
                                                df[df['Volatilies']  == float(min_vol.Volatilies)]['OIBR3.SA'], 
                                                df[df['Volatilies']  == float(min_vol.Volatilies)]['PETR4.SA']), 
                                                axis = -1),
                            hovertemplate=
                            "Volatilities: %{x:,.2f}<br>"+
                            "Returns: %{y:.5f}<br>"+
                            "Sharpe: %{text:.4f}<br>"+
                            "BPAC11.SA: %{customdata[0]}%<br>"+
                            "ITUB4.SA: %{customdata[1]}%<br>"+
                            "OIBR3.SA: %{customdata[2]}%<br>"+
                            "PETR4.SA: %{customdata[3]}%<br>",
                            mode = ' markers + lines',
                            name = 'Minimal Volatility')
        

        layout = go.Layout(title='Efficient Frontier',
                        title_x=0.5,
                        title_y=0.9,
                        yaxis={'title':'Returns','tickformat':'.5f'},
                        xaxis={'title':'Volatities', 'tickformat':'.2f'},
                        font=dict(size=18))

        fig = go.Figure(data = [Efficient_Frontier, Max_Sharpe, Min_Vol], layout = layout)


        fig.show()
