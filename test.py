from hack_itau_quant.optimization.quadratic_programming import QuadraticProgrammig
import numpy as np

B = np.array([[1, 2], [3, 4]])
A = np.array([[1, 1]])
c = np.array([[1]])

qp = QuadraticProgrammig(B = B, A = A, c = c)

w, l = qp.solve()

print("---------------- MATRIZ W:")
print(w)
print("---------------- MATRIZ lambda:")
print(l)

#print(np.linalg.inv(C) @ d)

from hack_itau_quant import EfficientFrontier
import yfinance as yf

tickers = ['BPAC11.SA', 'ITUB4.SA', 'OIBR3.SA', 'PETR4.SA']

close_prices = yf.download(tickers, start='2017-01-01', end='2021-01-01')['Close']

returns = close_prices.pct_change()[1:]

expected_returns = returns.mean()
cov_matrix = returns.cov()

ef = EfficientFrontier(expected_returns, cov_matrix)

w = ef.get_wallet(0.1)

print(w)

print(np.dot(w.T, expected_returns))
