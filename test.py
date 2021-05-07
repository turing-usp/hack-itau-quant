from hack_itau_quant import EfficientFrontier
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pypfopt import EfficientFrontier as ef_opt
from pypfopt import plotting

tickers = ['BPAC11.SA', 'ITUB4.SA', 'OIBR3.SA', 'PETR4.SA']

close_prices = yf.download(tickers, start='2017-01-01', end='2021-01-01')['Close']

returns = close_prices.pct_change()[1:]

expected_returns = returns.mean()
cov_matrix = returns.cov()

ef = EfficientFrontier(expected_returns, cov_matrix)

w1 = ef.efficient_return(0.1)

w2 = ef.efficient_risk(0.1)

print("EFFICIENT RETURN: ", w1)

print("EFFICIENT RISK: ", w2)



e = ef_opt(expected_returns, cov_matrix)
# e.min_volatility()

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(e, ax=ax, show_assets=False, ef_param ='risk')
plt.savefig('pyopt.png')

returns, risks = ef.plot_efficient_frontier(0.0005, 70)
plt.plot(risks, returns, label='Hack')
plt.savefig('test_hyperbola.png')



# returns, risks = [], []

# for r in tqdm(np.arange(0.02, 0.09, 0.0005)):

#     w = ef.efficient_return(r).reshape(4, 1)

#     rs = np.dot(w.T, expected_returns * 252)

#     sigma = np.dot(w.T, np.dot(cov_matrix * np.sqrt(252), w))

#     returns.append(float(rs))
#     risks.append(float(sigma))


# plt.plot(np.array(risks) * np.sqrt(252), np.array(returns) * 1)
# plt.savefig('test_solver.png')