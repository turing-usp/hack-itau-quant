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

target_return = 0.0014
target_risk = 0.024

w1 = ef.efficient_return(target_return)
w2 = ef.efficient_risk(target_risk)

print("EFFICIENT RETURN: ", w1)

print("EFFICIENT RISK: ", w2)



e = ef_opt(expected_returns, cov_matrix)

try:
    print("EFFICIENT RETURN: ", e.efficient_return(target_return))  
except:
    pass

try:
    print("EFFICIENT RISK: ", e.efficient_risk(target_risk**2))
except:
    pass

# fig, ax = plt.subplots()
# plotting.plot_efficient_frontier(e, ax=ax, show_assets=False)
# plt.savefig('pyopt.png')

returns, risks = ef.plot_efficient_frontier(0.00005, 30)
plt.plot(risks, returns, label='Otimizador Hack - Hyperbola')
plt.legend()
plt.savefig('test_hyperbola.png')



returns, risks = [], []

for r in tqdm(np.arange(0.0009, 0.0022, 0.00005)):

    w = ef.efficient_return(r).reshape(4, 1)

    rs = np.dot(w.T, expected_returns)

    sigma = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    returns.append(float(rs))
    risks.append(float(sigma))


plt.plot(np.array(risks), np.array(returns), label='Otimizador Hack - Solver')
plt.legend()
plt.savefig('test_solver.png')