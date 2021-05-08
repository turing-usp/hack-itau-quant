from hack_itau_quant import EfficientFrontier, BloombergData
import matplotlib.pyplot as plt
import numpy as np

close_prices = BloombergData.get()

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



# e = ef_opt(expected_returns, cov_matrix)

# fig, ax = plt.subplots()
# plotting.plot_efficient_frontier(e, ax=ax, show_assets=False)
# plt.savefig('pyopt.png')

ef.plot_efficient_frontier()



returns, risks = [], []

for r in tqdm(np.arange(0.0009, 0.0022, 0.00005)):

    try:
        w = ef.efficient_return(r).reshape(4, 1)

        rs = np.dot(w.T, expected_returns)

        sigma = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        returns.append(float(rs))
        risks.append(float(sigma))
    except:
        pass


# plt.plot(np.array(risks), np.array(returns), label='Otimizador Hack - Solver')
# plt.legend()
# plt.savefig('test_solver.png')

# print(ef.max_loss(-0.06, 1))