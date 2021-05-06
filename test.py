from hack_itau_quant import EfficientFrontier
import yfinance as yf

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