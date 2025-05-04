```Python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

tickers = sorted([])
data = yf.download(tickers, period='5y')['Adj Close']

def CAPM_plot(data, capital, allowable_risk, rf):
    n_assets = len(data.columns)
    pct_change = data.pct_change().dropna()

    mu = pct_change.mean() * 252
    A = pct_change.cov() * 252

    ones = np.ones(n_assets)
    inv_A = np.linalg.inv(A)

    a = ones.T @ inv_A @ ones
    b = ones.T @ inv_A @ mu
    c = mu.T @ inv_A @ mu

    w_G = inv_A @ ones / a

    mu_G = w_G.T @ mu
    var_G = w_G.T @ A @ w_G

    target_returns = np.linspace(mu_G, max(mu)*2, 100)

    risk = []
    expected_returns = []
    for r in target_returns:
        w = (inv_A @ ones * (c - b * r) + inv_A @ mu * (a * r - b)) / (a * c - b**2)
        std = np.sqrt(w.T @ A @ w)

        risk.append(std)
        expected_returns.append(r)

    risk_return = pd.DataFrame({
        'Risk': risk,
        'Expected Return': expected_returns
    })

    w_M = inv_A @ (mu - rf * ones) / (b - a * rf)
    return_M = w_M.T @ mu
    risk_M = np.sqrt(w_M.T @ A @ w_M)

    cml_risk = np.linspace(min(risk_return['Risk']), max(risk_return['Risk']), 100)
    cml_return = rf + (return_M - rf) * (cml_risk / risk_M)

    plt.figure(figsize=(10, 6))
    plt.plot(risk_return['Risk'], risk_return['Expected Return'], label='Efficient Frontier')
    plt.plot(cml_risk, cml_return, label='Capital Market Line', color='orange', linestyle='--')
    plt.scatter(risk_M, return_M, label='Market Portfolio', color='g')
    plt.scatter(np.sqrt(var_G), mu_G, label='Global Minimum Variance Portfolio', c='r')

    plt.title('Efficient Frontier and Capital Market Line')
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.grid(True)
    plt.legend()
    plt.show()

    riskless_ratio = allowable_risk / risk_M
    invested_capital = capital * riskless_ratio
    riskless_capital = capital - invested_capital
    allocated_capital = w_M * invested_capital

    expected_future_value = capital + (capital * (rf + (return_M - rf) * riskless_ratio))

    return riskless_capital, invested_capital, allocated_capital, expected_future_value

riskless_security = 0.05

capital = 1000
allowable_risk = 0.2

rc, ic, ac, efv = CAPM_plot(data, capital, allowable_risk, riskless_security)

shares = ac / data.iloc[-1]

print(f'Capital Allocation \n'
      f'- Riskless Capital: ${rc:.2f}')
print(f'- Invested Capital: ${ic:.2f}')
for ticker, allocation, share in zip(tickers, ac, shares):
    print(f'    {ticker}: ${allocation:.2f}, Shares: {share:.2f}')
print(f'- Expected Future Value: ${efv:.2f}')
```