```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

tickers = sorted([])
data = yf.download(tickers, period='1y')['Close']

log_returns = np.log(data / data.shift(1)).dropna()
mu = log_returns.mean() / (1 / len(log_returns))
A = log_returns.cov() / (1 / len(log_returns))

sqrt_A = np.linalg.cholesky(A)

def simulated_prices(mu, sqrt_A, S_0, T, dt, simulations):
    time_steps = int(T / dt)
    S = np.zeros((time_steps + 1, len(tickers), simulations))
    S[0, :, :] = S_0[:, np.newaxis]
    for t in range(time_steps):
        Z = np.random.normal(size=(len(tickers), simulations))
        drift_component = (mu - 0.5 * np.diag(A))[:, np.newaxis] * S[t, :, :] * dt
        volatility_component = sqrt_A @ Z * (np.sqrt(dt) * S[t, :, :])
        S[t + 1, :, :] = S[t, :, :] + drift_component + volatility_component
    return S

T = 1
dt = 1 / 252
simulations = 1000
S_0 = data.iloc[-1].values

trajectories = simulated_prices(mu.values, sqrt_A, S_0, T, dt, simulations)

mean_simulation = np.mean(trajectories, axis=2)
percentiles = np.percentile(trajectories, [5, 95], axis=2)

time_points = np.linspace(0, T, trajectories.shape[0])
for i, ticker in enumerate(data.columns):
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(-1, 0, len(data)), data[ticker].values, label='Real Price', color='black')
    plt.plot(time_points, mean_simulation[:, i], label='Mean Simulated Price', color='blue')
    plt.fill_between(
        time_points,
        percentiles[0, :, i],
        percentiles[1, :, i],
        color='blue',
        alpha=0.1,
        label='Confidence Interval (90%)'
    )
    plt.title(f'Stock Price Simulation')
    plt.xlabel('Time (Years)')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
```