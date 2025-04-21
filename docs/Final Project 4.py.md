```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

tickers = ['AAPL', 'MSFT', 'AMZN', 'JNJ', 'JPM', 'XOM', 'PFE', 'NVDA', 'WMT', 'DIS']
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

data = yf.download(tickers, start=start_date, end=end_date)['Close']
data.dropna(inplace=True)

mu = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)

ef = EfficientFrontier(mu, S)
ef_weights = ef.max_sharpe()
ef_cleaned_weights = ef.clean_weights()
ef_perf = ef.portfolio_performance(verbose=False)

hrp = HRPOpt(data)
hrp_weights = hrp.optimize()
hrp_perf = hrp.portfolio_performance(verbose=False)

ef_df = pd.DataFrame.from_dict(ef_cleaned_weights, orient='index', columns=['Efficient Frontier'])
hrp_df = pd.DataFrame.from_dict(hrp_weights, orient='index', columns=['HRP'])
weights_df = ef_df.join(hrp_df)

num_simulations = 1000
num_days_forward = 5

ef_sim_results = np.zeros((num_days_forward, num_simulations))
hrp_sim_results = np.zeros((num_days_forward, num_simulations))

last_prices = data.iloc[-1].values

for sim in range(num_simulations):
    future_returns = np.random.normal(loc=0.0005, scale=0.02, size=(num_days_forward, len(tickers)))

    future_prices = last_prices * np.exp(np.cumsum(future_returns, axis=0))
    future_prices_df = pd.DataFrame(future_prices, columns=tickers)
    
    ef_vals = future_prices_df @ weights_df['Efficient Frontier']
    hrp_vals = future_prices_df @ weights_df['HRP']
    
    ef_sim_results[:, sim] = ef_vals
    hrp_sim_results[:, sim] = hrp_vals

ef_avg = np.mean(ef_sim_results, axis=1)
hrp_avg = np.mean(hrp_sim_results, axis=1)

ef_avg /= ef_avg[0]
hrp_avg /= hrp_avg[0]

perf_df = pd.DataFrame({
    'EF': ef_avg,
    'HRP': hrp_avg
})

ef_vals = future_prices_df @ weights_df['Efficient Frontier']
hrp_vals = future_prices_df @ weights_df['HRP']

print(weights_df)
print(perf_df)
```