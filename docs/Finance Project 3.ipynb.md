```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Section 1 Problem 1a

T = 1  
M = 1000 
K = 10000 
dt = T / M 

dW = np.random.normal(0, np.sqrt(dt), (K, M))
W = np.hstack((np.zeros((K, 1)), np.cumsum(dW, axis=1)))

Ito_integral_x = np.sum(W[:, :-1] * dW, axis=1)

Ito_integral_x2 = np.sum((W[:, :-1]**2) * dW, axis=1)

time = np.linspace(0, T, M)

mean_I_x = np.mean(W[:, :-1] * dW, axis=0)
var_I_x = np.var(W[:, :-1] * dW, axis=0)

mean_I_x2 = np.mean((W[:, :-1]**2) * dW, axis=0)
var_I_x2 = np.var((W[:, :-1]**2) * dW, axis=0)

# Prepare data for visualization
df_results = pd.DataFrame({
    'Time': time,
    'Mean_I_x': mean_I_x,
    'Var_I_x': var_I_x,
    'Mean_I_x2': mean_I_x2,
    'Var_I_x2': var_I_x2
})

# Plot mean and variance of the Itô integrals
plt.figure(figsize=(8, 6))
plt.plot(time, mean_I_x, label='Mean x')
plt.plot(time, var_I_x, label='Variance x')
plt.plot(time, mean_I_x2, label='Mean x^2')
plt.plot(time, var_I_x2, label='Variance x^2')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Ito Mean and Variance')
plt.legend()
plt.show()
```

```Python
# Section 1 Problem 1b

strat_x = np.sum(0.5 * (W[:, :-1] + W[:, 1:]) * dW, axis=1)

strat_x2 = np.sum(0.5 * ((W[:, :-1]**2) + (W[:, 1:]**2)) * dW, axis=1)

mean_strat_x = np.mean(0.5 * (W[:, :-1] + W[:, 1:]) * dW, axis=0)
var_strat_x = np.var(0.5 * (W[:, :-1] + W[:, 1:]) * dW, axis=0)

mean_strat_x2 = np.mean(0.5 * ((W[:, :-1]**2) + (W[:, 1:]**2)) * dW, axis=0)
var_strat_x2 = np.var(0.5 * ((W[:, :-1]**2) + (W[:, 1:]**2)) * dW, axis=0)

df_strat_results = pd.DataFrame({
    'Time': time,
    'Mean_strat_x': mean_strat_x,
    'Var_strat_x': var_strat_x,
    'Mean_strat_x2': mean_strat_x2,
    'Var_strat_x2': var_strat_x2
})

plt.figure(figsize=(8, 6))
plt.plot(time, mean_strat_x, label='Mean x')
plt.plot(time, var_strat_x, label='Variance x')
plt.plot(time, mean_strat_x2, label='Mean x^2')
plt.plot(time, var_strat_x2, label='Variance x^2')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Stratonovich Mean and Variance')
plt.legend()
plt.show()
```

```Python
# Section 1 Problem 1c

r = 0.03 
sigma_A = 0.2  
sigma_B = 0.3 
rho = 0.5 
S_0_A = 50
S_0_B = 50 

W_1 = np.random.randn(K, M) 
W_2 = rho * W_1 + np.sqrt(1 - rho**2) * np.random.randn(K, M) 

dt_sqrt = np.sqrt(dt)

S_A = np.zeros((K, M + 1))
S_B = np.zeros((K, M + 1))
S_A[:, 0] = S_0_A
S_B[:, 0] = S_0_B

for j in range(M):
    S_A[:, j+1] = S_A[:, j] * np.exp((r - 0.5 * sigma_A**2) * dt + sigma_A * dt_sqrt * W_1[:, j])
    S_B[:, j+1] = S_B[:, j] * np.exp((r - 0.5 * sigma_B**2) * dt + sigma_B * dt_sqrt * W_2[:, j])

V_T = np.maximum(S_B[:, -1] - S_A[:, -1], 0)

exchange_option_value = np.exp(-r * T) * np.mean(V_T)

print(f'Exchange Option Value: {exchange_option_value}')
```

```Python
# Section 1 Problem 2a

S_0 = 50
mu = 0.05
sigma = 0.25
T = 1
dt = 1/252
M = int(T/dt)
n = 1000

np.random.seed(42)
dW = np.random.normal(0, np.sqrt(dt), (M, n))
S = np.zeros((M+1, n))
S[0] = S_0

for t in range(1, M+1):
    S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t-1])

plt.figure(figsize=(8, 6))
plt.plot(S[:, :1000], alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Simulated Price Paths')
plt.show()
```

```Python
# Section 1 Problem 2b

R_i = np.log(S[1:] / S[:-1])
R_hat = np.mean(R_i, axis=0)
RV = np.sum((R_i - R_hat)**2, axis=0)

variance = sigma**2 * T

df_variance = pd.DataFrame({
    'Realized Variance': RV,
    'Theoretical Variance': variance
})

print(df_variance['Realized Variance'].describe())
df_variance
```

```Python
# Section 1 Problem 2c

kappa = 1
theta = 0.02
sigma_V = 0.2
rho = -0.5
V_0 = 0.0625

np.random.seed(42)
dW_1 = np.random.normal(0, np.sqrt(dt), (M, n))
dW_2 = rho * dW_1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), (M, n))

V = np.zeros((M+1, n))
S_Heston = np.zeros((M+1, n))

V[0, :] = V_0
S_Heston[0, :] = S_0

for t in range(1, M+1):
    V[t, :] = np.maximum(V[t-1, :] + kappa * (theta - V[t-1, :]) * dt + sigma_V * np.sqrt(V[t-1, :]) * dW_2[t-1, :], 0)
    S_Heston[t, :] = S_Heston[t-1, :] * np.exp((mu - 0.5 * V[t-1, :]) * dt + np.sqrt(V[t-1, :]) * dW_1[t-1, :])

quadratic_variation = np.sum((np.sqrt(V[:-1, :]) * dW_1) ** 2, axis=0)
sigma_quadratic_variation = np.var(quadratic_variation)

print(f'Variance of Quadtatic Variation: {sigma_quadratic_variation}')
```

```Python
from scipy.stats import norm
from scipy.optimize import root_scalar
import yfinance as yf
import plotly.express as px

# Section 2 Problem 1

spx = yf.Ticker('^SPX')

expiration = ['2025-03-31', '2025-04-30', '2025-05-30', '2025-06-30']

call_options = pd.concat([spx.option_chain(exp).calls.assign(expiration=exp) for exp in expiration])
call_options = call_options[call_options['lastPrice'] > 800]
call_sample = call_options.sample(10)

put_options = pd.concat([spx.option_chain(exp).puts.assign(expiration=exp) for exp in expiration])
put_options = put_options[put_options['lastPrice'] > 800]
put_sample = put_options.sample(10)

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(S, K, T, r, market_price):
    try:
        result = root_scalar(lambda sigma: black_scholes_call(S, K, T, r, sigma) - market_price, bracket=[0.001, 5], method='brentq')
        return result.root
    except ValueError:
        return np.nan

r = 0.03
S_0 = spx.history(period='1d')['Close'].iloc[-1]

call_sample['T'] = (pd.to_datetime(call_sample['expiration']) - pd.Timestamp.today()).dt.days / 365
call_sample.loc[:, 'Implied Volatility'] = call_sample.apply(
    lambda row: implied_volatility(S_0, row['strike'], row['T'], r, row['lastPrice']), axis=1
)

put_sample['T'] = (pd.to_datetime(put_sample['expiration']) - pd.Timestamp.today()).dt.days / 365
put_sample.loc[:, 'Implied Volatility'] = put_sample.apply(
    lambda row: implied_volatility(S_0, row['strike'], row['T'], r, row['lastPrice']), axis=1
)

fig1 = px.scatter_3d(x=call_sample['T'], y=call_sample['strike'] / S_0, z=call_sample['Implied Volatility'],
                     labels={'x':'T', 'y':'K / S_0', 'z':'Implied Volatility'}, title='Call Implied Volatility Skew')
fig1.update_layout(width=800, height=600)
fig1.show()

fig2 = px.scatter_3d(x=put_sample['T'], y=put_sample['strike'] / S_0, z=put_sample['Implied Volatility'],
                     labels={'x':'T', 'y':'K / S_0', 'z':'Implied Volatility'}, title='Put Implied Volatility Skew')
fig2.update_layout(width=800, height=600)
fig2.show()
```

```Python
data = yf.download('^GSPC', start='2025-01-23', end='2025-03-06')

data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))

std_dev = data['Log Returns'].std()

annualized_volatility = std_dev * np.sqrt(252)

print(f'Annualized Historical Volatility: {annualized_volatility:.2%}')
```

```Python
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import pandas as pd

# Black-Scholes formula for option pricing
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    else:
        raise ValueError('Invalid option type. Choose 'call' or 'put'.')

    return price

# Delta calculation
def delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return si.norm.cdf(d1)
    else:
        return si.norm.cdf(d1) - 1

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * si.norm.pdf(d1)

np.random.seed(42)
days = 5  
S0 = 5000 
K1, K2 = 4900, 5100 
T = 0.25 
r = 0.02  
sigma1, sigma2 = 0.2, 0.25 

returns = np.random.normal(r / 252, sigma1 / np.sqrt(252), days)
SPX_prices = S0 * np.cumprod(1 + returns)

option1_price = black_scholes(S0, K1, T, r, sigma1, 'call')
option2_price = black_scholes(S0, K2, T, r, sigma2, 'call')

delta1 = delta(S0, K1, T, r, sigma1, 'call')
delta2 = delta(S0, K2, T, r, sigma2, 'call')

vega1 = vega(S0, K1, T, r, sigma1)
vega2 = vega(S0, K2, T, r, sigma2)

beta = -vega1 / vega2

alpha = delta1 + beta * delta2

portfolio_values = []
for S in SPX_prices:
    V1 = black_scholes(S, K1, T, r, sigma1, 'call')
    V2 = black_scholes(S, K2, T, r, sigma2, 'call')
    portfolio_value = V1 + beta * V2 - alpha * S
    portfolio_values.append(portfolio_value)

plt.figure(figsize=(8, 6))
plt.plot(portfolio_values, label='Portfolio')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.title('Performance of Hedged Portfolio')
plt.legend()
plt.grid(True)
plt.show()

# Display summary
df_summary = pd.DataFrame({
    'Day': np.arange(1, days + 1),
    'SPX Price': SPX_prices,
    'Portfolio Value': portfolio_values
})
```