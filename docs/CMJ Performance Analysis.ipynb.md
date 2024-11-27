```Python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')

sports = df[df['Sport'].isin(
    ['list of sports']
)]

cmj = sports[sports['testType'] == 'CMJ'].loc[:, :'Landing RFD 50ms [N/s]']
```

```Python
sports = ['list of sports']

metrics = [
    'Concentric Peak Force / BM [N/kg]',
    'Eccentric Peak Force / BM [N/kg]',
    'Peak Power / BM [W/kg]',
    'RSI-modified [m/s]'
]

sport_percentile_dfs = {}

for sport in sports:
    sport_df = cmj[cmj['Sport'] == sport]

    sport_metrics = {}
    for metric in metrics:
        metric_df = pd.to_numeric(sport_df[metric], errors='coerce')

        Q1 = metric_df.quantile(0.25)
        Q3 = metric_df.quantile(0.75)
        IQR = Q3 - Q1
        filtered = metric_df[~((metric_df < (Q1 - 1.5 * IQR)) | (metric_df > (Q3 + 1.5 * IQR)))]

        mean = filtered.mean()
        p_10 = filtered.quantile(.10)
        p_20 = filtered.quantile(.20)
        p_30 = filtered.quantile(.30)
        p_40 = filtered.quantile(.40)
        p_50 = filtered.quantile(.50)
        p_60 = filtered.quantile(.60)
        p_70 = filtered.quantile(.70)
        p_80 = filtered.quantile(.80)
        p_90 = filtered.quantile(.90)
        p_100 = filtered.quantile(1)

        sport_metrics[metric] = {
            'Mean': f'{mean:.2f}',
            '10th Percentile': f'{p_10:.2f}',
            '20th Percentile': f'{p_20:.2f}',
            '30th Percentile': f'{p_30:.2f}',
            '40th Percentile': f'{p_40:.2f}',
            '50th Percentile (Median)': f'{p_50:.2f}',
            '60th Percentile': f'{p_60:.2f}',
            '70th Percentile': f'{p_70:.2f}',
            '80th Percentile': f'{p_80:.2f}',
            '90th Percentile': f'{p_90:.2f}',
            '100th Percentile': f'{p_100:.2f}'
        }

    sport_percentile_df = pd.DataFrame(sport_metrics)
    sport_percentile_df = sport_percentile_df.apply(pd.to_numeric, errors='ignore')

    sport_percentile_dfs[sport] = sport_percentile_df.T.reset_index().rename(columns={'index': 'Metric'})
```

```Python
def scaled(sport=None):
    if sport:
        data = cmj[cmj['Sport'] == sport].loc[:, 'Start of Movement [s]':]
        sport = sport
    else:
        data = cmj.loc[:, 'Start of Movement [s]':]
        sport = 'All Sports'

    data = data.fillna(data.median())
    pattern = '|'.join(['Landing', 'Asymmetry'])

    data = data.loc[:, ~data.columns.str.contains(pattern, case=False, na=False)]

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1

    filtered = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    scaler = StandardScaler()
    data_transformed = scaler.fit_transform(filtered)

    return pd.DataFrame(data_transformed, columns=data.columns)

data = scaled()

pca = PCA()
pca.fit(data)

eigenvalues = pca.explained_variance_

n_factors = sum(eigenvalues > 1)
print('Number of factors:', n_factors)
```

```Python
fa = FactorAnalysis(n_components=n_factors)

fa.fit(data)

loadings = fa.components_.T

loading_df = pd.DataFrame(loadings, index=data.columns)
```

```Python
factors = loading_df.columns

factor_dict = {}

threshhold = 0.7
for factor in factors:
    condition = (loading_df[factor] >= threshhold) | (loading_df[factor] <= -threshhold)
    matching_indices = loading_df[condition].index.tolist()
    if matching_indices:
        metric_with_loading = [f'{metric_name} ({loading_df.loc[metric_name, factor]:.2f})' for metric_name in matching_indices]
        factor_name = f'Factor {factor}'
        factor_dict[factor_name] = metric_with_loading

factor_df = pd.DataFrame({factor: pd.Series(indices) for factor, indices in factor_dict.items()})
```

```Python
factor1 = [
    'factor1 metrics'
]

factor5 = [
    'factor5 metrics'
]

df_1 = data[factor1]
df_5 = data[factor5]

corr_1 = df_1.corr()
corr_5 = df_5.corr()
```

```Python
def metric_annual_mean(metric):
    catdata = cmj[['Athlete', 'Sport', 'date']]
    filtered = cmj.loc[:, 'Start of Movement [s]':]

    Q1 = filtered.quantile(0.25)
    Q3 = filtered.quantile(0.75)

    IQR = Q3 - Q1

    cleaned = filtered[~((filtered < (Q1 - 1.5 * IQR)) | (filtered > (Q3 + 1.5 * IQR))).any(axis=1)]
    cmj_cleaned = pd.concat([catdata.loc[cleaned.index], cleaned], axis=1)

    cmj_cleaned['date'] = pd.to_datetime(cmj_cleaned['date'])
    cmj_cleaned['year'] = cmj_cleaned['date'].dt.year

    unique_years = cmj_cleaned.drop_duplicates(subset=['Athlete' ,'year'])
    unique_years.sort_values(by=['Athlete', 'year'])
    unique_years['Year'] = unique_years.groupby('Athlete').cumcount() + 1

    year_index = cmj_cleaned.merge(unique_years[['Athlete', 'year', 'Year']], on=['Athlete', 'year'], how='left').sort_values(by=['Athlete', 'Year'])
    year_index = year_index[year_index['Year'] <= 4]

    athletes_with_4_years = year_index[year_index['Year'] == 4]['Athlete'].unique()
    
    athletes_with_4_years = year_index[year_index['Athlete'].isin(athletes_with_4_years)]
    annual_mean = athletes_with_4_years.groupby(['Athlete', 'Year']).median(numeric_only=True).reset_index().groupby('Year').mean(numeric_only=True).reset_index()

    year_1_value = annual_mean.loc[annual_mean['Year'] == 1, metric].values[0]
    annual_mean['Percent_Change'] = (annual_mean[metric] - year_1_value) / year_1_value * 100

    return annual_mean

jump = metric_annual_mean('Jump Height (Flight Time) in Inches [in]')
rsi = metric_annual_mean('RSI-modified [m/s]')
eccen_power = metric_annual_mean('Eccentric Mean Power / BM [W/kg]')
concen_power = metric_annual_mean('Concentric Mean Power / BM [W/kg]')
eccen_force = metric_annual_mean('Eccentric Peak Force / BM [N/kg]')
concen_force = metric_annual_mean('Concentric Peak Force / BM [N/kg]')
impulse = metric_annual_mean('Positive Takeoff Impulse [N s]')

plt.figure(figsize=(10, 6))
plt.plot(jump['Year'], jump['Percent_Change'], label='Jump Height')
plt.plot(rsi['Year'], rsi['Percent_Change'], label='RSI-mod')
plt.plot(eccen_power['Year'], eccen_power['Percent_Change'], label='Eccentric Mean Power')
plt.plot(concen_power['Year'], concen_power['Percent_Change'], label='Concentric Mean Power')
plt.plot(eccen_force['Year'], eccen_force['Percent_Change'], label='Eccentric Peak Force')
plt.plot(concen_force['Year'], concen_force['Percent_Change'], label='Concentric Peak Force')
plt.plot(impulse['Year'], impulse['Percent_Change'], label='Positive Takeoff Impulse')
plt.title('Percent Change from Year 1')
plt.xlabel('Year')
plt.ylabel('Percent Change')
plt.grid(True)
plt.legend()

con_time = metric_annual_mean('Contraction Time [ms]')
St_Pk_Fc = metric_annual_mean('Movement Start to Peak Force [s]')
St_Pk_Pw = metric_annual_mean('Movement Start to Peak Power [s]')
Eccen_dur = metric_annual_mean('Eccentric Duration [ms]')
Flt_time = metric_annual_mean('Flight Time [ms]')


plt.figure(figsize=(10, 6))
plt.plot(con_time['Year'], con_time['Percent_Change'], label='Contraction Time')
plt.plot(St_Pk_Fc['Year'], St_Pk_Fc['Percent_Change'], label='Start to Peak Force')
plt.plot(St_Pk_Pw['Year'], St_Pk_Pw['Percent_Change'], label='Start to Peak Power')
plt.plot(Eccen_dur['Year'], Eccen_dur['Percent_Change'], label='Eccentric Duration')
plt.plot(Flt_time['Year'], Flt_time['Percent_Change'], label='Flight Time')
plt.title('Percent Change from Year 1')
plt.xlabel('Year')
plt.ylabel('Percent Change')
plt.grid(True)
plt.legend()
```

```Python
def metric_annual_mean_sport(metric, sport):
    catdata = cmj[['Athlete', 'Sport', 'date']]
    filtered = cmj.loc[:, 'Start of Movement [s]':]

    Q1 = filtered.quantile(0.25)
    Q3 = filtered.quantile(0.75)

    IQR = Q3 - Q1

    cleaned = filtered[~((filtered < (Q1 - 1.5 * IQR)) | (filtered > (Q3 + 1.5 * IQR))).any(axis=1)]
    cmj_cleaned = pd.concat([catdata.loc[cleaned.index], cleaned], axis=1)

    cmj_cleaned['date'] = pd.to_datetime(cmj_cleaned['date'])
    cmj_cleaned['year'] = cmj_cleaned['date'].dt.year

    unique_years = cmj_cleaned.drop_duplicates(subset=['Athlete' ,'year'])
    unique_years.sort_values(by=['Athlete', 'year'])
    unique_years['Year'] = unique_years.groupby('Athlete').cumcount() + 1

    year_index = cmj_cleaned.merge(unique_years[['Athlete', 'year', 'Year']], on=['Athlete', 'year'], how='left').sort_values(by=['Athlete', 'Year'])
    year_index = year_index[year_index['Year'] <= 4]

    athletes_with_4_years = year_index[year_index['Year'] == 4]['Athlete'].unique()
    
    athletes_with_4_years = year_index[year_index['Athlete'].isin(athletes_with_4_years)]
    athletes_with_4_years = athletes_with_4_years[athletes_with_4_years['Sport'] == sport]

    annual_mean = athletes_with_4_years.groupby(['Athlete', 'Year']).median(numeric_only=True).reset_index().groupby('Year').mean(numeric_only=True).reset_index()

    year_1_value = annual_mean.loc[annual_mean['Year'] == 1, metric].values[0]
    annual_mean['Percent_Change'] = (annual_mean[metric] - year_1_value) / year_1_value * 100

    return annual_mean

def graph_function(sport):
    jump = metric_annual_mean_sport('Jump Height (Flight Time) in Inches [in]', sport)
    rsi = metric_annual_mean_sport('RSI-modified [m/s]', sport)
    eccen_power = metric_annual_mean_sport('Eccentric Mean Power / BM [W/kg]', sport)
    concen_power = metric_annual_mean_sport('Concentric Mean Power / BM [W/kg]', sport)
    eccen_force = metric_annual_mean_sport('Eccentric Peak Force / BM [N/kg]', sport)
    concen_force = metric_annual_mean_sport('Concentric Peak Force / BM [N/kg]', sport)
    impulse = metric_annual_mean_sport('Positive Takeoff Impulse [N s]', sport)

    plt.figure(figsize=(10, 6))
    plt.plot(jump['Year'], jump['Percent_Change'], label='Jump Height')
    plt.plot(rsi['Year'], rsi['Percent_Change'], label='RSI-mod')
    plt.plot(eccen_power['Year'], eccen_power['Percent_Change'], label='Eccentric Mean Power')
    plt.plot(concen_power['Year'], concen_power['Percent_Change'], label='Concentric Mean Power')
    plt.plot(eccen_force['Year'], eccen_force['Percent_Change'], label='Eccentric Peak Force')
    plt.plot(concen_force['Year'], concen_force['Percent_Change'], label='Concentric Peak Force')
    plt.plot(impulse['Year'], impulse['Percent_Change'], label='Positive Takeoff Impulse')
    plt.title(f'Percent Change from Year 1 for {sport}')
    plt.xlabel('Year')
    plt.ylabel('Percent Change')
    plt.grid(True)
    plt.legend()
```

```Python
def linear_regression(metric, year):
    catdata = cmj[['Athlete', 'Sport', 'date']]
    filtered = cmj.loc[:, 'Start of Movement [s]':]

    Q1 = filtered.quantile(0.25)
    Q3 = filtered.quantile(0.75)

    IQR = Q3 - Q1

    cleaned = filtered[~((filtered < (Q1 - 1.5 * IQR)) | (filtered > (Q3 + 1.5 * IQR))).any(axis=1)]
    cmj_cleaned = pd.concat([catdata.loc[cleaned.index], cleaned], axis=1)

    cmj_cleaned['date'] = pd.to_datetime(cmj_cleaned['date'])
    cmj_cleaned['year'] = cmj_cleaned['date'].dt.year

    unique_years = cmj_cleaned.drop_duplicates(subset=['Athlete' ,'year'])
    unique_years.sort_values(by=['Athlete', 'year'])
    unique_years['Year'] = unique_years.groupby('Athlete').cumcount() + 1

    year_index = cmj_cleaned.merge(unique_years[['Athlete', 'year', 'Year']], on=['Athlete', 'year'], how='left').sort_values(by=['Athlete', 'Year'])
    year_index = year_index[year_index['Year'] <= 4]

    athletes_with_4_years = year_index[year_index['Year'] == 4]['Athlete'].unique()
    
    data = year_index[year_index['Athlete'].isin(athletes_with_4_years)]

    data = data.sort_values(['Athlete', 'date'])

    first_test = data.groupby('Athlete')[metric].first().reset_index()
    first_test.columns = ['Athlete'] + [f'{name}_first' for name in first_test.columns[1:]]

    yearly_performance = data.groupby(['Athlete', 'Year']).median(numeric_only=True).reset_index()

    performance_pivot = yearly_performance.pivot(
        index='Athlete', columns='Year', values=metric
    ).reset_index()

    performance_pivot.columns = ['Athlete'] + [f'Performance_{year}' for year in performance_pivot.columns[1:]]

    data = pd.merge(
        first_test,
        performance_pivot,
        on='Athlete',
        how='inner'
    ).drop(columns=['Athlete'])

    slope, intercept, r_value, p_value, std_err = linregress(data[f'{metric}_first'], data[f'Performance_{year}'])

    return r_value**2, slope

metrics = ['Jump Height (Flight Time) in Inches [in]', 
           'RSI-modified [m/s]', 
           'Eccentric Mean Power / BM [W/kg]', 
           'Concentric Mean Power / BM [W/kg]',
           'Eccentric Peak Force / BM [N/kg]',
           'Concentric Peak Force / BM [N/kg]',
           'Positive Takeoff Impulse [N s]']

reg_data = []
for metric in metrics:
    for year in range(1, 5):
        r_squared, slope = linear_regression(metric, year)
        reg_data.append({
            'Metric': metric,
            'Year': year,
            'R_squared': r_squared
        })

regression_results = pd.DataFrame(reg_data)
regression_pivot = regression_results.pivot(index='Year', columns='Metric', values='R_squared')

regression_pivot
```