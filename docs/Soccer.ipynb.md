```Python
import pandas as pd
import os
import glob
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

csv_files = glob.glob(os.path.join(
    '/Users/isaacdclark02/Library/CloudStorage/OneDrive-UniversityofUtah/Internship/Soccer Project/Catapult Data', '*.csv'
))

catapult = []

for file in csv_files:
    df = pd.read_csv(file, skiprows=9)
    catapult.append(df)

catapult = pd.concat(catapult, ignore_index=True)

df = catapult[~catapult['Player Name'].isin(['Athletes to remove'])]

df = df[['Player Name', 'Date', 'Total Player Load']]
```

```Python
plt.figure()
sns.histplot(df['Total Player Load'], bins=30, color='r')
plt.title('Total Player Load')
plt.show()

skewness = stats.skew(df['Total Player Load'])

plt.show()
print(f'Skewness={skewness}')
```

```Python
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
players = df['Player Name'].unique()

results = []
for player in players:
    player_data = df[df['Player Name'] == player]

    most_recent_date = df['Date'].max()
    cutoff_date = most_recent_date - pd.DateOffset(months=12)

    sample_data = player_data[player_data['Date'] > cutoff_date]

    pop_mean = df['Total Player Load'].mean()
    pop_std = df['Total Player Load'].std()
    pop_size = df['Total Player Load'].count()

    sample_mean = sample_data['Total Player Load'].mean()
    sample_std = sample_data['Total Player Load'].std()
    sample_size = sample_data['Total Player Load'].count()

    percent = (sample_mean - pop_mean) / pop_mean

    results.append({
        'Player': player,
        'Percent Difference': percent,
        'Sample Mean': sample_mean,
        'Population Mean': "{:.6f}".format(pop_mean),
        'Sample Size': sample_size,
        'Population Size': pop_size
    })

df_results = pd.DataFrame(results).dropna()
df_results = df_results[df_results['Sample Size'] > 100]
df_results
```

```Python
plt.figure(figsize=(10,6))
ax = sns.barplot(df_results, x='Percent Difference', y='Player', color='r', zorder=2)
plt.title('Average Athlete Total Player Load Compared to Team Average')
plt.grid(True, linestyle='--', zorder=1)
for p in ax.patches:
    ax.text(p.get_width(),
            p.get_y() + p.get_height() / 2,
            '{:.2f}'.format(p.get_width()),
            ha='left', va='center',
            fontsize=10)

plt.show()
```