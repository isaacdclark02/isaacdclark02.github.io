```Python
import pandas as pd
import os
import glob
import warnings

warnings.filterwarnings('ignore')

csv_files = glob.glob(os.path.join(
    'Folder', '*.csv'
))

catapult = []

for file in csv_files:
    df = pd.read_csv(file, skiprows=9)
    catapult.append(df)

catapult = pd.concat(catapult, ignore_index=True)

df = catapult[~catapult['Player Name'].isin(['Players to remove from analysis'])]

df = df[['Player Name', 'Date', 'Total Player Load', 'Player Load Per Minute']]
```

```Python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

mu, std = stats.norm.fit(df['Total Player Load'])

plt.hist(df['Total Player Load'], bins=30, density=True, alpha=0.6, color='r')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

plt.title('Total Player Load')

scipy_skewness = stats.skew(df['Total Player Load'])

scipy_skewness
```

```Python
descriptive_stats = df.describe()

descriptive_stats
```

```Python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

most_recent_date = df['Date'].max()
time = most_recent_date - pd.DateOffset(months=12)
data = df[df['Date'] >= time]

athletes = data['Player Name'].unique()

means = []
for athlete in athletes:
    athlete_data = data[data['Player Name'] == athlete]
    if not athlete_data.empty:
        mean = athlete_data['Total Player Load'].mean()
        means.append((athlete, mean))

means = pd.DataFrame(means, columns=['Player Name', 'Player Load Average'])

means['Avg Ratio'] = (means['Player Load Average'] / 188.979266) - 1

means = means[~means['Player Name'].isin(['Outlier player'])]

plt.figure(figsize=(10, 6))
ax = sns.barplot(means, x='Avg Ratio', y='Player Name', color='r', zorder=3)

for index, value in enumerate(means['Avg Ratio']):
    plt.text(value, index, f'{value:.2f}', va='center', zorder=4)

ax.grid(True, axis='x', linestyle='--', alpha=0.7, zorder=1)
ax.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=1)

plt.title('Total Player Load Athlete Average Comparison', fontsize=14)
plt.xlabel('Total Player Load Percentage of Team Average', fontsize=12)
plt.ylabel('Player Name', fontsize=12)

plt.tight_layout()
plt.show()
```