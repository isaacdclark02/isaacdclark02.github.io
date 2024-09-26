```python
#Importing Packages
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#Importing Datasets
df_games = pd.read_excel('23-24 MBB Games All Players.xlsx')
df_FP = pd.read_excel('23-24 MBB FP All Players.xlsx')

#Identifying Variables of Interest
strive_variables = ['minutes_played',
                    'Decelerations (Event)',
                    'Low Decelerations (<250 cm/s²) (Event)',
                    'Medium Decelerations (<350 cm/s² >=250 cm/s²) (Event)',
                    'High Decelerations (>=350 cm/s²) (Event)',
                    'Max Deceleration (cm/s²) (Event)',
                    'Max Deceleration (m/s²) (Event)']

fp_variables = ['Contraction Time [ms]',
                'Contraction Time:Eccentric Duration [%]',
                'Countermovement Depth [cm]',
                'Eccentric Acceleration Phase Duration [s]',
                'Eccentric Braking Impulse [N s]',
                'Eccentric Braking RFD [N/s]',
                'Eccentric Braking RFD / BM [N/s/kg]',
                'Eccentric Braking RFD-100ms [N/s]',
                'Eccentric Braking RFD-100ms / BM [N/s/kg]',
                'Eccentric Deceleration Impulse [N s]',
                'Eccentric Deceleration Phase Duration [s]',
                'Eccentric Deceleration RFD [N/s]',
                'Eccentric Deceleration RFD / BM [N/s/kg]',
                'Eccentric Duration [ms]',
                'Eccentric Mean Braking Force [N]',
                'Eccentric Mean Deceleration Force [N]',
                'Eccentric Mean Force [N]',
                'Eccentric Mean Power [W]',
                'Eccentric Mean Power / BM [W/kg]',
                'Eccentric Peak Force [N]',
                'Eccentric Peak Force / BM [N/kg]',
                'Eccentric Peak Power [W]',
                'Eccentric Peak Power / BM [W/kg]',
                'Eccentric Peak Power:Concentric Peak Power',
                'Eccentric Peak Velocity [m/s]',
                'Eccentric Unloading Impulse [N s]',
                'Eccentric:Concentric Duration [%]',
                'Eccentric:Concentric Mean Force Ratio [%]',
                'Jump Height (Flight Time) [cm]',
                'RSI-modified [m/s]',
                'Peak Power / BM [W/kg]']
```


```python
#Cleaning and Preping Data
games['Date'] = pd.to_datetime(games['Date']).dt.strftime('%Y-%m-%d')
FP['Date'] = pd.to_datetime(FP['Date']).dt.strftime('%Y-%m-%d')

minutes = pd.read_excel('min_played.xlsx')
minutes['Date'] = pd.to_datetime(minutes['Date']).dt.strftime('%Y-%m-%d')
df_games = pd.merge(df_games, minutes, on=['Athlete Name', 'Date'], how='left')
df_games['minutes_played'] = df_games['minutes_played'].fillna(0)

#Denoting which games were conference and nonconference
df_games['Conference/Nonconference'] = np.where(df_games['Date'] <= '2023-12-28', 'Nonconference', 'Conference')

df_FP = df_FP[df_FP['Date'] >= '2023-10-30']

max_height_index = df_FP.groupby(['Athlete Name', 'Date'])['Jump Height (Flight Time) [cm]'].idxmax()
df_FP = df_FP.loc[max_height_index]
```


```python
#Averaging and Merging Data Per Month
#Function to get averages
def avg_month(data, variables):
    data['Date'] = pd.to_datetime(data['Date']).dt.to_period('M')
    avg = data.groupby(['Athlete Name', 'Date'])[variables].mean().reset_index()
    
    return avg

#Getting averages
games_month = avg_month(df_games, strive_variables)
FP_month = avg_month(df_FP, fp_variables)

#Merging FP data with accelerometer data
merged_week = pd.merge(games_month, FP_month, on=['Athlete Name', 'Date'])

#Creating columns for decels per min
merged_week['High Decel Per Min'] = (merged_week['High Decelerations (>=350 cm/s²) (Event)'])/(merged_week['minutes_played'])
merged_week['Med Decel Per Min'] = (merged_week['Medium Decelerations (<350 cm/s² >=250 cm/s²) (Event)'])/(merged_week['minutes_played'])
merged_week['Low Decel Per Min'] = (merged_week['Low Decelerations (<250 cm/s²) (Event)'])/(merged_week['minutes_played'])
```


```python
#Averaging and Merging Data Per Week
def avg_week(data, variables):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Week'] = data['Date'] - pd.to_timedelta(data['Date'].dt.dayofweek, unit='D')
    avg = data.groupby(['Athlete Name', 'Week'])[variables].mean().reset_index()
    return avg

#Getting averages
games_week = avg_week(df_games, strive_variables)
FP_week = avg_week(df_FP, fp_variables)

#Merging FP data with accelerometer data
merged_week = pd.merge(games_week, FP_week, on=['Athlete Name', 'Week'])

#Creating columns for decels per min
merged_week['High Decel Per Min'] = (merged_week['High Decelerations (>=350 cm/s²) (Event)'])/(merged_week['minutes_played'])
merged_week['Med Decel Per Min'] = (merged_week['Medium Decelerations (<350 cm/s² >=250 cm/s²) (Event)'])/(merged_week['minutes_played'])
merged_week['Low Decel Per Min'] = (merged_week['Low Decelerations (<250 cm/s²) (Event)'])/(merged_week['minutes_played'])
```


```python
#Grouping by Playtime and Avg Variabels Per Week
df_games['Date'] = pd.to_datetime(df_games['Date'])
df_games['Week'] = df_games['Date'] - pd.to_timedelta(df_games['Date'].dt.dayofweek, unit='D')

df_FP['Date'] = pd.to_datetime(df_FP['Date'])
df_FP['Week'] = df_FP['Date'] - pd.to_timedelta(df_FP['Date'].dt.dayofweek, unit='D')

merged_groups = pd.merge(df_games, df_FP, on=['Athlete Name', 'Week'])

def categorize_playtime(minutes_played):
    if minutes_played <= 10:
        return 'low'
    else:
        return 'high'

merged_groups['Group'] = merged_groups['minutes_played'].apply(categorize_playtime)

avg_week = merged_groups.groupby(['Week', 'Group'])[strive_variables + fp_variables].mean().reset_index()

#Creating columns for decels per min
avg_week['High Decel Per Min'] = (avg_week['High Decelerations (>=350 cm/s²) (Event)'])/(avg_week['minutes_played'])
avg_week['Med Decel Per Min'] = (avg_week['Medium Decelerations (<350 cm/s² >=250 cm/s²) (Event)'])/(avg_week['minutes_played'])
avg_week['Low Decel Per Min'] = (avg_week['Low Decelerations (<250 cm/s²) (Event)'])/(avg_week['minutes_played'])
```


```python
#Dropping Lovering due to injury
merged_week = merged_week[(~(merged_week['Athlete Name'] == 'Lawson Lovering'))]

#Graphing Data as a Lineplot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
fig = sns.lineplot(merged_week, x='Week', y='Max Deceleration (m/s²) (Event)', hue='Athlete Name')
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.title('Max Deceleration per Month')
plt.xlabel('Month')
plt.xticks(rotation=-30)
plt.show()
```


```python
#Plotting Athletes over time
import seaborn as sns
import matplotlib.pyplot as plt

def lineplot(athlete):
    data = merged_week[merged_week['Athlete Name'] == athlete]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data, x='Week', y='Eccentric Deceleration RFD [N/s]', label='Eccentric Deceleration RFD', ci=None, ax=ax1)

    ax2 = plt.twinx()
    sns.lineplot(data, x='Week', y='High Decel Per Min', color='red', label='High Decel Per Min', ci=None, ax=ax2)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.4, 1))
    ax1.get_legend().remove()

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=-30)

    plt.title(f'{athlete}')
    plt.show()

athlete_list = ['list of athlete names']

for athlete in athlete_list:
    lineplot(athlete)
```


```python
#Plotting Groups over time
import seaborn as sns
import matplotlib.pyplot as plt

def lineplot(group):
    data = avg_week[avg_week['Group'] == group]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data, x='Week', y='Eccentric Deceleration RFD / BM [N/s/kg]', label='Eccentric Deceleration RFD', ci=None, ax=ax1)

    ax2 = plt.twinx()
    sns.lineplot(data, x='Week', y='High Decel Per Min', color='red', label='High Decel Per Min', ci=None, ax=ax2)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.4, 1))
    ax1.get_legend().remove()

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=-30)
    
    plt.title(f'{group} playing time')
    plt.show()

for group in ['low', 'high']:
    lineplot(group)
```


```python
#Calculating the Correlation Tables for Each Dataset
practice_numeric = practice_merged.select_dtypes(include=['number']).columns
practice_corr = practice_merged.groupby('Athlete Name')[practice_numeric].corr()

games_numeric = games_merged.select_dtypes(include=['number']).columns
games_corr = games_merged.groupby('Athlete Name')[games_numeric].corr()
```


```python
#Visualize Eccentric Variabel Correlation
import seaborn as sns
import matplotlib.pyplot as plt

practice_merged = practice_merged[(~(practice_merged['Athlete Name'] == 'athlete name'))]
games_merged = games_merged[(~(games_merged['Athlete Name'] == 'athlete name'))]

fig = sns.lmplot(games_merged,
                 x='Max Deceleration (m/s²) (Event)',
                 y='Peak Power / BM [W/kg]',
                 hue='Athlete Name')
sns.set_style('white')
plt.title('Athlete Trendlines')
```
