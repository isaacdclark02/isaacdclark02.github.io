```python
# Importing Pandas
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Loading in dataset
df = pd.read_excel('FB Master Database.xlsx')

# Ensuring proper date formate
df['Date'] = pd.to_datetime(df['Date'])

# Creating temporary column containing date year
df['temp'] = df['Date'].dt.year

# Calculating number of years in program for each athlete
df_unique_years = df.drop_duplicates(subset=['Last, First', 'temp'])
df_unique_years = df_unique_years.sort_values(by=['Last, First', 'temp'])
df_unique_years['Year'] = df_unique_years.groupby('Last, First').cumcount() + 1

data = df.merge(df_unique_years[['Last, First', 'temp', 'Year']], on=['Last, First', 'temp'], how='left')

# Identifying and dropping athletes who went on missions
data = data.sort_values(by=['Last, First', 'Date'])
data['date_diff'] = data.groupby('Last, First')['Date'].diff().dt.days
data['left_and_returned'] = data['date_diff'] > 365

data = data[data['left_and_returned'] != True]

# Only keeping rows with dates after 2011-01-01
data = data[data['Date'] >= '2011-01-01']

# Sorting and renaming columns for readablility
data = data.filter(['Last, First', 'Date', 'Year', 'Position', 'Group', 'Age', 'Height(in)', 'Weight(lbs)', 
                    'Squat_1RM(lbs)', 'VJ(in)', 'Broad Jump', 'L-Drill', 'PRO'])

data = data.dropna(how='all', subset=['Squat_1RM(lbs)', 'VJ(in)', 'Broad Jump', 'L-Drill', 'PRO'])
data = data.rename(columns={
    'Last, First': 'Athlete',
    'Height(in)': 'Height_in',
    'Weight(lbs)': 'Weight_lbs',
    'Squat_1RM(lbs)': 'Squat_1RM_lbs',
    'VJ(in)': 'VJ_in',
    'Broad Jump': 'BJ_in',
    'L-Drill': 'L_Drill'
})

# Print results
data.head()
```


```python
# Importing Pandas
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Loading in dataset
df = pd.read_csv('FB Database.csv')

df.head()
```


```python
# Importing graphing packages
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up squat and l-drill data sets
squat = df[['Athlete', 'Year', 'Position', 'Group', 'Squat_1RM_lbs']]
squat = squat.dropna(subset=['Squat_1RM_lbs'])

l_drill = df[['Athlete', 'Year', 'Position', 'Group', 'L_Drill']]
l_drill = l_drill.dropna(subset=['L_Drill'])

pro = df[['Athlete', 'Year', 'Position', 'Group', 'PRO']]
pro = pro.dropna(subset=['PRO'])

v_jump = df[['Athlete', 'Year', 'Position', 'Group', 'VJ_in']]
v_jump = v_jump.dropna(subset=['VJ_in'])

b_jump = df[['Athlete', 'Year', 'Position', 'Group', 'BJ_in']]
b_jump = b_jump.dropna(subset=['BJ_in'])

# Filtering for just athletes in the 'Bigs' group
squat_B = squat[squat['Group'] == 'B']
l_drill_B = l_drill[l_drill['Group'] == 'B']
pro_B = pro[pro['Group'] == 'B']
v_jump_B = v_jump[v_jump['Group'] == 'B']
b_jump_B = b_jump[b_jump['Group'] == 'B']

# Forcing data to numerical values
squat_B['Squat_1RM_lbs'] = pd.to_numeric(squat_B['Squat_1RM_lbs'])
l_drill_B['L_Drill'] = pd.to_numeric(l_drill_B['L_Drill'])
pro_B['PRO'] = pd.to_numeric(pro_B['PRO'])
v_jump_B['VJ_in'] = pd.to_numeric(v_jump_B['VJ_in'])
b_jump_B['BJ_in'] = pd.to_numeric(b_jump_B['BJ_in'])

# Averaging values for each year in the program
avg_squat_B = squat_B.groupby('Year').mean(numeric_only=True)
avg_l_drill_B = l_drill_B.groupby('Year').mean(numeric_only=True)
avg_pro_B = pro_B.groupby('Year').mean(numeric_only=True)
avg_v_jump_B = v_jump_B.groupby('Year').mean(numeric_only=True)
avg_b_jump_B = b_jump_B.groupby('Year').mean(numeric_only=True)

# Droping data in years beyond year 4 in the program
avg_squat_B = avg_squat_B.iloc[:4]
avg_l_drill_B = avg_l_drill_B.iloc[:4]
avg_pro_B = avg_pro_B.iloc[:4]
avg_v_jump_B = avg_v_jump_B.iloc[:4]
avg_b_jump_B = avg_b_jump_B.iloc[:4]

# Ploting L-Drill data
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.regplot(avg_squat_B, x=avg_squat_B.index, y='Squat_1RM_lbs', label='Squat 1RM (lbs)', ci=None, ax=ax1)

ax2 = ax1.twinx()
sns.regplot(avg_l_drill_B, x=avg_l_drill_B.index, y='L_Drill', label='L-Drill (s)', color='red', ci=None, ax=ax2)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.29, 1))

plt.title('Bigs Average Squat v. L-Drill for Each Year in Program')

# Ploting PRO data
fig2, ax1 = plt.subplots(figsize=(10, 6))
sns.regplot(avg_squat_B, x=avg_squat_B.index, y='Squat_1RM_lbs', label='Squat 1RM (lbs)', ci=None, ax=ax1)

ax2 = ax1.twinx()
sns.regplot(avg_pro_B, x=avg_pro_B.index, y='PRO', label='PRO', color='red', ci=None, ax=ax2)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.29, 1))

plt.title('Bigs Average Squat v. PRO for Each Year in Program')

# Ploting Vertical Jump data
fig3, ax1 = plt.subplots(figsize=(10, 6))
sns.regplot(avg_squat_B, x=avg_squat_B.index, y='Squat_1RM_lbs', label='Squat 1RM (lbs)', ci=None, ax=ax1)

ax2 = ax1.twinx()
sns.regplot(avg_v_jump_B, x=avg_v_jump_B.index, y='VJ_in', label='Vertial Jump (in)', color='red', ci=None, ax=ax2)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.29, 1))

plt.title('Bigs Average Squat v. Vertical Jump for Each Year in Program')

# Ploting Broad Jump data
fig4, ax1 = plt.subplots(figsize=(10, 6))
sns.regplot(avg_squat_B, x=avg_squat_B.index, y='Squat_1RM_lbs', label='Squat 1RM (lbs)', ci=None, ax=ax1)

ax2 = ax1.twinx()
sns.regplot(avg_b_jump_B, x=avg_b_jump_B.index, y='BJ_in', label='Broad Jump (in)', color='red', ci=None, ax=ax2)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.29, 1))

plt.title('Bigs Average Squat v. Broad Jump for Each Year in Program')

plt.show()
```


```python
import statsmodels.api as sm

# Independent variable
X = avg_l_drill_B['L_Drill']
X = sm.add_constant(X) # Adding constant term for intercept

# Dependent variable
Y = avg_squat_B['Squat_1RM_lbs']

# Define the model
model = sm.OLS(Y, X)

# Fit the model
results = model.fit()

results.summary()
```
