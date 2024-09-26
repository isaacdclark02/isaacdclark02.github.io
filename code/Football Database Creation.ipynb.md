```python
#Football Body Weight Data Extraction and Sorting
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

#Function for parcing bodyweight data from excel
def excel_scraping(file_name):
    #Loading the data
    df = pd.read_excel(file_name, sheet_name='BW Annual Plan - TRAVEL')

    #Indexing cells to extract
    data = df.iloc[21:867, 11:74]
    data = pd.DataFrame(data)
    data.columns = range(1, 64)
    data = data.drop(columns=range(3, 12))

    #Set up for position column
    headers = ['CB', 'S', 'WR', 'QB', 'LB', 'TE', 'RB', 'DE', 'DT', 'OL', 'SP']
    processed_data = []
    current_header = None

    #Loop to create position column
    for index, row in data.iterrows():
        value = row[1]

        if value in headers:
            current_header = value
        else:
            data.at[index, 'Header'] = current_header

    #Moving and renaming position column
    cols = data.columns.tolist()
    cols.insert(1, cols.pop(cols.index('Header')))
    data = data[cols]
    data.columns = range(1, 56)

    #Shifting body weight data up one row
    data.iloc[:, 3:55] = data.iloc[:, 3:55].shift(-1)

    #Droping empty rows
    data = data.dropna(subset=[1])
    data = data[data[1].str.contains(',', na=False)]

    #Renaming columns
    week_dates = df.iloc[12, 22:74].values.tolist()
    week_dates = [date for date in week_dates if pd.notna(date)]
    week_dates = [pd.to_datetime(date).strftime("%m/%d/%Y") for date in week_dates]

    data.columns = ['Athlete', 'Position', 'Height'] + week_dates

    #Identifying variables
    id_vars = ['Athlete', 'Position', 'Height']
    value_vars = week_dates

    #Melting columns
    melted_df = pd.melt(data, id_vars=id_vars, value_vars=value_vars, var_name='Week', value_name='Body Weight')
    
    #Sorting rows
    sorted_df = melted_df.sort_values(by=['Athlete', 'Week'])

    return sorted_df
```


```python
import pandas as pd
import sqlite3
import warnings

warnings.filterwarnings('ignore')

# Connecting to sqlite3 database
try:
    conn = sqlite3.connect('FB_Database.db')
    cursor = conn.cursor()
    print('Connected')

except sqlite3.Error as e:
    conn.rollback()
    print(f"Error: {e}")

#Loading in data
db = pd.read_csv('FB Master Database.csv')

df = pd.read_csv('FB bodyweight.csv')
df['Body Weight'] = pd.to_numeric(df['Body Weight'], errors='coerce')
```


```python
import pandas as pd

column_mapping = {
    'Athlete': 'Last, First',
    'Position': 'Position',
    'Height': 'Height(in)',
    'Week': 'Date',
    'Body Weight' : 'Weight(lbs)'
}

df_aligned = pd.DataFrame(columns=db.columns)

for df_data, db_data in column_mapping.items():
    df_aligned[db_data] = df[df_data]

data = pd.concat([db, df_aligned], ignore_index=True)
data['Last, First'] = data['Last, First'].str.strip()
data = data.dropna(subset=['Last, First'])
athletes = data.drop_duplicates(subset=['Last, First'])

data.tail()
```


```python
try:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Athletes (
            Athlete_id INTEGER PRIMARY KEY AUTOINCREMENT,
            Athlete TEXT,
            Position TEXT,
            `Group` TEXT,
            Scholarship TEXT,
            Status TEXT,
            DOB TEXT,
            Age TEXT,
            Sex TEXT,
            Height_in TEXT,
            Wing_in TEXT,
            Standing_Reach_in TEXT,
            Hand_Span_in TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Athlete_id INTEGER,
            Date TEXT,
            Weight_lbs TEXT,
            BF_Percent TEXT,
            FFM_lbs TEXT,
            FM_lbs TEXT,
            BMC TEXT,
            A_G TEXT,
            T_Score TEXT,
            Bench_lbs TEXT,
            Bench_reps TEXT,
            Bench_1RM TEXT,
            `225_reps` TEXT,
            Squat_lbs TEXT,
            Squat_reps TEXT,
            Squat_1RM TEXT,
            MBDS_lbs TEXT,
            MBDS_reps TEXT,
            MBDS_1RM TEXT,
            HC_lbs TEXT,
            HC_reps TEXT,
            HC_1RM TEXT,
            RA_FM_lbs TEXT,
            RA_LM_lbs TEXT,
            LA_FM_lbs TEXT,
            LA_LM_lbs TEXT,
            RL_FM_lbs TEXT,
            RL_LM_lbs TEXT,
            LL_FM_lbs TEXT,
            LL_LM_lbs TEXT,
            SAFBar_lbs TEXT,
            SAFBar_reps TEXT,
            SAFBar_1RM TEXT,
            VJ_Reach_in TEXT,
            VJ_Height_in TEXT,
            VJ_in TEXT,
            Broad_Jump_in TEXT,
            PRO TEXT,
            L_Drill TEXT,
            `40_YD` TEXT,
            Mile TEXT,
            FOREIGN KEY (Athlete_id) REFERENCES Athletes(Athlete_id)
        )
    """)
    print('Success')

except sqlite3.Error as e:
    conn.rollback()
    print(f"Error: {e}")
```


```python
# Inserting athlete names into Athletes tables
try:
    for index, row in athletes.iterrows():
        cursor.execute("""
            INSERT INTO Athletes (Athlete, Position, `Group`, Scholarship, Status, DOB, Age, Sex,
                Height_in, Wing_in, Standing_Reach_in, Hand_Span_in)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (row['Last, First'], row['Position'], row['Group'], row['Scholarship'], row['Status'], row['DOB'], row['Age'], row['Sex'],
              row['Height(in)'], row['Wing(in)'], row['Standing_Reach(in)'], row['Hand Span(in)']))
        conn.commit()
    print('Success')
    
except sqlite3.Error as e:
    conn.rollback()
    print(f"Error: {e}")
```


```python
# Inserting athlete measurement data into Measurements table
try:
    for index, row in data.iterrows():
        cursor.execute("""
            SELECT Athlete_id FROM Athletes WHERE Athlete = ?
        """, (row['Last, First'],))
        athlete_id = cursor.fetchone()[0]

        cursor.execute("""
            INSERT INTO Measurements (Athlete_id, Date, 
                Weight_lbs, BF_Percent, FFM_lbs, FM_lbs, BMC, A_G, T_Score,
                Bench_lbs, Bench_reps, Bench_1RM, `225_Reps`, 
                Squat_lbs, Squat_reps, Squat_1RM,
                MBDS_lbs, MBDS_reps, MBDS_1RM,
                HC_lbs, HC_reps, HC_1RM,
                RA_FM_lbs, RA_LM_lbs, LA_FM_lbs, LA_LM_lbs,
                RL_FM_lbs, RL_LM_lbs, LL_FM_lbs, LL_LM_lbs,
                SAFBar_lbs, SAFBar_reps, SAFBar_1RM,
                VJ_Reach_in, VJ_Height_in, VJ_in, Broad_Jump_in,
                PRO, L_Drill, `40_YD`, Mile)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (athlete_id, row['Date'], row['Weight(lbs)'], row['BF%'],
              row['FFM(lbs)'], row['FM(lbs)'], row['BMC'], row['A/G'], row['Tscore'],
              row['Bench(lbs)'], row['Bench(reps)'], row['Bench_1RM(lbs)'], row['225 REP'],
              row['Squat(lbs)'], row['Squat(reps)'], row['Squat_1RM(lbs)'], 
              row['MBDS(lbs)'], row['MBDS(reps)'], row['MBDS_1RM(lbs)'],
              row['HC(lbs)'], row['HC(reps)'], row['HC(lbs)'],
              row['RA_FM(lbs)'], row['RA_LM(lbs)'], row['LA_FM(lbs)'], row['LA_LM(lbs)'],
              row['RL_FM(lbs)'], row['RL_LM(lbs)'], row['LL_FM(lbs)'], row['LL_LM(lbs)'],
              row['SAFBar(lbs)'], row['SAFBar(reps)'], row['SAFBar(lbs)'],
              row['VJ_Reach(in)'], row['VJ_Height(in)'], row['VJ(in)'], row['Broad Jump'],
              row['PRO'], row['L-Drill'], row['40 YD'], row['1 Mile']))
        conn.commit()
    print('Success')

except sqlite3.Error as e:
    conn.rollback()
    print(f"Error: {e}")
```


```python
cursor.close()
conn.close()
```
