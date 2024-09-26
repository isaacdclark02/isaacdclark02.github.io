```python
#Import Pandas
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

#Import data sets
df_merged = pd.read_excel('merged_best.xlsx')
```


```python
#Sorting and Filtering Injury Data set
def filter_injury(data):
    injury_data = pd.read_excel(data)

    #Converting dates to YYYY-MM-DD
    injury_data['incident_date'] = pd.to_datetime(injury_data['incident_date']).dt.strftime('%Y-%m-%d')
    injury_data['examination_date'] = pd.to_datetime(injury_data['examination_date']).dt.strftime('%Y-%m-%d')
    injury_data['md_release_date'] = pd.to_datetime(injury_data['md_release_date']).dt.strftime('%Y-%m-%d')

    #Sorting rows by dates decending
    injury_data.sort_values(by='incident_date', inplace=True, ascending=False)

    #Cleaning the data
    injury_data.drop_duplicates()
    filtered_injury_data = injury_data[
        (~(injury_data['body_area'].isin(['Neck', 'Shoulder', 'Elbow', 'Head', 'Forearm', 'Wrist/Hand', 'Chest', 'Upper Arm', 'Thoracic Spine']))) &
        (~(injury_data['issue_classification'].isin(['Disc', 'Dislocation', 'Apophysitis', 'Osteoarthritis', 'Organ Damage', 'Nerve', 'Osteochondral','Vascular', 'Structural Abnormality', 'Fracture', 'Laceration/ Abrasion']))) &
        (~(injury_data['incident_sport'].isin(['Cheerleading', 'other - sports']))) &  
        (~(injury_data['issue_type'] == 'Illness'))]

filter_injury('injury_data.xlsx')
```


```python
#Getting the Avg of Trials for each Test
FP_df = pd.read_excel('best_trial_FP.xlsx')

numeric_columns = FP_df.select_dtypes(include='number').columns
avg_FP = FP_df.groupby(['Athlete', 'Test Date']).agg({col: 'mean' for col in numeric_columns}).reset_index()
```


```python
#Merging Force Plate and Injury Data
def merged_data(injury, FP):
    injury_df = pd.read_excel(injury)
    FP_df = pd.read_excel(FP)

    #Merging injuries and best trial data into one data frame
    merge_df = pd.merge(injury_df, FP_df, left_on='athlete_name', right_on='Athlete', how='inner')

    #Filtering to only get FP data within six months of incident date
    merge_df['incident_date'] = pd.to_datetime(merge_df['incident_date'])
    merge_df['Test Date'] = pd.to_datetime(merge_df['Test Date'])

    merged_filtered = merge_df[
        (merge_df['Test Date'] >= (merge_df['incident_date'] - pd.DateOffset(months=6))) & 
        (merge_df['Test Date'] <= (merge_df['incident_date']))]
    return merged_filtered

merged = merged_data('injury_data.xlsx', 'best_trial_FP.xlsx')
```


```python
#Getting Rows With FP Data Closest to Injury
def closest_to_injury(data):
    # Ensure date columns are formatted correctly
    data['incident_date'] = pd.to_datetime(data['incident_date'])
    data['Test Date'] = pd.to_datetime(data['Test Date'])

    # Sorting and calculating days difference
    data['DaysDifference'] = (data['incident_date'] - data['Test Date']).dt.days.abs()
    index = data.groupby(['athlete_name', 'incident_date'])['DaysDifference'].idxmin()

    # Identifying rows closest to injury
    data['ClosestTestToInjury'] = 0
    data.loc[index, 'ClosestTestToInjury'] = 1
    return data

closest_df = closest_to_injury(df_merged)
```


```python
#Function to pull out specific sports data
def sport_LR(sport, body_area):
    #Fitering to just get specified Injuries
    df_sport = df_merged[df_merged['sport_identity'].str.contains(sport)]
    df_sport_area = df_sport[df_sport['body_area'] == body_area]

    df_sport_area = df_sport_area[df_sport_area['ClosestTestToInjury'] == 1]

    #Filtering for Left and Right Ankle injuries
    df_sport_area_L = df_sport_area[df_sport_area['issue_description'].str.contains('Left')]
    df_sport_area_R = df_sport_area[df_sport_area['issue_description'].str.contains('Right')]

    return df_sport_area_L, df_sport_area_R, df_sport_area

df_L, df_R, df_area = sport_LR('Gymnastics', 'Ankle')

#Function to plot chosen metrics
def line_plot(data, metric):
    #Formating dates and sorting values
    data['Test Date'] = pd.to_datetime(data['Test Date'])
    df = data.sort_values(by='Test Date')

    #Graphing data
    import plotly.express as px

    fig2 = px.line(df, x='Test Date', y=metric, 
                color='athlete_name', template='plotly_dark',
                labels={'athlete_name': 'Athlete Name'}).update_layout(
                title='%s Over Time' % (metric),
                xaxis=dict(title='Date', showgrid=False),
                yaxis=dict(title='Asymmetry Value', showgrid=False),
                showlegend=False,
                width=800,
                height=600)
    return fig2

line_plot(df_area, 'Takeoff Peak Force Asymmetry [% L,R]')
```


```python
#Visualizing time between incident date and md release by Sport
df_injury = pd.read_excel('injury_data.xlsx')
injury_df = df_injury.dropna(subset=['md_release_date'])

formated_date_df = injury_df[['incident_sport', 'incident_date', 'md_release_date', 'body_area', 'issue_classification']].copy()
formated_date_df['time_to_md_release'] = (pd.to_datetime(formated_date_df['md_release_date']) - pd.to_datetime(formated_date_df['incident_date'])).dt.days

#Calculating average time to md release by sport
avg_days = formated_date_df.groupby('incident_sport').agg(avg_days_to_md_release=('time_to_md_release', 'mean')).reset_index()

#Ploting graph
import plotly.express as px

fig = px.bar(avg_days, x='incident_sport', y='avg_days_to_md_release', 
            color_discrete_sequence=['crimson'], template='plotly_dark').update_layout(
            title='Average Days to MD Release Grouped by Sport',
            xaxis=dict(title='Sport', showgrid=False),
            yaxis=dict(title='Average Days to MD Released', showgrid=False),
            showlegend=False,
            width=800,
            height=600)
fig
```


```python
#Visualizing Injuries Over Time by Sport
def injury_over_time(sport):
    #Loading injury dataset
    df_injury = pd.read_excel('injury_data.xlsx')
    
    #Setting date type to show month and year
    df_injury['incident_date'] = pd.to_datetime(df_injury['incident_date']).dt.strftime('%Y-%m')

    #Grouping by 'incident_sport' and 'incident_date', and counting occurrences
    counts = df_injury.groupby(['incident_sport', 'incident_date']).size().reset_index(name='counts')

    #Filtering for only desired sport data
    sport_df = counts[counts['incident_sport'] == sport]

    #Graphing date
    import plotly.express as px

    fig = px.line(sport_df, x='incident_date', y='counts', template='plotly_dark', 
              color_discrete_sequence=['crimson']).update_layout(
              title='%s Injuries Over Time' % (sport),
              xaxis=dict(title='Date', showgrid=False),
              yaxis=dict(title='Number of Injuries', showgrid=False),
              width=800,
              height=600)
    return fig

injury_over_time('Gymnastics')
```


```python
#Visualizing Injury Counts by Sport
def injury_count(sport):
    #Loading injury dataset
    df_injury = pd.read_excel('injury_data.xlsx')
    
    #Setting date type to show just years
    df_injury['incident_date'] = pd.to_datetime(df_injury['incident_date']).dt.strftime('%Y')

    #Grouping by 'incident_sport', 'incident_date', and 'body_area', and counting occurrences
    counts = df_injury.groupby(['incident_sport', 'incident_date', 'body_area']).size().reset_index(name='body_area_counts')

    #Filtering for only desired sport data
    sport_df = counts[counts['incident_sport'] == sport]

    #Graphing data
    import plotly.express as px

    fig = px.bar(sport_df, x='body_area', y='body_area_counts', 
                color='incident_date', template='plotly_dark',
                labels={'incident_date': 'Year'}).update_layout(
                title='%s Injuries by Year' % (sport),
                xaxis=dict(title='Body Area', showgrid=False),
                yaxis=dict(title='Number of Injuries', showgrid=False),
                width=800,
                height=600)
    return fig

injury_count('Gymnastics')
```


```python
#Visualizing Injury Classification Counts by Sport
def classification_count(sport):
    #Loading injury dataset
    df_injury = pd.read_excel('injury_data.xlsx')
    
    #Setting date type to just show years
    df_injury['incident_date'] = pd.to_datetime(df_injury['incident_date']).dt.strftime('%Y')

    #Grouping by 'incident_sport', 'incident_date', and 'issue_classification', and counting occurrences
    counts = df_injury.groupby(['incident_sport','incident_date', 'issue_classification']).size().reset_index(name='counts')

    #Filtering for only desired sport data
    sport_df = counts[counts['incident_sport'] == sport]

    #Graphing data
    import plotly.express as px

    fig = px.bar(sport_df, x='issue_classification', y='counts',
              template='plotly_dark', color='incident_date',
              labels={'incident_date': 'Year'}).update_layout(
              title='%s Injury Classification Counts' % (sport),
              xaxis=dict(title='Classification', showgrid=False),
              yaxis=dict(title='Number of Injuries', showgrid=False),
              width=800,
              height=600)
    return fig

classification_count('Gymnastics')
```


```python
#Visualizing Where Athletes Are Getting Injured by Sport
def mechanism_count(sport):
    #Loading injury dataset
    df_injury = pd.read_excel('injury_data.xlsx')
    
    #Setting date type to just show years
    df_injury['incident_date'] = pd.to_datetime(df_injury['incident_date']).dt.strftime('%Y')
    df_injury['mechanism'] = df_injury['mechanism'].str.capitalize().str.replace('-', ' ')

    #Grouping by 'incident_sport', 'incident_date', and 'issue_classification', and counting occurrences
    counts = df_injury.groupby(['incident_sport','incident_date', 'mechanism']).size().reset_index(name='counts')

    #Filtering for only desired sport data
    sport_df = counts[counts['incident_sport'] == sport]

    #Graphing data
    import plotly.express as px

    fig = px.bar(sport_df, x='mechanism', y='counts',
              template='plotly_dark', color='incident_date',
              labels={'incident_date': 'Year'}).update_layout(
              title='%s Injury Mechanism Counts' % (sport),
              xaxis=dict(title='Mechanism', showgrid=False),
              yaxis=dict(title='Number of Injuries', showgrid=False),
              width=800,
              height=600)
    return fig

mechanism_count('Gymnastics')
```


```python
#Code to Get Graphs for All Sports
sports_list = ['Baseball', 'Basketball', 'Beach Volleyball', 'Cross Country',
               'Diving', 'American Football', 'Golf', 'Gymnastics', 'Lacrosse',
               'Skiing', 'Soccer', 'Softball', 'Swimming', 'Tennis', 'Track & Field',
               'Volleyball']

for sport in sports_list:
    fig = mechanism_count(sport)
    fig.write_image(f'{sport} Injury Mech.png')
```


```python
#Performing and Visualizing Linear Regression
import statsmodels.api as sm

def linear_regression(metric):
    filterd_df = df_merged[(df_merged['mechanism'] != 'Contact with Other Player') &
                           (df_merged['mechanism'] != 'Contact with Playing Device') &
                           (df_merged['mechanism'] != 'Contact with Person') &
                           (df_merged['mechanism'] != 'Contact with Apparatus')]

    x = sm.add_constant(filterd_df[metric])
    y = filterd_df['DaysDifference']

    model = sm.OLS(y, x).fit()

    import plotly.express as px

    fig = px.scatter(filterd_df, x=metric, y='DaysDifference', 
                     template='seaborn', trendline="ols", trendline_color_override="red")

    fig.update_layout(title='Force Plate Linear Regression Plot',
                      height=600,
                      width=800)
    
    return fig.show(), model.summary()

linear_regression('RSI-modified [m/s]')
```


```python
#Cox Hazard Model for Predicting Time to Injury
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

filterd_df = df_merged[(df_merged['mechanism'] != 'Contact with Other Player') &
                       (df_merged['mechanism'] != 'Contact with Playing Device') &
                       (df_merged['mechanism'] != 'Contact with Person') &
                       (df_merged['mechanism'] != 'Contact with Apparatus')]

label_encoder = LabelEncoder()

filterd_df['sport_num'] = label_encoder.fit_transform(filterd_df['incident_sport'])
filterd_df['body_area_num'] = label_encoder.fit_transform(filterd_df['body_area'])

filterd_df['intercept'] = 1

model = sm.PHReg(filterd_df['DaysDifference'], filterd_df[['intercept', 'sport_num', 'body_area_num',
                                                           'RSI-modified [m/s]', 'Takeoff Peak Force Asymmetry [% L,R]']], 
                 status=filterd_df['ClosestTestToInjury']).fit()

model.summary()
```
