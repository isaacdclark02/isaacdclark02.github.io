```Python
from dash import dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

#Initializing dash app
app = dash.Dash(__name__)

#Importing data and formating dates
df_injury = pd.read_excel('cleaned_injury_data.xlsx')
df_injury['incident_date_month'] = pd.to_datetime(df_injury['incident_date']).dt.strftime('%Y-%m')
df_injury['incident_date_year'] = pd.to_datetime(df_injury['incident_date']).dt.strftime('%Y')

#Getting injury counts for body area
def injury_count(sport):
    #Grouping by 'incident_sport', 'incident_date', and 'body_area', and counting occurrences
    counts = df_injury.groupby(['incident_sport', 'incident_date_year', 'body_area']).size().reset_index(name='body_area_counts')

    #Filtering for only desired sport data
    sport_df = counts[counts['incident_sport'] == sport]
    
    #Creating graph
    fig = px.bar(sport_df, x='body_area', y='body_area_counts', 
              color='incident_date_year', template='plotly_dark',
              labels={'incident_date_year': 'Year'}).update_layout(
              title='%s Injuries by Year' % (sport),
              xaxis=dict(title='Body Area', showgrid=False),
              yaxis=dict(title='Number of Injuries', showgrid=False),
              height=600)
    return fig

#Getting injury counts for injury classiification
def classification_count(sport):
    #Grouping by 'incident_sport', 'incident_date', and 'issue_classification', and counting occurrences
    counts = df_injury.groupby(['incident_sport','incident_date_year', 'issue_classification']).size().reset_index(name='counts')

    #Filtering for only desired sport data
    sport_df = counts[counts['incident_sport'] == sport]

    #Creating graph
    fig = px.bar(sport_df, x='issue_classification', y='counts',
              template='plotly_dark', color='incident_date_year',
              labels={'incident_date_year': 'Year'}).update_layout(
              title='%s Injury Classification Counts' % (sport),
              xaxis=dict(title='Classification', showgrid=False),
              yaxis=dict(title='Number of Injuries', showgrid=False),
              height=600)
    return fig

#Visualizing Where Athletes Are Getting Injured by Sport
def mechanism_count(sport):
    #Formating mechanisms to combine duplicates
    df_injury['mechanism'] = df_injury['mechanism'].str.capitalize().str.replace('-', ' ')

    #Grouping by 'incident_sport', 'incident_date', and 'issue_classification', and counting occurrences
    counts = df_injury.groupby(['incident_sport','incident_date_year', 'mechanism']).size().reset_index(name='counts')

    #Filtering for only desired sport data
    sport_df = counts[counts['incident_sport'] == sport]

    #Graphing data
    import plotly.express as px

    fig = px.bar(sport_df, x='mechanism', y='counts',
              template='plotly_dark', color='incident_date_year',
              labels={'incident_date_year': 'Year'}).update_layout(
              title='%s Injury Mechanism Counts' % (sport),
              xaxis=dict(title='Mechanism', showgrid=False),
              yaxis=dict(title='Number of Injuries', showgrid=False),
              height=600)
    return fig

#Getting the number of injuries per month over time
def injury_over_time(sport):
    #Grouping by 'incident_sport' and 'incident_date', and counting occurrences
    counts = df_injury.groupby(['incident_sport', 'incident_date_month']).size().reset_index(name='counts')

    #Filtering for only desired sport data
    sport_df = counts[counts['incident_sport'] == sport]
    
    #Creating graph
    fig = px.line(sport_df, x='incident_date_month', y='counts', template='plotly_dark',
              color_discrete_sequence=['crimson']).update_layout(
              title='%s Injuries Over Time' % (sport),
              xaxis=dict(title='Date', showgrid=False),
              yaxis=dict(title='Number of Injuries', showgrid=False),
              height=600)
    return fig

#Getting average time to MD release by sport
def recovery_time(sport):
    #Droping rows that don't contain data
    injury_df = df_injury.dropna(subset=['md_release_date'])

    #Grouping data and calculating the difference between incident data and md release date
    formated_date_df = injury_df[['incident_sport', 'incident_date', 'md_release_date', 'body_area', 'issue_classification']].copy()
    formated_date_df['time_to_md_release'] = (pd.to_datetime(formated_date_df['md_release_date']) - 
                                              pd.to_datetime(formated_date_df['incident_date'])).dt.days

    #Calculating the average days to md release
    avg_days = formated_date_df.groupby('incident_sport').agg(avg_days_to_md_release=('time_to_md_release', 'mean')).reset_index()

    #Creating graph
    fig = px.bar(avg_days, x='incident_sport', y='avg_days_to_md_release', 
            color_discrete_sequence=['crimson'], template='plotly_dark').update_layout(
            title='Average Days to MD Release Grouped by Sport',
            xaxis=dict(title='Sport', showgrid=False),
            yaxis=dict(title='Average Days to MD Released', showgrid=False),
            showlegend=False,
            height=600)
    return fig

#List of sports for drop down list
sports_options = ['Baseball', 'Basketball', 'Beach Volleyball', 'Cross Country',
               'Diving', 'American Football', 'Golf', 'Gymnastics', 'Lacrosse',
               'Skiing', 'Soccer', 'Softball', 'Swimming', 'Tennis', 'Track & Field',
               'Volleyball']

fig1 = injury_count('American Football')
fig2 = classification_count('American Football')
fig3 = mechanism_count('American Football')
fig4 = injury_over_time('American Football')
fig5 = recovery_time('American Football')

#Setting up dashboard layout
app.layout = html.Div(children=[html.Div([
    html.H1(children='Injury Dashboard', style={'font-family': 'Arial, sans-serif'}),
        dcc.Dropdown(id='sport-dropdown', style={'font-family': 'Arial, sans-serif'},
            options=[{'label': sport, 'value': sport} for sport in sports_options], value='American Football')]),
    html.Div([
        dcc.Graph(id='graph1', figure=fig1),], 
        style={'width': '50%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='graph2', figure=fig2),],
        style={'width': '50%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='graph3', figure=fig3),],
        style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='graph4', figure=fig4),],
        style={'width': '50%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='graph5', figure=fig5),],
        style={'width': '50%', 'display': 'inline-block'}),])

#Creating callback for interactivity
@app.callback([Output('graph1', 'figure'),
               Output('graph2', 'figure'),
               Output('graph3', 'figure'),
               Output('graph4', 'figure')],
              [Input('sport-dropdown', 'value')])

#Updating graphs with drop down selection
def update_graphs(selected_sport):
    fig1 = injury_count(selected_sport)
    fig2 = classification_count(selected_sport)
    fig3 = mechanism_count(selected_sport)
    fig4 = injury_over_time(selected_sport)
    return fig1, fig2, fig3, fig4

#Running the dash app
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
```
