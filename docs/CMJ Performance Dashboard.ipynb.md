```Python
# CMJ Performance Dashboard developed by Isaac Clark

# Import necessary packages
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import dash_auth
import secrets

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

# Initializing the dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Setting secret key for session
app.server.secret_key = secrets.token_hex(16)

# Setting up password protection
auth = dash_auth.BasicAuth(
    app,
    {'username': 'password'}  # username: password
)

# Importing data
df = pd.read_csv(
    'data.csv'
)

# Filtering for only wanted sports groups
df = df[df['Sport'].isin(
    ['list of sports']
)]

# Filtering for only CMJ tests
df = df[df['testType'] == 'CMJ']

# Ensuring time data is formatted correctly
df['date'] = pd.to_datetime(df['date'])

# Remove outliers
def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))]

# Time slicer
def time_filter(data, time_range):
    most_recent_date = data['date'].max()
    
    if time_range == '1 Year':
        start_date = most_recent_date - pd.DateOffset(years=1)
    elif time_range == '9 Months':
        start_date = most_recent_date - pd.DateOffset(months=6)
    elif time_range == '6 Months':
        start_date = most_recent_date - pd.DateOffset(months=6)
    elif time_range == '3 Months':
        start_date = most_recent_date - pd.DateOffset(months=3)
    elif time_range == '1 Month':
        start_date = most_recent_date - pd.DateOffset(months=1)
    else:  # All Time
        start_date = data['date'].min()

    return data[data['date'] >= start_date]

# Stats table
def stats_table(data, metric, athlete):
    data[metric] = pd.to_numeric(data[metric], errors='coerce')
    data[metric] = remove_outliers(data[metric])
    idx = data.groupby(['Athlete', 'date'])[metric].idxmax().dropna()
    data = data.loc[idx].reset_index(drop=True)
    
    team_stats = {
        'Average': data[metric].mean(),
        'Standard Deviation': data[metric].std(),
        'Median': data[metric].median(),
        'Min': data[metric].min(),
        'Max': data[metric].max()
    }
    
    team_stats = {k: round(v, 2) for k, v in team_stats.items()}
    team_stats['Title'] = 'Team Stats'

    athlete_data = data[data['Athlete'].isin([athlete])]

    athlete_stats = {
        'Average': athlete_data[metric].mean(),
        'Standard Deviation': athlete_data[metric].std(),
        'Median': athlete_data[metric].median(),
        'Min': athlete_data[metric].min(),
        'Max': athlete_data[metric].max()
    }
    
    athlete_stats = {k: round(v, 2) for k, v in athlete_stats.items()}
    athlete_stats['Title'] = athlete

    team_stats['Title'] = str(team_stats['Title'])
    athlete_stats['Title'] = str(athlete_stats['Title'])

    combined_stats = pd.DataFrame([team_stats, athlete_stats])
    
    return combined_stats[['Title', 'Average', 'Standard Deviation', 'Median', 'Min', 'Max']]

# Fig1
def team_comparison(data, metric):
    data[metric] = pd.to_numeric(data[metric], errors='coerce')
    data[metric] = remove_outliers(data[metric])
    idx = data.groupby(['Athlete', 'date'])[metric].idxmax().dropna()
    data = data.loc[idx].reset_index(drop=True)

    most_recent_date = data['date'].max()
    time = most_recent_date - pd.DateOffset(months=3)
    time_filtered = data[data['date'] >= time]

    athletes = time_filtered['Athlete'].unique()

    z_scores = []
    for athlete in athletes:
        athlete_data = data[data['Athlete'].isin([athlete])]
        if not athlete_data.empty:
            mean, std = athlete_data[metric].mean(), athlete_data[metric].std()
            athlete_metric_value = athlete_data.loc[athlete_data['date'].idxmax(), metric]
            athlete_z_score = (athlete_metric_value - mean) / std
            z_scores.append((athlete, athlete_z_score))

    z_scores = pd.DataFrame(z_scores, columns=['Athlete', 'Z-Score'])
    data = time_filtered.groupby('Athlete').apply(lambda x: x.loc[x['date'].idxmax()]).reset_index(drop=True)
    data = data.merge(z_scores, on='Athlete')
    data = data.sort_values(by=metric, ascending=True)

    num_athletes = len(data['Athlete'].unique())
    chart_height = max(600, num_athletes * 25)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data[metric],
        y=data['Athlete'],
        orientation='h',
        marker=dict(
            color=data['Z-Score'],
            colorscale='RdYlGn',
            cmin=-2, cmax=2,
            colorbar=dict(title='Z-Score')
        )
    ))
    fig.update_layout(
        title=f'Team Comparison of {metric}',
        template='plotly_dark',
        xaxis_title=f'Most Recent {metric}',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        height=chart_height
    )

    return fig

# Fig2
def team_dist(data, metric, athlete):
    data[metric] = pd.to_numeric(data[metric], errors='coerce')
    data[metric] = remove_outliers(data[metric])
    idx = data.groupby(['Athlete', 'date'])[metric].idxmax().dropna()
    data = data.loc[idx].reset_index(drop=True)

    mean, std = data[metric].mean(), data[metric].std()

    hist = go.Histogram(
        x=data[metric], 
        nbinsx=30, 
        name='Distribution', 
        histnorm='probability density', 
        marker=dict(color='#666666')
    )
    normal_curve = go.Scatter(
        x=np.linspace(data[metric].min(), data[metric].max(), 100),
        y=(1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.linspace(data[metric].min(), data[metric].max(), 100) - mean) / std) ** 2),
        mode='lines', name='Normal Curve', line=dict(width=3), marker=dict(color='#be0009')
    )

    fig = go.Figure(data=[hist, normal_curve])

    athlete_data = data[data['Athlete'].isin([athlete])]
    if not athlete_data.empty:
        athlete_metric_value = athlete_data.loc[athlete_data['date'].idxmax(), metric]
        z_score = (athlete_metric_value - mean) / std
        percentile = stats.norm.cdf(z_score) * 100
        point = go.Scatter(
            x=[athlete_metric_value],
            y=[(1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((athlete_metric_value - mean) / std) ** 2)],
            mode='markers+text',
            name=f'{athlete}\'s<br>Most Recent Test',
            marker=dict(color='#be0009', size=15),
            text=[f'{percentile:.1f} Percentile'],
            textposition='top right',
            textfont=dict(color='white')
        )
        fig.add_trace(point)

    fig.update_layout(
        title=f'Distribution of {metric}',
        template='plotly_dark',
        xaxis_title=metric,
        yaxis_title='Density',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    return fig

# Fig3
def athlete_comparison(data, metric, athlete):
    data[metric] = pd.to_numeric(data[metric], errors='coerce')
    data[metric] = remove_outliers(data[metric])
    idx = data.groupby(['Athlete', 'date'])[metric].idxmax().dropna()
    data = data.loc[idx].reset_index(drop=True)

    fig = go.Figure()
    
    athlete_data = data[data['Athlete'].isin([athlete])]
    if not athlete_data.empty:
        most_recent = athlete_data.loc[athlete_data['date'].idxmax(), metric]
        avg_previous = athlete_data[metric].mean()
        fig.add_trace(go.Bar(
            x=['Average'], 
            y=[avg_previous], 
            name='Average', 
            marker=dict(color='#666666'), 
            text=[f'{avg_previous:.2f}'], 
            textposition='auto', 
            textfont=dict(color='white')
        ))
        fig.add_trace(go.Bar(
            x=['Most Recent Test'], 
            y=[most_recent], 
            name='Most Recent Test', 
            marker=dict(color='#be0009'), 
            text=[f'{most_recent:.2f}'], 
            textposition='auto', 
            textfont=dict(color='white')
        ))
    
    fig.update_layout(
        title=f'{athlete}\'s Performance Comparison',
        template='plotly_dark',
        yaxis_title=metric,
        barmode='group',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    return fig

# Fig4
def z_score_comparison(data, metric, athlete):
    data[metric] = pd.to_numeric(data[metric], errors='coerce')
    data[metric] = remove_outliers(data[metric])
    idx = data.groupby(['Athlete', 'date'])[metric].idxmax().dropna()
    data = data.loc[idx].reset_index(drop=True)

    athlete_data = data[data['Athlete'].isin([athlete])]
    mean, std = athlete_data[metric].mean(), athlete_data[metric].std()

    athlete_data['Z-Score'] = (athlete_data[metric] - mean) / std

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=athlete_data['date'].astype(str),
        y=athlete_data['Z-Score'],
        marker=dict(
            color=athlete_data['Z-Score'],
            colorscale='RdYlGn',
            cmin=-2, cmax=2,
            colorbar=dict(title='Z-Score')
        )
    ))
    fig.update_layout(
        title=f'{athlete}\'s Z-Scores',
        template='plotly_dark',
        xaxis_title='date',
        yaxis_title=f'{metric}',
        yaxis=dict(showgrid=False),
        xaxis=dict(
            showgrid=False,
            type='category',
            title='date'
        )
    )

    return fig

# Fig5
def time_series(data, metric, athlete):
    data[metric] = pd.to_numeric(data[metric], errors='coerce')
    data[metric] = remove_outliers(data[metric])
    idx = data.groupby(['Athlete', 'date'])[metric].idxmax().dropna()
    data = data.loc[idx].reset_index(drop=True)

    athlete_data = data[data['Athlete'].isin([athlete])]
    athlete_avg = athlete_data[metric].mean()

    fig = go.Figure()
    if not athlete_data.empty:
        fig.add_trace(go.Scatter(
            x=athlete_data['date'], 
            y=athlete_data[metric], 
            name=athlete, 
            line=dict(color='#be0009', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=athlete_data['date'],
            y=[athlete_avg] * len(athlete_data),
            name='Average', 
            mode='lines',
            line=dict(color='#666666', width=2, dash='dash')
        ))

    fig.update_layout(
        title=f'{athlete}\'s Performance Over Time',
        template='plotly_dark',
        yaxis_title=metric,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
            
    return fig

data = time_filter(df[df['Sport'] == 'sport'], 'All Time')

fig1 = team_comparison(data, 'Jump Height (Flight Time) in Inches [in]')

data = time_filter(df[df['Sport'] == 'sport'], 'All Time')
first_athlete = data['Athlete'].iloc[0]

fig2 = team_dist(data, 'Jump Height (Flight Time) in Inches [in]', first_athlete)
fig3 = athlete_comparison(data, 'Jump Height (Flight Time) in Inches [in]', first_athlete)
fig4 = z_score_comparison(data, 'Jump Height (Flight Time) in Inches [in]', first_athlete)
fig5 = time_series(data, 'Jump Height (Flight Time) in Inches [in]', first_athlete)

app.layout = html.Div(style={'padding': '20px', 'font-family': 'Arial, sans-serif', 'background-color': '#111111'},
    children=[
        html.Div([  # Header and logo
            html.Img(src='https://upload.wikimedia.org/wikipedia/commons/b/be/Utah_Utes_logo.svg', style={'width': '120px', 'height': 'auto', 'margin-right': '20px'}),
            html.H1(children='CMJ Performance Dashboard', style={'color': 'white', 'display': 'inline-block', 'vertical-align': 'middle', 'font-size': '36px', 'font-weight': 'bold'})
        ], style={'display': 'flex', 'align-items': 'center'}
        ),
        dcc.Tabs([
            dcc.Tab(
                label='Team Performance',
                style={'font-size': '16px', 'color': 'white', 'background-color': '#333333', 'padding': '16px', 'border-radius': '5px', 'border': 'none', 'width': '200px', 'height': '55px'},
                selected_style={'font-size': '16px', 'color': 'white', 'background-color': '#be0009', 'padding': '16px', 'border-radius': '5px', 'border': 'none', 'width': '200px', 'height': '55px'},
                children=[
                    html.Div([
                        dcc.Dropdown(  # Sport dropdown 1
                            options=[{'label': 'Select All', 'value': 'select-all'}] + [{'label': sport, 'value': sport} for sport in df.Sport.unique()],
                            id='sport-dropdown-1',
                            style={'width': '100%', 'display': 'inline-block'},
                            value=['Men\'s Basketball'],
                            multi=True
                        ),
                        dcc.Dropdown(  # Metric dropdown 1
                            id='metric-dropdown-1',
                            style={'width': '100%', 'display': 'inline-block'},
                            options=[
                                {'label': 'Concentric Peak Force / BM [N/kg]', 'value': 'Concentric Peak Force / BM [N/kg]'},
                                {'label': 'Eccentric Peak Force / BM [N/kg]', 'value': 'Eccentric Peak Force / BM [N/kg]'},
                                {'label': 'Jump Height (Flight Time) in Inches [in]', 'value': 'Jump Height (Flight Time) in Inches [in]'},
                                {'label': 'Peak Power / BM [W/kg]', 'value': 'Peak Power / BM [W/kg]'},
                                {'label': 'RSI-modified [m/s]', 'value': 'RSI-modified [m/s]'},
                            ],
                            value='Jump Height (Flight Time) in Inches [in]',
                        ),
                        dcc.Dropdown(  # Time range dropdown 1
                            id='time-dropdown-1',
                            style={'width': '100%', 'display': 'inline-block'},
                            options=[
                                {'label': 'All Time', 'value': 'All Time'},
                                {'label': '1 Year', 'value': '1 Year'},
                                {'label': '9 Months', 'value': '9 Months'},
                                {'label': '6 Months', 'value': '6 Months'},
                                {'label': '3 Months', 'value': '3 Months'},
                                {'label': '1 Month', 'value': '1 Month'}
                            ],
                            value='3 Months',
                        )
                    ], style={'display': 'flex', 'color': 'black', 'justify-content': 'center', 'gap': '20px', 'max-width': '1100px', 'margin': '0 auto'}
                    ),
                    html.Div([  # fig1
                        dcc.Graph(id='graph1', figure=fig1)
                    ], style={'width': '100%', 'display': 'inline-block'}
                    )
                ]
            ),
            dcc.Tab(
                label='Athlete Performance', 
                style={'font-size': '16px', 'color': 'white', 'background-color': '#333333', 'padding': '16px', 'border-radius': '5px', 'border': 'none', 'width': '200px', 'height': '55px'},
                selected_style={'font-size': '16px', 'color': 'white', 'background-color': '#be0009', 'padding': '16px', 'border-radius': '5px', 'border': 'none', 'width': '200px', 'height': '55px'},
                children=[
                    html.Div([
                        dcc.Dropdown(  # Sport dropdown 2
                            options=[{'label': 'Select All', 'value': 'select-all'}] + [{'label': sport, 'value': sport} for sport in df.Sport.unique()],
                            id='sport-dropdown-2',
                            style={'width': '100%', 'display': 'inline-block'},
                            value=['Men\'s Basketball'],
                            multi=True
                        ),
                        dcc.Dropdown(  # Metric dropdown 2
                            id='metric-dropdown-2',
                            style={'width': '100%', 'display': 'inline-block'},
                            options=[
                                {'label': 'Concentric Peak Force / BM [N/kg]', 'value': 'Concentric Peak Force / BM [N/kg]'},
                                {'label': 'Eccentric Peak Force / BM [N/kg]', 'value': 'Eccentric Peak Force / BM [N/kg]'},
                                {'label': 'Jump Height (Flight Time) in Inches [in]', 'value': 'Jump Height (Flight Time) in Inches [in]'},
                                {'label': 'Peak Power / BM [W/kg]', 'value': 'Peak Power / BM [W/kg]'},
                                {'label': 'RSI-modified [m/s]', 'value': 'RSI-modified [m/s]'},
                            ],
                            value='Jump Height (Flight Time) in Inches [in]',
                        ),
                        dcc.Dropdown(  # Athlete dropdown
                            id='athlete-dropdown',
                            style={'width': '100%', 'display': 'inline-block'},
                            options=[],
                            value=None
                        ),
                        dcc.Dropdown(  # Time range dropdown 2
                            id='time-dropdown-2',
                            style={'width': '100%', 'display': 'inline-block'},
                            options=[
                                {'label': 'All Time', 'value': 'All Time'},
                                {'label': '1 Year', 'value': '1 Year'},
                                {'label': '9 Months', 'value': '9 Months'},
                                {'label': '6 Months', 'value': '6 Months'},
                                {'label': '3 Months', 'value': '3 Months'},
                                {'label': '1 Month', 'value': '1 Month'}
                            ],
                            value='3 Months',
                        )
                    ], style={'display': 'flex', 'color': 'black', 'justify-content': 'center', 'gap': '20px', 'margin': '0 auto'}
                    ),
                    html.Div([  # stats table
                        dash_table.DataTable(
                            id='stats-table',
                            columns=[{'name': col, 'id': col} for col in ['Title', 'Average', 'Standard Deviation', 'Median', 'Min', 'Max']],
                            data=[],
                            style_table={'height': 'auto', 'overflowY': 'auto'},
                            style_header={'backgroundColor': '#222222', 'color': 'white'},
                            style_cell={'backgroundColor': '#333333', 'color': 'white', 'textAlign': 'center', 'font-family': 'Arial, sans-serif'}
                        )
                    ], style={'width': '100%', 'display': 'inline-block'}
                    ),
                    html.Div([  # fig2
                        dcc.Graph(id='graph2', figure=fig2)
                    ], style={'width': '60%', 'display': 'inline-block'}
                    ),
                    html.Div([  # fig3
                        dcc.Graph(id='graph3', figure=fig3)
                    ], style={'width': '40%', 'display': 'inline-block'}
                    ),
                    html.Div([  # fig4
                        dcc.Graph(id='graph4', figure=fig4)
                    ], style={'width': '40%', 'display': 'inline-block'}
                    ),
                    html.Div([  # fig5
                        dcc.Graph(id='graph5', figure=fig5)
                    ], style={'width': '60%', 'display': 'inline-block'}
                    ),
                ]
            )
        ], style={'display': 'flex', 'justify-content': 'center', 'gap': '250px', 'padding-bottom': '20px', 'max-width': '1000px', 'margin': '20px auto',}
        )
    ]
)

# Callback to update fig1
@app.callback(
    [Output('graph1', 'figure')],
    [Input('sport-dropdown-1', 'value'),
     Input('metric-dropdown-1', 'value'),
     Input('time-dropdown-1', 'value')]
)
def update_team_graph(selected_sports, selected_metric, selected_time):
    if 'select-all' in selected_sports:
        selected_sports = df.Sport.unique().tolist()

    data = df[df['Sport'].isin(selected_sports)]
    data = time_filter(data, selected_time)

    fig1 = team_comparison(data, selected_metric)

    return [fig1]

# Callback to update fig2-fig5
@app.callback(
    [Output('graph2', 'figure'),
     Output('graph3', 'figure'),
     Output('graph4', 'figure'),
     Output('graph5', 'figure')],
    [Input('sport-dropdown-2', 'value'),
     Input('metric-dropdown-2', 'value'),
     Input('athlete-dropdown', 'value'),
     Input('time-dropdown-2', 'value')]
)
def update_athlete_graphs(selected_sports, selected_metric, selected_athlete, selected_time):
    if 'select-all' in selected_sports:
        selected_sports = df.Sport.unique().tolist()

    data = df[df['Sport'].isin(selected_sports)]
    data = time_filter(data, selected_time)

    fig2 = team_dist(data, selected_metric, selected_athlete)
    fig3 = athlete_comparison(data, selected_metric, selected_athlete)
    fig4 = z_score_comparison(data, selected_metric, selected_athlete)
    fig5 = time_series(data, selected_metric, selected_athlete)
    
    return fig2, fig3, fig4, fig5

# Callback to update stats table
@app.callback(
    Output('stats-table', 'data'),
    Input('sport-dropdown-2', 'value'),
    Input('metric-dropdown-2', 'value'),
    Input('time-dropdown-2', 'value'),
    Input('athlete-dropdown', 'value')
)
def update_stats_table(selected_sport, selected_metric, selected_time, selected_athlete):
    data = time_filter(df[df['Sport'].isin(selected_sport)], selected_time)
    stats = stats_table(data, selected_metric, selected_athlete)
    
    return stats.to_dict('records')

# Callback to update athlete dropdown
@app.callback(
    Output('athlete-dropdown', 'options'),
    Output('athlete-dropdown', 'value'),
    [Input('sport-dropdown-2', 'value'),
     Input('time-dropdown-2', 'value')]
)
def update_athlete_dropdown(selected_sports, selected_time):
    if 'select-all' in selected_sports:
        filtered_df = df
    else:
        filtered_df = df[df['Sport'].isin(selected_sports)]
    filtered_df = time_filter(filtered_df, selected_time)
    athletes = filtered_df['Athlete'].unique()

    first_athlete = athletes[0] if len(athletes) > 0 else None

    return [{'label': athlete, 'value': athlete} for athlete in athletes], first_athlete

# Start dashboard server
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=21050)
```
