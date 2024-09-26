```Python
# CMJ Performance Dashboard developed by Isaac Clark

# Import necessary packages
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_auth
import secrets

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

# Initializing the dash app
app = Dash(__name__)

# Setting secret key for session
app.server.secret_key = secrets.token_hex(16)

# Setting up password protection
auth = dash_auth.BasicAuth(
    app,
    {'username': 'password'}  # username: password
)

# Importing data
df = pd.read_csv('data.csv')

# Ensuring time data is formatted correctly
df['Test Date'] = pd.to_datetime(df['Test Date'])

# Time slicer
def time_filter(data, time_range):
    most_recent_date = data['Test Date'].max()
    
    if time_range == 'Last Year':
        start_date = most_recent_date - pd.DateOffset(years=1)
    elif time_range == 'Last 6 Months':
        start_date = most_recent_date - pd.DateOffset(months=6)
    elif time_range == 'Last 3 Months':
        start_date = most_recent_date - pd.DateOffset(months=3)
    else:  # All Time
        start_date = data['Test Date'].min()

    return data[data['Test Date'] >= start_date]

# Stats table
def stats_table(data, metric, athlete=None):
    idx = data.groupby(['Athlete', 'Test Date'])[metric].idxmax()
    data = data.loc[idx].reset_index(drop=True)
    data[metric] = pd.to_numeric(data[metric], errors='coerce')
    
    team_stats = {
        'Average': data[metric].mean(),
        'Standard Deviation': data[metric].std(),
        'Median': data[metric].median(),
        'Min': data[metric].min(),
        'Max': data[metric].max()
    }
    
    team_stats = {k: round(v, 2) for k, v in team_stats.items()}
    team_stats['Title'] = 'Team Stats'

    if athlete:
        athlete_data = data[data['Athlete'] == athlete]

        athlete_stats = {
            'Average': athlete_data[metric].mean(),
            'Standard Deviation': athlete_data[metric].std(),
            'Median': athlete_data[metric].median(),
            'Min': athlete_data[metric].min(),
            'Max': athlete_data[metric].max()
        }
        
        athlete_stats = {k: round(v, 2) for k, v in athlete_stats.items()}
        athlete_stats['Title'] = athlete
    
        combined_stats = pd.DataFrame([team_stats, athlete_stats])
    else:
        combined_stats = pd.DataFrame([team_stats])
    
    return combined_stats[['Title', 'Average', 'Standard Deviation', 'Median', 'Min', 'Max']]

# Fig1
def team_comparison(data, metric):
    idx = data.groupby(['Athlete', 'Test Date'])[metric].idxmax()
    data = data.loc[idx].reset_index(drop=True)
    data[metric] = pd.to_numeric(data[metric], errors='coerce')

    most_recent_date = data['Test Date'].max()
    time = most_recent_date - pd.DateOffset(months=3)
    data = data[data['Test Date'] >= time]

    athletes = data['Athlete'].unique()

    z_scores = []
    for athlete in athletes:
        athlete_data = data[data['Athlete'] == athlete]
        if not athlete_data.empty:
            mean, std = athlete_data[metric].mean(), athlete_data[metric].std()
            athlete_metric_value = athlete_data.loc[athlete_data['Test Date'].idxmax(), metric]
            athlete_z_score = (athlete_metric_value - mean) / std
            z_scores.append((athlete, athlete_z_score))

    z_scores = pd.DataFrame(z_scores, columns=['Athlete', 'Z-Score'])
    data = data.groupby('Athlete').apply(lambda x: x.loc[x['Test Date'].idxmax()]).reset_index(drop=True)
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
            colorbar=dict(title='Z-Score')
        )
    ))
    fig.update_layout(
        title=f'{data["Sport"].iloc[0]} Team Comparison of Most Recent Test',
        template='plotly_dark',
        xaxis_title=f'Most Recent {metric}',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        height=chart_height
    )

    return fig

# Fig2
def team_dist(data, metric, athlete=None):
    idx = data.groupby(['Athlete', 'Test Date'])[metric].idxmax()
    data = data.loc[idx].reset_index(drop=True)
    data[metric] = pd.to_numeric(data[metric], errors='coerce')

    mean, std = data[metric].mean(), data[metric].std()

    hist = go.Histogram(
        x=data[metric], 
        nbinsx=30, 
        name='Distribution', 
        histnorm='probability density', 
        marker=dict(color='#808080')
    )
    normal_curve = go.Scatter(
        x=np.linspace(data[metric].min(), data[metric].max(), 100),
        y=(1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.linspace(data[metric].min(), data[metric].max(), 100) - mean) / std) ** 2),
        mode='lines', name='Normal Curve', line=dict(width=3), marker=dict(color='#be0009')
    )

    fig = go.Figure(data=[hist, normal_curve])

    if athlete:
        athlete_data = data[data['Athlete'] == athlete]
        if not athlete_data.empty:
            athlete_metric_value = athlete_data.loc[athlete_data['Test Date'].idxmax(), metric]
            z_score = (athlete_metric_value - mean) / std
            percentile = stats.norm.cdf(z_score) * 100
            point = go.Scatter(
                x=[athlete_metric_value],
                y=[(1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((athlete_metric_value - mean) / std) ** 2)],
                mode='markers+text',
                name=f"{athlete}'s<br>Most Recent Test",
                marker=dict(color='#be0009', size=15),
                text=[f'{percentile:.2f} Percentile'],
                textposition='top right',
                textfont=dict(color='white')
            )
            fig.add_trace(point)

    fig.update_layout(
        title=f'Distribution of {metric} for {data["Sport"].iloc[0]}',
        template='plotly_dark',
        xaxis_title=metric,
        yaxis_title='Density',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    return fig

# Fig3
def athlete_comparison(data, metric, athlete=None):
    idx = data.groupby(['Athlete', 'Test Date'])[metric].idxmax()
    data = data.loc[idx].reset_index(drop=True)
    data[metric] = pd.to_numeric(data[metric], errors='coerce')

    fig = go.Figure()
    fig.update_layout(
        title='Athlete Performance Comparison',
        template='plotly_dark',
        yaxis_title=metric,
        barmode='group',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    if athlete:
        athlete_data = data[data['Athlete'] == athlete]
        if not athlete_data.empty:
            most_recent = athlete_data.loc[athlete_data['Test Date'].idxmax(), metric]
            avg_previous = athlete_data[metric].iloc[:-1].mean() if len(athlete_data) > 1 else 0
            fig.add_trace(go.Bar(
                x=['Average'], 
                y=[avg_previous], 
                name='Average', 
                marker=dict(color='#808080'), 
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
                title=f"{athlete}'s Performance Comparison"
            )

    return fig

# Fig4
def z_score_comparison(data, metric, athlete=None):
    idx = data.groupby(['Athlete', 'Test Date'])[metric].idxmax()
    data = data.loc[idx].reset_index(drop=True)
    data[metric] = pd.to_numeric(data[metric], errors='coerce')

    athlete_data = data[data['Athlete'] == athlete]
    mean, std = athlete_data[metric].mean(), athlete_data[metric].std()

    athlete_data['Z-Score'] = (athlete_data[metric] - mean) / std

    athlete_data['Test Date'] = athlete_data['Test Date'].dt.strftime('%Y-%m-%d')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=athlete_data['Test Date'].astype(str),
        y=athlete_data['Z-Score'],
        marker=dict(
            color=athlete_data['Z-Score'],
            colorscale='RdYlGn',
            colorbar=dict(title='Z-Score')
        )
    ))
    fig.update_layout(
        title=f'Z-Scores of {athlete} for {metric}',
        template='plotly_dark',
        xaxis_title='Test Date',
        yaxis_title=f'{metric}',
        yaxis=dict(showgrid=False),
        xaxis=dict(
            showgrid=False,
            type='category',
            title='Test Date'
        )
    )

    return fig

# Fig5
def time_series(data, metric, athlete=None):
    idx = data.groupby(['Athlete', 'Test Date'])[metric].idxmax()
    data = data.loc[idx].reset_index(drop=True)
    data[metric] = pd.to_numeric(data[metric], errors='coerce')

    fig = go.Figure()
    fig.update_layout(
        title='Athlete Performance Over Time',
        template='plotly_dark',
        yaxis_title=metric,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    if athlete:
        athlete_data = data[data['Athlete'] == athlete]
        if not athlete_data.empty:
            fig.add_trace(go.Scatter(
                x=athlete_data['Test Date'], 
                y=athlete_data[metric], 
                name=athlete, 
                line=dict(color='#be0009', width=3)
            ))
            fig.update_layout(
                title=f"{athlete}'s Performance Over Time"
            )
            
    return fig

fig1 = team_comparison(df[df['Sport'] == 'Mens Basketball'], 'Jump Height (Flight Time) [cm]')

data = time_filter(df[df['Sport'] == 'Mens Basketball'], 'All Time')

fig2 = team_dist(data, 'Jump Height (Flight Time) [cm]')
fig3 = athlete_comparison(data, 'Jump Height (Flight Time) [cm]')
fig4 = z_score_comparison(data, 'Jump Height (Flight Time) [cm]')
fig5 = time_series(data, 'Jump Height (Flight Time) [cm]')

app.layout = html.Div(
    style={'padding': '20px'},
    children=[
        html.Div([  # Header and logo
            html.Img(src='assets/logo.png', style={'width': '100px', 'height': 'auto', 'margin-right': '20px'}),
            html.H1(children='CMJ Performance Dashboard', style={'color': 'white', 'font-family': 'Arial, sans-serif', 'display': 'inline-block', 'vertical-align': 'middle'})
        ], style={'display': 'flex', 'align-items': 'center'}
        ),
        dcc.Tabs([
            dcc.Tab(
                label='Team Performance',
                style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': '#FFFFFF',
                    'background-color': '#505050',
                    'padding': '10px',
                    'border': 'none',
                },
                selected_style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': '#FFFFFF',
                    'background-color': '#be0009',
                    'padding': '10px',
                    'border': 'none',
                },
                children=[
                    html.Div([
                        dcc.Dropdown(  # Sport dropdown 1
                            df.Sport.unique(),
                            id='sport-dropdown-1',
                            style={'font-family': 'Arial, sans-serif', 'width': '100%', 'display': 'inline-block'},
                            value='Mens Basketball',
                        ),
                        dcc.Dropdown(  # Metric dropdown 1
                            id='metric-dropdown-1',
                            style={'font-family': 'Arial, sans-serif', 'width': '100%', 'display': 'inline-block'},
                            options=[
                                {'label': 'Concentric Peak Force / BM [N/kg]', 'value': 'Concentric Peak Force / BM [N/kg]'},
                                {'label': 'Eccentric Peak Force / BM [N/kg]', 'value': 'Eccentric Peak Force / BM [N/kg]'},
                                {'label': 'Jump Height (Flight Time) [cm]', 'value': 'Jump Height (Flight Time) [cm]'},
                                {'label': 'Peak Power / BM [W/kg]', 'value': 'Peak Power / BM [W/kg]'},
                                {'label': 'RSI-modified [m/s]', 'value': 'RSI-modified [m/s]'}
                            ],
                            value='Jump Height (Flight Time) [cm]',
                        ),
                    ], style={'display': 'flex', 'justify-content': 'space-between'}
                    ),
                    html.Div([  # fig1
                        dcc.Graph(id='graph1', figure=fig1),
                        html.P(
                            '*Z-Score calculated based on the last 3 months of available data',
                            style={'color': 'white', 'font-family': 'Arial, sans-serif', 'text-align': 'left'}
                        )
                    ], style={'width': '100%', 'display': 'inline-block'}
                    )
                ]
            ),
            dcc.Tab(
                label='Athlete Performance', 
                style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': '#FFFFFF',
                    'background-color': '#505050',
                    'padding': '10px',
                    'border': 'none',
                },
                selected_style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': '#FFFFFF',
                    'background-color': '#be0009',
                    'padding': '10px',
                    'border': 'none',
                },
                children=[
                    html.Div([
                        dcc.Dropdown(  # Sport dropdown 2
                            df.Sport.unique(),
                            id='sport-dropdown-2',
                            style={'font-family': 'Arial, sans-serif', 'width': '100%', 'display': 'inline-block'},
                            value='Mens Basketball',
                        ),
                        dcc.Dropdown(  # Metric dropdown 2
                            id='metric-dropdown-2',
                            style={'font-family': 'Arial, sans-serif', 'width': '100%', 'display': 'inline-block'},
                            options=[
                                {'label': 'Concentric Peak Force / BM [N/kg]', 'value': 'Concentric Peak Force / BM [N/kg]'},
                                {'label': 'Eccentric Peak Force / BM [N/kg]', 'value': 'Eccentric Peak Force / BM [N/kg]'},
                                {'label': 'Jump Height (Flight Time) [cm]', 'value': 'Jump Height (Flight Time) [cm]'},
                                {'label': 'Peak Power / BM [W/kg]', 'value': 'Peak Power / BM [W/kg]'},
                                {'label': 'RSI-modified [m/s]', 'value': 'RSI-modified [m/s]'}
                            ],
                            value='Jump Height (Flight Time) [cm]',
                        ),
                        dcc.Dropdown(  # Athlete dropdown
                            id='athlete-dropdown',
                            style={'font-family': 'Arial, sans-serif', 'width': '100%', 'display': 'inline-block'},
                            options=[],
                            value=None,
                            placeholder='Select an Athlete',
                        ),
                        dcc.Dropdown(  # Time range dropdown
                            id='time-dropdown',
                            style={'font-family': 'Arial, sans-serif', 'width': '100%', 'display': 'inline-block'},
                            options=[
                                {'label': 'All Time', 'value': 'All Time'},
                                {'label': 'Last Year', 'value': 'Last Year'},
                                {'label': 'Last 6 Months', 'value': 'Last 6 Months'},
                                {'label': 'Last 3 Months', 'value': 'Last 3 Months'}
                            ],
                            value='Last 3 Months',
                        )
                    ], style={'display': 'flex', 'justify-content': 'space-between'}
                    ),
                    html.Div([  # stats table
                        dash_table.DataTable(
                            id='stats-table',
                            columns=[{'name': col, 'id': col} for col in ['Title', 'Average', 'Standard Deviation', 'Median', 'Min', 'Max']],
                            data=[],
                            style_table={'height': 'auto', 'overflowY': 'auto'},
                            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                            style_cell={
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white',
                                'textAlign': 'center',
                                'fontFamily': 'Arial, sans-serif'
                            }
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
        ])
    ]
)

# Callback to update fig1
@app.callback(
    [Output('graph1', 'figure')],
    [Input('sport-dropdown-1', 'value'),
     Input('metric-dropdown-1', 'value')]
)
def update_team_graph(selected_sport, selected_metric):
    data = df[df['Sport'] == selected_sport]

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
     Input('time-dropdown', 'value')]
)
def update_athlete_graphs(selected_sport, selected_metric, selected_athlete, selected_time):
    data = time_filter(df[df['Sport'] == selected_sport], selected_time)
    
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
    Input('time-dropdown', 'value'),
    Input('athlete-dropdown', 'value')
)
def update_stats_table(selected_sport, selected_metric, selected_time, selected_athlete):
    data = time_filter(df[df['Sport'] == selected_sport], selected_time)
    stats = stats_table(data, selected_metric, selected_athlete)
    
    return stats.to_dict('records')

# Callback to update athlete dropdown
@app.callback(
    Output('athlete-dropdown', 'options'),
    Output('athlete-dropdown', 'value'),
    Input('sport-dropdown-2', 'value'),
    Input('time-dropdown', 'value')
)
def update_athlete_dropdown(selected_sport, selected_time):
    filtered_df = time_filter(df[df['Sport'] == selected_sport], selected_time)
    athletes = filtered_df['Athlete'].unique()
    
    return [{'label': athlete, 'value': athlete} for athlete in athletes], None

# Start dashboard server
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050)
```
