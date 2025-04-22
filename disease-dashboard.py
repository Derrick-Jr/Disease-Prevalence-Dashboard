import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np

# Add this near the top where you initialize your app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # Expose the Flask server for Render

# Load the data
data = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

# Color schemes for consistency
color_palette = px.colors.qualitative.Bold
disease_colors = {}

# Get top diseases by frequency for color mapping
top_diseases = data['Disease'].value_counts().nlargest(10).index
for i, disease in enumerate(top_diseases):
    disease_colors[disease] = color_palette[i % len(color_palette)]

# App layout with modern styling
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Disease Symptom Dashboard", className="display-4 text-center mb-4"),
        html.P("Interactive visualization of disease patterns, symptoms, & patient demographics @DevDK", 
               className="lead text-center mb-5")
    ], className="container mt-4"),
    
    # Main content
    dbc.Container([
        dbc.Row([
            # Filters sidebar
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Filter Options", className="text-white bg-primary"),
                    dbc.CardBody([
                        html.P("Select Age Range:", className="mb-1"),
                        dcc.RangeSlider(
                            id='age-slider',
                            min=data['Age'].min(),
                            max=data['Age'].max(),
                            value=[data['Age'].min(), data['Age'].max()],
                            marks={i: str(i) for i in range(data['Age'].min(), data['Age'].max()+1, 10)},
                            className="mb-4"
                        ),
                        
                        html.P("Select Gender:", className="mb-1"),
                        dcc.Dropdown(
                            id='gender-dropdown',
                            options=[
                                {'label': 'All', 'value': 'All'},
                                {'label': 'Female', 'value': 'Female'},
                                {'label': 'Male', 'value': 'Male'}
                            ],
                            value='All',
                            className="mb-4"
                        ),
                        
                        html.P("Select Outcome:", className="mb-1"),
                        dcc.Dropdown(
                            id='outcome-dropdown',
                            options=[
                                {'label': 'All', 'value': 'All'},
                                {'label': 'Positive', 'value': 'Positive'},
                                {'label': 'Negative', 'value': 'Negative'}
                            ],
                            value='All',
                            className="mb-4"
                        ),
                        
                        html.P("Select Diseases (Top 10):", className="mb-1"),
                        dcc.Dropdown(
                            id='disease-dropdown',
                            options=[{'label': disease, 'value': disease} for disease in top_diseases] + 
                                    [{'label': 'All', 'value': 'All'}],
                            value='All',
                            multi=True,
                            className="mb-4"
                        ),
                        
                        dbc.Button("Apply Filters", id="apply-button", color="primary", className="w-100")
                    ])
                ], className="mb-4 shadow")
            ], md=3),
            
            # Main visualization area
            dbc.Col([
                dbc.Row([
                    # Key metrics cards
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Total Cases", className="card-title text-center"),
                                html.H2(id="total-cases", className="text-center text-primary")
                            ])
                        ], className="shadow")
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Positive Rate", className="card-title text-center"),
                                html.H2(id="positive-rate", className="text-center text-success")
                            ])
                        ], className="shadow")
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Average Age", className="card-title text-center"),
                                html.H2(id="avg-age", className="text-center text-info")
                            ])
                        ], className="shadow")
                    ], width=4),
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Disease Distribution", className="text-white bg-primary"),
                            dbc.CardBody([
                                dcc.Graph(id="disease-pie-chart", config={'displayModeBar': False})
                            ])
                        ], className="shadow")
                    ], md=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Symptom Prevalence", className="text-white bg-primary"),
                            dbc.CardBody([
                                dcc.Graph(id="symptom-bar-chart", config={'displayModeBar': False})
                            ])
                        ], className="shadow")
                    ], md=6)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Age Distribution by Disease", className="text-white bg-primary"),
                            dbc.CardBody([
                                dcc.Graph(id="age-box-plot", config={'displayModeBar': False})
                            ])
                        ], className="shadow")
                    ])
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Symptom Co-occurrence Heatmap", className="text-white bg-primary"),
                            dbc.CardBody([
                                dcc.Graph(id="symptom-heatmap", config={'displayModeBar': False})
                            ])
                        ], className="shadow")
                    ], md=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Gender Distribution by Disease", className="text-white bg-primary"),
                            dbc.CardBody([
                                dcc.Graph(id="gender-disease-chart", config={'displayModeBar': False})
                            ])
                        ], className="shadow")
                    ], md=6)
                ], className="mb-4")
            ], md=9)
        ])
    ], fluid=True)
], className="bg-light min-vh-100 pb-5")

# Define callback to update visualizations
@app.callback(
    [Output("total-cases", "children"),
     Output("positive-rate", "children"),
     Output("avg-age", "children"),
     Output("disease-pie-chart", "figure"),
     Output("symptom-bar-chart", "figure"),
     Output("age-box-plot", "figure"),
     Output("symptom-heatmap", "figure"),
     Output("gender-disease-chart", "figure")],
    [Input("apply-button", "n_clicks")],
    [State("age-slider", "value"),
     State("gender-dropdown", "value"),
     State("outcome-dropdown", "value"),
     State("disease-dropdown", "value")]
)
def update_graphs(n_clicks, age_range, gender, outcome, selected_diseases):
    # Filter data based on user selections
    filtered_data = data.copy()
    
    # Filter by age range
    filtered_data = filtered_data[(filtered_data['Age'] >= age_range[0]) & 
                                 (filtered_data['Age'] <= age_range[1])]
    
    # Filter by gender
    if gender != 'All':
        filtered_data = filtered_data[filtered_data['Gender'] == gender]
    
    # Filter by outcome
    if outcome != 'All':
        filtered_data = filtered_data[filtered_data['Outcome Variable'] == outcome]
    
    # Filter by selected diseases
    if isinstance(selected_diseases, list) and 'All' not in selected_diseases and selected_diseases:
        filtered_data = filtered_data[filtered_data['Disease'].isin(selected_diseases)]
    
    # Calculate metrics
    total_cases = len(filtered_data)
    positive_rate = f"{(filtered_data['Outcome Variable'] == 'Positive').mean() * 100:.1f}%"
    avg_age = f"{filtered_data['Age'].mean():.1f}"
    
    # Create disease distribution pie chart
    disease_counts = filtered_data['Disease'].value_counts().nlargest(10)
    pie_chart = px.pie(
        values=disease_counts.values,
        names=disease_counts.index,
        title="Top 10 Diseases",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    pie_chart.update_traces(textposition='inside', textinfo='percent+label')
    pie_chart.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=350)
    
    # Create symptom prevalence bar chart
    symptom_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
    symptom_counts = filtered_data[symptom_columns].apply(lambda x: (x == 'Yes').sum())
    symptom_chart = px.bar(
        x=symptom_counts.index,
        y=symptom_counts.values,
        color=symptom_counts.index,
        labels={'x': 'Symptom', 'y': 'Count'},
        title="Symptom Prevalence",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    symptom_chart.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=350)
    
    # Age distribution by disease box plot
    top_diseases_data = filtered_data[filtered_data['Disease'].isin(disease_counts.index)]
    age_box = px.box(
        top_diseases_data,
        x="Disease",
        y="Age",
        color="Disease",
        title="Age Distribution by Disease",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    age_box.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=400)
    
    # Symptom co-occurrence heatmap
    symptom_matrix = filtered_data[symptom_columns].copy()
    symptom_matrix = symptom_matrix.replace({'Yes': 1, 'No': 0})
    correlation = symptom_matrix.corr()
    
    heatmap = px.imshow(
        correlation,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Symptom Correlation"
    )
    heatmap.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=350)
    
    # Gender distribution by disease
    gender_disease = pd.crosstab(filtered_data['Disease'], filtered_data['Gender'])
    gender_disease = gender_disease.loc[disease_counts.index]
    
    gender_chart = px.bar(
        gender_disease,
        title="Gender Distribution by Disease",
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    gender_chart.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=350)
    
    return total_cases, positive_rate, avg_age, pie_chart, symptom_chart, age_box, heatmap, gender_chart

# Run the app
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)
