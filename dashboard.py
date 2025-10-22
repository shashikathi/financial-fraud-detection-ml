import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import pickle

# Load sample data
df = pd.read_csv("data/sample_transactions.csv")

# Load trained model
model = pickle.load(open("models/fraud_model.pkl", "rb"))

app = dash.Dash(__name__)
app.title = "Fraud Detection Dashboard"

app.layout = html.Div([
    html.H1("Fraud Detection Dashboard", style={'textAlign': 'center'}),
    
    dcc.Tabs([
        dcc.Tab(label='EDA Overview', children=[
            html.Div([
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in df.select_dtypes(include='number').columns],
                    value='amount'
                ),
                dcc.Graph(id='feature-dist')
            ])
        ]),
        
        dcc.Tab(label='Model Prediction', children=[
            html.Div([
                html.H4("Enter Transaction Features:"),
                html.Div(id='input-fields'),
                html.Button("Predict Fraud", id='predict-btn', n_clicks=0),
                html.Div(id='prediction-result')
            ])
        ])
    ])
])

@app.callback(
    Output('feature-dist', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_graph(feature):
    fig = px.histogram(df, x=feature, color='is_fraud', barmode='overlay', nbins=40)
    fig.update_layout(title=f"Distribution of {feature}")
    return fig

# (You’ll later expand the prediction section with input fields matching your model features.)

if __name__ == "__main__":
    app.run_server(debug=True)
