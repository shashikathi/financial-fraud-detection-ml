import json
import joblib
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import io, base64

# -------------------------------------------------------------
# 🎯 APP CONFIG
# -------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "💸 Financial Fraud Detection System"
server = app.server

# -------------------------------------------------------------
# ⚙️ LOAD MODEL & METADATA
# -------------------------------------------------------------
MODEL_PATH = "fraud_model_slim.pkl"
META_PATH = "metadata.json"

try:
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    FEATURES = metadata.get("features", [])
    print("✅ Model and metadata loaded successfully.")
except Exception as e:
    print("❌ Error loading model or metadata:", e)
    model = None
    FEATURES = []

# -------------------------------------------------------------
# 📊 HELPERS
# -------------------------------------------------------------
def predict_fraud(df):
    """Run predictions using the trained model."""
    try:
        X = df[FEATURES]
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        df["Fraud_Probability"] = np.round(probs, 3)
        df["Prediction"] = np.where(preds == 1, "🚨 Fraud", "✅ Legit")
    except Exception as e:
        df["Prediction"] = f"Error: {e}"
        df["Fraud_Probability"] = np.nan
    return df

def read_html_file(path):
    """Load a local HTML file as iframe."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return html.Iframe(srcDoc=f.read(), height="480", width="100%")
    except FileNotFoundError:
        return html.Div(f"⚠️ Missing file: {path}", style={"color": "orange"})

# -------------------------------------------------------------
# 🧠 APP LAYOUT
# -------------------------------------------------------------
app.layout = dbc.Container([
    html.H1("💸 Financial Fraud Detection Dashboard", className="text-center my-3"),
    html.P("Detect and analyze fraudulent transactions using Machine Learning.",
           className="text-center text-muted"),

    dbc.Tabs([
        # ---------------- EDA TAB ----------------
        dbc.Tab(label="📊 EDA & Visuals", tab_id="eda", children=[
            dbc.Container([
                html.Br(),
                html.H4("Interactive Dashboards", className="text-info"),
                html.Hr(),
                dbc.Row([
                    dbc.Col(read_html_file("Graphs/transaction_types_distribution.html"), md=6),
                    dbc.Col(read_html_file("Graphs/fraud_rate_by_type.html"), md=6),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(read_html_file("Graphs/transaction_amount_distribution_animated.html"), md=6),
                    dbc.Col(read_html_file("Graphs/origin_account_balances_animated.html"), md=6),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(read_html_file("Graphs/feature_importance.html"), md=12),
                ])
            ])
        ]),

        # ---------------- PREDICTION TAB ----------------
        dbc.Tab(label="🔮 Fraud Prediction", tab_id="predict", children=[
            dbc.Container([
                html.Br(),
                html.H4("Upload Transaction Data", className="text-info"),
                html.P("Upload a CSV file with transaction details to predict potential frauds."),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div([
                        "📤 Drag & Drop or ",
                        html.A("Select File", style={"color": "#33C3F0", "fontWeight": "bold"})
                    ]),
                    style={
                        "width": "100%",
                        "height": "80px",
                        "lineHeight": "80px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": "10px",
                        "textAlign": "center",
                        "margin": "10px",
                        "cursor": "pointer"
                    },
                    multiple=False
                ),
                html.Div(id="output-prediction")
            ])
        ])
    ], id="tabs", active_tab="eda"),

    html.Hr(),
    html.Footer([
        html.P("Built with ❤️ by K. Shashi Preetham",
               className="text-center text-muted"),
        html.P("Data Analyst · ML Enthusiast", className="text-center text-secondary")
    ])
], fluid=True)

# -------------------------------------------------------------
# 📈 CALLBACK: HANDLE FILE UPLOAD + PREDICTION
# -------------------------------------------------------------
@app.callback(
    Output("output-prediction", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def update_predictions(contents, filename):
    if contents is None:
        return html.Div("📁 Upload a CSV file to begin.", className="text-muted")

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception as e:
        return html.Div(f"❌ Error reading file: {e}", style={"color": "red"})

    df_pred = predict_fraud(df.copy())

    # Summary
    fraud_count = (df_pred["Prediction"] == "🚨 Fraud").sum()
    total = len(df_pred)
    ratio = (fraud_count / total) * 100

    summary_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Total Transactions", className="card-title"),
            html.H2(f"{total:,}", className="text-primary")
        ])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Detected Frauds", className="card-title"),
            html.H2(f"🚨 {fraud_count}", className="text-danger")
        ])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Fraud Ratio", className="card-title"),
            html.H2(f"{ratio:.2f}%", className="text-warning")
        ])), md=4)
    ])

    fig = px.histogram(df_pred, x="Fraud_Probability", color="Prediction",
                       nbins=20, title="Fraud Probability Distribution",
                       color_discrete_map={"🚨 Fraud": "red", "✅ Legit": "green"})
    fig.update_layout(template="plotly_dark", height=400)

    return html.Div([
        html.Br(),
        html.H4(f"📊 Predictions for {filename}", className="text-success"),
        html.Br(),
        summary_cards,
        html.Br(),
        dcc.Graph(figure=fig),
        html.Br(),
        html.H5("🔍 Top 50 Predictions"),
        dash_table.DataTable(
            data=df_pred.head(50).to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_pred.columns],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#222", "color": "white", "fontWeight": "bold"},
            style_data={"backgroundColor": "#111", "color": "white"},
            page_size=10
        )
    ])

# -------------------------------------------------------------
# 🚀 RUN SERVER
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
