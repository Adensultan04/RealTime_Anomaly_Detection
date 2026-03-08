from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import joblib
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import time

# Load model and scaler
model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
df = pd.read_csv("transactions.csv")

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Real-Time Retail Anomaly Detection Dashboard"),

    html.Div(id="stats"),

    dcc.Graph(id="live-graph"),

    html.H2("Last 5 Suspicious Transactions"),
    html.Button("Download Anomalies", id="download-btn"),
    dcc.Download(id="download-data"),

    dash_table.DataTable(
        id="anomaly-table",
        columns=[
            {"name": "Transaction ID", "id": "transaction_id"},
            {"name": "Amount", "id": "amount"}
        ],
        style_cell={"textAlign": "center"},
        style_header={"fontWeight": "bold"},
    ),

    dcc.Interval(
        id="interval-component",
        interval=1000,
        n_intervals=0
    )
])

processed_data = []

@app.callback(
    [Output("stats", "children"),
     Output("live-graph", "figure"),
     Output("anomaly-table", "data")],
    [Input("interval-component", "n_intervals")]
)

def update_dashboard(n):

    if n < len(df):
        row = df.iloc[n]
        amount = row["amount"]

        scaled_amount = scaler.transform([[amount]])
        prediction = model.predict(scaled_amount)

        is_anomaly = prediction[0] == -1

        processed_data.append({
            "transaction_id": row["transaction_id"],
            "amount": amount,
            "anomaly": is_anomaly
        })

    display_df = pd.DataFrame(processed_data)

    total = len(display_df)
    anomalies = display_df["anomaly"].sum() if total > 0 else 0
    if anomalies > 0:
      alert = html.H2("⚠️ Suspicious Transactions Detected!", style={"color": "red"})
    else:
      alert = html.H2("System Normal", style={"color": "green"})

    fig = px.scatter(
    display_df,
    x=display_df.index,
    y="amount",
    color="anomaly",
    title="Transaction Amount with Anomalies",
)

    stats = html.Div([
    html.H3(f"Total Transactions: {total}"),
    html.H3(f"Anomalies Detected: {anomalies}"),
    alert
])

    # Get last 5 anomalies
    anomaly_df = display_df[display_df["anomaly"] == True].tail(5)

    return stats, fig, anomaly_df.to_dict("records")

@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_anomalies(n_clicks):

    df = pd.DataFrame(processed_data)

    anomaly_df = df[df["anomaly"] == True]

    return dcc.send_data_frame(anomaly_df.to_csv, "anomalies.csv")
if __name__ == "__main__":
    app.run(debug=True)
   