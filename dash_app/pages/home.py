import requests
from dash import html, dcc, Input, Output, State, callback

API_BASE = "http://127.0.0.1:8000"

layout = html.Div(
    style={"maxWidth": "900px", "margin": "40px auto", "fontFamily": "Arial"},
    children=[
        html.H1("PFE Acoustique & Physiologie"),
        html.H3("Créer une session"),

        dcc.Input(id="session-label", type="text", placeholder="Nom de la session", style={"width": "300px"}),
        html.Button("Créer session", id="btn-create-session", n_clicks=0, style={"marginLeft": "10px"}),

        html.Div(id="session-error", style={"color": "crimson", "marginTop": "10px"}),

        # ce composant sert à déclencher la redirection
        dcc.Location(id="redirect", refresh=True),
    ]
)

@callback(
    Output("redirect", "pathname"),
    Output("session-error", "children"),
    Input("btn-create-session", "n_clicks"),
    State("session-label", "value"),
    prevent_initial_call=True
)
def create_session(n, label):
    if not label:
        return dash.no_update, "Veuillez saisir un nom de session."

    r = requests.post(f"{API_BASE}/sessions", json={"label": label})
    if r.status_code != 200:
        return dash.no_update, f"Erreur création session: {r.text}"

    session_id = r.json()["id"]
    return f"/viewer/{session_id}", ""
