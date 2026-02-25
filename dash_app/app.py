import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output

from dash_app.pages.home import layout as home_layout
from dash_app.pages.viewer import layout as viewer_layout



from dash import clientside_callback, Output, Input




app = dash.Dash(
    __name__,
    update_title=None,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "PFE Acousti-Physio"

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname is None:
        return home_layout
    if pathname == "/" or pathname == "":
        return home_layout
    if pathname.startswith("/viewer/"):
        return viewer_layout
    return html.Div("404 - Page non trouvée")

#Ajout
clientside_callback(
    """
    function(n) {
        const audio = document.getElementById("viewer-audio");
        if (!audio) return 0;

        // si pause -> on renvoie la dernière position, elle n'avance plus
        if (audio.paused) return audio.currentTime || 0;

        return audio.currentTime || 0;
    }
    """,
    Output("audio-time", "data"),
    Input("audio-tick", "n_intervals"),
)





if __name__ == "__main__":
    app.run(debug=True, port=8050)

