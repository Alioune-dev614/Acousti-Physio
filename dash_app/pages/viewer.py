# /mnt/data/viewer.py
import base64
import io

import numpy as np
import requests
import plotly.graph_objects as go

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback

from scipy.signal import spectrogram

API_BASE = "http://127.0.0.1:8000"


def make_empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=30, b=20),
        title=title,
        xaxis_title="Temps (s)",
        yaxis_title="Amplitude",
    )
    return fig


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1]
    return s


def parse_textgrid_from_string(content_string: str) -> dict:
    """
    Parse un TextGrid Praat (format long "ooTextFile") depuis le contenu dcc.Upload.

    Retour:
        {
          "xmin": float,
          "xmax": float,
          "tiers": [
            {
              "class": "IntervalTier"|"TextTier"|...,
              "name": str,
              "xmin": float,
              "xmax": float,
              "intervals": [{"xmin":float,"xmax":float,"text":str}, ...]  # pour IntervalTier
              "points": [{"time":float,"mark":str}, ...]                 # pour TextTier
            }, ...
          ]
        }
    """
    if not content_string:
        return {"xmin": 0.0, "xmax": 0.0, "tiers": []}

    payload = content_string.split(",", 1)[1] if "," in content_string else content_string
    text = base64.b64decode(payload).decode("utf-8", errors="ignore")
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    def find_first_float(key: str, start: int = 0) -> float | None:
        for idx in range(start, len(lines)):
            s = lines[idx].strip()
            if s.startswith(key):
                try:
                    return float(s.split("=", 1)[1].strip())
                except Exception:
                    return None
        return None

    tg_xmin = find_first_float("xmin", 0) or 0.0
    tg_xmax = find_first_float("xmax", 0) or 0.0

    tiers: list[dict] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()

        if s.startswith("item [") and s.endswith("]:"):
            tier = {
                "class": None,
                "name": None,
                "xmin": None,
                "xmax": None,
                "intervals": [],
                "points": [],
            }

            i += 1
            while i < len(lines):
                s2 = lines[i].strip()

                if s2.startswith("item [") and s2.endswith("]:"):
                    i -= 1
                    break

                if s2.startswith("class"):
                    tier["class"] = _strip_quotes(s2.split("=", 1)[1].strip())
                elif s2.startswith("name"):
                    tier["name"] = _strip_quotes(s2.split("=", 1)[1].strip())
                elif s2.startswith("xmin"):
                    try:
                        tier["xmin"] = float(s2.split("=", 1)[1].strip())
                    except Exception:
                        pass
                elif s2.startswith("xmax"):
                    try:
                        tier["xmax"] = float(s2.split("=", 1)[1].strip())
                    except Exception:
                        pass

                # IntervalTier
                if s2.startswith("intervals [") and s2.endswith("]:"):
                    interval = {"xmin": None, "xmax": None, "text": ""}
                    j = i + 1
                    while j < len(lines):
                        s3 = lines[j].strip()
                        if s3.startswith("intervals [") and s3.endswith("]:"):
                            break
                        if s3.startswith("item [") and s3.endswith("]:"):
                            break

                        if s3.startswith("xmin"):
                            try:
                                interval["xmin"] = float(s3.split("=", 1)[1].strip())
                            except Exception:
                                pass
                        elif s3.startswith("xmax"):
                            try:
                                interval["xmax"] = float(s3.split("=", 1)[1].strip())
                            except Exception:
                                pass
                        elif s3.startswith("text"):
                            interval["text"] = _strip_quotes(s3.split("=", 1)[1].strip())
                            break
                        j += 1

                    if interval["xmin"] is not None and interval["xmax"] is not None:
                        tier["intervals"].append(interval)

                    i = j - 1

                # TextTier (points)
                if s2.startswith("points [") and s2.endswith("]:"):
                    point = {"time": None, "mark": ""}
                    j = i + 1
                    while j < len(lines):
                        s3 = lines[j].strip()
                        if s3.startswith("points [") and s3.endswith("]:"):
                            break
                        if s3.startswith("item [") and s3.endswith("]:"):
                            break

                        if s3.startswith("time"):
                            try:
                                point["time"] = float(s3.split("=", 1)[1].strip())
                            except Exception:
                                pass
                        elif s3.startswith("mark"):
                            point["mark"] = _strip_quotes(s3.split("=", 1)[1].strip())
                            break
                        j += 1

                    if point["time"] is not None:
                        tier["points"].append(point)

                    i = j - 1

                i += 1

            if tier["xmin"] is None:
                tier["xmin"] = tg_xmin
            if tier["xmax"] is None:
                tier["xmax"] = tg_xmax
            if tier["class"] is None:
                tier["class"] = "IntervalTier"
            if tier["name"] is None:
                tier["name"] = f"Tier {len(tiers) + 1}"

            tiers.append(tier)

        i += 1

    return {"xmin": tg_xmin, "xmax": tg_xmax, "tiers": tiers}


layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Location(id="viewer-url"),
        dcc.Store(id="viewer-session-id"),
        dcc.Store(id="viewer-audio-b64"),
        dcc.Store(id="store-fig-audio"),
        dcc.Store(id="store-fig-spec"),
        dcc.Store(id="store-fig-f0"),
        dcc.Store(id="store-textgrid", data=None),
        dcc.Store(id="audio-time", data=0),
        dcc.Interval(id="audio-tick", interval=200, n_intervals=0),
        dbc.Row(
            [
                dbc.Col(
                    width=3,
                    children=[
                        html.H4("Session"),
                        html.Div(id="viewer-session-label", style={"color": "#555", "marginBottom": "10px"}),
                        html.Hr(),
                        html.H5("Upload WAV"),
                        dcc.Upload(
                            id="viewer-upload",
                            children=html.Div(["Glisser un WAV ou cliquer"]),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "6px",
                                "textAlign": "center",
                                "marginBottom": "10px",
                            },
                            multiple=False,
                        ),
                        html.Div(id="viewer-upload-info", style={"fontSize": "0.9rem", "color": "#333"}),
                        html.Hr(),
                        html.H5("Upload TextGrid (optionnel)"),
                        dcc.Upload(
                            id="viewer-upload-textgrid",
                            children=html.Div(["Glisser un TextGrid ou cliquer"]),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "6px",
                                "textAlign": "center",
                                "marginBottom": "10px",
                            },
                            multiple=False,
                        ),
                        html.Div(id="viewer-textgrid-info", style={"fontSize": "0.9rem", "color": "#333"}),
                        html.Hr(),
                        html.H5("Audio"),
                        html.Audio(id="viewer-audio", controls=True, style={"width": "100%"}),
                        html.Div(style={"height": "10px"}),
                        dbc.Button("Mesurer durée", id="viewer-btn-duration", color="primary", n_clicks=0, className="w-100"),
                        html.Div(id="viewer-measure-out", style={"marginTop": "10px", "fontWeight": "600"}),
                        html.Hr(),
                        dbc.Button("⬅ Retour", href="/", color="secondary", className="w-100"),
                    ],
                ),
                dbc.Col(
                    width=9,
                    children=[
                        html.H3("Visualisation"),
                        dcc.Graph(id="fig-audio", figure=make_empty_fig("Signal acoustique (audio)")),
                        dcc.Graph(id="fig-spec", figure=make_empty_fig("Spectrogramme")),
                        dcc.Graph(id="fig-f0", figure=make_empty_fig("Fréquence fondamentale (f0)")),
                        dcc.Graph(id="fig-textgrid", figure=make_empty_fig("Annotations TextGrid")),
                        html.Div("", style={"color": "#777", "fontSize": "0.9rem"}),
                    ],
                ),
            ]
        ),
    ],
)


@callback(
    Output("viewer-session-id", "data"),
    Input("viewer-url", "pathname"),
)
def parse_session_id(pathname):
    if not pathname or not pathname.startswith("/viewer/"):
        return None
    return pathname.split("/viewer/")[1]


@callback(
    Output("viewer-upload-info", "children"),
    Output("viewer-audio", "src"),
    Output("store-fig-audio", "data"),
    Output("store-fig-spec", "data"),
    Output("store-fig-f0", "data"),
    Input("viewer-upload", "contents"),
    State("viewer-upload", "filename"),
    State("viewer-session-id", "data"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename, session_id):
    if not session_id:
        return "Session invalide.", None, dash.no_update, dash.no_update, dash.no_update

    if not contents:
        return "", None, dash.no_update, dash.no_update, dash.no_update

    content_type, content_string = contents.split(",")
    file_bytes = base64.b64decode(content_string)

    files = {"file": (filename, io.BytesIO(file_bytes), "audio/wav")}
    r = requests.post(f"{API_BASE}/sessions/{session_id}/upload", files=files)
    if r.status_code != 200:
        return f"Erreur upload: {r.text}", None, dash.no_update, dash.no_update, dash.no_update

    meta = r.json()
    info = f"{meta['original_filename']} | durée: {meta['duration_sec']:.2f}s | fs: {meta['sample_rate']:.0f}Hz"

    try:
        import soundfile as sf

        x, sr = sf.read(io.BytesIO(file_bytes), always_2d=True)
        x = x.mean(axis=1)
        def zero_crossing_rate(frame):
            """Calcule le taux de passage par zéro d'une trame"""
            return np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        def estimate_f0_autocorr(x, sr, frame_ms=40, hop_ms=10, fmin=70, fmax=300):
            frame_len = int(sr * frame_ms / 1000)
            hop = int(sr * hop_ms / 1000)

            lag_min = int(sr / fmax)
            lag_max = int(sr / fmin)

            n_frames = 1 + max(0, (len(x) - frame_len) // hop)
            times = np.zeros(n_frames)
            f0 = np.full(n_frames, np.nan)

            win = np.hanning(frame_len)

            energy_threshold = 0.015
            clarity_threshold = 0.6

            fmin_human = 100
            fmax_human = 240

            for i in range(n_frames):
                start = i * hop
                if start + frame_len > len(x):
                    break

                frame = x[start : start + frame_len]
                frame = frame - np.mean(frame)
                frame = frame * win

                rms = np.sqrt(np.mean(frame**2))
                if rms < energy_threshold:
                    times[i] = (start + frame_len / 2) / sr
                    f0[i] = np.nan
                    continue


                # Vérification du taux de passage par zéro (ZCR)
                zcr = zero_crossing_rate(frame)
                if zcr > 0.4:  # seuil empirique : >0.3 = bruit/non-voisé
                    times[i] = (start + frame_len / 2) / sr
                    f0[i] = np.nan
                    continue

                ac = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
                if len(ac) < lag_max + 1:
                    times[i] = (start + frame_len / 2) / sr
                    f0[i] = np.nan
                    continue

                seg = ac[lag_min:lag_max]
                if len(seg) == 0:
                    times[i] = (start + frame_len / 2) / sr
                    f0[i] = np.nan
                    continue

                peak_idx = np.argmax(seg)
                peak_lag = peak_idx + lag_min

                clarity = ac[peak_lag] / (ac[0] + 1e-12)
                if clarity < clarity_threshold:
                    times[i] = (start + frame_len / 2) / sr
                    f0[i] = np.nan
                    continue

                f0_value = sr / peak_lag
                if fmin_human <= f0_value <= fmax_human:
                    f0[i] = f0_value
                else:
                    f0[i] = np.nan

                times[i] = (start + frame_len / 2) / sr

            return times, f0

        def smooth_f0(t, f0, max_jump=50):
            valid = np.isfinite(f0)
            if valid.sum() < 3:
                return t, f0

            f0_clean = f0.copy()
            for i in range(1, len(f0) - 1):
                if not np.isfinite(f0[i]):
                    continue
                prev_valid = np.isfinite(f0[i - 1])
                next_valid = np.isfinite(f0[i + 1])

                if prev_valid and next_valid:
                    if abs(f0[i] - f0[i - 1]) > max_jump and abs(f0[i] - f0[i + 1]) > max_jump:
                        f0_clean[i] = np.nan
                elif prev_valid and abs(f0[i] - f0[i - 1]) > max_jump:
                    f0_clean[i] = np.nan
                elif next_valid and abs(f0[i] - f0[i + 1]) > max_jump:
                    f0_clean[i] = np.nan
            return t, f0_clean

        t_f0, f0 = estimate_f0_autocorr(x, sr)
        t_f0, f0 = smooth_f0(t_f0, f0, max_jump=40)
        valid = np.isfinite(f0)

        if valid.sum() < 3:
            fig_f0 = make_empty_fig("Fréquence fondamentale (f0)")
        else:
            fig_f0 = go.Figure()
            fig_f0.add_trace(go.Scatter(x=t_f0[valid], y=f0[valid], mode="lines", name="f0 (Hz)"))
            fig_f0.update_layout(
                height=220,
                margin=dict(l=20, r=20, t=30, b=20),
                title="Fréquence fondamentale (f0)",
                xaxis_title="Temps (s)",
                yaxis_title="f0 (Hz)",
                uirevision="f0",
            )
            fig_f0.update_yaxes(range=[50, 350], autorange=False)

        nperseg = int(0.025 * sr)
        noverlap = int(0.015 * sr)
        f, t, Sxx = spectrogram(
            x,
            fs=sr,
            window="hann",
            nperseg=max(256, nperseg),
            noverlap=max(128, noverlap),
            scaling="spectrum",
            mode="magnitude",
        )
        Sdb = 20 * np.log10(Sxx + 1e-12)

        MAX_T = 2000
        MAX_F = 512
        if Sdb.shape[1] > MAX_T:
            step_t = int(np.ceil(Sdb.shape[1] / MAX_T))
            Sdb = Sdb[:, ::step_t]
            t = t[::step_t]
        if Sdb.shape[0] > MAX_F:
            step_f = int(np.ceil(Sdb.shape[0] / MAX_F))
            Sdb = Sdb[::step_f, :]
            f = f[::step_f]

        fig_spec = go.Figure(data=go.Heatmap(x=t, y=f, z=Sdb, colorscale="Viridis"))
        fig_spec.update_layout(
            height=220,
            margin=dict(l=20, r=20, t=30, b=20),
            title="Spectrogramme",
            xaxis_title="Temps (s)",
            yaxis_title="Fréquence (Hz)",
        )
        fig_spec.update_yaxes(range=[0, min(8000, sr / 2)])

        n = len(x)
        step = max(1, n // 5000)
        xs = x[::step]
        ts = (np.arange(0, len(xs)) * step) / sr

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=xs, mode="lines", name="audio"))
        fig.update_layout(
            height=220,
            margin=dict(l=20, r=20, t=30, b=20),
            title="Signal acoustique (audio)",
            xaxis_title="Temps (s)",
            uirevision="audio",
        )
    except Exception:
        fig_f0 = make_empty_fig("Fréquence fondamentale (f0)")
        fig_spec = make_empty_fig("Spectrogramme")
        fig = make_empty_fig("Signal acoustique (audio)")

    return info, contents, fig.to_dict(), fig_spec.to_dict(), fig_f0.to_dict()


@callback(
    Output("store-textgrid", "data"),
    Output("viewer-textgrid-info", "children"),
    Input("viewer-upload-textgrid", "contents"),
    State("viewer-upload-textgrid", "filename"),
    prevent_initial_call=True,
)
def handle_textgrid_upload(contents, filename):
    if not contents:
        return None, "Aucun TextGrid chargé"

    try:
        tg = parse_textgrid_from_string(contents)
        n_tiers = len(tg.get("tiers", []))
        n_intervals = sum(len(tier.get("intervals", [])) for tier in tg.get("tiers", []))
        return tg, f"✅ TextGrid chargé: {filename} ({n_tiers} tiers, {n_intervals} intervalles)"
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"


@callback(
    Output("viewer-measure-out", "children"),
    Input("viewer-btn-duration", "n_clicks"),
    State("viewer-session-id", "data"),
    State("viewer-upload-info", "children"),
    prevent_initial_call=True,
)
def measure_duration(n, session_id, upload_info):
    return "À brancher : stocker upload_id et appeler /measures"


@callback(
    Output("fig-audio", "figure"),
    Input("audio-time", "data"),
    Input("store-fig-audio", "data"),
)
def draw_audio_with_cursor(t, fig_dict):
    if not fig_dict:
        return make_empty_fig("Signal acoustique (audio)")

    fig = go.Figure(fig_dict)
    if t is None:
        t = 0

    fig.update_layout(
        uirevision="audio",
        shapes=[
            dict(
                type="line",
                x0=t,
                x1=t,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(width=2),
            )
        ],
    )
    return fig


@callback(
    Output("fig-spec", "figure"),
    Input("audio-time", "data"),
    Input("store-fig-spec", "data"),
)
def draw_spec_with_cursor(t, fig_dict):
    if not fig_dict or "data" not in fig_dict or len(fig_dict["data"]) == 0:
        return make_empty_fig("Spectrogramme")

    if t is None:
        t = 0

    fig = go.Figure(fig_dict)
    fig.update_layout(
        uirevision="spec",
        shapes=[
            dict(
                type="line",
                x0=t,
                x1=t,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(width=2),
            )
        ],
    )
    return fig


@callback(
    Output("fig-f0", "figure"),
    Input("audio-time", "data"),
    Input("store-fig-f0", "data"),
)
def draw_f0_with_cursor(t, fig_dict):
    if not fig_dict or "data" not in fig_dict or len(fig_dict["data"]) == 0:
        return make_empty_fig("Fréquence fondamentale (f0)")

    if t is None:
        t = 0

    fig = go.Figure(fig_dict)
    fig.update_layout(
        uirevision="f0",
        shapes=[
            dict(
                type="line",
                x0=t,
                x1=t,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(width=2),
            )
        ],
    )
    fig.update_yaxes(range=[50, 350], autorange=False)
    return fig


@callback(
    Output("fig-textgrid", "figure"),
    Input("store-textgrid", "data"),
    Input("audio-time", "data"),
)
def draw_textgrid_annotations(textgrid, t):
    fig = go.Figure()

    if not textgrid or not textgrid.get("tiers"):
        fig.update_layout(
            height=140,
            title="Annotations TextGrid (aucun)",
            xaxis_title="Temps (s)",
            yaxis=dict(showticklabels=False, range=[0, 1]),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        return fig

    tiers = textgrid["tiers"]
    xmin = float(textgrid.get("xmin", 0.0))
    xmax = float(textgrid.get("xmax", 0.0))
    n_tiers = len(tiers)

    tier_colors = [
        "rgba(100, 149, 237, 0.35)",
        "rgba(60, 179, 113, 0.35)",
        "rgba(255, 165, 0, 0.35)",
        "rgba(186, 85, 211, 0.35)",
        "rgba(220, 20, 60, 0.35)",
        "rgba(0, 206, 209, 0.35)",
    ]
    empty_color = "rgba(180, 180, 180, 0.18)"

    # axis scaffolding
    fig.add_trace(
        go.Scatter(
            x=[xmin, xmax],
            y=[0, n_tiers],
            mode="markers",
            marker=dict(size=0, opacity=0),
            showlegend=False,
            hoverinfo="none",
        )
    )

    for row, tier in enumerate(tiers):
        tier_name = tier.get("name", f"Tier {row + 1}")
        tier_class = tier.get("class", "IntervalTier")
        y0 = n_tiers - 1 - row
        y1 = y0 + 0.85

        fig.add_annotation(
            x=xmin,
            y=y0 + 0.425,
            text=tier_name,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            xref="x",
            yref="y",
        )

        color = tier_colors[row % len(tier_colors)]

        if tier_class == "IntervalTier":
            for interval in tier.get("intervals", []):
                ix0 = float(interval["xmin"])
                ix1 = float(interval["xmax"])
                label = interval.get("text", "")
                is_empty = (label or "").strip() == ""

                fig.add_shape(
                    type="rect",
                    x0=ix0,
                    x1=ix1,
                    y0=y0,
                    y1=y1,
                    fillcolor=empty_color if is_empty else color,
                    line=dict(width=1, color="black"),
                    layer="below",
                )

                if not is_empty:
                    mid_x = (ix0 + ix1) / 2
                    fig.add_annotation(
                        x=mid_x,
                        y=y0 + 0.425,
                        text=label,
                        showarrow=False,
                        font=dict(size=9, color="black"),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        xref="x",
                        yref="y",
                    )

        elif tier_class == "TextTier":
            for pt in tier.get("points", []):
                xpt = float(pt["time"])
                mark = (pt.get("mark", "") or "").strip()
                fig.add_shape(
                    type="line",
                    x0=xpt,
                    x1=xpt,
                    y0=y0,
                    y1=y1,
                    line=dict(width=2, color="black"),
                )
                if mark:
                    fig.add_annotation(
                        x=xpt,
                        y=y0 + 0.425,
                        text=mark,
                        showarrow=False,
                        font=dict(size=9, color="black"),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                    )

    if t is not None:
        fig.add_shape(
            type="line",
            x0=t,
            x1=t,
            y0=0,
            y1=n_tiers,
            line=dict(width=2, color="black"),
            xref="x",
            yref="y",
        )

    fig.update_layout(
        height=max(160, 70 + 45 * n_tiers),
        title="Annotations TextGrid",
        xaxis_title="Temps (s)",
        xaxis=dict(range=[xmin, xmax]),
        yaxis=dict(
            showticklabels=False,
            range=[0, n_tiers],
            fixedrange=True,
        ),
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig