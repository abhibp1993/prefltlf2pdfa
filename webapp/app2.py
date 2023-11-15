import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import ast
import os
import dash
import dash_bootstrap_components as dbc
# import dash_core_components as dcc
import zipfile

import ioutils
import translate2
import vizutils

from dash import html
from dash import dcc
from loguru import logger
# logger.add(os.path.join("assets", "out", "app.log"), rotation="20 MB")
logger.info("Starting application...")

app = dash.Dash(
    __name__,
    # requests_pathname_prefix="/home/prefltlf",
    external_stylesheets=[dbc.themes.GRID, dbc.themes.BOOTSTRAP, "styles.css"]
)
server = app.server
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

# Reference the external CSS file
# app.css.append_css({"external_url": "styles.css"})

# Layout
app.layout = html.Div([
    # Title
    html.H1("prefltlf2pdfa translator"),
    # Author
    html.H3([
        "Author: ",
        html.A("Abhishek N. Kulkarni", href="http://www.akulkarni.me")
    ]),
    # Inputs
    html.Div(style={'height': '20px'}),
    html.Div([
        dbc.Row([
            html.Div(style={'width': '20px'}),
            dbc.Col([
                dbc.Row(html.Label("PrefLTLf Specification:", style={'text-decoration': 'underline'}, className="box-title")),
                dbc.Row(dcc.Textarea(id="text-spec", rows=10)),
            ], width=5),
            html.Div(style={'width': '20px'}),
            dbc.Col([
                dbc.Row(html.Label("Alphabet:", style={'text-decoration': 'underline'}, className="box-title"), ),
                dbc.Row(dcc.Textarea(id="text-alphabet", rows=10)),
            ], width=5),
        ]),
    ]),

    # Options
    html.Div(style={'height': '20px'}),
    html.Div([
        dbc.Row([
            html.Div(style={'width': '20px'}),
            dbc.Col([
                html.Div(style={'height': '5px'}),
                dbc.Row(html.Label("DFA to PNG Options:", style={'text-decoration': 'underline'}, className="box-title")),

                html.Div(style={'height': '5px'}),
                dbc.Row(
                    dcc.Checklist(
                        id='dfa2png-options',
                        options=[
                            {'label': 'Generate DFA Images', 'value': 'dfa2png-enable'},
                        ],
                        value=['dfa2png-enable']),
                ),

                html.Div(style={'height': '5px'}),
                dbc.Row(
                    dcc.Dropdown(
                        id='dfa2png-engine',
                        options=[
                            {'label': 'dot', 'value': 'dfa2png-dot'},
                            {'label': 'neato', 'value': 'dfa2png-neato'},
                            {'label': 'fdp', 'value': 'dfa2png-fdp'},
                        ],
                        value='dfa2png-dot',
                        style={'width': '150px'}
                    ),
                ),
                html.Div(style={'width': '20px'}),
            ]),  # End of Column
            dbc.Col([
                html.Div(style={'height': '5px'}),
                dbc.Row(html.Label("PrefLTLf Model Visualization Options:", style={'text-decoration': 'underline'},
                                   className="box-title")),

                html.Div(style={'height': '5px'}),
                dbc.Row(
                    dcc.Checklist(
                        id='spec2png-options',
                        options=[
                            {'label': 'Generate PrefLTLf Model', 'value': 'spec2png-enable'},
                            {'label': 'Show formulas in nodes', 'value': 'spec2png-show-formula'},
                        ],
                        value=['spec2png-enable']),
                ),

                html.Div(style={'height': '5px'}),
                dbc.Row(
                    dcc.Dropdown(
                        id='spec2png-engine',
                        options=[
                            {'label': 'dot', 'value': 'spec2png-dot'},
                            {'label': 'neato', 'value': 'spec2png-neato'},
                            {'label': 'fdp', 'value': 'spec2png-fdp'},
                        ],
                        value='spec2png-dot',
                        style={'width': '150px'}
                    ),
                ),
                html.Div(style={'width': '20px'}),
            ]),  # End of Column
            dbc.Col([
                html.Div(style={'height': '5px'}),
                dbc.Row(html.Label("PDFA Visualization Options:", style={'text-decoration': 'underline'}, className="box-title")),

                html.Div(style={'height': '5px'}),
                dbc.Row(
                    dcc.Checklist(
                        id='pdfa2png-options',
                        options=[
                            {'label': 'Show product state names', 'value': 'pdfa2png-state-names'},
                            {'label': 'Show preference graph class names', 'value': 'pdfa2png-node-class'},
                        ],
                        value=['pdfa2png-enable']),
                ),

                html.Div(style={'height': '5px'}),
                dbc.Row(
                    dcc.Dropdown(
                        id='pdfa2png-engine',
                        options=[
                            {'label': 'dot', 'value': 'pdfa2png-dot'},
                            {'label': 'neato', 'value': 'pdfa2png-neato'},
                            {'label': 'fdp', 'value': 'pdfa2png-fdp'},
                        ],
                        value='pdfa2png-dot',
                        style={'width': '150px'}
                    ),
                ),
                html.Div(style={'width': '20px'}),
            ]),  # End of Column
            html.Div(style={'width': '20px'}),
        ]),  # End of Row
    ]),  # End of Options

    # Submit button
    html.Div(style={'width': '20px'}),
    html.Div([
        dbc.Button("Translate", id="button-translate", className="box-title", n_clicks=0),
        html.A("Download Generated Files", id="dynamic-link", href="#", className="box-title"),
    ]),

    # Output images
    html.Div(style={'height': '20px'}),
    dbc.Row([
        dbc.Col([
            html.Div([  # Image box for "pic1"
                html.Div("PDFA Underlying Graph", className="image-box"),  # Title
                html.Img(id="image1", style={'max-width': '100%', 'max-height': '100%'}),
            ], className="image-container"),
        ]),
        dbc.Col([
            html.Div([  # Image box for "pic2"
                html.Div("PDFA Preference Graph", className="image-box"),  # Title
                html.Img(id="image2", style={'max-width': '100%', 'max-height': '100%'}),
            ], className="image-container"),
        ]),  # End of column
    ]),  # End of row

    # Show JSON output for copy-paste
    # Inputs
    html.Div(style={'height': '20px'}),
    html.Div([
        dbc.Row([
            html.Div(style={'width': '20px'}),
            dbc.Col([
                dbc.Row(
                    html.Label(
                        "Serialized PDFA (JSON):",
                        style={'text-decoration': 'underline'},
                        className="box-title",
                    ),
                ),
                dbc.Row(html.Textarea(id="output", rows=20, readOnly=True, style={'backgroundColor': 'LightGray'})),
            ], width=5),
        ]),
    ]),
    dbc.Alert(id="alert", is_open=False, duration=4000, color="danger", className="alert-top"),
])  # End of layout
logger.info("exiting layout")


@app.callback(
    [
        dash.dependencies.Output("image1", "src"),
        dash.dependencies.Output("image2", "src"),
        dash.dependencies.Output("output", "value"),
        dash.dependencies.Output("alert", "is_open"),
        dash.dependencies.Output("alert", "children"),
        dash.dependencies.Output("dynamic-link", "href"),
    ],
    [
        dash.dependencies.Input("button-translate", "n_clicks"),
    ],
    [
        dash.dependencies.State("text-spec", "value"),
        dash.dependencies.State("text-alphabet", "value"),
        dash.dependencies.State("dfa2png-options", "value"),
        dash.dependencies.State("dfa2png-engine", "value"),
        dash.dependencies.State("spec2png-options", "value"),
        dash.dependencies.State("spec2png-engine", "value"),
        dash.dependencies.State("pdfa2png-options", "value"),
        dash.dependencies.State("pdfa2png-engine", "value"),
    ]
)
def translate(n_clicks, text_spec, text_alphabet, dfa2png_options, dfa2png_engine, spec2png_options, spec2png_engine, pdfa2png_options,
              pdfa2png_engine):
    # Check if the button was clicked
    if n_clicks == 0 or n_clicks is None:
        logger.info("init button click")
        return "", "", "", False, "", ""

    try:
        # If no specification is given, terminate
        if text_spec is None:
            return "", "", "", True, dbc.Alert("No specification given", color="danger"), ""

        # Parse alphabet
        if text_alphabet is not None:
            alphabet = [ast.literal_eval(s.strip()) for s in text_alphabet.split("\n")]
        else:
            alphabet = None

        # Parse specification and generate model
        spec = translate2.PrefLTLf(text_spec, alphabet=alphabet)

        # Translate to PDFA
        pdfa = spec.translate(semantics=translate2.semantics_mp_forall_exists)

        # Generate images: determine user options
        dfa2png_enable = "dfa2png-enable" in dfa2png_options
        dfa2png_engine = dfa2png_engine.split("-")[1]

        spec2png_enable = "spec2png-enable" in spec2png_options
        spec2png_show_formula = "spec2png-show-formula" in spec2png_options
        spec2png_engine = spec2png_engine.split("-")[1]

        pdfa2png_state_names = "pdfa2png-state-names" in pdfa2png_options
        pdfa2png_node_class = "pdfa2png-node-class" in pdfa2png_options
        pdfa2png_engine = pdfa2png_engine.split("-")[1]

        # Set up folders for storing output
        out_dir = os.path.join("assets", "out")

        # Create a new folder
        existing_folders = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
        folder_indices = [int(folder.split('_')[1]) for folder in existing_folders if folder.startswith('out_')]
        next_index = max(folder_indices) + 1 if len(folder_indices) > 0 else 0
        os.mkdir(os.path.join(out_dir, f"out_{next_index}"))

        # Generate DFA images
        if dfa2png_enable:
            for i in range(len(spec.dfa)):
                vizutils.dfa2png(spec.dfa[i], os.path.join(out_dir, f"out_{next_index}", f"dfa_{i}.png"), engine=dfa2png_engine)

        # Generate PrefLTLf model image
        if spec2png_enable:
            vizutils.spec2png(spec, os.path.join(out_dir, f"out_{next_index}", f"model.png"), engine=spec2png_engine,
                              show_formula=spec2png_show_formula)

        # Save PDFA serialization
        ioutils.to_json(os.path.join(out_dir, f"out_{next_index}", f"pdfa.json"), pdfa)

        # Generate PDFA images
        vizutils.pdfa2png(
            pdfa,
            os.path.join(out_dir, f"out_{next_index}", f"pdfa.png"),
            engine=pdfa2png_engine,
            show_state_name=pdfa2png_state_names,
            show_node_class=pdfa2png_node_class
        )

        # Generate zip folder for download
        with zipfile.ZipFile(os.path.join(out_dir, f"out_{next_index}.zip"), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(os.path.join(out_dir, f"out_{next_index}")):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.join(out_dir, f"out_{next_index}"))
                    zipf.write(file_path, arcname)

        return (
            os.path.join(out_dir, f"out_{next_index}", f"pdfa_dfa.png"),
            os.path.join(out_dir, f"out_{next_index}", f"pdfa_pref_graph.png"),
            ioutils.to_json_str(pdfa),
            True,
            dbc.Alert("Translation Successful. Download link updated.", color="success"),
            os.path.join(out_dir, f"out_{next_index}.zip")
        )

    except Exception as e:
        logger.exception(str(e))
        return "", "", "", True, dbc.Alert(str(e), color="danger"), ""
