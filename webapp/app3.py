"""
TODO. Download files
    1. Parsed Input
    2. Parsed Options
    3.
TODO. Logging
"""

import base64
import json
import os
import pathlib
import sys
import seaborn as sns

import pygraphviz

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
# import dash_cytoscape as cyto
from loguru import logger



# logger.add(os.path.join("assets", "out", "app.log"), rotation="20 MB")
logger.info("Starting application...")

app = dash.Dash(
    __name__,
    # requests_pathname_prefix="/home/prefltlf",
    external_stylesheets=[dbc.themes.GRID, dbc.themes.COSMO, "styles.css"]
)
server = app.server
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

# Reference the external CSS file
# app.css.append_css({"external_url": "styles.css"})

# Components
# Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Docs", href="/docs", style={"color": "white"})),
        dbc.NavItem(
            dbc.NavLink(
                "GitHub",
                href="https://github.com/abhibp1993/prefltlf2pdfa/",
                style={"color": "white"}
            )
        ),
        dbc.NavItem(dbc.NavLink("About", href="/about", style={"color": "white"})),
    ],
    brand="prefltlf2pdfa translator",
    brand_href="/",
    color="primary",
    # color="#A7C7E7",
    dark=True,
    style={"font-weight": "bold", "color": "white"}
)

# Specification Input
spec_placeholder = """An example spec:
prefltlf 3
F(a)
F(b)
F(c)
>=, 0, 1
>, 0, 2
"""
spec = dcc.Textarea(
    id='txt_spec',
    placeholder=spec_placeholder,
    style={'width': '60%', 'height': '200px'}
)

# Atoms Input
atoms_placeholder = """Provide one Python parsable set of atoms per line. For example:
set()
{"a"}
{"b"}
{"a", "c"}
"""
atoms = dbc.Container(
    [
        dbc.Button(
            "Click to define alphabet",
            id="btn_collapse",
            color="primary",
            className="mb-3",
        ),
        dbc.Collapse(
            [
                html.Label(
                    "Acceptable Symbols",
                    style={'text-decoration': 'underline'},
                    className="box-title"
                ),
                html.Br(),
                dcc.Textarea(
                    id='txt_alphabet',
                    placeholder=atoms_placeholder,
                    style={'width': '60%', 'height': '200px'}
                )
            ],
            id="collapse",
        ),
    ]
)

# Options
options = dbc.Container(
    style={
        'borderRadius': '10px',
        'border': '2px solid #000',
        'padding': '20px',
        'width': '50%',
    },
    children=[
        html.Label(
            "Options",
            style={'text-decoration': 'underline'},
            className="box-title"
        ),
        html.Br(),
        dcc.Checklist(
            id='chklist_options',
            options=[
                {'label': '  Semi-automaton: Show components', 'value': 'chk_sa_state'},
                {'label': '  Semi-automaton: Show preference partition', 'value': 'chk_class'},
                {'label': '  Colored partitions', 'value': 'chk_color'},
                {'label': '  Preference Graph: Show components', 'value': 'chk_pg_state'}
            ],
            style={'display': 'inline-block', 'text-align': 'left'}
        ),
        html.Br(),
        html.Br(),
        dbc.Row(
            children=[
                dbc.Col(style={"width": "100px"}, children=[
                    html.Label(
                        "Semantics:",
                        style={'text-decoration': 'underline'},
                        className="box-title",
                    )
                ]),
                dbc.Col(style={"width": "True"}, children=[
                    dcc.Dropdown(
                        id='ddl_semantics',
                        options=[
                            {'label': 'forall-exists', 'value': 'semantics_ae'},
                            {'label': 'exists-forall', 'value': 'semantics_ea'},
                            {'label': 'forall-forall', 'value': 'semantics_aa'},
                            {'label': 'mp-forall-exists', 'value': 'semantics_mp_ae'},
                            {'label': 'mp-exists-forall', 'value': 'semantics_mp_ea'},
                            {'label': 'mp-forall-forall', 'value': 'semantics_mp_aa'},
                        ],
                        value=None,
                        style={'width': 'True', "text-align": 'left'}  # Align text to the left within the dropdown
                    )
                ])
            ]
        )
    ],
)

# Translate Button
translate_button = dbc.Container(
    style={'width': '200px', 'height': "30px"},
    children=[
        dbc.Button(
            "Translate to PDFA",
            id="btn_translate",
            color="primary",
            className="mb-3",
            # style={},
        ),
    ]
)

translate_and_download_button = dbc.Container(
    style={'width': '400px', 'height': "30px"},
    children=[
        dbc.Button(
            "Translate and Download PDFA Files",
            id="btn_translate_download",
            color="primary",
            className="mb-3",
            # style={},
        ),
        dcc.Download(id="download-json"),
    ]
)

# Output images
semi_aut = dbc.Container(
    style={'width': '80%', 'height': '600px'},
    children=[
        dbc.Row([
            dbc.Col([
                dbc.Container(
                    style={'max-width': '100%', 'max-height': '100%', 'text-align': 'center'},
                    children=[  # Image box for "pic1"
                        html.Label(
                            "Semi-automaton",
                            style={'text-decoration': 'underline'},
                            className="box-title"
                        ),
                        html.Img(
                            id="img_semi_aut",
                            style={'max-width': '100%', 'max-height': '100%', 'display': 'block'},
                            src='https://via.placeholder.com/700'
                        ),
                    ], className="image-container"
                ),
            ]),
            dbc.Col([
                dbc.Container(
                    style={'max-width': '100%', 'max-height': '100%', 'text-align': 'center'},
                    children=[  # Image box for "pic2"
                        html.Label(
                            "Preference Graph",
                            style={'text-decoration': 'underline'},
                            className="box-title"
                        ),
                        html.Img(
                            id="img_pref_graph",
                            style={'max-width': '100%', 'max-height': '100%', 'display': 'block'},
                            src='https://via.placeholder.com/500'
                        ),
                    ], className="image-container"),
            ]),  # End of column
        ]),  # End of row
    ]
)

# Footer
footer = html.Div(
    style={'flexShrink': '0', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f0f0f0'},
    children=[
        html.Footer("Copyright © 2024 Abhishek N. Kulkarni. All rights reserved.")
    ]
)

# ======================================================================================================================
# Layout
app.layout = html.Div(
    style={'textAlign': 'center', 'flex': '1'},
    children=[
        # Navbar
        navbar,

        # Specification Input
        html.Br(),
        dbc.Row(
            html.Label(
                "PrefLTLf Specification",
                style={'text-decoration': 'underline'},
                className="box-title"
            ),
        ),
        spec,

        # Atoms
        html.Br(),
        atoms,

        # Options
        html.Br(),
        options,

        # Options
        html.Br(),
        dbc.Row([
            translate_button,
            translate_and_download_button
        ]),
        # dbc.ButtonGroup(
        #     [
        #         translate_button,
        #         translate_and_download_button
        #     ]
        # ),

        # Semi-automaton
        html.Br(),
        html.Br(),
        html.Br(),
        semi_aut,

        # Footer
        html.Br(),
        footer,

        # Alert
        html.Div([
            dbc.Alert(
                id="alert",
                is_open=False,
                dismissable=True,
                # color="danger",
                className="alert-top",
                style={
                    'borderRadius': '10px',
                    'border': '2px solid #000',
                    'padding': '20px',
                    'width': '50%',
                    'height': "auto",
                    # 'opacity': "1",
                }
            )
        ]),
    ]
)


def generate_input_dict(text_spec, text_alphabet, chklist_options, ddl_semantics):
    input_dict = dict()
    input_dict["spec"] = text_spec
    input_dict["alphabet"] = text_alphabet

    if chklist_options is None:
        chklist_options = []

    input_dict["options"] = {
        "show_sa_state": "chk_sa_state" in chklist_options,
        "show_class": "chk_class" in chklist_options,
        "show_color": "chk_color" in chklist_options,
        "show_pg_state": "chk_pg_state" in chklist_options
    }

    input_dict["semantics"] = ddl_semantics

    return input_dict


def render2(pdfa: translate2.PrefAutomaton, **kwargs):
    pref_graph = pdfa.pref_graph

    # Create graph for underlying product DFA
    dot_dfa = pygraphviz.AGraph(directed=True)
    for st, name in pdfa.get_states(name=True):
        if kwargs.get("show_state_name", True):
            dot_dfa.add_node(st, **{"label": name})
        else:
            dot_dfa.add_node(st, **{"label": st})

    dot_dfa.add_node("init", **{"label": "", "shape": "plaintext"})

    for u, d in pdfa.transitions.items():
        for label, v in d.items():
            dot_dfa.add_edge(u, v, **{"label": label})
    dot_dfa.add_edge("init", pdfa.init_state, **{"label": ""})

    # Set drawing engine
    dot_dfa.layout(prog=kwargs.get("engine", "dot"))

    # Preference graph
    dot_pref = pygraphviz.AGraph(directed=True)
    for n, data in pref_graph.nodes(data=True):
        if kwargs.get("show_node_class", True):
            dot_pref.add_node(n, **{"label": data['name']})
        else:
            dot_pref.add_node(n, **{"label": n})

    for u, v in pref_graph.edges():
        dot_pref.add_edge(u, v)

    dot_pref.layout(prog=kwargs.get("engine", "dot"))

    sa = dot_dfa.draw(path=None, format="png")
    pg = dot_pref.draw(path=None, format="png")
    return base64.b64encode(sa), base64.b64encode(pg)


def render(pdfa: translate2.PrefAutomaton, phi, **kwargs):
    # Extract options
    sa_state = kwargs.get("show_sa_state", False)
    sa_class = kwargs.get("show_class", False)
    sa_color = kwargs.get("show_color", False)
    pg_state = kwargs.get("show_pg_state", False)
    logger.debug(f"Options: {sa_state}, {sa_class}, {sa_color}, {pg_state}")

    # State to class mapping
    if sa_class or sa_color:
        state2class = dict()
        for part_id, data in pdfa.pref_graph.nodes(data=True):
            for st in data["partition"]:
                state2class[st] = part_id

    # Create color palette
    parts = list(pdfa.pref_graph.nodes())
    colors = sns.color_palette("pastel", len(pdfa.pref_graph.nodes()))
    color_map = {part: colors[i] for i, part in enumerate(parts)}
    print(color_map)

    # Create graph to display semi-automaton
    dot_dfa = pygraphviz.AGraph(directed=True)
    for st, name in pdfa.get_states(name=True):
        # Determine state name
        st_label = name if sa_state else st
        # Append state class if option enabled
        st_label = f"{st_label}\n({state2class[name]})" if sa_class else st_label
        # Add node
        if sa_color:
            color = color_map[state2class[name]]
            color = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            dot_dfa.add_node(st, **{"label": st_label, "fillcolor": color, "style": "filled"})
        else:
            dot_dfa.add_node(st, **{"label": st_label})

    dot_dfa.add_node("init", **{"label": "", "shape": "plaintext"})

    for u, d in pdfa.transitions.items():
        for label, v in d.items():
            dot_dfa.add_edge(u, v, **{"label": label})
    dot_dfa.add_edge("init", pdfa.init_state, **{"label": ""})

    # Set drawing engine
    dot_dfa.layout(prog=kwargs.get("engine", "dot"))

    # Preference graph
    dot_pref = pygraphviz.AGraph(directed=True)
    for n, data in pdfa.pref_graph.nodes(data=True):
        n_label = set(phi[i] for i in range(len(phi)) if data['name'][i] == 1) if pg_state else n
        if sa_color:
            color = color_map[n]
            color = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        else:
            color = "white"

        dot_pref.add_node(n, **{"label": n_label, "fillcolor": color, "style": "filled"})

    for u, v in pdfa.pref_graph.edges():
        dot_pref.add_edge(u, v)

    dot_pref.layout(prog=kwargs.get("engine", "dot"))

    sa = dot_dfa.draw(path=None, format="png")
    pg = dot_pref.draw(path=None, format="png")
    return base64.b64encode(sa), base64.b64encode(pg)


def translate_to_pdfa(input_dict):
    # Parse alphabet
    if input_dict["alphabet"]:
        alphabet = [ast.literal_eval(s.strip()) for s in input_dict["alphabet"].split("\n")]
    else:
        alphabet = set()

    # Parse specification and generate model
    phi = translate2.PrefLTLf(input_dict["spec"], alphabet=alphabet)

    # Determine semantics function
    if input_dict["semantics"] == "semantics_ae":
        semantics = translate2.semantics_forall_exists
    elif input_dict["semantics"] == "semantics_ea":
        semantics = translate2.semantics_exists_forall
    elif input_dict["semantics"] == "semantics_aa":
        semantics = translate2.semantics_forall_forall
    elif input_dict["semantics"] == "semantics_mp_ae":
        semantics = translate2.semantics_mp_forall_exists
    elif input_dict["semantics"] == "semantics_mp_ea":
        semantics = translate2.semantics_mp_exists_forall
    elif input_dict["semantics"] == "semantics_mp_aa":
        semantics = translate2.semantics_mp_forall_forall
    else:
        raise ValueError("Invalid semantics selected.")

    # Translate PrefLTLf to PDFA
    pdfa = phi.translate(semantics=semantics)

    # Return PDFA
    return phi, pdfa


@app.callback(
    [
        dash.dependencies.Output("img_semi_aut", "src"),
        dash.dependencies.Output("img_pref_graph", "src"),
        dash.dependencies.Output("alert", "is_open"),
        dash.dependencies.Output("alert", "color"),
        dash.dependencies.Output("alert", "children"),
        dash.dependencies.Output("download-json", "data"),
    ],
    [
        dash.dependencies.Input("btn_translate", "n_clicks"),
        dash.dependencies.Input("btn_translate_download", "n_clicks"),
    ],
    [
        dash.dependencies.State("txt_spec", "value"),
        dash.dependencies.State("txt_alphabet", "value"),
        dash.dependencies.State("chklist_options", "value"),
        dash.dependencies.State("ddl_semantics", "value"),
    ]
)
def cb_btn_translate(
        btn_translate_clicks,
        btn_translate_download_clicks,
        text_spec,
        text_alphabet,
        chklist_options,
        ddl_semantics
):
    # Check if the button was clicked
    if (btn_translate_clicks == 0 or btn_translate_clicks is None) and \
            (btn_translate_download_clicks == 0 or btn_translate_download_clicks is None):
        logger.info("init button click")
        return 'https://via.placeholder.com/200', 'https://via.placeholder.com/200', False, "", "", ""

    # Identify which button was clicked
    changed_id = [p['prop_id'].split(".") for p in dash.callback_context.triggered][0][0]

    try:
        # Define input
        input_dict = generate_input_dict(text_spec, text_alphabet, chklist_options, ddl_semantics)

        # Input validation
        if not input_dict["spec"]:
            raise ValueError("No specification given.")
        if not input_dict["semantics"]:
            raise ValueError("No semantics selected.")

        # Generate images
        phi, pdfa = translate_to_pdfa(input_dict)
        semi_aut, pref_graph = render(pdfa, phi=phi.phi, **input_dict["options"])
        semi_aut = f"data:image/png;base64,{semi_aut.decode()}"
        pref_graph = f"data:image/png;base64,{pref_graph.decode()}"

        if changed_id == "btn_translate":
            return semi_aut, pref_graph, False, "", "", ""

        elif changed_id == "btn_translate_download":
            # Set up output as json
            output_dict = {
                "input": input_dict,
                "formula": phi.serialize(),
                "pdfa": pdfa.serialize()
            }
            # return semi_aut, pref_graph, False, "", "", output_dict
            return semi_aut, pref_graph, False, "", "", dict(content=f"{json.dumps(output_dict, indent=2)}", filename="hello.json")

        else:
            raise ValueError("Invalid button clicked.")

    except Exception as err:
        return "", "", True, "danger", f"{repr(err)}", ""


# @app.callback(
#     [
#         dash.dependencies.Output("img_semi_aut", "src"),
#         dash.dependencies.Output("img_pref_graph", "src"),
#         dash.dependencies.Output("alert", "is_open"),
#         dash.dependencies.Output("alert", "color"),
#         dash.dependencies.Output("alert", "children"),
#     ],
#     [
#         dash.dependencies.Input("btn_translate", "n_clicks"),
#     ],
#     [
#         dash.dependencies.State("txt_spec", "value"),
#         dash.dependencies.State("txt_alphabet", "value"),
#         dash.dependencies.State("chklist_options", "value"),
#         # dash.dependencies.State("ddl_semantics", "value"),
#         # dash.dependencies.State("dfa2png-engine", "value"),
#         # dash.dependencies.State("spec2png-options", "value"),
#         # dash.dependencies.State("spec2png-engine", "value"),
#         # dash.dependencies.State("pdfa2png-options", "value"),
#         # dash.dependencies.State("pdfa2png-engine", "value"),
#     ]
# )
# def translate(n_clicks, text_spec, text_alphabet, chklist_options, ddl_semantics):
# # def translate(n_clicks, text_spec, text_alphabet, chklist_options):
#     print("test")
#     # Check if the button was clicked
#     if n_clicks == 0 or n_clicks is None:
#         logger.info("init button click")
#         return "", "", False, "", ""
#
#     try:
#         # Define input
#         print(text_spec, text_alphabet, chklist_options)
#         # input_dict = generate_input_dict(text_spec, text_alphabet, chklist_options, ddl_semantics)
#         input_dict = generate_input_dict(text_spec, text_alphabet, chklist_options, None)
#         return "", "", True, "success", f"{input_dict}"
#
#     except Exception as err:
#         return "", "", True, "danger", f"{repr(err)}"
#         # return "", "", True, "danger", dbc.Alert(f"{repr(err)}", color="danger")


#
# @app.callback(
#     [
#         dash.dependencies.Output("img_semi_aut", "src"),
#         dash.dependencies.Output("img_pref_graph", "src"),
#         # dash.dependencies.Output("output", "value"),
#         dash.dependencies.Output("alert", "is_open"),
#         dash.dependencies.Output("alert", "color"),
#         dash.dependencies.Output("alert", "children"),
#         # dash.dependencies.Output("dynamic-link", "href"),
#     ],
#     [
#         dash.dependencies.Input("btn_translate_and_download", "n_clicks"),
#     ],
#     [
#         dash.dependencies.State("txt_spec", "value"),
#         dash.dependencies.State("txt_alphabet", "value"),
#         dash.dependencies.State("chklist_options", "value"),
#         dash.dependencies.State("ddl_semantics", "value"),
#         # dash.dependencies.State("dfa2png-engine", "value"),
#         # dash.dependencies.State("spec2png-options", "value"),
#         # dash.dependencies.State("spec2png-engine", "value"),
#         # dash.dependencies.State("pdfa2png-options", "value"),
#         # dash.dependencies.State("pdfa2png-engine", "value"),
#     ]
# )
# def translate_and_download(n_clicks, text_spec, text_alphabet, chklist_options, ddl_semantics):
#     # Check if the button was clicked
#     if n_clicks == 0 or n_clicks is None:
#         logger.info("init button click")
#         return "", "", False, "", ""
#
#     try:
#         # Define input
#         input_dict = generate_input_dict(text_spec, text_alphabet, chklist_options, ddl_semantics)
#         return "", "", True, "success", f"{input_dict}"
#
#     except Exception as err:
#         return "", "", True, "danger", f"{repr(err)}"
#         # return "", "", True, "danger", dbc.Alert(f"{repr(err)}", color="danger")
#
#     # try:
#     #     # If no specification is given, terminate
#     #     if text_spec is None:
#     #         return "", "", "", True, dbc.Alert("No specification given", color="danger"), ""
#     #
#     #     # Parse alphabet
#     #     if text_alphabet is not None:
#     #         alphabet = [ast.literal_eval(s.strip()) for s in text_alphabet.split("\n")]
#     #     else:
#     #         alphabet = None
#     #
#     #     # Parse specification and generate model
#     #     spec = translate2.PrefLTLf(text_spec, alphabet=alphabet)
#     #
#     #     # Translate to PDFA
#     #     pdfa = spec.translate(semantics=translate2.semantics_mp_forall_exists)
#     #
#     #     # Generate images: determine user options
#     #     dfa2png_enable = "dfa2png-enable" in dfa2png_options
#     #     dfa2png_engine = dfa2png_engine.split("-")[1]
#     #
#     #     spec2png_enable = "spec2png-enable" in spec2png_options
#     #     spec2png_show_formula = "spec2png-show-formula" in spec2png_options
#     #     spec2png_engine = spec2png_engine.split("-")[1]
#     #
#     #     pdfa2png_state_names = "pdfa2png-state-names" in pdfa2png_options
#     #     pdfa2png_node_class = "pdfa2png-node-class" in pdfa2png_options
#     #     pdfa2png_engine = pdfa2png_engine.split("-")[1]
#     #
#     #     # Set up folders for storing output
#     #     out_dir = os.path.join("assets", "out")
#     #
#     #     # Create a new folder
#     #     existing_folders = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
#     #     folder_indices = [int(folder.split('_')[1]) for folder in existing_folders if folder.startswith('out_')]
#     #     next_index = max(folder_indices) + 1 if len(folder_indices) > 0 else 0
#     #     os.mkdir(os.path.join(out_dir, f"out_{next_index}"))
#     #
#     #     # Generate DFA images
#     #     if dfa2png_enable:
#     #         for i in range(len(spec.dfa)):
#     #             vizutils.dfa2png(spec.dfa[i], os.path.join(out_dir, f"out_{next_index}", f"dfa_{i}.png"),
#     #                              engine=dfa2png_engine)
#     #
#     #     # Generate PrefLTLf model image
#     #     if spec2png_enable:
#     #         vizutils.spec2png(spec, os.path.join(out_dir, f"out_{next_index}", f"model.png"), engine=spec2png_engine,
#     #                           show_formula=spec2png_show_formula)
#     #
#     #     # Save PDFA serialization
#     #     ioutils.to_json(os.path.join(out_dir, f"out_{next_index}", f"pdfa.json"), pdfa)
#     #
#     #     # Generate PDFA images
#     #     vizutils.pdfa2png(
#     #         pdfa,
#     #         os.path.join(out_dir, f"out_{next_index}", f"pdfa.png"),
#     #         engine=pdfa2png_engine,
#     #         show_state_name=pdfa2png_state_names,
#     #         show_node_class=pdfa2png_node_class
#     #     )
#     #
#     #     # Generate zip folder for download
#     #     with zipfile.ZipFile(os.path.join(out_dir, f"out_{next_index}.zip"), 'w', zipfile.ZIP_DEFLATED) as zipf:
#     #         for root, _, files in os.walk(os.path.join(out_dir, f"out_{next_index}")):
#     #             for file in files:
#     #                 file_path = os.path.join(root, file)
#     #                 arcname = os.path.relpath(file_path, os.path.join(out_dir, f"out_{next_index}"))
#     #                 zipf.write(file_path, arcname)
#     #
#     #     return (
#     #         os.path.join(out_dir, f"out_{next_index}", f"pdfa_dfa.png"),
#     #         os.path.join(out_dir, f"out_{next_index}", f"pdfa_pref_graph.png"),
#     #         ioutils.to_json_str(pdfa),
#     #         True,
#     #         dbc.Alert("Translation Successful. Download link updated.", color="success"),
#     #         os.path.join(out_dir, f"out_{next_index}.zip")
#     #     )
#     #
#     # except Exception as e:
#     #     logger.exception(str(e))
#     #     return "", "", "", True, dbc.Alert(str(e), color="danger"), ""
#

@app.callback(
    dash.dependencies.Output("collapse", "is_open"),
    [dash.dependencies.Input("btn_collapse", "n_clicks")],
    [dash.dependencies.State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
