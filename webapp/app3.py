"""
TODO. Logging
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Imports
import ast
import base64
import dash
import dash_bootstrap_components as dbc
import json
import pygraphviz
import seaborn as sns
import translate

from dash import html
from dash import dcc
from loguru import logger

# Create dash app
app = dash.Dash(
    __name__,
    # requests_pathname_prefix="/home/prefltlf",
    external_stylesheets=[dbc.themes.GRID, dbc.themes.COSMO, "styles.css"]
)

# Set up server for gunicorn
server = app.server
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

# ======================================================================================================================
# COMPONENTS FOR LAYOUT
# ======================================================================================================================

# Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink(
                "Docs",
                href="/docs",
                style={"color": "white"}
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "GitHub",
                href="https://github.com/abhibp1993/prefltlf2pdfa/",
                style={"color": "white"}
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "About",
                href="/about",
                style={"color": "white"}
            )
        ),
    ],
    brand="prefltlf2pdfa translator",
    brand_href="/",
    color="primary",
    dark=True,
    style={
        "font-weight": "bold",
        "color": "white"
    }
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
    style={
        'width': '60%',
        'height': '200px'
    }
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
                    style={
                        'text-decoration': 'underline'
                    },
                    className="box-title"
                ),
                html.Br(),
                dcc.Textarea(
                    id='txt_alphabet',
                    placeholder=atoms_placeholder,
                    style={
                        'width': '60%',
                        'height': '200px'
                    }
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
            style={
                'text-decoration': 'underline'
            },
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
            style={
                'display': 'inline-block',
                'text-align': 'left'
            }
        ),
        html.Br(),
        html.Br(),
        dbc.Row(
            children=[
                dbc.Col(style={
                    "width": "100px"
                },
                    children=[
                        html.Label(
                            "Semantics:",
                            style={
                                'text-decoration': 'underline'
                            },
                            className="box-title",
                        )
                    ]
                ),
                dbc.Col(style={
                    "width": "True"
                },
                    children=[
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
                            style={
                                'width': 'True',
                                "text-align": 'left'
                            }  # Align text to the left within the dropdown
                        )
                    ])
            ]
        )
    ],
)

# Translate Button
translate_button = dbc.Container(
    style={
        'width': '200px',
        'height': "30px"
    },
    children=[
        dbc.Button(
            "Translate to PDFA",
            id="btn_translate",
            color="primary",
            className="mb-3",
        ),
    ]
)

translate_and_download_button = dbc.Container(
    style={
        'width': '400px',
        'height': "30px"
    },
    children=[
        dbc.Button(
            "Translate and Download PDFA Files",
            id="btn_translate_download",
            color="primary",
            className="mb-3",
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
                    children=[
                        html.Label(
                            "Semi-automaton",
                            style={'text-decoration': 'underline'},
                            className="box-title"
                        ),
                        html.Img(
                            id="img_semi_aut",
                            style={
                                'max-width': '100%',
                                'max-height': '100%',
                                'display': 'block'
                            },
                            src='https://via.placeholder.com/700'
                        ),
                    ], className="image-container"
                ),
            ]),
            dbc.Col([
                dbc.Container(
                    style={
                        'max-width': '100%',
                        'max-height': '100%',
                        'text-align': 'center'
                    },
                    children=[  # Image box for "pic2"
                        html.Label(
                            "Preference Graph",
                            style={
                                'text-decoration': 'underline'
                            },
                            className="box-title"
                        ),
                        html.Img(
                            id="img_pref_graph",
                            style={
                                'max-width': '100%',
                                'max-height': '100%',
                                'display': 'block'
                            },
                            src='https://via.placeholder.com/500'
                        ),
                    ], className="image-container"),
            ]),
        ]),
    ]
)

# Footer
footer = html.Div(
    style={
        'flexShrink': '0',
        'textAlign': 'center',
        'padding': '10px',
        'backgroundColor': '#f0f0f0'
    },
    children=[
        html.Footer("Copyright © 2024 Abhishek N. Kulkarni. All rights reserved.")
    ]
)

# ======================================================================================================================
# LAYOUT
# ======================================================================================================================
app.layout = html.Div(
    style={
        'textAlign': 'center',
        'flex': '1'
    },
    children=[
        # Navbar
        navbar,

        # Specification Input
        html.Br(),
        dbc.Row(
            html.Label(
                "PrefLTLf Specification",
                style={
                    'text-decoration': 'underline'
                },
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
                className="alert-top",
                style={
                    'borderRadius': '10px',
                    'border': '2px solid #000',
                    'padding': '20px',
                    'width': '50%',
                    'height': "auto",
                }
            )
        ]),
    ]
)


# ======================================================================================================================
# HELPER FUNCTIONS
# ======================================================================================================================
def generate_input_dict(text_spec, text_alphabet, chklist_options, ddl_semantics):
    """
    Generate dictionary of user input.

    :param text_spec:
    :param text_alphabet:
    :param chklist_options:
    :param ddl_semantics:
    :return:
    """
    # Create input dictionary
    input_dict = dict()

    # Populate dictionary
    input_dict["spec"] = text_spec
    input_dict["alphabet"] = text_alphabet

    # Extract checklist options
    if chklist_options is None:
        chklist_options = []

    input_dict["options"] = {
        "show_sa_state": "chk_sa_state" in chklist_options,
        "show_class": "chk_class" in chklist_options,
        "show_color": "chk_color" in chklist_options,
        "show_pg_state": "chk_pg_state" in chklist_options
    }

    # Extract dropdown value
    input_dict["semantics"] = ddl_semantics

    # Return value
    return input_dict


def translate_to_pdfa(input_dict):
    # Parse alphabet
    if input_dict["alphabet"]:
        stmts = (line.strip() for line in input_dict["alphabet"].split("\n"))
        stmts = [line for line in stmts if line and not line.startswith("#")]
        alphabet = [ast.literal_eval(s) for s in stmts]
    else:
        alphabet = set()

    # Parse specification and generate model
    phi = translate.PrefLTLf(input_dict["spec"], alphabet=alphabet)

    # Determine semantics function
    if input_dict["semantics"] == "semantics_ae":
        semantics = translate.semantics_forall_exists
    elif input_dict["semantics"] == "semantics_ea":
        semantics = translate.semantics_exists_forall
    elif input_dict["semantics"] == "semantics_aa":
        semantics = translate.semantics_forall_forall
    elif input_dict["semantics"] == "semantics_mp_ae":
        semantics = translate.semantics_mp_forall_exists
    elif input_dict["semantics"] == "semantics_mp_ea":
        semantics = translate.semantics_mp_exists_forall
    elif input_dict["semantics"] == "semantics_mp_aa":
        semantics = translate.semantics_mp_forall_forall
    else:
        raise ValueError("Invalid semantics selected.")

    # Translate PrefLTLf to PDFA
    pdfa = phi.translate(semantics=semantics)

    # Return PDFA
    return phi, pdfa


def render(pdfa: translate.PrefAutomaton, phi, **kwargs):
    """
    Generates images for semi-automaton and preference graph.

    :param pdfa: PrefAutomaton
    :param phi: PrefLTLf
    :param kwargs: dict of options
    :return: tuple[base64, base64] Two images as base64 encoded strings
    """

    # Extract options
    sa_state = kwargs.get("show_sa_state", False)
    sa_class = kwargs.get("show_class", False)
    sa_color = kwargs.get("show_color", False)
    pg_state = kwargs.get("show_pg_state", False)
    logger.info(f"Options: {sa_state}, {sa_class}, {sa_color}, {pg_state}")

    # State to class mapping
    if sa_class or sa_color:
        state2class = dict()
        for part_id, data in pdfa.pref_graph.nodes(data=True):
            for st in data["partition"]:
                state2class[st] = part_id

    # Create color palette
    parts = list(pdfa.pref_graph.nodes())
    if len(pdfa.pref_graph.nodes()) < 10:
        colors = sns.color_palette("pastel", n_colors=len(pdfa.pref_graph.nodes()))
    else:
        colors = sns.color_palette("viridis", n_colors=len(pdfa.pref_graph.nodes()))
    color_map = {part: colors[i] for i, part in enumerate(parts)}

    # Create graph to display semi-automaton
    dot_dfa = pygraphviz.AGraph(directed=True)

    # Add nodes to semi-automaton
    for st, name in pdfa.get_states(name=True):
        # Determine state name
        st_label = name if sa_state else st

        # Append state class if option enabled
        st_label = f"{st_label}\n[{state2class[name]}]" if sa_class else st_label

        # Add node
        if sa_color:
            color = color_map[state2class[name]]
            color = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            dot_dfa.add_node(st, **{"label": st_label, "fillcolor": color, "style": "filled"})
        else:
            dot_dfa.add_node(st, **{"label": st_label})

    # Add initial state to semi-automaton
    dot_dfa.add_node("init", **{"label": "", "shape": "plaintext"})

    # Add edges to semi-automaton
    for u, d in pdfa.transitions.items():
        for label, v in d.items():
            dot_dfa.add_edge(u, v, **{"label": label})
    dot_dfa.add_edge("init", pdfa.init_state, **{"label": ""})

    # Set drawing engine
    dot_dfa.layout(prog=kwargs.get("engine", "dot"))

    # Preference graph
    dot_pref = pygraphviz.AGraph(directed=True)

    # Add nodes to preference graph
    for n, data in pdfa.pref_graph.nodes(data=True):
        # n_label = set(phi[i] for i in range(len(phi)) if data['name'][i] == 1) if pg_state else n
        n_label = data['name'] if pg_state else n
        if sa_color:
            color = color_map[n]
            color = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        else:
            color = "white"

        dot_pref.add_node(n, **{"label": n_label, "fillcolor": color, "style": "filled"})

    # Add edges to preference graph
    for u, v in pdfa.pref_graph.edges():
        dot_pref.add_edge(u, v)

    # Set drawing engine
    dot_pref.layout(prog=kwargs.get("engine", "dot"))

    # Generate images (as bytes)
    sa = dot_dfa.draw(path=None, format="png")
    pg = dot_pref.draw(path=None, format="png")

    # Return images as base64 encoded strings
    return base64.b64encode(sa), base64.b64encode(pg)


# ======================================================================================================================
# CALLBACK FUNCTIONS
# ======================================================================================================================
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
        return 'https://via.placeholder.com/200', 'https://via.placeholder.com/200', False, "", "", ""

    # Identify which button was clicked
    changed_id = [p['prop_id'].split(".") for p in dash.callback_context.triggered][0][0]

    try:
        # Define input
        input_dict = generate_input_dict(text_spec, text_alphabet, chklist_options, ddl_semantics)
        logger.info(f"Input dictionary: {input_dict}")

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

        # Set up output as json
        output_dict = {
            "input": input_dict,
            "formula": phi.serialize(),
            "pdfa": pdfa.serialize()
        }
        logger.info(f"Output dictionary: {output_dict}")

        if changed_id == "btn_translate":
            return semi_aut, pref_graph, False, "", "", ""

        elif changed_id == "btn_translate_download":
            return (
                semi_aut,
                pref_graph,
                False,
                "",
                "",
                dict(content=f"{json.dumps(output_dict, indent=2)}", filename="hello.json")
            )

        else:
            raise ValueError("Invalid button clicked.")

    except Exception as err:
        logger.exception(f"{err}")
        return "", "", True, "danger", f"{repr(err)}", ""


@app.callback(
    dash.dependencies.Output("collapse", "is_open"),
    [dash.dependencies.Input("btn_collapse", "n_clicks")],
    [dash.dependencies.State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
