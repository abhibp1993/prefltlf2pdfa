import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from PIL import Image
import io
import base64

import ioutils
from translate import *
from loguru import logger
import time

app = dash.Dash(__name__)
server = app.server

# Reference the external CSS file
app.css.append_css({"external_url": "styles.css"})
# app.css.append_css({"external_url": "header.css"})
# app.css.append_css({"external_url": "typography.css"})

# Create a download link for the images
download_zip = dcc.Download(id="download_output")


app.layout = html.Div([
    html.H1("prefltlf2pdfa translator"),
    html.Div([  # Text area container
        html.Label("Author: "),
        dcc.Link("Abhishek N. Kulkarni", href="https://www.akulkarni.me"),
    ], className="author"),
    html.Label("PrefLTLf Specification", className="box-title"),
    html.Div([  # Text area container
        dcc.Textarea(id="text-input", placeholder="Enter text here", rows=4, cols=50),
    ], className="text-area-container"),
    html.Label("Alphabet (optional)", className="box-title"),
    html.Div([  # Text area container
        dcc.Textarea(id="text-input-right", placeholder="Enter text here", rows=4, cols=50),
    ], className="text-area-container"),
    html.Div([  # Container div for "Semantics" and radio buttons
        html.Div("Semantics", className="box-title"),  # Title
        dcc.RadioItems(
            id="radio-buttons",
            options=[
                {'label': 'forall_exists', 'value': 'forall_exists'},
                {'label': 'exists_forall 2', 'value': 'exists_forall'},
                {'label': 'forall_forall', 'value': 'forall_forall'},
                {'label': 'mp_forall_exists', 'value': 'mp_forall_exists'},
                {'label': 'mp_exists_forall', 'value': 'mp_exists_forall'},
                {'label': 'mp_forall_forall', 'value': 'mp_forall_forall'},
            ],
            labelStyle={'display': 'block'},
            value="forall_exists"
        ),
    ], className="box"),
    html.Button("Translate", id="submit-button", className="box-title"),
    html.Button("Translate and Download", id="submit-and-download-button", className="box-title"),
    # dcc.Download(id="download-text"),  # Include the download link in the layout
    # html.Div([  # Image containers side by side
        html.Div([  # Image box for "pic1"
            html.Div("PDFA Underlying Graph", className="image-box"),  # Title
            html.Img(id="image1", style={'max-width': '100%', 'max-height': '100%'}),
        ], className="image-container"),
        html.Div([  # Image box for "pic2"
            html.Div("PDFA Preference Graph", className="image-box"),  # Title
            html.Img(id="image2", style={'max-width': '100%', 'max-height': '100%'}),
        ], className="image-container"),
    # ], className="image-boxes"),
    html.Label("pdfa.json", className="box-title"),
    html.Div([  # Text area container
        # html.Label("pdfa.json"),
        dcc.Textarea(id="text-output", placeholder="Enter text here", rows=4, cols=50, readOnly=True),
    ], className="author"),
    html.Label("error messages", className="box-title"),
    html.Div([  # Text area container
        # html.Label("pdfa.json"),
        dcc.Textarea(id="text-error", placeholder="Enter text here", rows=4, cols=50, readOnly=True),
    ], className="author"),
    download_zip,
])


# Function to generate images
def cb_func(text):
    # Your image generation logic here (simplified example)
    image1 = Image.new("RGB", (300, 300), "red")
    image2 = Image.new("RGB", (300, 300), "blue")

    # Convert images to bytes
    image1_bytes = io.BytesIO()
    image2_bytes = io.BytesIO()
    image1.save(image1_bytes, format="PNG")
    image2.save(image2_bytes, format="PNG")

    # Encode images as base64 for display
    image1_base64 = base64.b64encode(image1_bytes.getvalue()).decode('utf-8')
    image2_base64 = base64.b64encode(image2_bytes.getvalue()).decode('utf-8')

    return image1_base64, image2_base64


# @app.callback(
#     [
#         Output("image1", "src"),
#         Output("image2", "src"),
#         Output("text-output", "value"),
#         Output("text-error", "value"),
#         Output("download_output", "data")
#     ],
#     Input("submit-button", "n_clicks"),
#     [dash.dependencies.State("text-input", "value"),
#      dash.dependencies.State("text-input-right", "value"),  # Include the second text area
#      dash.dependencies.State("radio-buttons", "value") # Include the radio button value
#      ]
# )
# def update_images(n_clicks, text, text_right, radio_value):
#     print(text, text_right, radio_value)
#     if n_clicks is not None and text:
#         # image1, image2 = cb_func(text)
#         image1 = "assets/dfa_0.png"
#         image2 = "/assets/dfa_1.png"
#         # return "data:image/png;base64,{}".format(image1), "data:image/png;base64,{}".format(image2), "output", "error"
#         return image1, image2, "output", "error", dcc.send_file(image1)
#     return "", "", "", "", None


@app.callback(
    [
        Output("image1", "src"),
        Output("image2", "src"),
        Output("text-output", "value"),
        Output("text-error", "value")
    ],
    Input("submit-button", "n_clicks"),
    [dash.dependencies.State("text-input", "value"),
     dash.dependencies.State("text-input-right", "value"),  # Include the second text area
     dash.dependencies.State("radio-buttons", "value") # Include the radio button value
     ]
)
def update_images(n_clicks, formula, alphabet, semantics):

    if n_clicks is not None and formula:
        base_dir = pathlib.Path(__file__).parent
        out_dir = os.path.join(base_dir, "assets", "out")

        # Find existing folders and get their indices
        existing_folders = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
        folder_indices = [int(folder.split('_')[1]) for folder in existing_folders if folder.startswith('out_')]
        next_index = max(folder_indices) + 1 if len(folder_indices) > 0 else 0
        os.mkdir(os.path.join(out_dir, f"out_{next_index}"))
        ifiles = os.path.join(out_dir, f"out_{next_index}")

        formula = formula.split("\n")
        formula, phi = parse_prefltlf(formula)
        with open(os.path.join(ifiles, "formula.txt"), 'w') as f:
            if len(phi) == 0:
                f.write("prefltlf\n")
                for triple in formula:
                    f.write(",".join((str(e) for e in triple)) + "\n")
            else:
                f.write(f"prefltlf {len(phi)}\n")
                for ltlf in phi:
                    f.write(f"{ltlf}\n")
                for triple in formula:
                    f.write(",".join((str(e) for e in triple)) + "\n")

        # Start timer
        start_time = time.time()

        # Build preference model
        model = build_prefltlf_model(formula, phi)
        model = index_model(model)

        logger.info(f"Model: \n{prettystring_prefltlf_model(model)}")
        with open(os.path.join(ifiles, "model.txt"), 'w') as f:
            atoms, phi, preorder = model
            f.write("prefltlf model\n")
            f.write("atoms: " + ",".join(atoms) + "\n")
            f.write("phi: ")
            for idx in range(len(phi)):
                if idx > 0:
                    f.write(",")
                f.write(f"{idx}:{phi[idx]}")
            f.write("\n")
            for element in preorder:
                f.write(",".join((str(e) for e in element)) + "\n")

        # Translate to PDFA
        debug = True
        if semantics == "forall_exists":
            pdfa = translate(model, semantics=semantics_forall_exists, **{"debug": debug, "ifiles": ifiles})
        elif semantics == "exists_forall":
            pdfa = translate(model, semantics=semantics_exists_forall, **{"debug": debug, "ifiles": ifiles})
        elif semantics == "forall_forall":
            pdfa = translate(model, semantics=semantics_forall_forall, **{"debug": debug, "ifiles": ifiles})
        elif semantics == "mp_forall_exists":
            pdfa = translate(model, semantics=semantics_mp_forall_forall, **{"debug": debug, "ifiles": ifiles})
        elif semantics == "mp_forall_forall":
            pdfa = translate(model, semantics=semantics_mp_forall_forall, **{"debug": debug, "ifiles": ifiles})
        elif semantics == "mp_exists_forall":
            pdfa = translate(model, semantics=semantics_mp_forall_forall, **{"debug": debug, "ifiles": ifiles})
        else:
            raise ValueError("Semantics must be one of forall_exists, exists_forall, forall_forall, "
                             f"mp_forall_exists, mp_forall_forall, mp_exists_forall. {semantics} is unsupported.")
        # pdfa = translate(model, semantics=mp_semantics, **{"debug": debug, "ifiles": ifiles})

        # Stop timer
        end_time = time.time()

        # Save PDFA to intermediate files
        if ifiles:
            ioutils.to_json(os.path.join(ifiles, "pdfa.json"), pdfa)

        # Save files as per flags
        ioutils.to_json(os.path.join(ifiles, "pdfa.json"), pdfa)
        pdfa_to_png(pdfa, os.path.join(ifiles, "out.png"))

        # Print to stdout
        logger.info(f"====== TRANSLATION COMPLETED IN {round((end_time - start_time) * 10 ** 3, 4)} MILLISECONDS =====")
        logger.info(prettystring_pdfa(pdfa))

        out_str = f"====== TRANSLATION COMPLETED IN {round((end_time - start_time) * 10 ** 3, 4)} MILLISECONDS =====\n\n"
        out_str += prettystring_pdfa(pdfa)

        # return "/" + str(os.path.join("out", f"out_{next_index}", "out_dfa.png")), os.path.join("out", f"out_{next_index}", "out_pref_graph.png"), out_str, "error"
        return f"/assets/out/out_{next_index}/out_dfa.png", f"/assets/out/out_{next_index}/out_pref_graph.png", out_str, "error"

    return "", "", "", ""


if __name__ == '__main__':
    app.run_server(debug=True)
