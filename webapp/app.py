import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from PIL import Image
import io
import base64

app = dash.Dash(__name__)
server = app.server

# Reference the external CSS file
app.css.append_css({"external_url": "styles.css"})
# app.css.append_css({"external_url": "header.css"})
# app.css.append_css({"external_url": "typography.css"})

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
    html.Button("Submit", id="submit-button", className="box-title"),
    html.Div([  # Image containers side by side
        html.Div([  # Image box for "pic1"
            html.Div("PDFA Underlying Graph", className="image-box"),  # Title
            html.Img(id="image1", width=300),
        ], className="image-container"),
        html.Div([  # Image box for "pic2"
            html.Div("PDFA Preference Graph", className="image-box"),  # Title
            html.Img(id="image2", width=300),
        ], className="image-container"),
    ], className="image-boxes"),
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


@app.callback(
    [Output("image1", "src"), Output("image2", "src")],
    Input("submit-button", "n_clicks"),
    [dash.dependencies.State("text-input", "value"),
     dash.dependencies.State("text-input-right", "value"),  # Include the second text area
     dash.dependencies.State("radio-buttons", "value")]  # Include the radio button value
)
def update_images(n_clicks, text, text_right, radio_value):
    print(text, text_right, radio_value)
    if n_clicks is not None and text:
        image1, image2 = cb_func(text)
        return "data:image/png;base64,{}".format(image1), "data:image/png;base64,{}".format(image2)
    return "", ""


if __name__ == '__main__':
    app.run_server(debug=True)
