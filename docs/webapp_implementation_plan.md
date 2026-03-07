# PrefLTLfSpec Web IDE Implementation Plan

**Date:** March 7, 2026  
**Goal:** Build a web-based IDE for PrefLTLfSpec DSL with live parsing, error annotations, and PDFA generation.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Dash Web App                             │
│  ┌───────────────────────────────────────┐  ┌────────────────┐  │
│  │         dash-ace Editor               │  │   Output Pane  │  │
│  │  • Syntax highlighting                │  │   • PDFA viz   │  │
│  │  • Live error squiggles               │  │   • Logs       │  │
│  │  • Line numbers                       │  │   • Errors     │  │
│  └───────────────────────────────────────┘  └────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  [Run ▶]   [Clear]   [Load Example ▼]   [Export .prefltlf]│  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                                         ▲
         │ on keystroke (debounced)                │
         ▼                                         │
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (Flask/Dash)                        │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ Lark Parser │───▶│ Semantic     │───▶│ Lowering +        │  │
│  │ (grammar)   │    │ Analyzer     │    │ PrefLTLf.translate│  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
│         │                  │                      │             │
│         ▼                  ▼                      ▼             │
│   Parse errors      Semantic errors         PDFA result         │
│   (line, col)       (line, message)         (JSON/graph)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Lark Parser + Semantic Analyzer

### 1.1 Grammar File (`grammar.lark`)

```lark
// PrefLTLfSpec Grammar
// Compatible with language spec v2 (March 2026)

start: propositions_block alphabet_block? options_block? formulas_block preferences_block

// ═══════════════════════════════════════════════════════════════
// PROPOSITIONS BLOCK
// ═══════════════════════════════════════════════════════════════
propositions_block: "propositions" proposition_items "end" "propositions"
proposition_items: proposition_line+
proposition_line: IDENT ("," IDENT)*

// ═══════════════════════════════════════════════════════════════
// ALPHABET BLOCK (optional)
// ═══════════════════════════════════════════════════════════════
alphabet_block: "alphabet" alphabet_stmt* "end" "alphabet"

alphabet_stmt: singletons_stmt
             | emptyset_stmt
             | powerset_stmt
             | exclude_stmt

singletons_stmt: "singletons" ["(" [prop_list] ")"]
emptyset_stmt: "emptyset"
powerset_stmt: "powerset" ["(" [prop_list] ")"]
exclude_stmt: "exclude" (symbol | IDENT)

prop_list: IDENT ("," IDENT)*
symbol: "{" [prop_list] "}"

// ═══════════════════════════════════════════════════════════════
// OPTIONS BLOCK (optional)
// ═══════════════════════════════════════════════════════════════
options_block: "options" option_stmt* "end" "options"

option_stmt: semantics_opt | auto_complete_opt

semantics_opt: "semantics" "=" SEMANTICS_VALUE
auto_complete_opt: "auto-complete" "=" AUTO_COMPLETE_VALUE

SEMANTICS_VALUE: "EA" | "exists-forall"
               | "AE" | "forall-exists"
               | "AA" | "forall-forall"
               | "EE" | "exists-exists"
               | "MaxEA" | "max-exists-forall"
               | "MaxAE" | "max-forall-exists"
               | "MaxAA" | "max-forall-forall"
               | "MaxEE" | "max-exists-exists"

AUTO_COMPLETE_VALUE: "incomparable" | "minimal"

// ═══════════════════════════════════════════════════════════════
// LTLf FORMULAS BLOCK
// ═══════════════════════════════════════════════════════════════
formulas_block: "ltlf-formulas" formula_decl+ "end" "ltlf-formulas"
formula_decl: IDENT ":" ltlf_expr

// LTLf expression grammar (simplified - delegates complex parsing to ltlf2dfa)
ltlf_expr: ltlf_or

ltlf_or: ltlf_and ("|" ltlf_and)*
ltlf_and: ltlf_until ("&" ltlf_until)*
ltlf_until: ltlf_unary ("U" ltlf_unary | "R" ltlf_unary)*
ltlf_unary: "!" ltlf_unary
          | "F" ltlf_unary
          | "G" ltlf_unary
          | "X" ltlf_unary
          | ltlf_atom
ltlf_atom: "true" | "false" | IDENT | "(" ltlf_expr ")"

// ═══════════════════════════════════════════════════════════════
// PREFERENCES BLOCK
// ═══════════════════════════════════════════════════════════════
preferences_block: "preferences" preference_stmt+ "end" "preferences"

preference_stmt: pref_chain | pref_verbatim

pref_chain: pref_term (PREF_OP pref_term)+
PREF_OP: ">" | "<" | ">=" | "<=" | "~" | "<>"

pref_term: IDENT | "(" ltlf_expr ")"

pref_verbatim: pref_term "is" VERB_PHRASE "to" pref_term
VERB_PHRASE: "strictly preferred"
           | "weakly preferred"
           | "indifferent"
           | "incomparable"

// ═══════════════════════════════════════════════════════════════
// TERMINALS
// ═══════════════════════════════════════════════════════════════
IDENT: /[a-zA-Z_][a-zA-Z0-9_]*/
COMMENT: /#[^\n]*/

%import common.WS
%ignore WS
%ignore COMMENT
```

### 1.2 AST Node Classes (`ast_nodes.py`)

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union
from enum import Enum

class SemanticsType(Enum):
    EA = "exists-forall"
    AE = "forall-exists"
    AA = "forall-forall"
    EE = "exists-exists"
    MAX_EA = "max-exists-forall"
    MAX_AE = "max-forall-exists"
    MAX_AA = "max-forall-forall"
    MAX_EE = "max-exists-exists"

class AutoCompleteType(Enum):
    INCOMPARABLE = "incomparable"
    MINIMAL = "minimal"

@dataclass
class SourceLocation:
    line: int
    column: int
    end_line: int = None
    end_column: int = None

@dataclass
class AlphabetStmt:
    kind: str  # "singletons", "emptyset", "powerset", "exclude"
    props: Optional[List[str]] = None  # for singletons/powerset with args
    symbol: Optional[frozenset] = None  # for exclude
    loc: SourceLocation = None

@dataclass
class OptionsBlock:
    semantics: Optional[SemanticsType] = None
    auto_complete: Optional[AutoCompleteType] = None
    loc: SourceLocation = None

@dataclass
class FormulaDecl:
    name: str
    ltlf_expr: str  # raw LTLf string for ltlf2dfa
    loc: SourceLocation = None

@dataclass
class PreferenceStmt:
    # Normalized form: list of (op, left, right) tuples
    relations: List[Tuple[str, str, str]]  # (op, left_name, right_name)
    loc: SourceLocation = None

@dataclass
class Spec:
    propositions: List[str]
    alphabet: Optional[List[AlphabetStmt]] = None
    options: Optional[OptionsBlock] = None
    formulas: Dict[str, FormulaDecl] = field(default_factory=dict)
    preferences: List[PreferenceStmt] = field(default_factory=list)
    
    # Source locations for error reporting
    prop_locs: Dict[str, SourceLocation] = field(default_factory=dict)
```

### 1.3 Semantic Analyzer (`analyzer.py`)

```python
from dataclasses import dataclass
from typing import List, Set
from ast_nodes import Spec, SourceLocation

@dataclass
class SemanticError:
    message: str
    line: int
    column: int
    end_line: int = None
    end_column: int = None
    severity: str = "error"  # "error", "warning", "info"

class SemanticAnalyzer:
    def __init__(self, spec: Spec):
        self.spec = spec
        self.errors: List[SemanticError] = []
        self.warnings: List[SemanticError] = []
        
    def analyze(self) -> List[SemanticError]:
        """Run all semantic checks. Returns list of errors."""
        self._check_duplicate_propositions()
        self._check_duplicate_formula_names()
        self._check_undefined_formula_refs()
        self._check_undefined_props_in_alphabet()
        self._check_undefined_props_in_formulas()
        self._check_preference_cycles()
        self._check_conflicting_preferences()
        return self.errors + self.warnings
    
    def _check_duplicate_propositions(self):
        seen = {}
        for prop in self.spec.propositions:
            if prop in seen:
                self.errors.append(SemanticError(
                    message=f"Duplicate proposition '{prop}'",
                    line=self.spec.prop_locs[prop].line,
                    column=self.spec.prop_locs[prop].column,
                ))
            seen[prop] = True
    
    def _check_duplicate_formula_names(self):
        # Already handled by dict, but track for better errors
        pass
    
    def _check_undefined_formula_refs(self):
        """Check that all formula names in preferences are defined."""
        defined = set(self.spec.formulas.keys())
        for pref in self.spec.preferences:
            for op, left, right in pref.relations:
                if left not in defined and not left.startswith("("):
                    self.errors.append(SemanticError(
                        message=f"Undefined formula '{left}'",
                        line=pref.loc.line,
                        column=pref.loc.column,
                    ))
                if right not in defined and not right.startswith("("):
                    self.errors.append(SemanticError(
                        message=f"Undefined formula '{right}'",
                        line=pref.loc.line,
                        column=pref.loc.column,
                    ))
    
    def _check_undefined_props_in_alphabet(self):
        """Check alphabet references only declared propositions."""
        if not self.spec.alphabet:
            return
        defined = set(self.spec.propositions)
        for stmt in self.spec.alphabet:
            if stmt.props:
                for prop in stmt.props:
                    if prop not in defined:
                        self.errors.append(SemanticError(
                            message=f"Undefined proposition '{prop}' in alphabet",
                            line=stmt.loc.line,
                            column=stmt.loc.column,
                        ))
    
    def _check_undefined_props_in_formulas(self):
        """Warn if LTLf formula uses undeclared propositions."""
        defined = set(self.spec.propositions)
        for name, decl in self.spec.formulas.items():
            # Extract identifiers from LTLf (simplified)
            # Full check delegated to ltlf2dfa
            pass
    
    def _check_preference_cycles(self):
        """Detect strict preference cycles (would make spec inconsistent)."""
        # Build graph of strict preferences
        # Check for cycles using DFS
        pass
    
    def _check_conflicting_preferences(self):
        """Detect conflicting preferences like a > b and b > a."""
        strict = set()
        for pref in self.spec.preferences:
            for op, left, right in pref.relations:
                if op == ">":
                    if (right, left) in strict:
                        self.errors.append(SemanticError(
                            message=f"Conflicting preferences: '{left} > {right}' and '{right} > {left}'",
                            line=pref.loc.line,
                            column=pref.loc.column,
                        ))
                    strict.add((left, right))
```

### 1.4 Parser Driver with Error Recovery (`parser.py`)

```python
from lark import Lark, Transformer, v_args, UnexpectedToken, UnexpectedCharacters
from lark.exceptions import VisitError
from typing import List, Tuple
from ast_nodes import *
from analyzer import SemanticAnalyzer, SemanticError

GRAMMAR_PATH = "grammar.lark"

class ParseError:
    def __init__(self, message: str, line: int, column: int, 
                 end_line: int = None, end_column: int = None):
        self.message = message
        self.line = line
        self.column = column
        self.end_line = end_line or line
        self.end_column = end_column or column

class PrefLTLfTransformer(Transformer):
    """Transforms Lark parse tree into AST nodes."""
    
    def start(self, items):
        # Combine all blocks into Spec
        spec = Spec(propositions=[])
        for item in items:
            if isinstance(item, list) and all(isinstance(x, str) for x in item):
                spec.propositions = item
            elif isinstance(item, list) and all(isinstance(x, AlphabetStmt) for x in item):
                spec.alphabet = item
            elif isinstance(item, OptionsBlock):
                spec.options = item
            elif isinstance(item, dict):
                spec.formulas = item
            elif isinstance(item, list) and all(isinstance(x, PreferenceStmt) for x in item):
                spec.preferences = item
        return spec
    
    @v_args(meta=True)
    def proposition_line(self, meta, items):
        return [str(token) for token in items]
    
    @v_args(meta=True)
    def formula_decl(self, meta, items):
        name = str(items[0])
        ltlf_str = self._reconstruct_ltlf(items[1])
        return (name, FormulaDecl(
            name=name,
            ltlf_expr=ltlf_str,
            loc=SourceLocation(meta.line, meta.column)
        ))
    
    def _reconstruct_ltlf(self, tree):
        """Reconstruct LTLf expression as string for ltlf2dfa."""
        # Flatten the parse tree back to string
        if hasattr(tree, 'children'):
            return " ".join(self._reconstruct_ltlf(c) for c in tree.children)
        return str(tree)
    
    # ... more transformer methods

class PrefLTLfParser:
    def __init__(self):
        with open(GRAMMAR_PATH, 'r') as f:
            grammar = f.read()
        self.parser = Lark(
            grammar,
            start='start',
            parser='lalr',
            propagate_positions=True,  # For line numbers
            maybe_placeholders=False,
        )
        self.transformer = PrefLTLfTransformer()
    
    def parse(self, source: str) -> Tuple[Spec, List[ParseError]]:
        """
        Parse source code and return (AST, errors).
        Returns partial AST even on errors for live feedback.
        """
        errors = []
        
        try:
            tree = self.parser.parse(source)
            spec = self.transformer.transform(tree)
            
            # Run semantic analysis
            analyzer = SemanticAnalyzer(spec)
            semantic_errors = analyzer.analyze()
            
            for err in semantic_errors:
                errors.append(ParseError(
                    message=err.message,
                    line=err.line,
                    column=err.column,
                ))
            
            return spec, errors
            
        except UnexpectedToken as e:
            errors.append(ParseError(
                message=f"Unexpected token '{e.token}'. Expected one of: {', '.join(e.expected)}",
                line=e.line,
                column=e.column,
            ))
            return None, errors
            
        except UnexpectedCharacters as e:
            errors.append(ParseError(
                message=f"Unexpected character '{source[e.pos_in_stream]}' at position {e.column}",
                line=e.line,
                column=e.column,
            ))
            return None, errors
```

---

## Component 2: Dash Web App

### 2.1 Project Structure

```
prefltlf2pdfa/
├── webapp/
│   ├── __init__.py
│   ├── app.py                 # Main Dash app
│   ├── callbacks.py           # Dash callbacks
│   ├── layout.py              # UI layout
│   ├── ace_mode_prefltlf.js   # Custom syntax highlighting
│   └── assets/
│       ├── style.css
│       └── prefltlf_mode.js   # ACE mode definition
├── prefltlf/
│   ├── __init__.py
│   ├── grammar.lark
│   ├── parser.py
│   ├── ast_nodes.py
│   ├── analyzer.py
│   └── lowering.py
└── prefltlf2pdfa/
    └── (existing code)
```

### 2.2 Main App (`app.py`)

```python
import dash
from dash import html, dcc
import dash_ace
from dash.dependencies import Input, Output, State
from layout import create_layout
from callbacks import register_callbacks

app = dash.Dash(
    __name__,
    title="PrefLTLf IDE",
    suppress_callback_exceptions=True,
)

app.layout = create_layout()
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
```

### 2.3 Layout (`layout.py`)

```python
from dash import html, dcc
import dash_ace
import dash_bootstrap_components as dbc

EXAMPLE_CODE = """propositions
  clean, charged, safe
end propositions

alphabet
  powerset()
end alphabet

options
  semantics = MaxAE
  auto-complete = minimal
end options

ltlf-formulas
  safe_first: G safe
  eventually_clean: F clean
  charge_until_clean: charged U clean
end ltlf-formulas

preferences
  safe_first > eventually_clean >= charge_until_clean
end preferences
"""

def create_layout():
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2("PrefLTLf IDE", className="text-primary"),
                html.P("Write preference specifications, get PDFAs.", className="text-muted"),
            ])
        ], className="mb-3 mt-3"),
        
        # Main content
        dbc.Row([
            # Editor pane
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("Editor", className="fw-bold"),
                        dbc.ButtonGroup([
                            dbc.Button("Run ▶", id="run-btn", color="success", size="sm"),
                            dbc.Button("Clear", id="clear-btn", color="secondary", size="sm"),
                            dbc.DropdownMenu(
                                label="Examples",
                                children=[
                                    dbc.DropdownMenuItem("Simple", id="example-simple"),
                                    dbc.DropdownMenuItem("Robot Navigation", id="example-robot"),
                                    dbc.DropdownMenuItem("Complex", id="example-complex"),
                                ],
                                size="sm",
                            ),
                        ], className="float-end"),
                    ]),
                    dbc.CardBody([
                        dash_ace.DashAceEditor(
                            id='editor',
                            value=EXAMPLE_CODE,
                            theme='monokai',
                            mode='prefltlf',  # Custom mode
                            tabSize=2,
                            fontSize=14,
                            showGutter=True,
                            showPrintMargin=False,
                            highlightActiveLine=True,
                            wrapEnabled=True,
                            style={'height': '500px', 'width': '100%'},
                            annotations=[],  # Error markers
                            markers=[],       # Inline highlights
                        ),
                    ], className="p-0"),
                ]),
                
                # Error panel
                dbc.Card([
                    dbc.CardHeader("Problems"),
                    dbc.CardBody(id="error-panel", style={'maxHeight': '150px', 'overflowY': 'auto'}),
                ], className="mt-2"),
                
            ], width=6),
            
            # Output pane
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("Output", className="fw-bold"),
                        dbc.Tabs([
                            dbc.Tab(label="PDFA Graph", tab_id="tab-graph"),
                            dbc.Tab(label="JSON", tab_id="tab-json"),
                            dbc.Tab(label="Legacy Format", tab_id="tab-legacy"),
                            dbc.Tab(label="Logs", tab_id="tab-logs"),
                        ], id="output-tabs", active_tab="tab-graph"),
                    ]),
                    dbc.CardBody([
                        html.Div(id="output-content", style={'height': '600px', 'overflowY': 'auto'}),
                    ]),
                ]),
            ], width=6),
        ]),
        
        # Hidden stores
        dcc.Store(id='parse-result'),
        dcc.Store(id='pdfa-result'),
        dcc.Interval(id='parse-interval', interval=500, n_intervals=0),  # Debounced parsing
        
    ], fluid=True)
```

### 2.4 Callbacks (`callbacks.py`)

```python
from dash import html, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import json
import traceback

from prefltlf.parser import PrefLTLfParser
from prefltlf.lowering import lower_to_legacy
from prefltlf2pdfa import PrefLTLf

parser = PrefLTLfParser()

def register_callbacks(app):
    
    @app.callback(
        [Output('parse-result', 'data'),
         Output('editor', 'annotations'),
         Output('error-panel', 'children')],
        [Input('editor', 'value')],
        prevent_initial_call=False,
    )
    def live_parse(source):
        """Parse on every keystroke (debounced by interval)."""
        if not source or not source.strip():
            return None, [], html.Span("No input", className="text-muted")
        
        spec, errors = parser.parse(source)
        
        # Convert errors to ACE annotations
        annotations = []
        error_items = []
        
        for err in errors:
            annotations.append({
                'row': err.line - 1,  # ACE is 0-indexed
                'column': err.column,
                'text': err.message,
                'type': 'error',  # 'error', 'warning', 'info'
            })
            error_items.append(
                html.Div([
                    html.Span(f"Line {err.line}: ", className="text-danger fw-bold"),
                    html.Span(err.message),
                ], className="mb-1")
            )
        
        if not errors:
            error_items = [html.Span("✓ No problems", className="text-success")]
        
        result = {
            'valid': len(errors) == 0,
            'spec': spec.to_dict() if spec else None,
        }
        
        return result, annotations, error_items
    
    @app.callback(
        [Output('pdfa-result', 'data'),
         Output('output-content', 'children')],
        [Input('run-btn', 'n_clicks')],
        [State('editor', 'value'),
         State('output-tabs', 'active_tab')],
        prevent_initial_call=True,
    )
    def run_translation(n_clicks, source, active_tab):
        """Run full translation when Run button is clicked."""
        if not n_clicks:
            raise PreventUpdate
        
        try:
            # Parse DSL
            spec, errors = parser.parse(source)
            if errors:
                return None, html.Pre(
                    "\n".join(f"Error: {e.message}" for e in errors),
                    className="text-danger"
                )
            
            # Lower to legacy format
            legacy_str, alphabet = lower_to_legacy(spec)
            
            # Run translation
            pref = PrefLTLf(spec=legacy_str, alphabet=alphabet)
            pdfa = pref.translate()
            
            # Format output based on active tab
            if active_tab == "tab-graph":
                # Return Cytoscape graph or image
                return pdfa.serialize(), render_pdfa_graph(pdfa)
            elif active_tab == "tab-json":
                return pdfa.serialize(), html.Pre(
                    json.dumps(pdfa.serialize(), indent=2),
                    style={'fontSize': '12px'}
                )
            elif active_tab == "tab-legacy":
                return pdfa.serialize(), html.Pre(legacy_str)
            else:
                return pdfa.serialize(), html.Pre("Translation complete!")
                
        except Exception as e:
            return None, html.Pre(
                f"Translation error:\n{str(e)}\n\n{traceback.format_exc()}",
                className="text-danger",
                style={'fontSize': '11px'}
            )
    
    @app.callback(
        Output('editor', 'value'),
        [Input('clear-btn', 'n_clicks'),
         Input('example-simple', 'n_clicks'),
         Input('example-robot', 'n_clicks'),
         Input('example-complex', 'n_clicks')],
        prevent_initial_call=True,
    )
    def handle_buttons(clear, ex1, ex2, ex3):
        """Handle toolbar button clicks."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger == 'clear-btn':
            return ""
        elif trigger == 'example-simple':
            return EXAMPLE_SIMPLE
        elif trigger == 'example-robot':
            return EXAMPLE_ROBOT
        elif trigger == 'example-complex':
            return EXAMPLE_COMPLEX
        
        raise PreventUpdate


def render_pdfa_graph(pdfa):
    """Render PDFA as interactive graph using Cytoscape or Graphviz."""
    # Implementation depends on visualization preference
    # Option 1: dash-cytoscape
    # Option 2: Graphviz SVG rendering
    pass
```

---

## Component 3: Syntax Highlighting (ACE Mode)

### 3.1 Custom ACE Mode (`assets/prefltlf_mode.js`)

```javascript
ace.define("ace/mode/prefltlf_highlight_rules", ["require", "exports", "module", "ace/lib/oop", "ace/mode/text_highlight_rules"], function(require, exports, module) {
    "use strict";

    var oop = require("../lib/oop");
    var TextHighlightRules = require("./text_highlight_rules").TextHighlightRules;

    var PrefLTLfHighlightRules = function() {

        // Keywords for block delimiters
        var blockKeywords = (
            "propositions|end|alphabet|options|ltlf-formulas|preferences"
        );

        // Alphabet keywords
        var alphabetKeywords = (
            "powerset|singletons|emptyset|exclude"
        );

        // Options keywords
        var optionsKeywords = (
            "semantics|auto-complete"
        );

        // Semantics values
        var semanticsValues = (
            "EA|AE|AA|EE|MaxEA|MaxAE|MaxAA|MaxEE|" +
            "exists-forall|forall-exists|forall-forall|exists-exists|" +
            "max-exists-forall|max-forall-exists|max-forall-forall|max-exists-exists|" +
            "incomparable|minimal"
        );

        // LTLf temporal operators
        var ltlfOperators = (
            "F|G|X|U|R"
        );

        // Preference operators
        var prefOperators = (
            ">|<|>=|<=|~|<>"
        );

        // Verbatim preference phrases
        var verbatimKeywords = (
            "is|to|strictly|weakly|preferred|indifferent|incomparable"
        );

        // Boolean constants
        var boolConstants = (
            "true|false"
        );

        this.$rules = {
            "start": [
                {
                    token: "comment",
                    regex: "#.*$"
                },
                {
                    token: "keyword.control",
                    regex: "\\b(" + blockKeywords + ")\\b"
                },
                {
                    token: "keyword.other",
                    regex: "\\b(" + alphabetKeywords + ")\\b"
                },
                {
                    token: "keyword.operator",
                    regex: "\\b(" + optionsKeywords + ")\\b"
                },
                {
                    token: "constant.language",
                    regex: "\\b(" + semanticsValues + ")\\b"
                },
                {
                    token: "keyword.operator.temporal",
                    regex: "\\b(" + ltlfOperators + ")\\b"
                },
                {
                    token: "keyword.operator.preference",
                    regex: "(" + prefOperators + ")"
                },
                {
                    token: "keyword.other.verbatim",
                    regex: "\\b(" + verbatimKeywords + ")\\b"
                },
                {
                    token: "constant.language.boolean",
                    regex: "\\b(" + boolConstants + ")\\b"
                },
                {
                    token: "punctuation.definition.set",
                    regex: "[{}\\[\\](),:]"
                },
                {
                    token: "variable.parameter",  // Formula names (after colon context)
                    regex: "[a-zA-Z_][a-zA-Z0-9_]*(?=\\s*:)"
                },
                {
                    token: "variable.other",  // General identifiers
                    regex: "[a-zA-Z_][a-zA-Z0-9_]*"
                },
                {
                    token: "keyword.operator",
                    regex: "[=|&!]"
                },
                {
                    token: "text",
                    regex: "\\s+"
                }
            ]
        };

        this.normalizeRules();
    };

    oop.inherits(PrefLTLfHighlightRules, TextHighlightRules);
    exports.PrefLTLfHighlightRules = PrefLTLfHighlightRules;
});

ace.define("ace/mode/prefltlf", ["require", "exports", "module", "ace/lib/oop", "ace/mode/text", "ace/mode/prefltlf_highlight_rules"], function(require, exports, module) {
    "use strict";

    var oop = require("../lib/oop");
    var TextMode = require("./text").Mode;
    var PrefLTLfHighlightRules = require("./prefltlf_highlight_rules").PrefLTLfHighlightRules;

    var Mode = function() {
        this.HighlightRules = PrefLTLfHighlightRules;
        this.$behaviour = this.$defaultBehaviour;
    };

    oop.inherits(Mode, TextMode);

    (function() {
        this.lineCommentStart = "#";
        this.$id = "ace/mode/prefltlf";
    }).call(Mode.prototype);

    exports.Mode = Mode;
});
```

### 3.2 Color Theme Customization

Add to `assets/style.css`:

```css
/* PrefLTLf syntax highlighting colors */
.ace-monokai .ace_keyword.ace_control {
    color: #F92672;  /* Block keywords: propositions, end, etc. */
    font-weight: bold;
}

.ace-monokai .ace_keyword.ace_operator.ace_temporal {
    color: #66D9EF;  /* LTLf operators: F, G, X, U, R */
    font-weight: bold;
}

.ace-monokai .ace_keyword.ace_operator.ace_preference {
    color: #FD971F;  /* Preference operators: >, <, >=, etc. */
    font-weight: bold;
}

.ace-monokai .ace_keyword.ace_other {
    color: #AE81FF;  /* Alphabet keywords: powerset, singletons */
}

.ace-monokai .ace_keyword.ace_other.ace_verbatim {
    color: #E6DB74;  /* Verbatim: is, strictly preferred, etc. */
    font-style: italic;
}

.ace-monokai .ace_constant.ace_language {
    color: #A6E22E;  /* Semantics values, true/false */
}

.ace-monokai .ace_variable.ace_parameter {
    color: #FD971F;  /* Formula names (definitions) */
    font-weight: bold;
}

.ace-monokai .ace_variable.ace_other {
    color: #F8F8F2;  /* General identifiers/propositions */
}

.ace-monokai .ace_comment {
    color: #75715E;
    font-style: italic;
}

/* Error annotation styling */
.ace_error {
    background: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAYAAAC09K7GAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9sJFhQXEbhTg7YAAAAZdEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIEdJTVBXgQ4XAAAAMklEQVQI12NsZGT8z8jIyMDAwPCfgYHhP8MuBgaGJxBmYGBgYGZg+M8ABDgAAJYFA0VMQPUAAAAASUVORK5CYII=") bottom repeat-x;
}
```

---

## Component 4: Live Error Annotations

### 4.1 Error Annotation Flow

```
User types → Editor onChange → Debounce (300ms) → Parse → Annotate
                                                      ↓
                                              Semantic Analysis
                                                      ↓
                                              Build annotations[]
                                                      ↓
                                              Update editor.annotations
```

### 4.2 Annotation Types

```python
# ACE annotation format
annotation = {
    'row': 0,        # 0-indexed line number
    'column': 0,     # Column position
    'text': "Error message here",
    'type': 'error'  # 'error' | 'warning' | 'info'
}

# Full gutter marker (optional, for more visibility)
marker = {
    'startRow': 0,
    'startCol': 0,
    'endRow': 0,
    'endCol': 10,
    'className': 'error-marker',
    'type': 'text'
}
```

### 4.3 Debouncing Strategy

```python
# In callbacks.py
from dash import dcc

# Use dcc.Interval with shorter interval + clientside callback for debouncing
app.clientside_callback(
    """
    function(value, n_intervals) {
        // Only trigger if value hasn't changed in 300ms
        if (window.lastValue === value) {
            return value;
        }
        window.lastValue = value;
        window.lastTime = Date.now();
        return dash_clientside.no_update;
    }
    """,
    Output('debounced-value', 'data'),
    [Input('editor', 'value'), Input('debounce-interval', 'n_intervals')]
)
```

---

## Implementation Phases

### Phase 1: Core Parser (3-4 days)
- [ ] Write `grammar.lark` from EBNF spec
- [ ] Implement `ast_nodes.py` dataclasses
- [ ] Implement `PrefLTLfTransformer`
- [ ] Basic error recovery in parser
- [ ] Unit tests for grammar

### Phase 2: Semantic Analyzer (2-3 days)
- [ ] Symbol table construction
- [ ] All semantic checks (see analyzer.py above)
- [ ] Error message formatting with locations
- [ ] Unit tests for each check

### Phase 3: Lowering (2 days)
- [ ] Implement `lower_to_legacy()`
- [ ] Desugaring rules (chains, verbatim, `<`/`<=`)
- [ ] Alphabet construction
- [ ] Integration tests vs. existing specs

### Phase 4: Dash App Skeleton (2-3 days)
- [ ] Basic layout with dash-ace
- [ ] Run button callback → translation
- [ ] Output tabs (JSON, graph, legacy)
- [ ] Error panel

### Phase 5: Syntax Highlighting (1-2 days)
- [ ] Write ACE mode JS file
- [ ] CSS theming
- [ ] Test with all language constructs

### Phase 6: Live Annotations (1-2 days)
- [ ] Wire up live parsing callback
- [ ] Debouncing
- [ ] Annotation → ACE format conversion
- [ ] Inline squiggles CSS

### Phase 7: Polish (2-3 days)
- [ ] Example dropdown
- [ ] Export to `.prefltlf` button
- [ ] PDFA visualization (Cytoscape or Graphviz)
- [ ] Responsive layout
- [ ] Documentation

**Total: ~15-20 days**

---

## Open Questions

1. **Dash-ace vs. Monaco?**
   - dash-ace: simpler, works with Dash out of box
   - Monaco: richer features (autocomplete, hover), but needs more integration work
   - **Recommendation:** Start with dash-ace, migrate later if needed

2. **PDFA visualization preference?**
   - Cytoscape.js (interactive, zoomable)
   - Graphviz SVG (cleaner for publications)
   - Both? (tabs)

3. **Should the web app support file upload/download?**
   - Upload `.prefltlf` or `.pltlf` files
   - Download generated PDFA as JSON/PNG/DOT

4. **Authentication/multi-user?**
   - For now: single-user local dev
   - Future: optional save/share URLs

5. **Error recovery granularity?**
   - Option A: Stop at first error (simpler)
   - Option B: Continue parsing, collect multiple errors (better UX)
   - **Recommendation:** Option B with Lark's error recovery

6. **LTLf sub-expression errors?**
   - Should we parse LTLf ourselves (full control) or delegate to ltlf2dfa?
   - **Recommendation:** Parse in grammar for syntax highlighting, validate with ltlf2dfa for semantics

---

## Dependencies

```txt
# requirements.txt additions
lark>=1.1.0
dash>=2.14.0
dash-ace>=0.2.0
dash-bootstrap-components>=1.5.0
dash-cytoscape>=0.3.0  # optional, for PDFA visualization
```

---

## Next Steps

1. **Approve this plan** or provide feedback
2. **Answer open questions** above
3. **I'll implement Phase 1** (grammar + parser) first
4. **Iterative demos** after each phase

Ready to start coding when you give the go-ahead!

