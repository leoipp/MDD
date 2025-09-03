# app.py
import importlib
import inspect
from dash import Dash, dcc, html

# =========================
# Utils: carregar Eq.py
# =========================
def load_equation_module():
    try:
        mod = importlib.import_module("Eq")
        importlib.reload(mod)
        return mod
    except Exception as e:
        print(f"[WARN] Não foi possível importar Eq.py: {e}")
        return None

def collect_equations(mod):
    if mod is None:
        return {}
    if hasattr(mod, "EQUATIONS") and isinstance(mod.EQUATIONS, dict):
        # Usa o registry explícito do Eq.py
        return {str(k): v for k, v in mod.EQUATIONS.items() if callable(v)}
    # Fallback: coleta funções livres (não recomendado se você usa registry)
    eqs = {}
    for name, func in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_"):
            continue
        eqs[name] = func
    return eqs

def get_vars_spec(mod, eq_funcs):
    # Usa VARS_SPEC do Eq.py se existir; caso contrário, não define nada (somente univariadas)
    if mod is not None and hasattr(mod, "VARS_SPEC"):
        vs = getattr(mod, "VARS_SPEC")
        if isinstance(vs, dict):
            return vs
    return {}

# =========================
# Equações (sem dados ainda)
# =========================
eq_mod = load_equation_module()
EQ_FUNCS = collect_equations(eq_mod)
VARS_SPEC = get_vars_spec(eq_mod, EQ_FUNCS)
eq_names = sorted(EQ_FUNCS.keys())

# =========================
# App e template HTML externo
# =========================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Ajustes WB - @ipp"

# Usa um template HTML (opcional). Garanta que templates/index.html exista.
try:
    with open("templates/index.html", "r", encoding="utf-8") as f:
        app.index_string = f.read()
except Exception:
    # Se não houver template, segue com o default do Dash
    pass

# =========================
# Layout
# =========================
app.layout = html.Div([
    # Stores base
    dcc.Store(id="orig-data", data=None),
    dcc.Store(id="base-data", data=None),
    dcc.Store(id="removed-ids", data=[]),
    dcc.Store(id="fit-state", data=None),

    # Stores dinâmicos
    dcc.Store(id="store-num-cols", data=[]),
    dcc.Store(id="store-all-cols", data=[]),

    html.H2("Ajustes WB - @ipp", style={"margin": "12px 0 18px"}),

    # =======================
    # 1) PRIMEIRO CONTÊINER: FILE (Upload)
    # =======================
    html.Div([
        html.Div([
            dcc.Upload(
                id="file-upload",
                children=html.Button("File"),
                multiple=False,
            ),
            html.Small("  Selecione um .xlsx ou .csv", style={"marginLeft": "8px", "color": "#666"})
        ], className="row"),
    ], className="panel"),

    # =======================
    # 2) CONTÊINER: ESTRATIFICAÇÃO (separado)
    # =======================
    html.Div([
        html.Label("Estratificação"),
        dcc.Checklist(
            id="strat-check",
            options=[{"label": " habilitar", "value": "on"}],
            value=[],  # desabilitado por padrão
            style={"marginTop": "6px"}
        ),
        html.Div([
            html.Small("Coluna de estratos"),
            dcc.Dropdown(
                id="estrato-col-dd",
                options=[], value=None, clearable=True,
                className="minw-220"
            ),
        ], className="mt-8"),
        html.Div([
            html.Small("Valor do estrato"),
            dcc.Dropdown(
                id="estrato-val-dd",
                options=[], value=None, clearable=True,
                className="minw-220"
            ),
        ], className="mt-8"),
    ], className="panel mt-14"),

    # =======================
    # 3) CONTÊINER: EIXOS + EQUAÇÃO
    # =======================
    html.Div([
        html.Div([
            html.Div([
                html.Label("Eixo X"),
                dcc.Dropdown(id="x-dd", options=[], value=None, clearable=False, className="minw-220"),
            ], className="minw-220"),
            html.Div([
                html.Label("Eixo Y"),
                dcc.Dropdown(id="y-dd", options=[], value=None, clearable=False, className="minw-220"),
            ], className="minw-220"),
            html.Div([
                html.Label("Equação (Eq.py)"),
                dcc.Dropdown(
                    id="eq-dd",
                    options=[{"label": n, "value": n} for n in eq_names],
                    value=(eq_names[0] if eq_names else None),
                    clearable=True,
                    className="minw-260"
                ),
            ], className="minw-260"),
        ], className="row", style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ], className="panel mt-14"),

    # =======================
    # 4) CONTÊINER: MAPEAMENTO DINÂMICO
    # =======================
    html.Div([
        html.Label("Mapeamento de variáveis (dinâmico conforme a equação)"),
        html.Div(id="vars-mapping", className="row mt-8"),
        html.Div([
            html.Small("Target Y (coluna alvo)"),
            dcc.Dropdown(id="map-target", options=[], value=None, clearable=True, className="minw-260"),
        ], className="mt-8"),
    ], className="panel mt-14"),

    # =======================
    # 5) CONTÊINER: GRÁFICO + AÇÕES
    # =======================
    html.Div([
        dcc.Graph(
            id="scatter-plot",
            figure={},  # vazio até carregar dados
            config={"modeBarButtonsToAdd": ["lasso2d", "select2d"]}
        ),
        html.Div([
            html.Button("Remover selecionados", id="remove-btn", n_clicks=0),
            html.Button("Mostrar IDs removidos", id="show-removed-btn", n_clicks=0),
            html.Button("Resetar base", id="reset-btn", n_clicks=0),
            html.Button("Ajustar equação", id="fit-btn", n_clicks=0),
        ], className="mt-8 row", style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
    ], className="panel mt-14"),

    # =======================
    # Mensagens
    # =======================
    html.Div(id="status", className="mono mt-14"),   # mensagens do upload
    html.Div(id="output", className="mono mt-14"),   # mensagens de remover/fit/reset
], className="container")

# =========================
# Callbacks
# =========================
from callbacks import register_callbacks
register_callbacks(app, context=dict(
    EQ_FUNCS=EQ_FUNCS,
    VARS_SPEC=VARS_SPEC,
))

if __name__ == "__main__":
    app.run(debug=True)
