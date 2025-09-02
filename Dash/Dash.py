import importlib, inspect
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from dash.dependencies import ALL
from scipy.optimize import curve_fit

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
        return {str(k): v for k, v in mod.EQUATIONS.items() if callable(v)}
    eqs = {}
    for name, func in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_"):
            continue
        eqs[name] = func
    return eqs

def get_vars_spec(mod, eq_funcs):
    """
    Prioriza VARS_SPEC em Eq.py. Se não houver, cria um fallback
    com base em nomes comuns das suas equações.
    """
    if mod is not None and hasattr(mod, "VARS_SPEC"):
        vs = getattr(mod, "VARS_SPEC")
        if isinstance(vs, dict):
            return vs

    # fallback para as equações citadas
    spec = {}
    for name in eq_funcs.keys():
        if name in ("DMAXPROJETADO_EQ1", "DMAXPROJETADO_EQ2"):
            spec[name] = ["IDADE1", "IDADE2", "DMAX1"]
        elif name == "DMINPROJETADO_EQ3":
            spec[name] = ["IDADE1", "IDADE2", "DMIN1"]
        elif name == "BETAPROJETADO_EQ4":
            spec[name] = ["IDADE1", "IDADE2", "DMAX2", "BETA1"]
        elif name in ("ESCALAPROJETADO_EQ5", "ESCALAPROJETADO_EQ6", "ESCALAPROJETADO_EQ7"):
            spec[name] = ["IDADE1", "IDADE2", "ESCALA1"]
        elif name in ("GOMPERTZ_EQ8", "MODIFIED_EXP_EQ9"):
            spec[name] = ["IDADE1", "IDADE2", "VAR1"]
        # senão, assume univariada (sem Vars)
    return spec

def get_initial_guess(mod, name, func, X, y):
    if mod is not None and hasattr(mod, "INITIAL_GUESSES"):
        ig = getattr(mod, "INITIAL_GUESSES")
        if isinstance(ig, dict) and name in ig:
            return list(ig[name])
    if mod is not None and hasattr(mod, "initial_guess") and callable(mod.initial_guess):
        try:
            g = mod.initial_guess(name, np.asarray(X), np.asarray(y))
            if g is not None:
                return list(g)
        except Exception:
            pass
    # fallback simples pelo nº de parâmetros
    try:
        sig = inspect.signature(func)
        n_params = max(0, len(sig.parameters) - 1)
    except Exception:
        n_params = 2
    if n_params <= 0:
        return []
    y = np.asarray(y)
    slope = 0.1
    medy = float(np.nanmedian(y)) if np.isfinite(np.nanmedian(y)) else 1.0
    base = [medy, slope]
    while len(base) < n_params:
        base.append(1.0)
    return base[:n_params]

# métricas
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask]
    yhat = y_pred[mask]
    if y.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    if y.size > 1 and np.nanstd(y) > 0 and np.nanstd(yhat) > 0:
        r = float(np.corrcoef(y, yhat)[0, 1])
    else:
        r = np.nan
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2)) if y.size > 1 else np.nan
    r2 = float(1 - sse / sst) if sst and np.isfinite(sst) and sst > 0 else np.nan
    rmse = float(np.sqrt(sse / y.size))
    bias = float(np.mean(yhat - y))
    return r, r2, rmse, bias

# =========================
# Dados
# =========================
df_raw = pd.read_excel("dash.xlsx")
df_raw["estrato"] = df_raw["estrato"].astype(str).str.strip()

# ID estável
df_raw = df_raw.reset_index(drop=True)
df_raw.insert(0, "id", range(len(df_raw)))

# Opções estrato
estratos = sorted(df_raw["estrato"].dropna().unique().tolist())
estrato_options = [{"label": "Todos", "value": "__ALL__"}] + [{"label": v, "value": v} for v in estratos]

# Colunas numéricas p/ eixos
num_cols = [c for c in df_raw.select_dtypes(include="number").columns if c != "id"]
default_x = num_cols[0] if num_cols else None
default_y = num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else None)

# Todas colunas para mapeamento
all_cols = [c for c in df_raw.columns if c not in ["id"]]

# Carrega Eq.py
eq_mod = load_equation_module()
EQ_FUNCS = collect_equations(eq_mod)
VARS_SPEC = get_vars_spec(eq_mod, EQ_FUNCS)
eq_names = sorted(EQ_FUNCS.keys())
eq_options = [{"label": n, "value": n} for n in eq_names]

# Conjunto de equações que usam Vars (tem entrada multivariada)
EQS_WITH_VARS = set(VARS_SPEC.keys())

# =========================
# Gráfico
# =========================
def make_scatter(dframe: pd.DataFrame, x_col: str, y_col: str):
    d_plot = dframe.dropna(subset=[x_col, y_col])
    fig = px.scatter(d_plot, x=x_col, y=y_col, custom_data=["id"])
    fig.update_layout(dragmode="lasso")
    return fig

def add_fit_curve_generic(fig: go.Figure, func, params, df_view, x_col, y_col, vars_map):
    """
    Plot genérico para equações com Vars:
    - Se o eixo X coincide com alguma coluna mapeada, variamos essa coluna.
    - As demais variáveis ficam fixas na mediana (visão atual).
    """
    if not vars_map:
        return fig

    # colunas mapeadas -> lista (na ordem de VARS_SPEC)
    mapped_cols = list(vars_map.values())
    if x_col not in mapped_cols:
        return fig  # não dá para varrer o eixo X se ele não é nenhuma var do modelo

    # dados válidos
    needed = mapped_cols + ([y_col] if y_col else [])
    d = df_view.dropna(subset=[c for c in needed if c in df_view.columns])
    if d.empty:
        return fig

    # grade no eixo X
    xmin, xmax = float(d[x_col].min()), float(d[x_col].max())
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        return fig
    x_grid = np.linspace(xmin, xmax, 200)

    # construir Xg com mediana nas variáveis que não são o eixo X
    Xg_cols = []
    for var_name, col in vars_map.items():
        if col == x_col:
            Xg_cols.append(x_grid)
        else:
            Xg_cols.append(np.full_like(x_grid, float(np.nanmedian(d[col]))))
    # empilhar na ordem
    Xg = np.column_stack(Xg_cols)

    try:
        y_hat = func(Xg, *params)
        fig.add_trace(go.Scatter(x=x_grid, y=y_hat, mode="lines",
                                 name=f"Fit ({x_col} variável)"))
    except Exception as e:
        print(f"[WARN] Falha ao plotar curva: {e}")
    return fig

def add_fit_curve_univar(fig: go.Figure, func, params, df_view, x_col, y_col):
    d = df_view.dropna(subset=[x_col, y_col])
    if d.empty:
        return fig
    xmin, xmax = float(d[x_col].min()), float(d[x_col].max())
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        return fig
    x_grid = np.linspace(xmin, xmax, 200)
    try:
        y_hat = func(x_grid, *params)
        fig.add_trace(go.Scatter(x=x_grid, y=y_hat, mode="lines", name="Fit"))
    except Exception as e:
        print(f"[WARN] Falha ao plotar curva univar: {e}")
    return fig

# =========================
# App
# =========================
app = Dash(__name__)
app.title = "Ajustes WB - @ipp"

container_style = {
    "maxWidth": "1200px",
    "margin": "0 auto",
    "padding": "16px 20px 28px",
    "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial"
}
panel_style = {"padding": "12px", "border": "1px solid #eee", "borderRadius": "10px", "background": "#fafafa"}

app.layout = html.Div([
    html.H2("Ajustes WB - @ipp", style={"margin": "12px 0 18px"}),

    dcc.Store(id="orig-data", data=df_raw.to_dict("records")),
    dcc.Store(id="base-data", data=df_raw.to_dict("records")),
    dcc.Store(id="removed-ids", data=[]),
    dcc.Store(id="fit-state", data=None),  # {'name', 'params', 'uses_vars', 'maps', 'target'}

    # Linha de controles principais
    html.Div([
        html.Div([
            html.Div([
                html.Label("Estrato"),
                dcc.Dropdown(id="estrato-dd", options=estrato_options, value="__ALL__", clearable=False),
            ], style={"minWidth": 220}),
            html.Div([
                html.Label("Eixo X"),
                dcc.Dropdown(id="x-dd", options=[{"label": c, "value": c} for c in num_cols],
                             value=default_x, clearable=False),
            ], style={"minWidth": 220}),
            html.Div([
                html.Label("Eixo Y"),
                dcc.Dropdown(id="y-dd", options=[{"label": c, "value": c} for c in num_cols],
                             value=default_y, clearable=False),
            ], style={"minWidth": 220}),
            html.Div([
                html.Label("Equação (Eq.py)"),
                dcc.Dropdown(id="eq-dd", options=[{"label": n, "value": n} for n in eq_names],
                             value=(eq_names[0] if eq_names else None), clearable=True),
            ], style={"minWidth": 260}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"})
    ], style=panel_style),

    # ---- Mapeamento DINÂMICO de variáveis ----
    html.Div([
        html.Label("Mapeamento de variáveis (dinâmico conforme a equação)"),
        html.Div(id="vars-mapping", style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "8px"}),
        html.Div([
            html.Small("Target Y (coluna alvo)"),
            dcc.Dropdown(id="map-target", options=[{"label": c, "value": c} for c in all_cols],
                         value=None, clearable=True, style={"minWidth": 260}),
        ], style={"marginTop": "10px"})
    ], style={**panel_style, "marginTop": "14px"}),

    # ---- Gráfico e botões ----
    html.Div([
        dcc.Graph(
            id="scatter-plot",
            figure=make_scatter(df_raw, default_x, default_y),
            config={"modeBarButtonsToAdd": ["lasso2d", "select2d"]}
        ),
        html.Div([
            html.Button("Remover selecionados", id="remove-btn", n_clicks=0, style={"marginRight": 8}),
            html.Button("Mostrar IDs removidos", id="show-removed-btn", n_clicks=0, style={"marginRight": 8}),
            html.Button("Resetar base", id="reset-btn", n_clicks=0, style={"marginRight": 16}),
            html.Button("Ajustar equação", id="fit-btn", n_clicks=0),
        ], style={"marginTop": "8px"}),
    ], style={**panel_style, "marginTop": "14px"}),

    html.Div(id="output", style={"marginTop": "12px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"})
], style=container_style)

# -------- (A) Renderiza mapeamento DINÂMICO conforme a equação --------
@app.callback(
    Output("vars-mapping", "children"),
    Input("eq-dd", "value")
)
def render_vars_mapping(eq_name):
    # Quais variáveis essa equação exige?
    vars_needed = VARS_SPEC.get(eq_name, [])
    if not vars_needed:
        return [html.Div("Esta equação não requer mapeamento de variáveis adicionais.", style={"color": "#666"})]

    children = []
    for var in vars_needed:
        children.append(
            html.Div([
                html.Small(f"{var} (coluna)"),
                dcc.Dropdown(
                    id={"role": "var-map", "name": var},
                    options=[{"label": c, "value": c} for c in all_cols],
                    value=None,
                    clearable=True,
                    style={"minWidth": 220}
                )
            ])
        )
    return children

# -------- (B) Atualiza gráfico quando base/filtro/eixos/fit mudam --------
@app.callback(
    Output("scatter-plot", "figure"),
    Input("estrato-dd", "value"),
    Input("x-dd", "value"),
    Input("y-dd", "value"),
    Input("base-data", "data"),
    Input("fit-state", "data"),
)
def update_graph(sel_estrato, x_col, y_col, base_data, fit_state):
    df_base = pd.DataFrame(base_data)
    df_view = df_base if sel_estrato == "__ALL__" else df_base[df_base["estrato"] == sel_estrato]
    fig = make_scatter(df_view, x_col, y_col)

    if fit_state and isinstance(fit_state, dict):
        name = fit_state.get("name")
        params = fit_state.get("params")
        uses_vars = fit_state.get("uses_vars", False)
        func = EQ_FUNCS.get(name)
        if func and params:
            if uses_vars:
                vars_map = fit_state.get("maps") or {}    # dict var_name -> column
                fig = add_fit_curve_generic(fig, func, params, df_view, x_col, y_col, vars_map)
            else:
                fig = add_fit_curve_univar(fig, func, params, df_view, x_col, y_col)
    return fig

# -------- (C) Unificar ações (remover / mostrar / resetar / ajustar) --------
@app.callback(
    Output("base-data", "data"),
    Output("removed-ids", "data"),
    Output("output", "children"),
    Output("fit-state", "data"),
    Input("remove-btn", "n_clicks"),
    Input("reset-btn", "n_clicks"),
    Input("show-removed-btn", "n_clicks"),
    Input("fit-btn", "n_clicks"),
    State("scatter-plot", "selectedData"),
    State("estrato-dd", "value"),
    State("x-dd", "value"),
    State("y-dd", "value"),
    State("eq-dd", "value"),
    State({"role": "var-map", "name": ALL}, "id"),
    State({"role": "var-map", "name": ALL}, "value"),
    State("map-target", "value"),
    State("base-data", "data"),
    State("removed-ids", "data"),
    State("orig-data", "data"),
    prevent_initial_call=True
)
def handle_actions(n_remove, n_reset, n_show, n_fit,
                   selectedData, sel_estrato, x_col, y_col, eq_name,
                   var_ids, var_vals, map_target,
                   base_data, removed_ids, orig_data):

    trig = ctx.triggered_id

    # RESET
    if trig == "reset-btn":
        return orig_data, [], "Base resetada; IDs removidos zerados.", None

    # MOSTRAR IDs
    if trig == "show-removed-btn":
        if not removed_ids:
            return no_update, no_update, "Nenhum ID removido até agora.", no_update
        return no_update, no_update, f"IDs removidos acumulados: {sorted(set(removed_ids))}", no_update

    df_base = pd.DataFrame(base_data)

    # REMOVER
    if trig == "remove-btn":
        if not selectedData or "points" not in selectedData:
            return base_data, removed_ids, "Nenhum ponto selecionado.", no_update
        picked_ids = []
        for p in selectedData["points"]:
            cd = p.get("customdata")
            if cd is not None:
                pid = cd[0] if isinstance(cd, (list, tuple)) else cd
                try:
                    picked_ids.append(int(pid))
                except Exception:
                    pass
        if not picked_ids:
            d = df_base if sel_estrato == "__ALL__" else df_base[df_base["estrato"] == sel_estrato]
            d_view = d.dropna(subset=[x_col, y_col]).reset_index(drop=True)
            idxs = [int(p["pointIndex"]) for p in selectedData["points"]]
            picked_ids = d_view.loc[idxs, "id"].tolist()
        if not picked_ids:
            return base_data, removed_ids, "Nenhum ID capturado.", no_update
        df_new = df_base[~df_base["id"].isin(picked_ids)]
        removed_ids_new = (removed_ids or []) + picked_ids
        return df_new.to_dict("records"), removed_ids_new, f"Removidos {len(picked_ids)}: {picked_ids}", no_update

    # AJUSTAR (fit + métricas)
    if trig == "fit-btn":
        if not eq_name:
            return no_update, no_update, "Selecione uma equação em Eq.py.", no_update
        func = EQ_FUNCS.get(eq_name)
        if func is None:
            return no_update, no_update, f"Equação '{eq_name}' não encontrada.", no_update

        df_view = df_base if sel_estrato == "__ALL__" else df_base[df_base["estrato"] == sel_estrato]

        # Montar dict de mapeamentos dinâmicos: {var_name -> coluna}
        vars_needed = VARS_SPEC.get(eq_name, [])
        vars_map = {}
        if vars_needed:
            # var_ids é uma lista de dicts {'role':'var-map','name':<var>}
            # var_vals é a lista de valores selecionados nas mesmas posições
            for comp_id, val in zip(var_ids, var_vals):
                if isinstance(comp_id, dict) and comp_id.get("role") == "var-map":
                    vars_map[comp_id.get("name")] = val

        uses_vars = bool(vars_needed)

        if uses_vars:
            # Validar mapeamentos + target
            missing = [v for v in vars_needed if not vars_map.get(v)]
            if missing:
                return no_update, no_update, f"Faltam mapeamentos: {missing}", no_update
            if not map_target or map_target not in df_view.columns:
                return no_update, no_update, "Selecione a coluna alvo (Target Y).", no_update

            cols_need = [vars_map[v] for v in vars_needed] + [map_target]
            df_fit = df_view.dropna(subset=cols_need)
            if df_fit.empty:
                return no_update, no_update, "Sem dados (NaN) após mapeamento para ajuste.", no_update

            # Montar X na ORDEM de vars_needed
            X = df_fit[[vars_map[v] for v in vars_needed]].to_numpy(dtype=float, copy=True)
            y = df_fit[map_target].to_numpy(dtype=float, copy=True)

            p0 = get_initial_guess(eq_mod, eq_name, func, X, y)
            try:
                popt, _ = curve_fit(func, X, y, p0=p0, maxfev=20000)
                y_hat = func(X, *popt)
                r, r2, rmse, bias = regression_metrics(y, y_hat)
                fit_info = {
                    "name": eq_name,
                    "params": [float(v) for v in popt],
                    "uses_vars": True,
                    "maps": vars_map,
                    "target": map_target,
                }
                msg = (
                    f"Ajuste OK: {eq_name}\n"
                    f"Parâmetros: {np.round(fit_info['params'], 6).tolist()}\n"
                    f"r={r:.4f} | r²={r2:.4f} | RMSE={rmse:.4f} | Bias={bias:.4f}"
                )
                return no_update, no_update, msg, fit_info
            except Exception as e:
                return no_update, no_update, f"Falha no ajuste: {e}", no_update

        # Caso univariado (sem Vars_SPEC)
        else:
            if x_col is None or y_col is None:
                return no_update, no_update, "Defina Eixo X e Eixo Y para ajuste univariado.", no_update
            df_fit = df_view.dropna(subset=[x_col, y_col])
            if df_fit.empty:
                return no_update, no_update, "Sem dados válidos (NaN) para ajuste univariado.", no_update
            X = df_fit[x_col].to_numpy(dtype=float, copy=True)
            y = df_fit[y_col].to_numpy(dtype=float, copy=True)
            p0 = get_initial_guess(eq_mod, eq_name, func, X, y)
            try:
                popt, _ = curve_fit(func, X, y, p0=p0, maxfev=20000)
                y_hat = func(X, *popt)
                r, r2, rmse, bias = regression_metrics(y, y_hat)
                fit_info = {"name": eq_name, "params": [float(v) for v in popt], "uses_vars": False}
                msg = (
                    f"Ajuste OK: {eq_name}\n"
                    f"Parâmetros: {np.round(fit_info['params'], 6).tolist()}\n"
                    f"r={r:.4f} | r²={r2:.4f} | RMSE={rmse:.4f} | Bias={bias:.4f}"
                )
                return no_update, no_update, msg, fit_info
            except Exception as e:
                return no_update, no_update, f"Falha no ajuste: {e}", no_update

    return no_update, no_update, no_update, no_update

# =========================
if __name__ == "__main__":
    app.run(debug=True)
