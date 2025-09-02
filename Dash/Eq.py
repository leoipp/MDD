# Eq.py
import numpy as np

# =========================
# Suas funções originais
# =========================
def DMAXPROJETADO_EQ1(Vars, b0):
    IDADE1, IDADE2, DMAX1 = Vars
    return DMAX1 * (IDADE1 / IDADE2) + b0 * (1 - IDADE1 / IDADE2)

def DMAXPROJETADO_EQ2(Vars, b0, b1):
    IDADE1, IDADE2, DMAX1 = Vars
    return DMAX1 * np.exp(-b0 ** ((IDADE2 ** b1) - (IDADE1 ** b1)))

def DMINPROJETADO_EQ3(Vars, b0, b1):
    IDADE1, IDADE2, DMIN1 = Vars
    return DMIN1 * np.exp(-b0 ** ((IDADE2 ** b1) - (IDADE1 ** b1)))

def BETAPROJETADO_EQ4(Vars, b0):
    IDADE1, IDADE2, DMAX2, BETA1 = Vars
    return BETA1 * (IDADE1 / IDADE2) + b0 * (1 - IDADE1 / IDADE2) * DMAX2

def ESCALAPROJETADO_EQ5(Vars, b0):
    IDADE1, IDADE2, ESCALA1 = Vars
    return ESCALA1 * (IDADE1 / IDADE2) + b0 * (1 - IDADE1 / IDADE2)

def ESCALAPROJETADO_EQ6(Vars, b0, b1):
    IDADE1, IDADE2, ESCALA1 = Vars
    return np.exp(np.log(ESCALA1) * np.exp(-(b0) * ((IDADE2 ** b1) - (IDADE1 ** b1))))

def ESCALAPROJETADO_EQ7(Vars, b0, b1, b2):
    IDADE1, IDADE2, ESCALA1 = Vars
    return np.exp((np.log(ESCALA1) ** b0) * np.exp(-b1 * ((IDADE2 ** b2) - (IDADE1 ** b2))))

def GOMPERTZ_EQ8(Vars, b1, b2):
    IDADE1, IDADE2, D = Vars
    return D * np.exp(np.exp(b1-b2*IDADE1)-np.exp(b1-b2*IDADE2))

def MODIFIED_EXP_EQ9(Vars, b0):
    IDADE1, IDADE2, D = Vars
    return D * np.exp(b0*(1/IDADE2 - 1/IDADE1))

# =========================
# Helpers compatíveis com curve_fit
# =========================
def _to_Vars_generic(X, n_vars):
    """
    Converte X em tupla de n_vars vetores (cada um shape (n_obs,)):
    - Aceita X 1D com tamanho múltiplo de n_vars, (n_obs, n_vars) ou (n_vars, n_obs).
    """
    X = np.asarray(X)
    if X.ndim == 1:
        if X.size % n_vars != 0:
            raise ValueError(f"Esperado múltiplo de {n_vars} valores em X; recebi {X.size}.")
        X = X.reshape(-1, n_vars)
    elif X.ndim == 2:
        # Se vier (n_vars, n_obs), transpõe para (n_obs, n_vars)
        if X.shape[0] == n_vars and X.shape[1] != n_vars:
            X = X.T
        if X.shape[1] != n_vars:
            raise ValueError(f"X deve ter {n_vars} colunas; shape={X.shape}")
    else:
        raise ValueError(f"X deve ter 1 ou 2 dimensões; shape={X.shape}")

    return tuple(X[:, i] for i in range(n_vars))

# =========================
# Wrappers f(X, *params) para curve_fit
# (Respeitam a ORDEM de VARS especificada em VARS_SPEC)
# =========================
def DMAXPROJETADO_EQ1_fit(X, b0):
    # Vars: [IDADE1, IDADE2, DMAX1]
    Vars = _to_Vars_generic(X, 3)
    return DMAXPROJETADO_EQ1(Vars, b0)

def DMAXPROJETADO_EQ2_fit(X, b0, b1):
    # Vars: [IDADE1, IDADE2, DMAX1]
    Vars = _to_Vars_generic(X, 3)
    return DMAXPROJETADO_EQ2(Vars, b0, b1)

def DMINPROJETADO_EQ3_fit(X, b0, b1):
    # Vars: [IDADE1, IDADE2, DMIN1]
    Vars = _to_Vars_generic(X, 3)
    return DMINPROJETADO_EQ3(Vars, b0, b1)

def BETAPROJETADO_EQ4_fit(X, b0):
    # Vars: [IDADE1, IDADE2, DMAX2, BETA1]
    Vars = _to_Vars_generic(X, 4)
    return BETAPROJETADO_EQ4(Vars, b0)

def ESCALAPROJETADO_EQ5_fit(X, b0):
    # Vars: [IDADE1, IDADE2, ESCALA1]
    Vars = _to_Vars_generic(X, 3)
    return ESCALAPROJETADO_EQ5(Vars, b0)

def ESCALAPROJETADO_EQ6_fit(X, b0, b1):
    # Vars: [IDADE1, IDADE2, ESCALA1]
    Vars = _to_Vars_generic(X, 3)
    return ESCALAPROJETADO_EQ6(Vars, b0, b1)

def ESCALAPROJETADO_EQ7_fit(X, b0, b1, b2):
    # Vars: [IDADE1, IDADE2, ESCALA1]
    Vars = _to_Vars_generic(X, 3)
    return ESCALAPROJETADO_EQ7(Vars, b0, b1, b2)

def GOMPERTZ_EQ8_fit(X, b1, b2):
    # Vars: [IDADE1, IDADE2, D]
    Vars = _to_Vars_generic(X, 3)
    return GOMPERTZ_EQ8(Vars, b1, b2)

def MODIFIED_EXP_EQ9_fit(X, b0):
    # Vars: [IDADE1, IDADE2, D]
    Vars = _to_Vars_generic(X, 3)
    return MODIFIED_EXP_EQ9(Vars, b0)

# =========================
# Registry (usado pelo Dash)
# =========================
EQUATIONS = {
    "DMAXPROJETADO_EQ1": DMAXPROJETADO_EQ1_fit,
    "DMAXPROJETADO_EQ2": DMAXPROJETADO_EQ2_fit,
    "DMINPROJETADO_EQ3": DMINPROJETADO_EQ3_fit,
    "BETAPROJETADO_EQ4": BETAPROJETADO_EQ4_fit,
    "ESCALAPROJETADO_EQ5": ESCALAPROJETADO_EQ5_fit,
    "ESCALAPROJETADO_EQ6": ESCALAPROJETADO_EQ6_fit,
    "ESCALAPROJETADO_EQ7": ESCALAPROJETADO_EQ7_fit,
    "GOMPERTZ_EQ8": GOMPERTZ_EQ8_fit,
    "MODIFIED_EXP_EQ9": MODIFIED_EXP_EQ9_fit,
}

# Especificação da ORDEM de variáveis esperada por equação (para montar X no Dash)
VARS_SPEC = {
    "DMAXPROJETADO_EQ1": ["IDADE1", "IDADE2", "DMAX1"],
    "DMAXPROJETADO_EQ2": ["IDADE1", "IDADE2", "DMAX1"],
    "DMINPROJETADO_EQ3": ["IDADE1", "IDADE2", "DMIN1"],
    "BETAPROJETADO_EQ4": ["IDADE1", "IDADE2", "DMAX2", "BETA1"],
    "ESCALAPROJETADO_EQ5": ["IDADE1", "IDADE2", "ESCALA1"],
    "ESCALAPROJETADO_EQ6": ["IDADE1", "IDADE2", "ESCALA1"],
    "ESCALAPROJETADO_EQ7": ["IDADE1", "IDADE2", "ESCALA1"],
    "GOMPERTZ_EQ8": ["IDADE1", "IDADE2", "D"],
    "MODIFIED_EXP_EQ9": ["IDADE1", "IDADE2", "D"],
}

# =========================
# Chutes iniciais
# =========================
INITIAL_GUESSES = {
    "DMAXPROJETADO_EQ1": [0.5],           # b0
    "DMAXPROJETADO_EQ2": [0.5, 1.0],      # b0, b1
    "DMINPROJETADO_EQ3": [0.5, 1.0],      # b0, b1
    "BETAPROJETADO_EQ4": [0.1],           # b0
    "ESCALAPROJETADO_EQ5": [0.5],         # b0
    "ESCALAPROJETADO_EQ6": [0.1, 1.0],    # b0, b1
    "ESCALAPROJETADO_EQ7": [1.0, 0.1, 1.0],  # b0, b1, b2
    "GOMPERTZ_EQ8": [0.01, 0.1],          # b1, b2
    "MODIFIED_EXP_EQ9": [0.1],            # b0
}

def initial_guess(name, X, y):
    """
    Heurísticas simples. Se quiser algo mais esperto por equação, personalize aqui.
    """
    # exemplo especial p/ DMAXPROJETADO_EQ1 (estimativa fechada aproximada)
    if name == "DMAXPROJETADO_EQ1":
        try:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 3)
            IDADE1, IDADE2, DMAX1 = X[:, 0], X[:, 1], X[:, 2]
            denom = (1 - IDADE1 / IDADE2)
            denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
            r = (y - DMAX1 * (IDADE1 / IDADE2)) / denom
            b0 = float(np.nanmedian(r))
            if np.isfinite(b0):
                return [b0]
        except Exception:
            pass
        return INITIAL_GUESSES["DMAXPROJETADO_EQ1"]

    # Default: usar os chutes do dicionário
    if name in INITIAL_GUESSES:
        return list(INITIAL_GUESSES[name])

    return None
