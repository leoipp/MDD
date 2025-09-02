# Eq.py
import numpy as np

# =========================
# Suas funções originais
# =========================
def DMAXPROJETADO_EQ1(Vars, b0):
    IDADE1, IDADE2, DMAX1 = Vars
    return DMAX1 * (IDADE1/IDADE2) + b0 * (1 - IDADE1/IDADE2)

def DMAXPROJETADO_EQ2(Vars, b0, b1):
    IDADE1, IDADE2, DMAX1 = Vars
    return DMAX1 * np.exp(-b0 ** ((IDADE2 ** b1) - (IDADE1 ** b1)))

def DMINPROJETADO_EQ3(Vars, b0, b1):
    IDADE1, IDADE2, DMIN1 = Vars
    return DMIN1 * np.exp(-b0 ** ((IDADE2 ** b1) - (IDADE1 ** b1)))

def BETAPROJETADO_EQ4(Vars, b0):
    IDADE1, IDADE2, DMAX2, BETA1 = Vars
    return BETA1 * (IDADE1/IDADE2) + b0 * (1 - IDADE1/IDADE2) * DMAX2

def ESCALAPROJETADO_EQ5(Vars, b0):
    IDADE1, IDADE2, ESCALA1 = Vars
    return ESCALA1 * (IDADE1/IDADE2) + b0 * (1 - IDADE1/IDADE2)

def ESCALAPROJETADO_EQ6(Vars, b0, b1):
    IDADE1, IDADE2, ESCALA1 = Vars
    return np.log()
# =========================
# Wrappers compatíveis com curve_fit
#  - Aceitam X como array/matriz com 3 colunas:
#    [IDADE1, IDADE2, DMAX1]
#  - Retornam vetor y (shape (n_obs,))
# =========================
def _to_Vars(X, n_vars=3):
    """
    Converte X (n_obs, n_vars) ou (n_vars, n_obs) em tuple Vars=(IDADE1, IDADE2, DMAX1),
    cada um como vetor (n_obs,).
    """
    X = np.asarray(X)
    if X.ndim == 1:
        # caso raro: uma observação só -> (3,)
        if X.size != n_vars:
            raise ValueError(f"Esperado {n_vars} variáveis em X; recebi {X.size}.")
        # transforma em (1,3)
        X = X.reshape(1, -1)
    # Se veio transposto (n_vars, n_obs), ajusta
    if X.shape[0] == n_vars and X.shape[1] != n_vars:
        # (3, n_obs) -> (n_obs, 3)
        X = X.T
    if X.shape[1] != n_vars:
        raise ValueError(f"X deve ter {n_vars} colunas (IDADE1, IDADE2, DMAX1); shape={X.shape}")
    IDADE1 = X[:, 0]
    IDADE2 = X[:, 1]
    DMAX1  = X[:, 2]
    return (IDADE1, IDADE2, DMAX1)

def DMAXPROJETADO_EQ1_fit(X, b0):
    Vars = _to_Vars(X, n_vars=3)
    return DMAXPROJETADO_EQ1(Vars, b0)

def DMAXPROJETADO_EQ2_fit(X, b0, b1):
    Vars = _to_Vars(X, n_vars=3)
    return DMAXPROJETADO_EQ2(Vars, b0, b1)

# =========================
# Registry para o Dash
# =========================
EQUATIONS = {
    "DMAXPROJETADO_EQ1": DMAXPROJETADO_EQ1_fit,
    "DMAXPROJETADO_EQ2": DMAXPROJETADO_EQ2_fit,
}

# Chutes iniciais (ajuste conforme seus dados)
INITIAL_GUESSES = {
    "DMAXPROJETADO_EQ1": [0.5],      # b0
    "DMAXPROJETADO_EQ2": [0.5, 1.0], # b0, b1
}

def initial_guess(name, X, y):
    """
    Opcional: heurística de chute inicial mais esperta.
    X chega como array (n_obs, 3) com [IDADE1, IDADE2, DMAX1] (ou equivalente).
    """
    if name == "DMAXPROJETADO_EQ1":
        # b0 ~ mediana(y - DMAX1*(IDADE1/IDADE2) + ... ) simplificada:
        try:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            IDADE1, IDADE2, DMAX1 = X[:,0], X[:,1], X[:,2]
            r = (y - DMAX1 * (IDADE1/IDADE2)) / (1 - IDADE1/IDADE2 + 1e-9)
            b0 = float(np.nanmedian(r))
            if np.isfinite(b0):
                return [b0]
        except Exception:
            pass
        return INITIAL_GUESSES["DMAXPROJETADO_EQ1"]

    if name == "DMAXPROJETADO_EQ2":
        # ponto de partida genérico
        return INITIAL_GUESSES["DMAXPROJETADO_EQ2"]

    return None
