"""
DeePC Playground — Backend Utilities
======================================
Implements:
  • Data generation & system simulation (PRBS)
  • Optimized Hankel matrix construction
  • DeePCSolver  : Data-Enabled Predictive Control (Coulson et al., 2019)
  • ClassicMPCSolver : Model-based MPC benchmark
  • Performance metrics (ISE, IAE, TV, Overshoot, Settling Time)

References
----------
[1] J.C. Willems et al., "A Note on Persistency of Excitation," 2005.
[2] J. Coulson et al., "Data-Enabled Predictive Control," ECC 2019.
[3] I. Markovsky & F. Dörfler, "Behavioral systems theory in data-driven
    analysis, signal processing, and control," Annual Reviews in Control, 2021.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import cvxpy as cp
import numpy as np
import scipy.signal as signal


# ═══════════════════════════════════════════════════════════════
# SYSTEM LIBRARY
# ═══════════════════════════════════════════════════════════════

SYSTEMS: Dict[str, Dict] = {
    "2. Derece: Kütle-Yay-Sönüm": {
        "num":   [0.0, 0.0, 0.15],
        "den":   [1.0, -1.5, 0.7],
        "dt":    1.0,
        "order": 2,
        "desc":  "Endüstriyel robot eklemi · Araç süspansiyonu · Yapısal titreşim sönümleme",
        "u_min": -3.0,
        "u_max":  3.0,
        "ref":    2.0,
    },
    "1. Derece: DC Motor": {
        "num":   [0.0, 0.2],
        "den":   [1.0, -0.8],
        "dt":    1.0,
        "order": 1,
        "desc":  "Drone motor hızı · Konveyör bant · Pompa kontrolü",
        "u_min": -3.0,
        "u_max":  3.0,
        "ref":    1.5,
    },
    "2. Derece: Yavaş Termal Proses": {
        "num":   [0.0, 0.0, 0.05],
        "den":   [1.0, -1.8, 0.82],
        "dt":    1.0,
        "order": 2,
        "desc":  "Fırın sıcaklık kontrolü · Kimyasal reaktör · HVAC sistemi",
        "u_min":  0.0,
        "u_max":  5.0,
        "ref":    3.0,
    },
}

PRESETS: Dict[str, Dict] = {
    "⚡ Hızlı & Agresif": {
        "T_ini": 8,  "N_p": 15, "Q": 30.0, "R": 0.1,  "lambda_g":  5.0,
        "N_total": 250, "noise": 0.02,
        "info": "Yüksek Q/R → hızlı yanıt, daha büyük kontrol hareketleri",
    },
    "⚖️ Dengeli (Varsayılan)": {
        "T_ini": 10, "N_p": 20, "Q": 10.0, "R": 0.5,  "lambda_g": 10.0,
        "N_total": 300, "noise": 0.05,
        "info": "Hız ve enerji arasında optimal denge — başlangıç noktası olarak ideal",
    },
    "🛡️ Enerji Tasarruflu": {
        "T_ini": 12, "N_p": 25, "Q":  5.0, "R": 5.0,  "lambda_g": 20.0,
        "N_total": 350, "noise": 0.05,
        "info": "Yüksek R → kontrol eforu minimize; yavaş ama aktuatör dostu",
    },
    "🏭 Gürültülü Endüstriyel": {
        "T_ini": 15, "N_p": 20, "Q": 15.0, "R": 1.0,  "lambda_g": 50.0,
        "N_total": 400, "noise": 0.15,
        "info": "Yüksek λ önceliği → gürültülü sensör ortamında kararlılık",
    },
}


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def build_hankel(data: np.ndarray, L: int) -> np.ndarray:
    """
    Construct a depth-L Hankel matrix from a 1-D data array.

    Uses ``numpy.lib.stride_tricks`` — O(1) memory allocation, no loops.

    Parameters
    ----------
    data : array_like, shape (N,)
    L    : int — number of rows  (= T_ini + N_p)

    Returns
    -------
    H : ndarray, shape (L, N-L+1)

    Raises
    ------
    ValueError if data is shorter than L.
    """
    data = np.asarray(data, dtype=np.float64)
    N    = len(data)
    cols = N - L + 1
    if cols <= 0:
        raise ValueError(
            f"Veri çok kısa: N={N}, L={L} için en az {L} örnek gereklidir."
        )
    stride = data.strides[0]
    H = np.lib.stride_tricks.as_strided(
        data, shape=(L, cols), strides=(stride, stride)
    )
    return np.array(H, copy=True)


def simulate_system(
    N_total: int,
    noise_level: float = 0.01,
    system_name: str = "2. Derece: Kütle-Yay-Sönüm",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a selected LTI system driven by PRBS (Pseudo-Random Binary Sequence).
    PRBS ensures *persistent excitation* across all frequencies, a prerequisite
    for Willems' Fundamental Lemma.

    Parameters
    ----------
    N_total     : number of samples
    noise_level : std-dev of additive Gaussian measurement noise
    system_name : key in ``SYSTEMS``
    seed        : optional random seed for reproducibility

    Returns
    -------
    (u_data, y_data) : each ndarray of shape (N_total,)
    """
    if seed is not None:
        np.random.seed(seed)

    p      = SYSTEMS[system_name]
    sys_tf = signal.TransferFunction(p["num"], p["den"], dt=p["dt"])

    # PRBS ∈ {-1, +1}
    u_data = (np.random.randint(0, 2, N_total) * 2 - 1).astype(np.float64)

    _, y_out = signal.dlsim(sys_tf, u_data)
    y_data   = y_out.flatten()

    if noise_level > 0:
        y_data = y_data + np.random.normal(0.0, noise_level, N_total)

    return u_data, y_data


def get_ss_matrices(
    system_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return discrete-time state-space matrices (A, B, C, D) for a given system.
    Used by ``ClassicMPCSolver``.
    """
    p = SYSTEMS[system_name]
    A, B, C, D = signal.tf2ss(p["num"], p["den"])
    return (
        np.asarray(A, dtype=float),
        np.asarray(B, dtype=float),
        np.asarray(C, dtype=float),
        np.asarray(D, dtype=float),
    )


def simulate_step(
    y_hist: np.ndarray,
    u_hist: np.ndarray,
    u_new: float,
    system_name: str,
    noise_level: float = 0.0,
) -> float:
    """
    Advance the system one step via its ARX difference equation.

    Supports 1st- and 2nd-order systems from SYSTEMS dict.
    y[k] is computed from past y and u values only (no model object needed).
    """
    p   = SYSTEMS[system_name]
    den = p["den"]
    num = p["num"]

    if p["order"] == 1:
        # y[k] = -a1·y[k-1] + b1·u[k-1]
        y_new = -den[1] * y_hist[-1] + num[1] * u_hist[-1]
    else:  # order == 2
        # y[k] = -a1·y[k-1] - a2·y[k-2] + b2·u[k-2]
        y_new = (
            -den[1] * y_hist[-1]
            - den[2] * y_hist[-2]
            + num[-1] * (u_hist[-2] if len(u_hist) >= 2 else 0.0)
        )

    if noise_level > 0:
        y_new += float(np.random.normal(0.0, noise_level))

    return float(y_new)


def generate_reference(
    ref_type: str, ref_value: float, steps: int
) -> np.ndarray:
    """
    Generate a reference trajectory.

    Supported types
    ---------------
    "Sabit Adım"   — constant setpoint
    "Kare Dalga"   — square wave toggling between 0 and ref_value
    "Sinüzoidal"   — sinusoidal around ref_value/2
    "Ramp"         — linearly increasing to ref_value
    """
    t = np.arange(steps)
    if ref_type == "Sabit Adım":
        return np.ones(steps) * ref_value
    elif ref_type == "Kare Dalga":
        period = max(steps // 3, 10)
        sq     = signal.square(2 * np.pi * t / period)
        return ref_value * (0.5 + 0.5 * sq)
    elif ref_type == "Sinüzoidal":
        period = max(steps // 2, 20)
        return ref_value * (0.5 + 0.5 * np.sin(2 * np.pi * t / period))
    elif ref_type == "Ramp":
        return np.clip(ref_value * t / max(steps // 2, 1), 0.0, ref_value)
    else:
        return np.ones(steps) * ref_value


def compute_metrics(
    y_sim: np.ndarray,
    u_sim: np.ndarray,
    ref:   np.ndarray | float,
    T_ini: int,
    solve_times: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Standard closed-loop performance metrics.

    ISE  — Integral Squared Error      : penalises large deviations heavily
    IAE  — Integral Absolute Error     : penalises all deviations equally
    TV   — Total Variation of u        : measures control effort smoothness
    Overshoot (%) — peak beyond setpoint
    Settling step — last step with |e| > 5 % band
    Avg solve ms  — mean solver wall-time per step (if provided)
    """
    # Referans sinyalinin uzunluğu kadar (sim_steps) veriyi al, sondaki fazlalığı kırp
    n_steps = len(ref) if not np.isscalar(ref) else (len(y_sim) - T_ini)
    y = y_sim[T_ini : T_ini + n_steps]
    u = u_sim[T_ini : T_ini + n_steps]

    if np.isscalar(ref):
        r_arr = np.ones(len(y)) * ref
        ref_scalar = float(ref)
    else:
        r_arr      = np.asarray(ref)[: len(y)]
        ref_scalar = float(np.mean(ref))

    e   = y - r_arr
    ISE = float(np.sum(e ** 2))
    IAE = float(np.sum(np.abs(e)))
    TV  = float(np.sum(np.abs(np.diff(u))))

    if abs(ref_scalar) > 1e-9:
        overshoot = float(max(0.0, (np.max(y) - ref_scalar) / abs(ref_scalar) * 100.0))
    else:
        overshoot = 0.0

    band      = max(0.05 * abs(ref_scalar), 0.01)
    settling  = len(y)
    for i in range(len(y) - 1, -1, -1):
        if abs(e[i]) > band:
            settling = i + 1
            break
    else:
        settling = 0

    result: Dict[str, Any] = {
        "ISE":               round(ISE, 4),
        "IAE":               round(IAE, 4),
        "TV (Kontrol Eforu)": round(TV, 4),
        "Aşım (%)":          round(overshoot, 2),
        "Oturma Adımı":      int(settling),
    }
    if solve_times:
        result["Ort. Çözüm (ms)"] = round(float(np.mean(solve_times)), 2)
    return result


def auto_interpret(
    deepc_metrics: Dict[str, Any],
    mpc_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a plain-language interpretation of the simulation results.
    If both DeePC and Classic MPC metrics are provided, produce a comparison.
    """
    lines: list[str] = []
    ise   = deepc_metrics.get("ISE", 0)
    ov    = deepc_metrics.get("Aşım (%)", 0)
    tv    = deepc_metrics.get("TV (Kontrol Eforu)", 0)
    sett  = deepc_metrics.get("Oturma Adımı", 0)

    if ise < 1.0:
        lines.append("✅ **Mükemmel takip:** ISE < 1 — sistem referansı çok küçük hatayla izliyor.")
    elif ise < 10.0:
        lines.append("🟡 **Kabul edilebilir takip:** ISE makul aralıkta, Q parametresini artırarak geliştirilebilir.")
    else:
        lines.append("⚠️ **Yüksek hata:** ISE yüksek — λ veya N_p değerlerini, ya da veri kalitesini kontrol edin.")

    if ov > 20:
        lines.append(f"⚠️ **Aşım yüksek ({ov:.1f}%):** Q/R oranını düşürün veya N_p'yi artırın.")
    elif ov > 5:
        lines.append(f"🟡 **Orta aşım ({ov:.1f}%):** Çoğu endüstriyel uygulama için kabul edilebilir.")
    else:
        lines.append(f"✅ **Düşük aşım ({ov:.1f}%):** Hassas kontrol gereksinimlerini karşılıyor.")

    if tv > 20:
        lines.append("⚠️ **Kontrol sinyali çok hareketli:** R değerini artırarak aktuatör yorulmasını azaltın.")
    else:
        lines.append("✅ **Kontrol sinyali düzgün:** Aktuatör ömrü açısından sağlıklı.")

    if mpc_metrics:
        d_ise = deepc_metrics.get("ISE", 1)
        m_ise = mpc_metrics.get("ISE", 1)
        ratio = (d_ise - m_ise) / max(m_ise, 1e-9) * 100
        if abs(ratio) < 10:
            lines.append(
                "📊 **DeePC ↔ Klasik MPC:** Performanslar birbirine çok yakın "
                f"(ISE farkı {abs(ratio):.1f}%). Model olmadan eşdeğer sonuç!"
            )
        elif ratio < 0:
            lines.append(
                f"🏆 **DeePC kazandı:** Klasik MPC'ye kıyasla ISE %{abs(ratio):.1f} daha düşük — "
                "model belirsizliği olmayan sistemlerde DeePC avantajlı olabilir."
            )
        else:
            lines.append(
                f"📊 **Klasik MPC avantajlı:** ISE %{abs(ratio):.1f} daha düşük — "
                "ancak Klasik MPC'nin tam model bilgisine ihtiyaç duyduğunu unutmayın."
            )

    return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# DeePC SOLVER
# ═══════════════════════════════════════════════════════════════

class DeePCSolver:
    """
    Data-Enabled Predictive Control (DeePC).

    Encodes Willems' Fundamental Lemma as an optimisation constraint:
    the past & future trajectories of any LTI system live in the column
    space of its Hankel matrix — no (A,B,C,D) model required.

    The QP is built **once** with CVXPY Parameters; subsequent ``solve()``
    calls only update parameter values → fast warm-started solves.

    Cost
    ----
    min  Q·‖y_f − r‖² + R·‖u_f‖² + λ·‖g‖²
    s.t. [U_p; Y_p; U_f; Y_f] g = [u_ini; y_ini; u_f; y_f]
         u_min ≤ u_f ≤ u_max
    """

    def __init__(
        self,
        u_data:   np.ndarray,
        y_data:   np.ndarray,
        T_ini:    int,
        N_p:      int,
        lambda_g: float = 10.0,
        Q:        float = 1.0,
        R:        float = 0.1,
        u_min:    float = -2.0,
        u_max:    float =  2.0,
    ) -> None:
        self.T_ini    = T_ini
        self.N_p      = N_p
        self.lambda_g = lambda_g
        self.Q        = Q
        self.R        = R
        self.u_min    = u_min
        self.u_max    = u_max

        L = T_ini + N_p

        # ── Hankel matrices (built once) ──────────────────────
        Hu = build_hankel(u_data, L)
        Hy = build_hankel(y_data, L)

        self.U_p = Hu[:T_ini, :]
        self.U_f = Hu[T_ini:,  :]
        self.Y_p = Hy[:T_ini, :]
        self.Y_f = Hy[T_ini:,  :]
        self.N_g = Hu.shape[1]

        # Store full matrices for visualisation
        self._Hu  = Hu
        self._Hy  = Hy

        # ── Rank / excitation analysis ────────────────────────
        H_full = np.vstack([Hu, Hy])
        self.hankel_rank     = int(np.linalg.matrix_rank(H_full, tol=1e-8))
        self.hankel_cols     = H_full.shape[1]
        self.hankel_rows     = H_full.shape[0]
        # Heuristic: rank should cover at least T_ini + N_p
        self.is_pe = self.hankel_rank >= min(L, self.hankel_cols) * 0.85

        # ── Build parametric CVXPY problem ────────────────────
        self._build_problem()

    # ----------------------------------------------------------
    def _build_problem(self) -> None:
        g   = cp.Variable(self.N_g, name="g")
        u_f = cp.Variable(self.N_p, name="u_f")
        y_f = cp.Variable(self.N_p, name="y_f")

        self._u_ini_p = cp.Parameter(self.T_ini, name="u_ini")
        self._y_ini_p = cp.Parameter(self.T_ini, name="y_ini")
        self._r_p     = cp.Parameter(self.N_p,   name="r")

        cost = (
            self.Q        * cp.sum_squares(y_f - self._r_p)
            + self.R      * cp.sum_squares(u_f)
            + self.lambda_g * cp.sum_squares(g)
        )
        constraints = [
            self.U_p @ g == self._u_ini_p,
            self.Y_p @ g == self._y_ini_p,
            self.U_f @ g == u_f,
            self.Y_f @ g == y_f,
            u_f >= self.u_min,
            u_f <= self.u_max,
        ]

        self._prob = cp.Problem(cp.Minimize(cost), constraints)
        self._g    = g
        self._u_f  = u_f
        self._y_f  = y_f

    # ----------------------------------------------------------
    def solve(
        self,
        u_ini: np.ndarray,
        y_ini: np.ndarray,
        r:     Any,
        solver_name: str = "OSQP",
    ) -> Tuple[Optional[float], Optional[np.ndarray], Optional[float], str, float]:
        """
        Solve DeePC for one time step (receding-horizon principle).

        Parameters
        ----------
        u_ini       : past inputs,  shape (T_ini,)
        y_ini       : past outputs, shape (T_ini,)
        r           : reference — scalar or array (N_p,)
        solver_name : "OSQP" | "ECOS" | "SCS"

        Returns
        -------
        (u_opt_first, y_f_predicted, cost_value, status_str, solve_time_ms)
        """
        if np.isscalar(r):
            r = np.full(self.N_p, float(r))

        self._u_ini_p.value = np.asarray(u_ini, dtype=float)
        self._y_ini_p.value = np.asarray(y_ini, dtype=float)
        self._r_p.value     = np.asarray(r,     dtype=float)

        _solvers = {"OSQP": cp.OSQP, "ECOS": cp.ECOS, "SCS": cp.SCS}
        cvx_s    = _solvers.get(solver_name, cp.OSQP)

        t0 = time.perf_counter()
        try:
            self._prob.solve(
                solver=cvx_s,
                warm_start=True,
                max_iter=10_000,
                eps_abs=1e-5,
                eps_rel=1e-5,
            )
        except Exception as exc:
            dt = (time.perf_counter() - t0) * 1e3
            return None, None, None, f"SolverError: {str(exc)[:60]}", dt

        dt     = (time.perf_counter() - t0) * 1e3
        status = str(self._prob.status)

        if status in ("optimal", "optimal_inaccurate"):
            return (
                float(self._u_f.value[0]),
                self._y_f.value.copy(),
                float(self._prob.value),
                status,
                dt,
            )
        return None, None, None, status, dt


# ═══════════════════════════════════════════════════════════════
# CLASSIC MPC SOLVER  (model-based benchmark)
# ═══════════════════════════════════════════════════════════════

class ClassicMPCSolver:
    """
    Classic Model-Based Predictive Control (MPC) — benchmarking reference.

    Requires exact knowledge of (A, B, C, D).
    Unlike DeePC, it cannot operate without a system model.

    The state-space dynamics form the equality constraints.
    The CVXPY problem is parametric in (x0, r) for efficiency.
    """

    def __init__(
        self,
        A:    np.ndarray,
        B:    np.ndarray,
        C:    np.ndarray,
        N_p:  int,
        Q:    float = 1.0,
        R:    float = 0.1,
        u_min: float = -2.0,
        u_max: float =  2.0,
    ) -> None:
        self.A     = np.asarray(A, dtype=float)
        self.B     = np.asarray(B, dtype=float)
        self.C     = np.asarray(C, dtype=float)
        self.N_p   = N_p
        self.Q     = Q
        self.R     = R
        self.u_min = u_min
        self.u_max = u_max
        self.nx    = A.shape[0]

        self._build_problem()

    # ----------------------------------------------------------
    def _build_problem(self) -> None:
        x     = cp.Variable((self.nx, self.N_p + 1), name="x")
        u     = cp.Variable(self.N_p,                name="u")
        x0_p  = cp.Parameter(self.nx,               name="x0")
        r_p   = cp.Parameter(self.N_p,              name="r")

        cost = 0.0
        cons = [x[:, 0] == x0_p]

        for k in range(self.N_p):
            y_k   = self.C @ x[:, k]
            cost += self.Q * cp.sum_squares(y_k - r_p[k])
            cost += self.R * cp.square(u[k])
            cons += [
                x[:, k + 1] == self.A @ x[:, k] + self.B.flatten() * u[k],
                u[k] >= self.u_min,
                u[k] <= self.u_max,
            ]
        # Terminal cost
        cost += self.Q * cp.sum_squares(self.C @ x[:, self.N_p] - r_p[-1])

        self._prob  = cp.Problem(cp.Minimize(cost), cons)
        self._x     = x
        self._u     = u
        self._x0_p  = x0_p
        self._r_p   = r_p

    # ----------------------------------------------------------
    def solve(
        self,
        x0:  np.ndarray,
        r:   Any,
        solver_name: str = "OSQP",
    ) -> Tuple[Optional[float], Optional[np.ndarray], str, float]:
        """
        Parameters
        ----------
        x0  : current state, shape (nx,)
        r   : reference — scalar or array (N_p,)

        Returns
        -------
        (u_opt_first, y_predicted, status_str, solve_time_ms)
        """
        if np.isscalar(r):
            r = np.full(self.N_p, float(r))

        self._x0_p.value = np.asarray(x0, dtype=float)
        self._r_p.value  = np.asarray(r,  dtype=float)

        _solvers = {"OSQP": cp.OSQP, "ECOS": cp.ECOS, "SCS": cp.SCS}
        t0       = time.perf_counter()

        try:
            self._prob.solve(
                solver=_solvers.get(solver_name, cp.OSQP),
                warm_start=True,
                max_iter=10_000,
                eps_abs=1e-5,
                eps_rel=1e-5,
            )
        except Exception as exc:
            dt = (time.perf_counter() - t0) * 1e3
            return None, None, f"SolverError: {str(exc)[:60]}", dt

        dt     = (time.perf_counter() - t0) * 1e3
        status = str(self._prob.status)

        if status in ("optimal", "optimal_inaccurate"):
            y_pred = np.array(
                [(self.C @ self._x.value[:, k]).item() for k in range(self.N_p)]
            )
            return float(self._u.value[0]), y_pred, status, dt

        return None, None, status, dt
