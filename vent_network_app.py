"""
OSV Motor Ventilation Network Analyser
======================================
Streamlit app with interactive pipe-circuit builder (LTSpice-style workflow),
Hardy-Cross solver, and 2D motor diagram.

Install:
    pip install streamlit plotly numpy pandas streamlit-plotly-events

Run:
    streamlit run vent_network_app.py
"""

import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Try to import streamlit-plotly-events (optional but enables click-to-build) ──
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except ImportError:
    HAS_PLOTLY_EVENTS = False

# ─────────────────────────────────────────────────────────────────────────────
# Physics constants
# ─────────────────────────────────────────────────────────────────────────────
RHO_AIR = 1.18      # kg/m³ at ~30 °C
MU_AIR  = 1.86e-5   # Pa·s dynamic viscosity
ROUGHNESS = 5e-4    # m  — painted steel / aluminium casting


def hydro_diam(w, h):
    return 2 * w * h / (w + h) if (w + h) > 0 else 1e-6


def friction_factor(Re, eps=ROUGHNESS, Dh=1.0):
    if Re < 1:
        return 1.0
    if Re < 2300:
        return 64 / Re
    # Swamee-Jain explicit approx to Colebrook-White
    return 0.25 / (math.log10(eps / (3.7 * Dh) + 5.74 / Re**0.9)) ** 2


def branch_resistance(Q, L, Dh, A, k_minor):
    """Return (dP, R) where dP = R * Q^n, n≈2 (turbulent)."""
    if A <= 0 or Dh <= 0 or abs(Q) < 1e-12:
        return 0.0, 1e6
    V  = Q / A
    Re = RHO_AIR * abs(V) * Dh / MU_AIR
    f  = friction_factor(Re, ROUGHNESS, Dh)
    dP = (f * L / Dh + k_minor) * 0.5 * RHO_AIR * V * abs(V)
    R  = abs(dP / Q) if abs(Q) > 1e-12 else 1e6
    return dP, R


# ─────────────────────────────────────────────────────────────────────────────
# Hardy-Cross solver
# ─────────────────────────────────────────────────────────────────────────────
def hardy_cross_solve(branches, loops, Q_init, fan_dP, n_iter=300, tol=1e-6):
    """
    branches : list of dicts  {id, L, Dh, A, k}
    loops    : list of lists  [{id, sign}]  — independent loops
    Q_init   : dict {branch_id: Q}
    fan_dP   : driving pressure [Pa]

    For a simple parallel network (all branches share same ΔP = fan_dP)
    Hardy-Cross converges in <50 iterations.
    """
    flows = dict(Q_init)

    for _ in range(n_iter):
        max_dQ = 0
        for loop in loops:
            hL, dhL = 0.0, 0.0
            for item in loop:
                bid, sign = item["id"], item["sign"]
                b = next((x for x in branches if x["id"] == bid), None)
                if b is None:
                    continue
                Q_signed = flows[bid] * sign
                dP, _ = branch_resistance(abs(Q_signed), b["L"], b["Dh"], b["A"], b["k"])
                signed_dP = math.copysign(dP, Q_signed)
                hL  += signed_dP
                dhL += dP / max(abs(Q_signed), 1e-9)
            dQ = -hL / (2 * dhL) if dhL > 1e-12 else 0
            max_dQ = max(max_dQ, abs(dQ))
            for item in loop:
                flows[item["id"]] = flows[item["id"]] + item["sign"] * dQ
        if max_dQ < tol:
            break

    # Compute per-branch results
    results = []
    for b in branches:
        Q  = flows[b["id"]]
        V  = Q / b["A"] if b["A"] > 0 else 0
        Re = RHO_AIR * abs(V) * b["Dh"] / MU_AIR if b["Dh"] > 0 else 0
        dP, _ = branch_resistance(abs(Q), b["L"], b["Dh"], b["A"], b["k"])
        results.append({
            **b,
            "Q_m3s": Q,
            "Q_Ls":  Q * 1000,
            "V_ms":  V,
            "Re":    Re,
            "dP_Pa": dP,
        })
    return results, flows


# ─────────────────────────────────────────────────────────────────────────────
# Session-state: circuit graph
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    if "nodes" not in st.session_state:
        # Default 2-node parallel network matching the motor topology
        st.session_state.nodes = {
            "N0": {"x": 1, "y": 3, "label": "Inlet\n(shaft end)", "type": "source"},
            "N1": {"x": 5, "y": 3, "label": "Outlet\n(mid stator)", "type": "sink"},
        }
    if "branches" not in st.session_state:
        st.session_state.branches = {
            "AG": {"from": "N0", "to": "N1", "label": "Airgap",
                   "L": 0.65, "w": 0.009, "h": 0.009, "type": "annular",
                   "k": 1.0, "color": "#2f81f7"},
            "SV": {"from": "N0", "to": "N1", "label": "Stator vents",
                   "L": 0.035, "w": 0.010, "h": 0.030, "type": "rect",
                   "k": 0.5, "color": "#3fb950", "n_parallel": 8},
            "RV": {"from": "N0", "to": "N1", "label": "Rotor vents",
                   "L": 0.030, "w": 0.008, "h": 0.025, "type": "rect",
                   "k": 0.5, "color": "#d29922", "n_parallel": 6},
        }
    if "add_mode" not in st.session_state:
        st.session_state.add_mode = None   # None | "node" | "branch_from" | "branch_to"
    if "pending_from" not in st.session_state:
        st.session_state.pending_from = None
    if "node_counter" not in st.session_state:
        st.session_state.node_counter = 2
    if "branch_counter" not in st.session_state:
        st.session_state.branch_counter = 3
    if "selected" not in st.session_state:
        st.session_state.selected = None


# ─────────────────────────────────────────────────────────────────────────────
# Build physics branches from session state
# ─────────────────────────────────────────────────────────────────────────────
def build_physics_branches():
    phys = []
    for bid, b in st.session_state.branches.items():
        n = b.get("n_parallel", 1)
        if b["type"] == "annular":
            # Treat as single annular channel — user sets w=gap, h=unused
            Dh = b["w"] * 2          # annular Dh ≈ 2g for thin gap
            A  = math.pi * Dh * b.get("mean_r", 0.19)   # π * Dh * r_mean
            A  = max(A, b["w"] * b["h"])  # fallback
        else:
            Dh = hydro_diam(b["w"], b["h"])
            A  = b["w"] * b["h"] * n
        phys.append({"id": bid, "label": b["label"], "L": b["L"],
                     "Dh": Dh, "A": A, "k": b["k"]})
    return phys


def build_loops():
    """Auto-detect independent loops for parallel networks."""
    # For a simple parallel network (all branches share 2 nodes)
    # generate loops as pairs: branch_0 vs branch_i
    bids = list(st.session_state.branches.keys())
    if len(bids) < 2:
        return []
    loops = []
    b0 = bids[0]
    for bi in bids[1:]:
        loops.append([{"id": b0, "sign": 1}, {"id": bi, "sign": -1}])
    return loops


def initial_flows(phys, fan_dP):
    """Distribute fan_dP equally across branches as starting guess."""
    flows = {}
    for b in phys:
        V_est = math.sqrt(2 * fan_dP / RHO_AIR) * 0.3
        flows[b["id"]] = max(V_est * b["A"], 1e-6)
    return flows


# ─────────────────────────────────────────────────────────────────────────────
# Circuit diagram (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
def draw_circuit(results_map=None):
    nodes  = st.session_state.nodes
    branches = st.session_state.branches

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=20, r=20, t=30, b=20),
        height=420,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#1c2330", zeroline=False,
                   range=[-0.5, 7.5], tickfont=dict(color="#484f58")),
        yaxis=dict(showgrid=True, gridcolor="#1c2330", zeroline=False,
                   range=[0, 6], tickfont=dict(color="#484f58"),
                   scaleanchor="x", scaleratio=1),
        clickmode="event",
        title=dict(text="Circuit Builder  —  click nodes/branches to inspect",
                   font=dict(color="#8b949e", size=12), x=0.01),
    )

    # ── Draw branches ────────────────────────────────────────────────────
    for bid, b in branches.items():
        n0 = nodes.get(b["from"])
        n1 = nodes.get(b["to"])
        if not n0 or not n1:
            continue
        x0, y0 = n0["x"], n0["y"]
        x1, y1 = n1["x"], n1["y"]

        # Offset parallel branches so they don't overlap
        bids_list = list(branches.keys())
        idx = bids_list.index(bid)
        offset = (idx - (len(bids_list) - 1) / 2) * 0.5

        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + offset

        # Bezier-ish route via midpoint
        pts_x = [x0, mx, x1]
        pts_y = [y0, my, y1]

        r = results_map.get(bid) if results_map else None
        color = b["color"]
        lw = 3 if r else 2
        label = b["label"]
        if r:
            label += f"<br>Q={r['Q_Ls']:.2f} L/s  V={r['V_ms']:.2f} m/s<br>ΔP={r['dP_Pa']:.1f} Pa  Re={r['Re']:.0f}"

        fig.add_trace(go.Scatter(
            x=pts_x, y=pts_y, mode="lines",
            line=dict(color=color, width=lw),
            hoverinfo="text", hovertext=label,
            name=bid,
        ))

        # Arrow at midpoint
        dx = x1 - x0; dy = my - y0
        mag = math.sqrt(dx**2 + dy**2) or 1
        ax = dx / mag * 0.25; ay = dy / mag * 0.25
        fig.add_annotation(
            x=mx + ax, y=my + ay, ax=mx - ax, ay=my - ay,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowcolor=color, arrowwidth=2,
        )

        # Branch label
        fig.add_trace(go.Scatter(
            x=[mx], y=[my + 0.18],
            mode="text",
            text=[b["label"]],
            textfont=dict(color=color, size=11),
            hoverinfo="skip",
        ))

    # ── Draw nodes ───────────────────────────────────────────────────────
    for nid, n in nodes.items():
        col = {"source": "#3fb950", "sink": "#f85149"}.get(n["type"], "#2f81f7")
        fig.add_trace(go.Scatter(
            x=[n["x"]], y=[n["y"]],
            mode="markers+text",
            marker=dict(size=22, color=col, line=dict(color="#e6edf3", width=2)),
            text=[nid],
            textfont=dict(color="#e6edf3", size=10),
            textposition="middle center",
            hovertext=n["label"],
            hoverinfo="text",
            name=nid,
            customdata=[nid],
        ))

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2D Motor Cross-section (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
def draw_motor(geom):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=420, showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False,
                   range=[-0.35, 0.35], scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.35, 0.35]),
        title=dict(text="Motor End View", font=dict(color="#8b949e", size=13), x=0.01),
        legend=dict(font=dict(color="#8b949e", size=11), bgcolor="#161b22",
                    bordercolor="#30363d", borderwidth=1),
    )

    def hex_to_rgba(hex_color, alpha=0.15):
        """Convert #RRGGBB to rgba(r,g,b,alpha) string for Plotly."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    def circle_trace(r, color, fill, name, lw=1.5, dash="solid"):
        theta = np.linspace(0, 2 * np.pi, 200)
        x = r * np.cos(theta); y = r * np.sin(theta)
        return go.Scatter(x=list(x)+[None], y=list(y)+[None],
                          mode="lines", line=dict(color=color, width=lw, dash=dash),
                          fill=fill, fillcolor=hex_to_rgba(color, 0.15),
                          name=name, showlegend=True)

    rSO = geom["stator_od"] / 2
    rSI = geom["stator_id"] / 2
    rRO = geom["rotor_od"]  / 2
    rRI = geom["rotor_id"]  / 2

    fig.add_trace(circle_trace(rSO, "#607080", "toself",  "Stator frame", lw=2))
    fig.add_trace(circle_trace(rSI, "#8b949e", "toself",  "Stator bore",  lw=1))
    fig.add_trace(circle_trace(rRO, "#4a9eff", "toself",  "Rotor OD",     lw=2))
    fig.add_trace(circle_trace(rRI, "#30363d", "toself",  "Shaft/spider", lw=1))

    # Stator vent slots
    nSV = geom["n_stator_vents"]
    for i in range(nSV):
        ang = i / nSV * 2 * np.pi
        x0 = rSI * math.cos(ang); y0 = rSI * math.sin(ang)
        x1 = rSO * math.cos(ang); y1 = rSO * math.sin(ang)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1],
                                 mode="lines", line=dict(color="#2f81f7", width=3),
                                 showlegend=(i == 0), name="Stator vents",
                                 hoverinfo="skip"))

    # Rotor vent slots
    nRV = geom["n_rotor_vents"]
    for i in range(nRV):
        ang = (i + 0.5) / nRV * 2 * np.pi
        x0 = rRI * math.cos(ang); y0 = rRI * math.sin(ang)
        x1 = rRO * math.cos(ang); y1 = rRO * math.sin(ang)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1],
                                 mode="lines", line=dict(color="#3fb950", width=2),
                                 showlegend=(i == 0), name="Rotor vents",
                                 hoverinfo="skip"))

    # Airgap annotation
    fig.add_annotation(x=0, y=(rSI + rRO) / 2,
                       text=f"Airgap {(rSI-rRO)*1000:.1f} mm",
                       font=dict(color="#2f81f7", size=10),
                       showarrow=False)
    return fig


def draw_axial(geom):
    """Axial half-section using Plotly shapes."""
    L     = geom["active_length"]
    rSO   = geom["stator_od"] / 2
    rSI   = geom["stator_id"] / 2
    rRO   = geom["rotor_od"]  / 2
    rRI   = geom["rotor_id"]  / 2

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=300, showlegend=False,
        margin=dict(l=60, r=30, t=40, b=40),
        xaxis=dict(title="Axial position (m)", range=[-0.05, L + 0.05],
                   gridcolor="#1c2330", color="#8b949e"),
        yaxis=dict(title="Radius (m)", range=[0, rSO * 1.1],
                   gridcolor="#1c2330", color="#8b949e", scaleanchor="x", scaleratio=1),
        title=dict(text="Axial Half-Section", font=dict(color="#8b949e", size=13), x=0.01),
    )

    # Layer rectangles [y_inner, y_outer, color, name]
    layers = [
        (rSI, rSO, "rgba(96,112,128,0.4)",  "Stator frame"),
        (rRO, rSI, "rgba(47,129,247,0.10)", "Airgap"),
        (rRI, rRO, "rgba(74,158,255,0.25)", "Rotor"),
        (0,   rRI, "rgba(20,30,40,0.8)",    "Shaft"),
    ]
    for y0, y1, col, name in layers:
        fig.add_shape(type="rect", x0=0, x1=L, y0=y0, y1=y1,
                      fillcolor=col, line=dict(color="#30363d", width=1))
        fig.add_annotation(x=-0.02, y=(y0 + y1) / 2, text=name,
                           font=dict(color="#8b949e", size=9),
                           showarrow=False, xanchor="right")

    # Airgap flow arrow
    y_ag = (rSI + rRO) / 2
    fig.add_annotation(x=L * 0.6, y=y_ag, ax=0.02, ay=y_ag,
                       xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowcolor="#2f81f7", arrowwidth=2,
                       text="Axial flow", font=dict(color="#2f81f7", size=10))

    # Stator vent exit arrows (upward through stator at ~L/3 & 2L/3)
    for xv in [L / 3, 2 * L / 3]:
        fig.add_annotation(x=xv, y=rSO, ax=xv, ay=rSI,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowcolor="#3fb950", arrowwidth=2,
                           text="", )

    fig.add_annotation(x=L / 2, y=rSO * 1.05, text="Radial vent exit →",
                       font=dict(color="#3fb950", size=10), showarrow=False)

    # Inlet / outlet labels
    fig.add_annotation(x=0, y=rSI, text="INLET", font=dict(color="#3fb950", size=10, family="monospace"),
                       showarrow=False, yshift=6)
    fig.add_annotation(x=L, y=rSO, text="OUTLET", font=dict(color="#d29922", size=10, family="monospace"),
                       showarrow=False, yshift=8)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="OSV Motor · Ventilation Network Analyser",
        page_icon="🌀",
        layout="wide",
    )

    init_state()

    # ── Custom CSS ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    body, [data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    h1,h2,h3 { color: #e6edf3; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #2f81f7 !important; border-bottom: 2px solid #2f81f7; }
    .metric-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                  padding: 12px 16px; margin: 4px 0; }
    .metric-val { font-size: 22px; font-weight: 700; font-family: monospace; }
    .metric-lbl { font-size: 11px; color: #8b949e; font-family: monospace; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🌀 OSV Motor · Ventilation Network Analyser")
    st.caption("Parsons Peebles — 800 kW induction motor  ·  Hardy-Cross pipe network solver")

    if not HAS_PLOTLY_EVENTS:
        st.info(
            "**Tip:** Install `streamlit-plotly-events` for click-to-build circuit interaction:\n"
            "```\npip install streamlit-plotly-events\n```"
        )

    # ─────────────────────────────────────────────────────────────────────
    # SIDEBAR — global parameters + circuit editor
    # ─────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Global Parameters")
        fan_dP = st.number_input("Fan / inlet total pressure (Pa)", 10, 2000, 200, step=10)
        rho    = st.number_input("Air density ρ (kg/m³)", 0.9, 1.3, 1.18, step=0.01)
        mu     = st.number_input("Dynamic viscosity μ (×10⁻⁵ Pa·s)", 1.0, 2.5, 1.86, step=0.01)

        st.divider()
        st.markdown("### 🔧 Motor Geometry (for diagram)")
        stator_od = st.number_input("Stator OD (m)",      0.2, 2.0, 0.560, step=0.005, format="%.3f")
        stator_id = st.number_input("Stator bore ID (m)", 0.1, 1.5, 0.380, step=0.005, format="%.3f")
        rotor_od  = st.number_input("Rotor OD (m)",       0.1, 1.4, 0.376, step=0.005, format="%.3f")
        rotor_id  = st.number_input("Rotor ID (m)",       0.01, 0.5, 0.120, step=0.005, format="%.3f")
        active_l  = st.number_input("Active length (m)",  0.1, 3.0, 0.650, step=0.010, format="%.3f")
        n_sv      = st.number_input("No. stator vent packets", 1, 48, 8, step=1)
        n_rv      = st.number_input("No. rotor vent channels", 1, 48, 6, step=1)

        geom = {
            "stator_od": stator_od, "stator_id": stator_id,
            "rotor_od":  rotor_od,  "rotor_id":  rotor_id,
            "active_length": active_l,
            "n_stator_vents": n_sv,  "n_rotor_vents": n_rv,
        }

        st.divider()
        st.markdown("### 🧱 Circuit Editor")
        st.caption("Add/remove nodes and branches. Click a branch to edit its properties.")

        # Add node
        with st.expander("➕ Add node"):
            nn_label = st.text_input("Node label", "New node")
            nn_x     = st.slider("X position", 0.0, 7.0, 3.0, 0.5)
            nn_y     = st.slider("Y position", 0.0, 6.0, 1.5, 0.5)
            nn_type  = st.selectbox("Type", ["junction", "source", "sink"])
            if st.button("Add node"):
                nid = f"N{st.session_state.node_counter}"
                st.session_state.node_counter += 1
                st.session_state.nodes[nid] = {
                    "x": nn_x, "y": nn_y, "label": nn_label, "type": nn_type
                }
                st.rerun()

        # Add branch
        with st.expander("➕ Add branch"):
            node_ids = list(st.session_state.nodes.keys())
            nb_from  = st.selectbox("From node", node_ids, key="nb_from")
            nb_to    = st.selectbox("To node",   node_ids, key="nb_to", index=min(1, len(node_ids)-1))
            nb_label = st.text_input("Branch label", "New duct")
            nb_type  = st.selectbox("Duct cross-section", ["rect", "annular", "circular"])
            nb_L     = st.number_input("Length L (m)",      0.001, 10.0, 0.5,  step=0.01,  format="%.3f")
            nb_w     = st.number_input("Width / gap (m)",   0.001, 1.0,  0.01, step=0.001, format="%.4f")
            nb_h     = st.number_input("Height (m)",        0.001, 1.0,  0.03, step=0.001, format="%.4f")
            nb_k     = st.number_input("Minor loss coeff k", 0.0, 10.0, 0.5, step=0.1)
            nb_n     = st.number_input("No. in parallel",   1, 100, 1, step=1)
            nb_color = st.color_picker("Branch colour", "#2f81f7")
            if st.button("Add branch"):
                bid = f"B{st.session_state.branch_counter}"
                st.session_state.branch_counter += 1
                st.session_state.branches[bid] = {
                    "from": nb_from, "to": nb_to, "label": nb_label,
                    "L": nb_L, "w": nb_w, "h": nb_h, "type": nb_type,
                    "k": nb_k, "n_parallel": nb_n, "color": nb_color,
                }
                st.rerun()

        # Remove
        with st.expander("🗑 Remove element"):
            rm_type = st.radio("Remove", ["Branch", "Node"])
            if rm_type == "Branch":
                rm_id = st.selectbox("Branch", list(st.session_state.branches.keys()))
                if st.button("Remove branch"):
                    st.session_state.branches.pop(rm_id, None)
                    st.rerun()
            else:
                rm_id = st.selectbox("Node", list(st.session_state.nodes.keys()))
                if st.button("Remove node"):
                    st.session_state.nodes.pop(rm_id, None)
                    st.rerun()

        # Edit branch properties
        with st.expander("✏️ Edit branch properties"):
            if st.session_state.branches:
                eb_id  = st.selectbox("Branch to edit", list(st.session_state.branches.keys()))
                eb     = st.session_state.branches[eb_id]
                new_L  = st.number_input("L (m)",  0.001, 10.0, float(eb["L"]),  step=0.01,  key="eb_L",  format="%.3f")
                new_w  = st.number_input("w (m)",  0.001, 1.0,  float(eb["w"]),  step=0.001, key="eb_w",  format="%.4f")
                new_h  = st.number_input("h (m)",  0.001, 1.0,  float(eb["h"]),  step=0.001, key="eb_h",  format="%.4f")
                new_k  = st.number_input("k",      0.0,  10.0,  float(eb["k"]),  step=0.1,   key="eb_k")
                new_n  = st.number_input("N par.", 1,    100,   int(eb.get("n_parallel", 1)), step=1, key="eb_n")
                if st.button("Update branch"):
                    st.session_state.branches[eb_id].update(
                        {"L": new_L, "w": new_w, "h": new_h, "k": new_k, "n_parallel": new_n}
                    )
                    st.rerun()

        st.divider()
        # Export / Import
        with st.expander("💾 Save / Load circuit"):
            circuit_json = json.dumps({
                "nodes": st.session_state.nodes,
                "branches": st.session_state.branches,
            }, indent=2)
            st.download_button("⬇️ Download circuit JSON", circuit_json,
                               "circuit.json", "application/json")
            uploaded = st.file_uploader("⬆️ Load circuit JSON", type="json")
            if uploaded:
                data = json.load(uploaded)
                st.session_state.nodes    = data["nodes"]
                st.session_state.branches = data["branches"]
                st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    # Solve network
    # ─────────────────────────────────────────────────────────────────────
    phys_branches = build_physics_branches()
    loops         = build_loops()

    solve_ok = False
    results  = []
    results_map = {}
    total_Q = 0.0

    if phys_branches and loops:
        try:
            Q0 = initial_flows(phys_branches, fan_dP)
            results, flows = hardy_cross_solve(phys_branches, loops, Q0, fan_dP)
            results_map = {r["id"]: r for r in results}
            total_Q = sum(r["Q_m3s"] for r in results)
            solve_ok = True
        except Exception as e:
            st.error(f"Solver error: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Summary metrics
    # ─────────────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total flow", f"{total_Q*1000:.2f} L/s" if solve_ok else "—")
    with m2:
        st.metric("Fan ΔP", f"{fan_dP} Pa")
    with m3:
        ag = results_map.get("AG")
        st.metric("Airgap velocity", f"{ag['V_ms']:.2f} m/s" if ag else "—")
    with m4:
        st.metric("Branches", len(st.session_state.branches))

    # ─────────────────────────────────────────────────────────────────────
    # Tabs
    # ─────────────────────────────────────────────────────────────────────
    tab_circuit, tab_results, tab_motor, tab_theory = st.tabs(
        ["🔌 Circuit Builder", "📊 Network Results", "🔩 Motor Diagram", "📐 Theory"]
    )

    # ── CIRCUIT BUILDER ────────────────────────────────────────────────
    with tab_circuit:
        st.markdown("**Pipe network topology** — branches are parallel flow paths between nodes.")
        st.caption("Edit branches and nodes in the sidebar ↖  ·  Hover over branches to see solved flow values.")

        circuit_fig = draw_circuit(results_map if solve_ok else None)

        if HAS_PLOTLY_EVENTS:
            clicked = plotly_events(circuit_fig, click_event=True, key="circuit_click")
            if clicked:
                st.info(f"Clicked: {clicked}")
        else:
            st.plotly_chart(circuit_fig, use_container_width=True)

        # Node table
        with st.expander("Node positions"):
            node_rows = [{"ID": k, "Label": v["label"], "X": v["x"], "Y": v["y"], "Type": v["type"]}
                         for k, v in st.session_state.nodes.items()]
            st.dataframe(pd.DataFrame(node_rows), use_container_width=True, hide_index=True)

    # ── NETWORK RESULTS ────────────────────────────────────────────────
    with tab_results:
        if not solve_ok:
            st.warning("Network needs at least 2 branches sharing 2 nodes to solve.")
        else:
            # Results table
            df_rows = []
            for r in results:
                pct = r["Q_m3s"] / total_Q * 100 if total_Q > 0 else 0
                df_rows.append({
                    "Branch":      r["label"],
                    "Q (L/s)":     round(r["Q_Ls"], 3),
                    "V (m/s)":     round(r["V_ms"], 3),
                    "Re":          int(r["Re"]),
                    "ΔP (Pa)":     round(r["dP_Pa"], 2),
                    "Dh (mm)":     round(r["Dh"] * 1000, 2),
                    "A (cm²)":     round(r["A"] * 1e4, 3),
                    "Flow share %": round(pct, 1),
                })
            df = pd.DataFrame(df_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Bar chart — flow distribution
            fig_bar = go.Figure(go.Bar(
                x=[r["label"] for r in results],
                y=[r["Q_Ls"] for r in results],
                marker_color=[st.session_state.branches.get(r["id"], {}).get("color", "#2f81f7")
                              for r in results],
                text=[f"{r['Q_Ls']:.2f} L/s" for r in results],
                textposition="outside",
            ))
            fig_bar.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e"),
                yaxis=dict(title="Flow rate (L/s)", gridcolor="#1c2330"),
                xaxis=dict(gridcolor="#1c2330"),
                title="Flow distribution by branch",
                height=320, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Velocity / pressure drop comparison
            c1, c2 = st.columns(2)
            with c1:
                fig_v = go.Figure(go.Bar(
                    x=[r["label"] for r in results],
                    y=[r["V_ms"] for r in results],
                    marker_color="#d29922",
                    text=[f"{r['V_ms']:.2f}" for r in results],
                    textposition="outside",
                ))
                fig_v.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                                    font=dict(color="#8b949e"), height=260,
                                    yaxis=dict(title="m/s", gridcolor="#1c2330"),
                                    xaxis=dict(gridcolor="#1c2330"),
                                    title="Velocity", margin=dict(t=40, b=10))
                st.plotly_chart(fig_v, use_container_width=True)
            with c2:
                fig_dp = go.Figure(go.Bar(
                    x=[r["label"] for r in results],
                    y=[r["dP_Pa"] for r in results],
                    marker_color="#f85149",
                    text=[f"{r['dP_Pa']:.1f}" for r in results],
                    textposition="outside",
                ))
                fig_dp.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                                     font=dict(color="#8b949e"), height=260,
                                     yaxis=dict(title="Pa", gridcolor="#1c2330"),
                                     xaxis=dict(gridcolor="#1c2330"),
                                     title="Pressure drop", margin=dict(t=40, b=10))
                st.plotly_chart(fig_dp, use_container_width=True)

            st.caption(
                f"Air properties: ρ = {rho} kg/m³ · μ = {mu:.2e} Pa·s · "
                f"ε = {ROUGHNESS*1000:.1f} mm · Fan ΔP = {fan_dP} Pa"
            )

    # ── MOTOR DIAGRAM ──────────────────────────────────────────────────
    with tab_motor:
        d1, d2 = st.columns([1, 1])
        with d1:
            st.plotly_chart(draw_motor(geom), use_container_width=True)
        with d2:
            st.plotly_chart(draw_axial(geom), use_container_width=True)

        # Computed geometry
        airgap = (stator_id - rotor_od) / 2
        st.divider()
        st.markdown("**Derived geometry**")
        gcols = st.columns(4)
        derived = [
            ("Airgap", f"{airgap*1000:.2f} mm"),
            ("Airgap Dh", f"{airgap*2*1000:.2f} mm"),
            ("Stator frame t", f"{(stator_od-stator_id)/2*1000:.1f} mm"),
            ("Rotor body t", f"{(rotor_od-rotor_id)/2*1000:.1f} mm"),
        ]
        for col, (k, v) in zip(gcols, derived):
            with col:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-lbl">{k}</div>
                    <div class="metric-val">{v}</div>
                </div>""", unsafe_allow_html=True)

    # ── THEORY ────────────────────────────────────────────────────────
    with tab_theory:
        st.markdown("""
### Hardy-Cross Method for Pipe Networks

The ventilation circuit is modelled as a **fluid resistance network**, directly analogous
to an electrical circuit:

| Electrical | Fluid |
|---|---|
| Voltage (V) | Pressure (Pa) |
| Current (A) | Flow rate Q (m³/s) |
| Resistance (Ω) | Flow resistance R |
| EMF source | Fan / inlet head |

#### Branch pressure–flow relation

For turbulent flow (Re > 2300, which is typical in motor ducts):

$$\\Delta P = R \\cdot Q^2 \\quad \\text{where} \\quad R = \\left(f \\frac{L}{D_h} + k\\right) \\frac{\\rho}{2A^2}$$

Friction factor $f$ from **Swamee-Jain** (explicit Colebrook-White):

$$f = \\frac{0.25}{\\left[\\log_{10}\\!\\left(\\frac{\\varepsilon}{3.7 D_h} + \\frac{5.74}{Re^{0.9}}\\right)\\right]^2}$$

#### Hardy-Cross iteration

For each independent loop, the correction $\\Delta Q$ is applied:

$$\\Delta Q = -\\frac{\\sum h_L}{2 \\sum |h_L / Q|}$$

where $h_L$ is the signed head loss around the loop. Repeated until $|\\Delta Q| < 10^{-6}$ m³/s.

#### Hydraulic diameter

For rectangular ducts:
$$D_h = \\frac{2wh}{w+h}$$

For the annular airgap ($r_i$ = stator bore, $r_o$ = rotor OD):
$$D_h \\approx 2(r_i - r_o)$$

#### Parallel branch check

All parallel branches between the same two nodes must satisfy the same pressure drop
(Kirchhoff's pressure law). The solver enforces this by driving $\\sum h_L \\rightarrow 0$
for every loop.
        """)


if __name__ == "__main__":
    main()
