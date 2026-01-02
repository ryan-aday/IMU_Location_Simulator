import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

CONFIG_DIR = Path("saved_configs")
CONFIG_DIR.mkdir(exist_ok=True)

DEFAULT_DRIFT = {
    # TDK InvenSense ICM-45686 published noise/accuracy references
    # Gyro noise: 3.8 mdps/√Hz → 0.0038 °/s/√Hz (used as noise proxy)
    # Accel noise: 70 µg/√Hz → ~6.9e-4 m/s²/√Hz
    "gyro_drift_dps": 0.02,  # °/s proxy bias/accuracy
    "accel_bias_mps2": 0.002,  # m/s^2 proxy bias/accuracy
    "noise_density": 0.00069,  # m/s^2 sigma proxy for weighting
}


def _symmetric_positions(count: int, radius: float) -> List[List[float]]:
    base = {
        1: [[radius, 0.0, 0.0]],
        2: [[radius, 0.0, 0.0], [-radius, 0.0, 0.0]],
        4: [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
        ],
        6: [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
    }

    if count in base:
        return base[count]

    if count in {8, 10, 12}:
        # Evenly distribute around a circle in the XY plane and add mirrored Z layers for coverage.
        angles = np.linspace(0, 2 * np.pi, num=count // 2, endpoint=False)
        upper = [[radius * np.cos(a), radius * np.sin(a), radius * 0.35] for a in angles]
        lower = [[radius * np.cos(a), radius * np.sin(a), -radius * 0.35] for a in angles]
        return upper + lower

    return base[1]


def _asymmetric_positions(count: int) -> List[List[float]]:
    presets = {
        1: [[0.25, -0.1, 0.15]],
        2: [[0.25, -0.1, 0.15], [-0.15, 0.18, -0.05]],
        4: [
            [0.25, -0.1, 0.15],
            [-0.15, 0.18, -0.05],
            [-0.2, -0.22, 0.12],
            [0.1, 0.2, -0.18],
        ],
        6: [
            [0.25, -0.1, 0.15],
            [-0.15, 0.18, -0.05],
            [-0.2, -0.22, 0.12],
            [0.1, 0.2, -0.18],
            [0.32, 0.05, 0.08],
            [-0.28, -0.05, -0.16],
        ],
        8: [
            [0.25, -0.1, 0.15],
            [-0.15, 0.18, -0.05],
            [-0.2, -0.22, 0.12],
            [0.1, 0.2, -0.18],
            [0.32, 0.05, 0.08],
            [-0.28, -0.05, -0.16],
            [0.05, -0.32, 0.1],
            [-0.12, 0.26, -0.22],
        ],
        10: [
            [0.25, -0.1, 0.15],
            [-0.15, 0.18, -0.05],
            [-0.2, -0.22, 0.12],
            [0.1, 0.2, -0.18],
            [0.32, 0.05, 0.08],
            [-0.28, -0.05, -0.16],
            [0.05, -0.32, 0.1],
            [-0.12, 0.26, -0.22],
            [0.22, -0.18, -0.14],
            [-0.3, 0.12, 0.18],
        ],
        12: [
            [0.25, -0.1, 0.15],
            [-0.15, 0.18, -0.05],
            [-0.2, -0.22, 0.12],
            [0.1, 0.2, -0.18],
            [0.32, 0.05, 0.08],
            [-0.28, -0.05, -0.16],
            [0.05, -0.32, 0.1],
            [-0.12, 0.26, -0.22],
            [0.22, -0.18, -0.14],
            [-0.3, 0.12, 0.18],
            [0.16, 0.28, 0.04],
            [-0.24, -0.14, 0.26],
        ],
    }
    return presets.get(count, presets[1])


def _plot_positions(positions: List[List[float]]):
    positions = np.array(positions)
    fig = go.Figure()
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "teal",
        "magenta",
        "brown",
        "gold",
        "navy",
        "darkgreen",
        "gray",
    ]
    for i, pos in enumerate(positions):
        fig.add_trace(
            go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode="markers+text",
                marker=dict(size=6, color=colors[i % len(colors)]),
                text=[f"IMU #{i+1}"],
                textposition="top center",
            )
        )
    # axes
    axes_len = max(0.5, float(np.max(np.abs(positions)) + 0.2))
    fig.add_trace(
        go.Scatter3d(x=[0, axes_len], y=[0, 0], z=[0, 0], mode="lines", line=dict(color="black"), name="x")
    )
    fig.add_trace(
        go.Scatter3d(x=[0, 0], y=[0, axes_len], z=[0, 0], mode="lines", line=dict(color="black"), name="y")
    )
    fig.add_trace(
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axes_len], mode="lines", line=dict(color="black"), name="z")
    )
    fig.update_layout(
        scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]"),
        margin=dict(l=0, r=0, b=0, t=20),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)


def _drift_entries(count: int, unique: bool) -> List[Dict[str, float]]:
    if unique:
        drift_entries: List[Dict[str, float]] = []
        for i in range(count):
            col1, col2, col3 = st.columns(3)
            with col1:
                gyro = st.number_input(
                    f"IMU #{i+1} gyro drift (°/s)",
                    value=DEFAULT_DRIFT["gyro_drift_dps"],
                    key=f"gyro_{i}",
                    format="%0.7f",
                )
            with col2:
                accel = st.number_input(
                    f"IMU #{i+1} accel bias (m/s²)",
                    value=DEFAULT_DRIFT["accel_bias_mps2"],
                    key=f"accel_{i}",
                    format="%0.7f",
                )
            with col3:
                noise = st.number_input(
                    f"IMU #{i+1} noise density",
                    value=DEFAULT_DRIFT["noise_density"],
                    key=f"noise_{i}",
                    format="%0.7f",
                )
            drift_entries.append(
                {"gyro_drift_dps": gyro, "accel_bias_mps2": accel, "noise_density": noise}
            )
        return drift_entries
    return [DEFAULT_DRIFT.copy() for _ in range(count)]


def _gyro_weights(noise: List[float]) -> List[float]:
    """Weights that minimize combined gyro variance with sum-to-one constraint."""
    if len(noise) == 1:
        return [1.0]
    sigma_sq = np.square(np.array(noise))
    inv_sigma_sq = 1.0 / sigma_sq
    weights = inv_sigma_sq / np.sum(inv_sigma_sq)
    return weights.tolist()


def _accel_weights(positions: List[List[float]], noise: List[float]) -> List[float]:
    """Closed-form accelerometer weights enforcing R w = 0 and 1^T w = 1 (Eq. 18–25)."""
    n = len(positions)
    if n == 1:
        return [1.0]

    pos = np.array(positions)  # (n, 3)
    sigma_sq = np.square(np.array(noise))
    sigma_sq = np.where(sigma_sq <= 0, 1e-12, sigma_sq)

    Sigma_inv = np.diag(1.0 / sigma_sq)
    R = pos.T  # (3, n)
    R_bar = R @ Sigma_inv
    R_bar_RT = R_bar @ R.T  # (3, 3)
    r_bar = R_bar @ np.ones(n)

    correction = R.T @ (np.linalg.pinv(R_bar_RT) @ r_bar)
    w_hat = Sigma_inv @ (np.ones(n) - correction)

    denom = np.sum(w_hat)
    if np.isclose(denom, 0.0):
        return [1.0 / n for _ in range(n)]

    w_star = w_hat / denom
    return w_star.tolist()


def main():
    st.title("IMU Network Builder")
    st.sidebar.header("Layout Controls")
    symmetric = st.sidebar.checkbox("Symmetric layout", value=True)
    num_imus = st.sidebar.selectbox("Number of IMUs", options=[1, 2, 4, 6, 8, 10, 12], index=3)

    if symmetric:
        radius = st.sidebar.slider(
            "Radial placement (m)",
            min_value=0.03,
            max_value=5.0,
            value=1.0,
            step=0.0001,
            format="%0.7f",
        )
        positions = _symmetric_positions(num_imus, radius)
    else:
        st.sidebar.markdown("Customize each IMU position (m)")
        default_positions = _asymmetric_positions(num_imus)
        positions = []
        for i in range(num_imus):
            colx, coly, colz = st.sidebar.columns(3)
            with colx:
                x = st.number_input(
                    f"IMU #{i+1} x", value=float(default_positions[i][0]), key=f"x_{i}", format="%0.7f"
                )
            with coly:
                y = st.number_input(
                    f"IMU #{i+1} y", value=float(default_positions[i][1]), key=f"y_{i}", format="%0.7f"
                )
            with colz:
                z = st.number_input(
                    f"IMU #{i+1} z", value=float(default_positions[i][2]), key=f"z_{i}", format="%0.7f"
                )
            positions.append([x, y, z])

    st.sidebar.header("Sensor Model")
    homogeneous = st.sidebar.checkbox("Homogeneous sensors", value=True)
    drifts = _drift_entries(num_imus, unique=not homogeneous)

    # use noise_density as sigma proxy for both gyro and accel weighting
    noise_sigmas = [entry["noise_density"] for entry in drifts]
    gyro_weights = _gyro_weights(noise_sigmas)
    accel_weights = _accel_weights(positions, noise_sigmas)

    st.markdown(
        """
        Default coordinates mirror the placement examples in the paper: symmetric layouts
        sit along the vehicle axes with equal radius; asymmetric layouts scatter IMUs near
        corners and rails to create leverage for cross terms.
        """
    )
    _plot_positions(positions)

    st.subheader("Estimated VIMU weights")
    st.markdown(
        """
        Weights are estimated from the paper's Section IV-D/IV-E: gyroscope weights minimize
        combined variance with a sum-to-one constraint, while accelerometer weights additionally
        satisfy \(\sum_j w_j r_j = 0\) to place the virtual IMU at the vehicle frame. Identical
        sensors collapse to equal weights; heterogeneous noise tilts weights toward quieter units.
        """
    )
    weight_rows = []
    for idx in range(num_imus):
        weight_rows.append(
            {
                "IMU": f"#{idx+1}",
                "Accel weight": f"{accel_weights[idx]:0.4f}",
                "Gyro weight": f"{gyro_weights[idx]:0.4f}",
            }
        )
    st.table(weight_rows)

    config = {
        "symmetric": symmetric,
        "num_imus": num_imus,
        "positions": positions,
        "homogeneous": homogeneous,
        "drift_models": drifts,
        "weights": {"accelerometer": accel_weights, "gyroscope": gyro_weights},
    }

    st.subheader("Export configuration")
    filename = st.text_input("Filename", value="imu_network.json")
    if st.button("Save JSON"):
        target = CONFIG_DIR / filename
        target.write_text(json.dumps(config, indent=2))
        st.success(f"Saved configuration to {target}")

    st.caption(
        "ICM-45686 defaults derive from TDK InvenSense datasheets and community comparisons; override with your own calibration values."
    )


if __name__ == "__main__":
    main()
