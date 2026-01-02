import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

CONFIG_DIR = Path("saved_configs")
CONFIG_DIR.mkdir(exist_ok=True)

DEFAULT_DRIFT = {
    "gyro_drift_dps": 0.5,  # degrees per second zero-rate level (approx from BNO086 typical)
    "accel_bias_mps2": 0.08,  # m/s^2 bias
    "noise_density": 0.003,  # approximate rad/s / sqrt(Hz)
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
    return base.get(count, base[1])


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
    }
    return presets.get(count, presets[1])


def _plot_positions(positions: List[List[float]]):
    positions = np.array(positions)
    fig = go.Figure()
    colors = ["red", "blue", "green", "purple", "orange", "teal"]
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
                gyro = st.number_input(f"IMU #{i+1} gyro drift (°/s)", value=DEFAULT_DRIFT["gyro_drift_dps"], key=f"gyro_{i}")
            with col2:
                accel = st.number_input(
                    f"IMU #{i+1} accel bias (m/s²)", value=DEFAULT_DRIFT["accel_bias_mps2"], key=f"accel_{i}"
                )
            with col3:
                noise = st.number_input(
                    f"IMU #{i+1} noise density", value=DEFAULT_DRIFT["noise_density"], key=f"noise_{i}"
                )
            drift_entries.append(
                {"gyro_drift_dps": gyro, "accel_bias_mps2": accel, "noise_density": noise}
            )
        return drift_entries
    return [DEFAULT_DRIFT for _ in range(count)]


def main():
    st.title("IMU Network Builder")
    st.sidebar.header("Layout Controls")
    symmetric = st.sidebar.checkbox("Symmetric layout", value=True)
    num_imus = st.sidebar.selectbox("Number of IMUs", options=[1, 2, 4, 6], index=3)

    if symmetric:
        radius = st.sidebar.slider(
            "Radial placement (m)", min_value=0.03, max_value=5.0, value=1.0, step=0.01
        )
        positions = _symmetric_positions(num_imus, radius)
    else:
        st.sidebar.markdown("Customize each IMU position (m)")
        default_positions = _asymmetric_positions(num_imus)
        positions = []
        for i in range(num_imus):
            colx, coly, colz = st.sidebar.columns(3)
            with colx:
                x = st.number_input(f"IMU #{i+1} x", value=float(default_positions[i][0]), key=f"x_{i}")
            with coly:
                y = st.number_input(f"IMU #{i+1} y", value=float(default_positions[i][1]), key=f"y_{i}")
            with colz:
                z = st.number_input(f"IMU #{i+1} z", value=float(default_positions[i][2]), key=f"z_{i}")
            positions.append([x, y, z])

    st.sidebar.header("Sensor Model")
    homogeneous = st.sidebar.checkbox("Homogeneous sensors", value=True)
    drifts = _drift_entries(num_imus, unique=not homogeneous)

    st.markdown(
        """
        Default coordinates mirror the placement examples in the paper: symmetric layouts
        sit along the vehicle axes with equal radius; asymmetric layouts scatter IMUs near
        corners and rails to create leverage for cross terms.
        """
    )
    _plot_positions(positions)

    config = {
        "symmetric": symmetric,
        "num_imus": num_imus,
        "positions": positions,
        "homogeneous": homogeneous,
        "drift_models": drifts,
    }

    st.subheader("Export configuration")
    filename = st.text_input("Filename", value="imu_network.json")
    if st.button("Save JSON"):
        target = CONFIG_DIR / filename
        target.write_text(json.dumps(config, indent=2))
        st.success(f"Saved configuration to {target}")

    st.caption(
        "BNO086 defaults come from SparkFun's breakout documentation; feel free to override with your own calibration values."
    )


if __name__ == "__main__":
    main()
