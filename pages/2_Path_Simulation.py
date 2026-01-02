import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CONFIG_DIR = Path("saved_configs")
CONFIG_DIR.mkdir(exist_ok=True)


def _load_config(uploaded_file) -> Dict:
    if uploaded_file is None:
        default_file = CONFIG_DIR / "imu_network.json"
        if default_file.exists():
            return json.loads(default_file.read_text())
        return {}
    return json.load(uploaded_file)


def _simulate_path(duration_s: float, step: float = 0.01):
    t = np.arange(0, duration_s, step)
    radius = 2.0
    z_amp = 0.5
    ground_truth = np.stack(
        [radius * np.cos(0.5 * t), radius * np.sin(0.5 * t), z_amp * np.sin(0.25 * t)], axis=1
    )
    return t, ground_truth


def _apply_imu_bias(path: np.ndarray, config: Dict, step: float = 0.01):
    if not config:
        return path.copy()
    noise_scale = 0.01
    drift_sum = 0.0
    for model in config.get("drift_models", []):
        drift_sum += model.get("accel_bias_mps2", 0.0)
    drift_mean = drift_sum / max(1, len(config.get("drift_models", [])))
    noisy = path + np.cumsum(np.random.normal(scale=noise_scale, size=path.shape) * step, axis=0)
    noisy += drift_mean * step * np.arange(path.shape[0])[:, None]
    return noisy


def _plot_paths(t: np.ndarray, truth: np.ndarray, estimate: np.ndarray):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(x=truth[:, 0], y=truth[:, 1], z=truth[:, 2], mode="lines", name="Ground truth", line=dict(color="black"))
    )
    fig.add_trace(
        go.Scatter3d(x=estimate[:, 0], y=estimate[:, 1], z=estimate[:, 2], mode="lines", name="Simulated", line=dict(color="tomato"))
    )
    fig.update_layout(
        scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]"),
        margin=dict(l=0, r=0, b=0, t=40),
        height=650,
        title="Trajectory comparison",
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def _load_penn_metadata():
    return {
        "description": "Penn COSYVIO provides stereo + IMU traces for indoor/outdoor trajectories.",
        "url": "https://daniilidis-group.github.io/penncosyvio/",
    }


def main():
    st.title("Path Simulation & Error Review")
    st.sidebar.header("Input selection")
    uploaded = st.sidebar.file_uploader("Load IMU network JSON", type="json")
    config = _load_config(uploaded)

    dataset_choice = st.sidebar.radio("Trajectory source", ["Random synthetic", "Penn COSYVIO (describe)"])
    duration = st.sidebar.slider("Synthetic path duration (s)", min_value=5.0, max_value=60.0, value=20.0, step=1.0)

    if dataset_choice == "Penn COSYVIO (describe)":
        meta = _load_penn_metadata()
        st.info(
            f"Penn COSYVIO dataset: {meta['description']} — download at {meta['url']}."
            " Use its provided ground truth with your exported JSON to replay a full evaluation."
        )

    st.subheader("Generate trajectory")
    t, ground_truth = _simulate_path(duration)
    estimate = _apply_imu_bias(ground_truth, config)
    _plot_paths(t, ground_truth, estimate)

    position_error = np.linalg.norm(estimate - ground_truth, axis=1)
    rmse = float(np.sqrt(np.mean(position_error**2)))
    mae = float(np.mean(np.abs(position_error)))
    summary = pd.DataFrame(
        {
            "metric": ["RMSE [m]", "MAE [m]", "Final drift [m]"],
            "value": [rmse, mae, float(position_error[-1])],
        }
    )
    st.subheader("Error summary")
    st.table(summary)

    export = {
        "time": t.tolist(),
        "ground_truth": ground_truth.tolist(),
        "estimate": estimate.tolist(),
        "config": config,
        "metrics": {"rmse": rmse, "mae": mae, "final_drift": float(position_error[-1])},
    }

    st.download_button(
        "Download run as JSON",
        data=json.dumps(export, indent=2),
        file_name="imu_path_eval.json",
        mime="application/json",
    )

    st.caption(
        "Synthetic path uses a gentle Lissajous curve; error accumulation is influenced by your drift settings."
    )


if __name__ == "__main__":
    main()
