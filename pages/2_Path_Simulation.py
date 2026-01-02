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


def _simulate_path(duration_s: float, radius: float, avg_speed: float, step: float = 0.01):
    """Generate a gentle Lissajous-like path with configurable spatial range and speed."""
    t = np.arange(0, duration_s, step)

    # Approximate angular rate to match desired average speed around the XY loop
    safe_radius = max(radius, 0.1)
    omega_xy = max(avg_speed / safe_radius, 0.05)
    omega_z = 0.5 * omega_xy
    z_amp = 0.25 * safe_radius

    ground_truth = np.stack(
        [safe_radius * np.cos(omega_xy * t), safe_radius * np.sin(omega_xy * t), z_amp * np.sin(omega_z * t)],
        axis=1,
    )
    return t, ground_truth


def _apply_imu_bias(path: np.ndarray, config: Dict, step: float = 0.01):
    """Apply weighted bias/noise derived from the saved IMU network configuration.

    The saved JSON carries per-IMU accelerometer/gyroscope weights and noise models. We
    collapse the accelerometer terms into a fused bias and fused noise using the same
    weighting that placed the VIMU frame (Eq. 18–25). This keeps the simulator aligned
    with the configuration authoring page instead of averaging biases naively.
    """

    if not config:
        return path.copy()

    drift_models: List[Dict[str, float]] = config.get("drift_models", [])
    n = len(drift_models)
    if n == 0:
        return path.copy()

    # Pull accelerometer weights; if missing or mismatched, fall back to uniform weights.
    accel_weights = np.array(config.get("weights", {}).get("accelerometer", []), dtype=float)
    if accel_weights.size != n or np.isclose(accel_weights.sum(), 0.0):
        accel_weights = np.full(n, 1.0 / n)
    else:
        accel_weights = accel_weights / accel_weights.sum()

    biases = np.array([model.get("accel_bias_mps2", 0.0) for model in drift_models], dtype=float)
    noise_sigmas = np.array([model.get("noise_density", 0.003) for model in drift_models], dtype=float)

    fused_bias = float(np.dot(accel_weights, biases))
    fused_sigma = float(np.sqrt(np.sum(np.square(accel_weights * noise_sigmas))))

    # Weighted noise and bias accumulation applied to position proxy.
    noise = np.random.normal(scale=fused_sigma, size=path.shape)
    noisy = path + np.cumsum(noise * step, axis=0)
    noisy += fused_bias * step * np.arange(path.shape[0])[:, None]
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
    radius = st.sidebar.slider("Path range / radius (m)", min_value=0.5, max_value=20.0, value=3.0, step=0.1)
    avg_speed = st.sidebar.slider("Average speed (m/s)", min_value=0.1, max_value=12.0, value=2.0, step=0.1)

    if dataset_choice == "Penn COSYVIO (describe)":
        meta = _load_penn_metadata()
        st.info(
            f"Penn COSYVIO dataset: {meta['description']} — download at {meta['url']}."
            " Use its provided ground truth with your exported JSON to replay a full evaluation."
        )

    st.subheader("Generate trajectory")
    t, ground_truth = _simulate_path(duration, radius=radius, avg_speed=avg_speed)
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

    st.info(
        "Why can errors still exceed the paper? The simulator now respects your saved accelerometer weights when"
        " fusing biases/noise, but it still injects random-walk noise on position directly instead of performing"
        " full strapdown integration with gyro/accel coupling and fused-bias tracking (Eq. 15). Lever-arm removal is"
        " approximated through the weights, yet attitude error, gravity alignment, and filter dynamics from the paper"
        " remain simplified, so results are conservative upper bounds."
    )

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
        "Synthetic path uses a gentle Lissajous curve; error accumulation is influenced by your drift settings,"
        " the simplified random-walk bias model here, and the lack of attitude/lever-arm compensation used in the paper."
    )


if __name__ == "__main__":
    main()
