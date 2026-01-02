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


def _simulate_path(duration_s: float, spatial_span: float, avg_speed: float, step: float = 0.01):
    """Smooth 3D random walk using correlated turn rates (FANET-inspired smooth model).

    The smooth-turn mobility idea (see "A 3D Smooth Random Walk Mobility Model for FANETs")
    is approximated by shaping angular velocity with an Ornstein–Uhlenbeck process so that
    heading changes are gradual instead of jittery. Speed is held near the chosen average
    with small noise, and Z motion drifts slowly to avoid sudden jumps.
    """

    t = np.arange(0, duration_s, step)
    n = t.size

    # Ornstein–Uhlenbeck parameters for smooth turning
    beta = 0.6  # mean reversion rate
    sigma_turn = 0.4  # turn rate volatility (rad/s)
    dt = step

    rng = np.random.default_rng()
    omega = np.zeros((n, 3))
    for i in range(1, n):
        dW = rng.normal(scale=np.sqrt(dt), size=3)
        omega[i] = omega[i - 1] + beta * (-omega[i - 1]) * dt + sigma_turn * dW

    # Integrate orientation as a heading vector
    heading = np.zeros((n, 3))
    heading[0] = np.array([1.0, 0.0, 0.05])
    for i in range(1, n):
        heading[i] = heading[i - 1] + np.cross(heading[i - 1], omega[i]) * dt
        norm = np.linalg.norm(heading[i])
        if norm > 0:
            heading[i] /= norm

    # Speed profile with mild noise, clipped to non-negative
    speed = np.clip(avg_speed + rng.normal(scale=0.05 * avg_speed, size=n), a_min=0.0, a_max=None)

    # Position integration with spatial clamping to the chosen span
    pos = np.zeros((n, 3))
    for i in range(1, n):
        vel = heading[i] * speed[i]
        pos[i] = pos[i - 1] + vel * dt
        # softly push back toward center if we exceed the span
        radius = np.linalg.norm(pos[i])
        if radius > spatial_span:
            pos[i] -= 0.2 * (radius - spatial_span) * pos[i] / (radius + 1e-6)

    return t, pos


def _apply_imu_bias(path: np.ndarray, config: Dict, step: float = 0.01):
    """Apply weighted bias/noise derived from the saved IMU network configuration.

    The saved JSON carries per-IMU accelerometer/gyroscope weights and noise models. We
    collapse the accelerometer terms into a fused bias and fused noise using the same
    weighting that placed the VIMU frame (Eq. 18–25). Noise is injected as a small
    acceleration error and integrated to velocity/position to reduce over-inflated drift
    while still reflecting how additional IMUs (and optimal weights) shrink \(\sigma^2\).
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

    # Treat fused bias/noise as acceleration disturbance and integrate to velocity/position.
    accel_error = fused_bias + np.random.normal(scale=fused_sigma, size=path.shape)
    vel_error = np.cumsum(accel_error * step, axis=0)
    pos_error = np.cumsum(vel_error * step, axis=0)
    return path + pos_error


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
    duration = st.sidebar.slider("Synthetic path duration (s)", min_value=5.0, max_value=97200.0, value=60.0, step=1.0)
    avg_speed = st.sidebar.slider("Average speed (m/s)", min_value=0.1, max_value=150.0, value=2.0, step=0.1)

    max_span = max(0.5, min(10000.0, 0.25 * avg_speed * duration))
    spatial_span = st.sidebar.slider(
        "Path spatial span (m)", min_value=0.5, max_value=float(max_span), value=min(3.0, float(max_span)), step=0.1
    )

    if dataset_choice == "Penn COSYVIO (describe)":
        meta = _load_penn_metadata()
        st.info(
            f"Penn COSYVIO dataset: {meta['description']} — download at {meta['url']}."
            " Use its provided ground truth with your exported JSON to replay a full evaluation."
        )

    st.subheader("Generate trajectory")
    t, ground_truth = _simulate_path(duration, spatial_span=spatial_span, avg_speed=avg_speed)
    estimate = _apply_imu_bias(ground_truth, config)
    _plot_paths(t, ground_truth, estimate)

    position_error = np.linalg.norm(estimate - ground_truth, axis=1)
    pos_rmse = float(np.sqrt(np.mean(position_error**2)))
    pos_mae = float(np.mean(np.abs(position_error)))

    truth_vel = np.gradient(ground_truth, t, axis=0)
    est_vel = np.gradient(estimate, t, axis=0)
    truth_dir = truth_vel / (np.linalg.norm(truth_vel, axis=1, keepdims=True) + 1e-8)
    est_dir = est_vel / (np.linalg.norm(est_vel, axis=1, keepdims=True) + 1e-8)
    cos_angles = np.sum(truth_dir * est_dir, axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    ang_err = np.arccos(cos_angles)
    rot_rmse = float(np.sqrt(np.mean(ang_err**2)))
    rot_mae = float(np.mean(np.abs(ang_err)))

    summary = pd.DataFrame(
        {
            "metric": ["Positional RMSE [m]", "Positional MAE [m]", "Rotational RMSE [rad]", "Rotational MAE [rad]"],
            "value": [pos_rmse, pos_mae, rot_rmse, rot_mae],
        }
    )
    st.subheader("Error summary (separate position vs. rotation)")
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
        "metrics": {
            "pos_rmse": pos_rmse,
            "pos_mae": pos_mae,
            "rot_rmse": rot_rmse,
            "rot_mae": rot_mae,
        },
    }

    st.download_button(
        "Download run as JSON",
        data=json.dumps(export, indent=2),
        file_name="imu_path_eval.json",
        mime="application/json",
    )

    st.caption(
        "Synthetic path uses a smooth random walk inspired by the FANET mobility model; drift follows your weights and"
        " bias settings but still omits full strapdown/lever-arm dynamics from the paper."
    )


if __name__ == "__main__":
    main()
