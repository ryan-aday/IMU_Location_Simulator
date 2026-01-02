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


def _simulate_path(duration_s: float, avg_speed: float, step: float = 0.01):
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

    # Position integration from heading + speed
    pos = np.zeros((n, 3))
    for i in range(1, n):
        vel = heading[i] * speed[i]
        pos[i] = pos[i - 1] + vel * dt

    return t, pos, heading


def _rotation_matrix_from_heading(heading: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Build a body-to-world rotation from a heading vector, keeping axes orthonormal."""

    eps = 1e-8
    h = heading
    if np.linalg.norm(h) < eps:
        h = fallback
    h = h / (np.linalg.norm(h) + eps)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, h)
    if np.linalg.norm(right) < eps:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(world_up, h)
    right = right / (np.linalg.norm(right) + eps)

    body_z = np.cross(h, right)
    body_z = body_z / (np.linalg.norm(body_z) + eps)

    R_bw = np.stack([h, right, body_z], axis=1)  # columns are body axes in world frame
    return R_bw


def _angular_velocity(R_prev: np.ndarray, R_curr: np.ndarray, dt: float) -> np.ndarray:
    """Approximate body angular velocity from successive rotation matrices."""

    delta = R_prev.T @ R_curr
    trace = np.trace(delta)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)

    skew = (delta - delta.T) / (2.0 * np.sin(angle) + 1e-9)
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return axis * angle / dt


def _strapdown_path(t: np.ndarray, pos_true: np.ndarray, heading: np.ndarray, config: Dict) -> np.ndarray:
    """Perform strapdown integration using fused IMU biases/weights (Eq. 15)."""

    if t.size < 2:
        return pos_true.copy()

    drift_models: List[Dict[str, float]] = config.get("drift_models", []) if config else []
    n = len(drift_models)

    if n == 0:
        return pos_true.copy()

    accel_weights = np.array(config.get("weights", {}).get("accelerometer", []), dtype=float)
    gyro_weights = np.array(config.get("weights", {}).get("gyroscope", []), dtype=float)
    if accel_weights.size != n or np.isclose(accel_weights.sum(), 0.0):
        accel_weights = np.full(n, 1.0 / n)
    else:
        accel_weights = accel_weights / accel_weights.sum()
    if gyro_weights.size != n or np.isclose(gyro_weights.sum(), 0.0):
        gyro_weights = np.full(n, 1.0 / n)
    else:
        gyro_weights = gyro_weights / gyro_weights.sum()

    accel_biases = np.array([model.get("accel_bias_mps2", 0.0) for model in drift_models], dtype=float)
    gyro_biases = np.array([model.get("gyro_drift_dps", 0.0) for model in drift_models], dtype=float)
    accel_noises = np.array([model.get("noise_density", 0.003) for model in drift_models], dtype=float)
    gyro_noises = accel_noises  # reuse density as proxy

    fused_accel_bias = float(np.dot(accel_weights, accel_biases))
    fused_gyro_bias = float(np.dot(gyro_weights, gyro_biases))
    fused_accel_sigma = float(np.sqrt(np.sum(np.square(accel_weights * accel_noises))))
    fused_gyro_sigma = float(np.sqrt(np.sum(np.square(gyro_weights * gyro_noises))))

    dt = np.diff(t)
    g = np.array([0.0, 0.0, -9.81])

    # Build reference rotations
    R_list = []
    fallback = np.array([1.0, 0.0, 0.0])
    for h in heading:
        R_list.append(_rotation_matrix_from_heading(h, fallback))
        fallback = h if np.linalg.norm(h) > 1e-8 else fallback

    pos_est = np.zeros_like(pos_true)
    vel_est = np.zeros_like(pos_true)
    R_est = R_list[0].copy()

    rng = np.random.default_rng()

    # Compute true kinematics for measurement synthesis
    vel_true = np.gradient(pos_true, t, axis=0)
    accel_true = np.gradient(vel_true, t, axis=0)

    for i in range(1, t.size):
        dt_i = dt[i - 1]
        R_true_prev = R_list[i - 1]
        R_true_curr = R_list[i]
        omega_true = _angular_velocity(R_true_prev, R_true_curr, dt_i)

        # Measurements
        gyro_meas = omega_true + np.deg2rad(fused_gyro_bias) + rng.normal(scale=np.deg2rad(fused_gyro_sigma), size=3)
        accel_body_true = R_true_curr.T @ (accel_true[i] - g)
        accel_meas = accel_body_true + fused_accel_bias + rng.normal(scale=fused_accel_sigma, size=3)

        # Bias-compensated estimates (Eq. 15 simplified)
        omega_est = gyro_meas - np.deg2rad(fused_gyro_bias)
        accel_body_est = accel_meas - fused_accel_bias

        # Orientation update (first-order integration)
        angle = np.linalg.norm(omega_est) * dt_i
        if angle > 0:
            axis = omega_est / (np.linalg.norm(omega_est) + 1e-9)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R_est = R_est @ (np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K))

        accel_world = R_est @ accel_body_est + g
        vel_est[i] = vel_est[i - 1] + accel_world * dt_i
        pos_est[i] = pos_est[i - 1] + vel_est[i] * dt_i

    return pos_est


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

    if dataset_choice == "Penn COSYVIO (describe)":
        meta = _load_penn_metadata()
        st.info(
            f"Penn COSYVIO dataset: {meta['description']} — download at {meta['url']}."
            " Use its provided ground truth with your exported JSON to replay a full evaluation."
        )

    st.subheader("Generate trajectory")
    t, ground_truth, heading = _simulate_path(duration, avg_speed=avg_speed)
    estimate = _strapdown_path(t, ground_truth, heading, config)
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
        "Errors may still exceed the paper if the simplified strapdown (first-order integration, coarse angular"
        " rates) drifts or if the heading proxy deviates from your intended body frame. Leverage your exported"
        " weights in a higher-fidelity strapdown to reduce drift further."
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
        "Synthetic path uses a smooth random walk inspired by the FANET mobility model; bias and noise follow your"
        " saved weights through strapdown integration."
    )


if __name__ == "__main__":
    main()
