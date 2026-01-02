import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def _ou_process(n: int, dt: float, beta: float, sigma: float) -> np.ndarray:
    rng = np.random.default_rng()
    x = np.zeros((n, 3))
    for i in range(1, n):
        dW = rng.normal(scale=np.sqrt(dt), size=3)
        x[i] = x[i - 1] + beta * (-x[i - 1]) * dt + sigma * dW
    return x


def _quat_mult(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def _simulate_path(duration_s: float, avg_speed: float, step: float = 0.01) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """Smooth 3D random walk with strapdown-ready outputs.

    Generates correlated angular rates (OU process) and body specific force, integrates
    quaternions for attitude, then integrates velocity/position so the strapdown replay
    can synthesize IMU measurements along the same ground truth.
    """

    t = np.arange(0, duration_s, step)
    n = t.size
    dt = step

    # Smooth angular motion and specific force noise
    omega_true = _ou_process(n, dt, beta=0.6, sigma=0.3)  # rad/s
    specific_force_body = _ou_process(n, dt, beta=0.8, sigma=0.8)  # m/s^2 body-frame specific force

    # Initialize attitude quaternion and integrate
    q = np.zeros((n, 4))
    q[0] = np.array([1.0, 0.0, 0.0, 0.0])
    for i in range(1, n):
        dw = omega_true[i] * dt
        angle = np.linalg.norm(dw)
        dq = _quat_from_axis_angle(dw if angle > 0 else np.array([1.0, 0.0, 0.0]), angle)
        q[i] = _quat_mult(q[i - 1], dq)
        q[i] = q[i] / (np.linalg.norm(q[i]) + 1e-12)

    R_list = [_quat_to_rotmat(qi) for qi in q]

    # Integrate translational motion
    g = np.array([0.0, 0.0, -9.81])
    vel = np.zeros((n, 3))
    pos = np.zeros((n, 3))

    for i in range(1, n):
        force_world = R_list[i] @ specific_force_body[i] + g
        # Encourage speed to stay near avg_speed via damping toward target magnitude
        speed = np.linalg.norm(vel[i - 1])
        if speed > 1e-6:
            correction_dir = vel[i - 1] / speed
        else:
            correction_dir = R_list[i][:, 0]
        speed_error = avg_speed - speed
        accel_correction = 0.3 * speed_error * correction_dir
        accel_world = force_world + accel_correction
        vel[i] = vel[i - 1] + accel_world * dt
        pos[i] = pos[i - 1] + vel[i] * dt
        # Enforce Reaper-like altitude envelope [0 m, 15,000 m]
        clipped_z = np.clip(pos[i, 2], 0.0, 15000.0)
        if not np.isclose(clipped_z, pos[i, 2]):
            pos[i, 2] = clipped_z
            vel[i, 2] = 0.0

    return t, pos, R_list, omega_true, specific_force_body


def _recompute_weights(config: Dict, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recompute weights from config noise/positions to mimic on-vehicle recalibration."""

    accel_weights = np.array(config.get("weights", {}).get("accelerometer", []), dtype=float)
    gyro_weights = np.array(config.get("weights", {}).get("gyroscope", []), dtype=float)

    drift_models: List[Dict[str, float]] = config.get("drift_models", []) if config else []
    noise_sigmas = np.array([model.get("noise_density", 0.003) for model in drift_models], dtype=float)
    noise_sigmas = np.where(noise_sigmas <= 0, 1e-6, noise_sigmas)

    if accel_weights.size != n or np.isclose(accel_weights.sum(), 0.0):
        positions = np.array(config.get("positions", []), dtype=float)
        if positions.shape[0] != n:
            accel_weights = np.full(n, 1.0 / n)
        else:
            sigma_sq = np.square(noise_sigmas)
            Sigma_inv = np.diag(1.0 / sigma_sq)
            R = positions.T
            R_bar = R @ Sigma_inv
            R_bar_RT = R_bar @ R.T
            r_bar = R_bar @ np.ones(n)
            correction = R.T @ (np.linalg.pinv(R_bar_RT) @ r_bar)
            w_hat = Sigma_inv @ (np.ones(n) - correction)
            denom = np.sum(w_hat)
            accel_weights = np.full(n, 1.0 / n) if np.isclose(denom, 0.0) else w_hat / denom
    else:
        accel_weights = accel_weights / accel_weights.sum()

    if gyro_weights.size != n or np.isclose(gyro_weights.sum(), 0.0):
        inv_sigma_sq = 1.0 / np.square(noise_sigmas)
        gyro_weights = inv_sigma_sq / np.sum(inv_sigma_sq)
    else:
        gyro_weights = gyro_weights / gyro_weights.sum()

    return accel_weights, gyro_weights


def _strapdown_path(
    t: np.ndarray,
    pos_true: np.ndarray,
    rotations_true: List[np.ndarray],
    omega_true: np.ndarray,
    specific_force_body: np.ndarray,
    config: Dict,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Perform strapdown integration using fused IMU biases/weights (Eq. 15) with bias relaxation."""

    if t.size < 2:
        return pos_true.copy(), rotations_true

    drift_models: List[Dict[str, float]] = config.get("drift_models", []) if config else []
    n = len(drift_models)

    if n == 0:
        return pos_true.copy(), rotations_true

    accel_weights, gyro_weights = _recompute_weights(config, n)

    accel_biases = np.array([model.get("accel_bias_mps2", 0.0) for model in drift_models], dtype=float)
    gyro_biases = np.array([model.get("gyro_drift_dps", 0.0) for model in drift_models], dtype=float)
    accel_noises = np.array([model.get("noise_density", 0.003) for model in drift_models], dtype=float)
    gyro_noises = accel_noises  # reuse density as proxy

    fused_accel_bias = float(np.dot(accel_weights, accel_biases))
    fused_gyro_bias = float(np.dot(gyro_weights, gyro_biases))
    fused_accel_sigma = float(np.sqrt(np.sum(np.square(accel_weights * accel_noises))))
    fused_gyro_sigma = float(np.sqrt(np.sum(np.square(gyro_weights * gyro_noises))))

    # allow slow bias relaxation toward zero to mimic online recalibration
    bias_relax_hz = float(config.get("bias_relax_hz", 0.05))
    bias_relax_hz = max(bias_relax_hz, 1e-6)

    dt = np.diff(t)
    g = np.array([0.0, 0.0, -9.81])

    pos_est = np.zeros_like(pos_true)
    vel_est = np.zeros_like(pos_true)
    q_est = np.array([1.0, 0.0, 0.0, 0.0])
    R_est_list: List[np.ndarray] = [_quat_to_rotmat(q_est)]

    fused_accel_bias_est = fused_accel_bias
    fused_gyro_bias_est = fused_gyro_bias
    accel_weights_est = accel_weights.copy()
    gyro_weights_est = gyro_weights.copy()
    recal_window = max(1, int(np.round(1.0 / bias_relax_hz)))

    rng = np.random.default_rng()

    for i in range(1, t.size):
        dt_i = dt[i - 1]

        # Measurements synthesized from truth + fused bias/noise
        gyro_meas = omega_true[i] + np.deg2rad(fused_gyro_bias_est) + rng.normal(scale=np.deg2rad(fused_gyro_sigma), size=3)
        accel_meas = specific_force_body[i] + fused_accel_bias_est + rng.normal(scale=fused_accel_sigma, size=3)

        # Bias-compensated estimates (Eq. 15)
        omega_est = gyro_meas - np.deg2rad(fused_gyro_bias_est)
        accel_body_est = accel_meas - fused_accel_bias_est

        # Quaternion update using exponential map
        angle = np.linalg.norm(omega_est) * dt_i
        dq = _quat_from_axis_angle(omega_est if angle > 0 else np.array([1.0, 0.0, 0.0]), angle)
        q_est = _quat_mult(q_est, dq)
        q_est = q_est / (np.linalg.norm(q_est) + 1e-12)
        R_est = _quat_to_rotmat(q_est)
        R_est_list.append(R_est)

        accel_world = R_est @ accel_body_est + g
        vel_est[i] = vel_est[i - 1] + accel_world * dt_i
        pos_est[i] = pos_est[i - 1] + vel_est[i] * dt_i
        pos_est[i, 2] = np.clip(pos_est[i, 2], 0.0, 15000.0)
        if pos_est[i, 2] in {0.0, 15000.0}:
            vel_est[i, 2] = 0.0

        # Self-correct fused bias toward slow-drift assumption and refresh weights periodically
        decay = np.exp(-bias_relax_hz * dt_i)
        fused_accel_bias_est *= decay
        fused_gyro_bias_est *= decay
        if i % recal_window == 0:
            accel_weights_est, gyro_weights_est = _recompute_weights(config, n)
            fused_accel_bias = float(np.dot(accel_weights_est, accel_biases))
            fused_gyro_bias = float(np.dot(gyro_weights_est, gyro_biases))
            fused_accel_sigma = float(np.sqrt(np.sum(np.square(accel_weights_est * accel_noises))))
            fused_gyro_sigma = float(np.sqrt(np.sum(np.square(gyro_weights_est * gyro_noises))))
            fused_accel_bias_est = fused_accel_bias
            fused_gyro_bias_est = fused_gyro_bias

    return pos_est, R_est_list


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
    duration = st.sidebar.slider(
        "Synthetic path duration (s)", min_value=5.0, max_value=97200.0, value=60.0, step=0.1, format="%0.7f"
    )
    avg_speed = st.sidebar.slider(
        "Average speed (m/s)", min_value=0.1, max_value=150.0, value=2.0, step=0.01, format="%0.7f"
    )

    if dataset_choice == "Penn COSYVIO (describe)":
        meta = _load_penn_metadata()
        st.info(
            f"Penn COSYVIO dataset: {meta['description']} — download at {meta['url']}."
            " Use its provided ground truth with your exported JSON to replay a full evaluation."
        )

    st.subheader("Generate trajectory")
    t, ground_truth, rotations_true, omega_true, specific_force_body = _simulate_path(duration, avg_speed=avg_speed)
    estimate, rotations_est = _strapdown_path(t, ground_truth, rotations_true, omega_true, specific_force_body, config)
    _plot_paths(t, ground_truth, estimate)

    position_error = np.linalg.norm(estimate - ground_truth, axis=1)
    pos_rmse = float(np.sqrt(np.mean(position_error**2)))
    pos_mae = float(np.mean(np.abs(position_error)))

    ang_err = []
    for r_true, r_est in zip(rotations_true, rotations_est):
        delta = r_true.T @ r_est
        trace = np.trace(delta)
        angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        ang_err.append(angle)
    ang_err = np.array(ang_err)
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

    export = {
        "time": t.tolist(),
        "ground_truth": ground_truth.tolist(),
        "estimate": estimate.tolist(),
        "omega_true": omega_true.tolist(),
        "specific_force_body": specific_force_body.tolist(),
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
