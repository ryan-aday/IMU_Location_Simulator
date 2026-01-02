import streamlit as st

st.title("Paper Math Overview & Alternatives")
st.caption(
    "Based on Kan et al. (2025) — IMU network localization with cross-axis coupling; see the "
    "[arXiv preprint](https://arxiv.org/html/2506.00371v1)."
)

st.header("Sensor & Motion Model")
st.markdown(
    r"""Each IMU \(i\) sits at lever arm position \(r_i\) in the vehicle frame. Body-frame
    specific force \(a_i\) and angular rate \(\omega_i\) are corrupted by bias and noise, then
    mapped to the world via the rotation matrix \(R_{wb}(t)\). Gyroscope measurements are
    location independent, while accelerometers sense both translational and apparent acceleration
    from rotation."""
)
st.latex(r"R_{wb}(t+\Delta t) = R_{wb}(t)\,\exp\!\left([\omega(t)\Delta t]_\times\right)")
st.latex(r"v(t+\Delta t) = v(t) + \big(R_{wb}(t)a_{trans}(t) + g\big)\Delta t")
st.latex(r"p(t+\Delta t) = p(t) + v(t)\Delta t + \tfrac{1}{2}\big(R_{wb}(t)a_{trans}(t) + g\big)\Delta t^2")
st.caption(
    "The exponential map integrates angular velocity; translational acceleration is derived from"
    " the virtual-IMU fusion below rather than per-IMU lever-arm corrections."
)

st.subheader("Virtual IMU via weighted averaging (lever-arm neutralized)")
st.markdown(
    r"""Section IV-C shows that choosing weights \(w_j\) such that \(\sum_j w_j r_j = 0\) makes
    the accelerometer fusion insensitive to individual lever arms. Setting the VIMU frame equal to
    the vehicle frame yields a single virtual sensor whose apparent acceleration terms cancel: the
    fused accelerometer behaves as if it were co-located at the origin, while gyroscopes are already
    location independent."""
)
st.latex(r"\bar{y}_a(t) = \sum_j w_j\,y^a_j(t), \quad \bar{y}_\omega(t) = \sum_j v_j\,y^\omega_j(t)")
st.latex(r"\sum_j w_j r_j = 0 \;\Rightarrow\; \bar{y}_a(t) \text{ free of lever-arm terms}")
st.latex(r"\hat{\omega}(t) = \bar{y}_\omega(t) - \hat{b}^\omega(t), \quad \hat{a}(t) = \bar{y}_a(t) - \hat{b}^a(t)")
st.caption(
    "Equation (15) in the paper: combined biases remain slowly varying, so the filter tracks only the"
    " fused bias terms instead of per-IMU biases, reducing compute while improving noise rejection"
    " (\(\sigma_{\text{fused}} = \sigma / n\) for \(n\) identical IMUs with equal weights)."
)

st.subheader("Bias, noise, and drift")
st.markdown(
    r"""Biases \(b^a_i, b^g_i\) are modeled as random walks, while measurement noise enters the
    covariance matrices \(\Sigma_i\). The paper highlights that distributing the sensors reduces
    the per-unit bias impact because the common motion term is shared, but uncorrelated noise is
    down-weighted through the fusion weights."""
)

st.header("Observability & Error Dynamics")
st.markdown(
    r"""Cross-axis coupling introduces additional constraints: the centripetal and Euler terms
    depend on \(r_i\), so different baselines make gyroscope biases more observable. The paper
    linearizes the error state \(\delta x = [\delta R, \delta v, \delta p, b^g, b^a]^T\) and
    propagates it with first-order approximations."""
)
st.latex(r"\delta \dot{R} = -[\omega - b^g]_\times \delta R - [\delta b^g]_\times R")
st.latex(r"\delta \dot{v} = R\,\delta a_{trans} + [R a_{trans}]_\times \delta R + \delta g")
st.latex(r"\delta \dot{p} = \delta v")
st.caption(
    "By injecting multiple lever arms, the Jacobians gain rank compared with co-located IMUs, improving bias convergence and lowering long-term drift."
)

st.header("Why alternatives underperform")
cols = st.columns(2)
with cols[0]:
    st.subheader("Single or co-located IMUs")
    st.markdown(
        r"""* **Issue:** Apparent acceleration terms vanish when \(r_i = 0\), leaving yaw unobservable
        during low dynamics and letting biases dominate.\n"
        r"* **Consequence:** Requires aggressive filtering or external aiding; drift grows \(\propto t^2\)."""
    )
    st.subheader("Heuristic averaging stacks")
    st.markdown(
        """* **Issue:** Naively averaging co-located IMUs improves noise but not observability; biases remain.
        \n* **Consequence:** Diminishing returns and higher compute than the proposed weighted, lever-arm-aware fusion."""
    )
    st.subheader("Vision-IMU (VIO) only")
    st.markdown(
        """* **Issue:** Sensitive to lighting and texture; failure cases force reinitialization and high CPU/GPU load.
        \n* **Consequence:** Poor reliability in low light or high vibration compared with inertial-only redundancy."""
    )
with cols[1]:
    st.subheader("Wheel odometry + IMU")
    st.markdown(
        """* **Issue:** Wheel slip breaks the nonholonomic constraint; side-slips and jumps inject bias.
        \n* **Consequence:** Fused estimate drifts or lags when slip occurs; less robust than spatial IMU cues."""
    )
    st.subheader("GNSS/RTK dependence")
    st.markdown(
        """* **Issue:** Urban canyons and indoor operation lose lock; multi-path corrupts updates.
        \n* **Consequence:** Requires fallback; the paper's network improves standalone performance without constant GNSS."""
    )
    st.subheader("Dense filter banks")
    st.markdown(
        """* **Issue:** High-rate EKF or optimization across many states increases compute and tuning burden.
        \n* **Consequence:** Proposed approach keeps a compact error state while gaining observability from geometry."""
    )

st.header("How this app relates")
st.markdown(
    """Use this page as a reference for the derivations on the first paper page. Subsequent pages let you
    instantiate the IMU network, set drift models, and simulate trajectories with the same lever-arm-aware
    fusion, highlighting the improvements over the alternative baselines above."""
)
