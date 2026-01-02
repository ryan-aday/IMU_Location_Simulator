import streamlit as st

st.title("Paper Math Overview & Alternatives")
st.caption(
    "Based on Kan et al. (2025) — IMU network localization with cross-axis coupling; see the "
    "[arXiv preprint](https://arxiv.org/html/2506.00371v1)."
)

st.header("Sensor & Motion Model")
st.markdown(
    r"""The paper models each IMU \(i\) at lever arm position \(r_i\) in the vehicle frame. "
    r"Body-frame specific force \(a_i\) and angular rate \(\omega_i\) are corrupted by bias "
    r"and noise, then mapped to the world via the rotation matrix \(R_{wb}(t)\)."""
)
st.latex(r"R_{wb}(t+\Delta t) = R_{wb}(t)\,\exp\!\left([\omega(t)\Delta t]_\times\right)")
st.latex(r"v(t+\Delta t) = v(t) + \big(R_{wb}(t)a_{trans}(t) + g\big)\Delta t")
st.latex(r"p(t+\Delta t) = p(t) + v(t)\Delta t + \tfrac{1}{2}\big(R_{wb}(t)a_{trans}(t) + g\big)\Delta t^2")
st.caption(
    "The exponential map integrates angular velocity; translational acceleration comes from "
    "fusing lever-arm-compensated IMU readings."
)

st.subheader("Lever-arm compensation")
st.markdown(
    r"""Each IMU experiences apparent acceleration from rotation; removing it exposes the
    translational component that the paper fuses across the network."""
)
st.latex(
    r"a_{trans,i}(t) = a_i(t) - \dot{\omega}(t) \times r_i - \omega(t) \times (\omega(t) \times r_i)"
)
st.latex(r"\hat{a}(t) = \sum_i w_i\,a_{trans,i}(t), \quad w_i \propto \Sigma_i^{-1}")
st.caption(
    "Weights follow the inverse covariance of each IMU; spatial diversity makes the centrifugal term informative for attitude and bias estimation."
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
