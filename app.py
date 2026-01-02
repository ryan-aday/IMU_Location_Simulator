import streamlit as st

st.set_page_config(
    page_title="IMU Network Vehicle Localization",
    page_icon="🛰️",
    layout="wide",
)

st.title("IMU Network Vehicle Localization Companion App")
st.markdown(
    """
This Streamlit experience walks through the IMU network modeling approach from
[Kan et al. (2025)](https://arxiv.org/html/2506.00371v1). It includes a math-first explanation,
interactive network construction, trajectory simulation, and ready-to-use C/C++ code
for real-time localization.
    """
)

st.header("Why another IMU network?")
st.markdown(
    """
Traditional single-IMU odometry accumulates bias rapidly, while naive IMU stacking
magnifies noise and drift without guaranteeing complementary coverage. The paper argues
for spatially distributed IMUs with cross-axis coupling terms that improve observability
and reduce per-sensor load. This app mirrors those ideas so you can prototype placements
and error models before bringing hardware online.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Kinematic backbone")
    st.markdown(
        "The paper derives vehicle position by integrating accelerometer readings in the body frame,"
        " aligning them to an inertial frame via gyroscope-driven rotation updates."
    )
    st.latex(r"R_{wb}(t+\Delta t) = R_{wb}(t) \cdot \exp\left([\omega(t)\Delta t]_\times\right)")
    st.latex(r"p(t+\Delta t) = p(t) + v(t)\Delta t + \tfrac{1}{2}\big(R_{wb}(t)a_b(t) + g\big)\Delta t^2")
    st.latex(r"v(t+\Delta t) = v(t) + \big(R_{wb}(t)a_b(t) + g\big)\Delta t")
    st.caption(
        "Skew-symmetric matrix [·]_× converts angular velocity to rotation; each IMU contributes"
        " its own a_b(t) and ω(t)."
    )

with col2:
    st.subheader("Networked IMU fusion")
    st.markdown(
        """
        The paper aggregates multiple IMUs by projecting each accelerometer reading into the
        vehicle frame at its mounting position **r_i**, adding centrifugal and Euler terms to
        isolate translational acceleration:
        """
    )
    st.latex(
        r"a_{trans}(t) = a_i(t) - \dot{\omega}(t) \times r_i - \omega(t) \times (\omega(t) \times r_i)"
    )
    st.markdown(
        """
        Weighted fusion minimizes drift and exploits geometric diversity:
        """
    )
    st.latex(r"\hat{a}(t) = \sum_i w_i\,a_{trans,i}(t),\quad w_i \propto \Sigma_i^{-1}")
    st.caption(
        "Compared with single-IMU or rigidly co-located arrays, spatial separation makes"
        " the centrifugal term informative, improving attitude recovery and reducing bias accumulation."
    )

st.header("How to use this app")
st.markdown(
    """
    * **IMU Network Builder** – pick symmetric/asymmetric layouts, edit drift/error models, and export JSON.
    * **Path Simulation & Error Review** – replay Penn COSYVIO or synthesize a path, then compare against the fused estimate.
    * **C/C++ Integration** – drop-in snippets to consume the JSON configuration in real-time software.
    """
)

st.info(
    "Equations are formatted in LaTeX; defaults borrow measurements from the SparkFun BNO086 breakout "
    "([link](https://www.sparkfun.com/sparkfun-vr-imu-breakout-bno086-qwiic.html))."
)
