# IMU Location Simulator Streamlit App

This project prototypes the IMU network approach described in [Kan et al. (2025)](https://arxiv.org/html/2506.00371v1),
providing interactive placement, simulation, and integration helpers grounded in that formulation.

## Features
- **Math overview** with LaTeX equations for networked IMU fusion and motivation versus single-IMU baselines.
- **IMU network builder** with symmetric/asymmetric layouts supporting 1–19 sensors, tunable ICM-45686-style drift models,
  live 3D placement visualization, per-IMU weight display, and JSON export (weights included). Symmetric layouts allow radii down to 0.03 m.
- **Path simulation** that loads your JSON, generates a smooth random-walk trajectory (FANET-style continuous turning) or links
  to Penn COSYVIO data, replays weighted strapdown integration with fused biases/noise, and reports positional/rotational RMSE/MAE
  along with downloadable results.
- **C/C++ snippets** showing how to read the exported JSON in embedded or desktop code.

## Getting started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```
3. Navigate pages from the left bar:
   - **IMU Network Builder** to configure placements, view computed weights, and export `saved_configs/imu_network.json`.
   - **Path Simulation & Error Review** to generate trajectories, select Penn COSYVIO or synthetic data, and view error metrics.
   - **C / C++ Integration** to copy parsing boilerplate.

## Notes
- Default drift/error values now follow the TDK InvenSense ICM-45686 specs (3.8 mdps/√Hz gyro noise, 70 µg/√Hz accel noise, small bias proxies), drawing on the datasheet and comparisons such as https://docs.slimevr.dev/diy/imu-comparison.html and https://invensense.tdk.com/products/motion-tracking/6-axis/icm-45686/.
- Synthetic motion follows a 3D smooth random walk inspired by FANET mobility modeling ([Barrado et al., 2019](https://www.researchgate.net/publication/333199745_A_3D_Smooth_Random_Walk_Mobility_Model_for_FANETs)).
- The simulator performs weighted strapdown integration using exported accelerometer/gyroscope weights (Equations 15–24 in [Kan et al. (2025)](https://arxiv.org/html/2506.00371v1)).
- Random-flight trajectories clamp altitude to the 0–15,000 m envelope (MQ-9/Reaper-like), while speed remains user-tunable up to 150 m/s.
- Saved configurations land in `saved_configs/` by default.
- Penn COSYVIO dataset links are provided for real data download alongside the synthetic generator.
