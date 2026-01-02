# IMU Location Simulator Streamlit App

This project prototypes the IMU network approach described in [Kan et al. (2025)](https://arxiv.org/html/2506.00371v1),
providing interactive placement, simulation, and integration helpers.

## Features
- **Math overview** with LaTeX equations for networked IMU fusion and motivation versus single-IMU baselines.
- **IMU network builder** with symmetric/asymmetric layouts (1, 2, 4, or 6 sensors), tunable BNO086-style drift models,
  live 3D placement visualization, and JSON export.
- **Path simulation** that loads your JSON, generates a synthetic trajectory or links to Penn COSYVIO data,
  propagates drift/noise, and reports RMSE/MAE along with downloadable results.
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
   - **IMU Network Builder** to configure placements and export `saved_configs/imu_network.json`.
   - **Path Simulation & Error Review** to generate trajectories and view error metrics.
   - **C / C++ Integration** to copy parsing boilerplate.

## Notes
- Default drift/error values come from the SparkFun BNO086 breakout documentation: https://www.sparkfun.com/sparkfun-vr-imu-breakout-bno086-qwiic.html.
- Saved configurations land in `saved_configs/` by default.
- Synthetic paths use a Lissajous-like 3D curve; Penn COSYVIO links are provided for real data download.
