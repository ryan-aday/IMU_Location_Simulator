import streamlit as st

C_SNIPPET = r'''
#include <stdio.h>
#include <stdlib.h>
#include "cJSON.h"

typedef struct {
  double x, y, z;
  double gyro_drift_dps;
  double accel_bias_mps2;
  double noise_density;
} imu_t;

typedef struct {
  int num_imus;
  int homogeneous;
  imu_t *imus;
} imu_network_t;

imu_network_t *load_network(const char *path) {
  FILE *fp = fopen(path, "r");
  if (!fp) return NULL;
  fseek(fp, 0, SEEK_END);
  long len = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char *buffer = (char *)malloc(len + 1);
  fread(buffer, 1, len, fp);
  buffer[len] = '\0';
  fclose(fp);

  cJSON *root = cJSON_Parse(buffer);
  free(buffer);
  if (!root) return NULL;

  int num_imus = cJSON_GetObjectItem(root, "num_imus")->valueint;
  imu_network_t *net = calloc(1, sizeof(imu_network_t));
  net->num_imus = num_imus;
  net->homogeneous = cJSON_IsTrue(cJSON_GetObjectItem(root, "homogeneous"));
  net->imus = calloc(num_imus, sizeof(imu_t));

  cJSON *positions = cJSON_GetObjectItem(root, "positions");
  cJSON *drifts = cJSON_GetObjectItem(root, "drift_models");
  for (int i = 0; i < num_imus; ++i) {
    cJSON *p = cJSON_GetArrayItem(positions, i);
    cJSON *d = cJSON_GetArrayItem(drifts, net->homogeneous ? 0 : i);
    net->imus[i].x = cJSON_GetArrayItem(p, 0)->valuedouble;
    net->imus[i].y = cJSON_GetArrayItem(p, 1)->valuedouble;
    net->imus[i].z = cJSON_GetArrayItem(p, 2)->valuedouble;
    net->imus[i].gyro_drift_dps = cJSON_GetObjectItem(d, "gyro_drift_dps")->valuedouble;
    net->imus[i].accel_bias_mps2 = cJSON_GetObjectItem(d, "accel_bias_mps2")->valuedouble;
    net->imus[i].noise_density = cJSON_GetObjectItem(d, "noise_density")->valuedouble;
  }
  cJSON_Delete(root);
  return net;
}
'''

CPP_SNIPPET = r'''
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

struct Imu {
  double x{}; double y{}; double z{};
  double gyro_drift_dps{0.0};
  double accel_bias_mps2{0.0};
  double noise_density{0.0};
};

struct ImuNetwork {
  bool symmetric{true};
  bool homogeneous{true};
  std::vector<Imu> imus;
};

ImuNetwork load_network(const std::string &path) {
  std::ifstream f(path);
  nlohmann::json j; f >> j;
  ImuNetwork net;
  net.symmetric = j.value("symmetric", true);
  net.homogeneous = j.value("homogeneous", true);
  const auto &positions = j["positions"];
  const auto &drifts = j["drift_models"];
  for (size_t i = 0; i < positions.size(); ++i) {
    Imu imu;
    imu.x = positions[i][0];
    imu.y = positions[i][1];
    imu.z = positions[i][2];
    const auto &d = net.homogeneous ? drifts[0] : drifts[i];
    imu.gyro_drift_dps = d.value("gyro_drift_dps", 0.0);
    imu.accel_bias_mps2 = d.value("accel_bias_mps2", 0.0);
    imu.noise_density = d.value("noise_density", 0.0);
    net.imus.push_back(imu);
  }
  return net;
}

int main() {
  auto net = load_network("imu_network.json");
  std::cout << "Loaded " << net.imus.size() << " IMUs" << std::endl;
  // TODO: feed IMU streaming data and propagate state using the equations
  // from app.py (rotation update + translational fusion).
  return 0;
}
'''


def main():
    st.title("C / C++ Integration")
    st.markdown(
        """
        Drop these snippets into your embedded or robotics stack to consume the exported JSON. The
        C version uses cJSON for tiny-footprint microcontrollers, while the C++ version targets
        desktop or ROS-style builds with [nlohmann/json](https://github.com/nlohmann/json).
        """
    )

    st.subheader("C (cJSON)")
    st.code(C_SNIPPET, language="c")

    st.subheader("C++ (nlohmann/json)")
    st.code(CPP_SNIPPET, language="cpp")

    st.info(
        "Hook the parsed network into your real-time estimator: use each IMU's position vector to compute "
        "centrifugal/Euler terms and fuse accelerations as outlined on the landing page."
    )


if __name__ == "__main__":
    main()
