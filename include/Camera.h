#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

struct Camera {
  glm::vec3 target = glm::vec3(0.0f);
  float radius;
  float azimuth = 0.5f;
  float elevation = 0.4f;
  bool dragging = false;
  bool panning = false;
  double lastX = 0.0;
  double lastY = 0.0;

  static constexpr float PI_F = 3.14159265358979323846f;

  Camera(float initialRadius) : radius(initialRadius) {}

  glm::vec3 position() const {
    float clampedEl = std::max(0.05f, std::min(elevation, PI_F - 0.05f));
    glm::vec3 offset(radius * sin(clampedEl) * cos(azimuth), radius * cos(clampedEl),
                     radius * sin(clampedEl) * sin(azimuth));
    return target + offset;
  }

  glm::mat4 getViewProj(float aspect) const {
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspect, 1e9f, 5e13f);
    glm::mat4 view = glm::lookAt(position(), target, glm::vec3(0, 1, 0));
    return proj * view;
  }
};
