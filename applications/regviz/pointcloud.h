#pragma once

#include <GL/glew.h>
#include <types/worldpoint.h>

class Pointcloud {
 public:
  Pointcloud(const std::vector<WorldPoint>& pointcloud, float pointSize = 1.f);

  void setupGL();
  void draw();

 private:
  GLuint vertBuff;
  GLuint colorBuff;
  std::vector<GLfloat> vertices;
  std::vector<GLfloat> colors;
  float pointSize;
};
