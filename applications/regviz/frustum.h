#pragma once

#include <GL/glew.h>
#include <types/extrinsics.h>
#include <types/intrinsics.h>

#include <QMatrix4x4>

class Frustum {
 public:
  const static int vertices = 5;
  const static int indices = 16;

 public:
  void setup(Extrinsics pose, Intrinsics intr, cv::Vec3b color);

  void setupGL(float* dataVert, float* dataColor, GLuint* dataIndex,
               int offset);

 private:
  Extrinsics mExtrinsics;
  Intrinsics mIntrinsics;
  cv::Vec3b mColor;
};
