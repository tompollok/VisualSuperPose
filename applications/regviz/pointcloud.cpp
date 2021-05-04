#include "pointcloud.h"

Pointcloud::Pointcloud(const std::vector<WorldPoint>& pointcloud,
                       float pointSize)
    : pointSize(pointSize) {
  vertices.resize(pointcloud.size() * 3);
  colors.resize(pointcloud.size() * 3);
  for (size_t i = 0; i < pointcloud.size(); ++i) {
    vertices[i * 3 + 0] = static_cast<float>(pointcloud[i].pos.x);
    vertices[i * 3 + 1] = static_cast<float>(pointcloud[i].pos.y);
    vertices[i * 3 + 2] = static_cast<float>(pointcloud[i].pos.z);

    colors[i * 3 + 0] = pointcloud[i].color[0] / 255.f;
    colors[i * 3 + 1] = pointcloud[i].color[1] / 255.f;
    colors[i * 3 + 2] = pointcloud[i].color[2] / 255.f;
  }
}

void Pointcloud::setupGL() {
  glGenBuffers(1, &vertBuff);
  glBindBuffer(GL_ARRAY_BUFFER, vertBuff);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat),
               vertices.data(), GL_STATIC_DRAW);
  glGenBuffers(1, &colorBuff);
  glBindBuffer(GL_ARRAY_BUFFER, colorBuff);
  glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(GLfloat), colors.data(),
               GL_STATIC_DRAW);
}

void Pointcloud::draw() {
  glPointSize(pointSize);

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vertBuff);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, colorBuff);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
  glDrawArrays(GL_POINTS, 0, vertices.size());
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);

  glPointSize(1.f);
}
