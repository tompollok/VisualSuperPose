#pragma once

#include <GL/glew.h>
#include <types/worldpoint.h>

#include <QElapsedTimer>
#include <QOpenGLWindow>

#include "camera.h"
#include "frustum.h"
#include "pointcloud.h"

class Window : public QOpenGLWindow {
 public:
  Window();

  void addPointcloud(const std::vector<WorldPoint>& pointcloud,
                     float pointSize = 1.f);
  void addFrustum(const Frustum& frustum);

  void addCameraListener(const Window* w) { other = w; }
  const Camera* getCamera() const { return &cam; }

 protected:
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;

  void keyPressEvent(QKeyEvent* e) override;
  void keyReleaseEvent(QKeyEvent* e) override;
  void mouseMoveEvent(QMouseEvent* e) override;

 private:
  void keyEvent(int key, bool pressed);

  void createScene();

 private:
  Camera cam;

  bool mouseGrabbed = false;

  GLuint programID;
  GLuint frustumBufferVert;
  GLuint frustumBufferColor;
  GLuint frustumBufferIndex;

  std::vector<Pointcloud> mPointclouds;
  std::vector<Frustum> mFrustums;

  const Window* other = nullptr;

  QElapsedTimer timer;
  quint64 frameTimeAcc = 0.0;
  int frames = 100;

  bool skipMME = false;
};
