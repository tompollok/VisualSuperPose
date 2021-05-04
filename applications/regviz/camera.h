#pragma once

#include <QVector4D>

class Camera {
 public:
  constexpr static double camSpeed = 2e-9;  // units / nsec
  constexpr static double camTurnSpeed = 0.05;
  constexpr static double camRollSpeed = 45e-9;  // degree / nsec

  Camera();

  QVector4D getCamPos() const { return m_pos; }
  QVector4D getCamU() const { return m_u; }
  QVector4D getCamV() const { return m_v; }
  QVector4D getCamW() const { return m_w; }
  QVector4D getCamUp() const { return m_up; }

  void resize(int w, int h);
  void render(quint32 programID);
  void move(quint64 frameTime);

  void drawFrustum() const;

  int getWidth() const { return m_width; }
  int getHeight() const { return m_height; }
  double getFovy() const { return m_fovy; }

  void rotate(int deltaX, int deltaY);

  void setRoll(int roll) { m_roll = camRollSpeed * roll; }
  void moveX(bool right) { m_velocity.setX(m_velocity.x() + (right ? 1 : -1)); }
  void moveY(bool up) { m_velocity.setY(m_velocity.y() + (up ? 1 : -1)); }
  void moveZ(bool back) { m_velocity.setZ(m_velocity.z() + (back ? 1 : -1)); }

 private:
  void rotateFree(int deltaX, int deltaY);
  void rotateFixedPitch(int deltaX, int deltaY);

 private:
  QVector4D m_pos = QVector4D(0.0, 0.0, 0.0, 1.0);
  QVector4D m_u = QVector4D(1.0, 0.0, 0.0, 0.0);
  QVector4D m_v = QVector4D(0.0, 1.0, 0.0, 0.0);
  QVector4D m_w = QVector4D(0.0, 0.0, 1.0, 0.0);
  QVector4D m_up = QVector4D(0.0, 1.0, 0.0, 0.0);

  QVector4D m_velocity = QVector4D(0.0, 0.0, 0.0, 0.0);
  float m_roll = 0.f;

  double m_fovy = 65.0;
  double m_zNear = 1.f;
  double m_zFar = 1000.f;
  int m_width = 800;
  int m_height = 600;
};
