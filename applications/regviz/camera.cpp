#include "camera.h"

#include <GL/glew.h>
#include <math.h>

#include <QMatrix4x4>
#include <QVector3D>

#include "util.h"

Camera::Camera() {}

void Camera::resize(int w, int h) {
  m_width = w;
  m_height = h;
}

void Camera::render(quint32 programID) {
  /* GL 1
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluPerspective( m_fovy, static_cast<double>(m_width) /
  static_cast<double>(m_height), m_zNear, m_zFar); glViewport( 0, 0, m_width,
  m_height );

  glMatrixMode( GL_MODELVIEW );
  QMatrix4x4 m(m_u.x(), m_u.y(), m_u.z(), QVector4D::dotProduct(m_u, -m_pos),
               m_v.x(), m_v.y(), m_v.z(), QVector4D::dotProduct(m_v, -m_pos),
               m_w.x(), m_w.y(), m_w.z(), QVector4D::dotProduct(m_w, -m_pos),
               0,        0,        0,        1);

  glLoadMatrixf(m.constData());
  */

  QMatrix4x4 proj;
  proj.perspective(m_fovy,
                   static_cast<double>(m_width) / static_cast<double>(m_height),
                   m_zNear, m_zFar);

  QMatrix4x4 view(m_u.x(), m_u.y(), m_u.z(), QVector4D::dotProduct(m_u, -m_pos),
                  m_v.x(), m_v.y(), m_v.z(), QVector4D::dotProduct(m_v, -m_pos),
                  m_w.x(), m_w.y(), m_w.z(), QVector4D::dotProduct(m_w, -m_pos),
                  0, 0, 0, 1);

  glUniformMatrix4fv(glGetUniformLocation(programID, "P"), 1, GL_FALSE,
                     proj.constData());
  glUniformMatrix4fv(glGetUniformLocation(programID, "V"), 1, GL_FALSE,
                     view.constData());
}

void Camera::move(quint64 frameTime) {
  if (m_velocity.x() != 0 || m_velocity.y() != 0 || m_velocity.z() != 0) {
    QMatrix4x4 m(m_u.x(), m_v.x(), m_w.x(), 0, m_u.y(), m_v.y(), m_w.y(), 0,
                 m_u.z(), m_v.z(), m_w.z(), 0, 0, 0, 0, 1);

    m_pos += camSpeed * frameTime * (m * m_velocity);
  }

  if (m_roll != 0.f) {
    QMatrix4x4 roll;
    roll.rotate(m_roll * frameTime, QVector3D(m_w.x(), m_w.y(), m_w.z()));
    m_u = roll * m_u;
    m_v = roll * m_v;
  }

  m_up = m_v;
}

void Camera::rotate(int deltaX, int deltaY) {
  rotateFree(deltaX, deltaY);
  // rotateFixedPitch(deltaX, deltaY);
}

void Camera::rotateFree(int deltaX, int deltaY) {
  float alpha = -deltaX * camTurnSpeed;
  float beta = -deltaY * camTurnSpeed;

  QMatrix4x4 m;
  m.rotate(alpha, QVector3D(m_up.x(), m_up.y(), m_up.z()));
  m.rotate(beta, QVector3D(m_u.x(), m_u.y(), m_u.z()));
  m_u = m * m_u;
  m_v = m * m_v;
  m_w = m * m_w;
}

void Camera::rotateFixedPitch(int deltaX, int deltaY) {
  float alpha = -deltaX * camTurnSpeed;
  float beta = -deltaY * camTurnSpeed;

  float betaB =
      safeAcos(QVector4D::dotProduct(m_w, QVector4D(0.0, 1.0, 0.0, 0.0)));
  if (beta < 0.f)
    beta = std::max(beta, -betaB);
  else
    beta = std::min(beta, PI - betaB);

  QMatrix4x4 m;
  m.rotate(alpha, QVector3D(0.0, 1.0, 0.0));
  m.rotate(beta, QVector3D(m_u.x(), m_u.y(), m_u.z()));
  m_u = m * m_u;
  m_v = m * m_v;
  m_w = m * m_w;
}

void Camera::drawFrustum() const {
  glPushMatrix();
  QMatrix4x4 m(m_u.x(), m_v.x(), m_w.x(), m_pos.x(), m_u.y(), m_v.y(), m_w.y(),
               m_pos.y(), m_u.z(), m_v.z(), m_w.z(), m_pos.z(), 0, 0, 0, 1);
  glMultMatrixf(m.constData());

  float back = 0.0f, front = -m_zNear;
  float top = front * tan(m_fovy / 2.0 / 180.0 * PI);
  float bottom = -top;
  float right =
      (static_cast<float>(m_width) / static_cast<float>(m_height)) * top;
  float left = -right;
  glBegin(GL_LINE_STRIP);
  glVertex3f(left, bottom, front);
  glVertex3f(0.0f, 0.0f, back);
  glVertex3f(left, top, front);
  glVertex3f(right, top, front);
  glVertex3f(0.0f, 0.0f, back);
  glVertex3f(right, bottom, front);
  glVertex3f(left, bottom, front);
  glVertex3f(left, top, front);
  glEnd();
  glBegin(GL_LINES);
  glVertex3f(right, top, front);
  glVertex3f(right, bottom, front);
  glEnd();

  float color[4];
  glGetFloatv(GL_CURRENT_COLOR, color);

  glBegin(GL_QUADS);
  float before_alpha = color[2];
  color[2] = 0.f;
  glColor4fv(color);

  glVertex3f(left, top, front);
  glVertex3f(left, bottom, front);
  glVertex3f(right, bottom, front);
  glVertex3f(right, top, front);

  color[2] = before_alpha;
  glColor4fv(color);
  glEnd();
  glPopMatrix();

  glBegin(GL_LINES);
  QVector4D t1 = m_pos + 50 * m_w;
  QVector4D t2 = m_pos - 50 * m_w;

  glVertex3f(t1.x(), t1.y(), t1.z());
  glVertex3f(t2.x(), t2.y(), t2.z());
  glEnd();
}
