#include "util.h"

#include <GL/glew.h>
#include <math.h>

double safeAcos(double x) {
  if (x < -1.0)
    x = -1.0;
  else if (x > 1.0)
    x = 1.0;
  return acos(x);
}

float safeAcos(float x) {
  if (x < -1.0f)
    x = -1.0f;
  else if (x > 1.0f)
    x = 1.0f;
  return acos(x);
}

void cube() {
  glBegin(GL_TRIANGLE_STRIP);
  glVertex3f(1.f, 1.f, 1.f);
  glVertex3f(1.f, -1.f, 1.f);
  glVertex3f(1.f, 1.f, -1.f);
  glVertex3f(1.f, -1.f, -1.f);

  glVertex3f(-1.f, 1.f, -1.f);
  glVertex3f(-1.f, -1.f, -1.f);

  glVertex3f(-1.f, 1.f, 1.f);
  glVertex3f(-1.f, -1.f, 1.f);

  glVertex3f(1.f, 1.f, 1.f);
  glVertex3f(1.f, -1.f, 1.f);
  glEnd();

  // bottom
  glBegin(GL_TRIANGLE_STRIP);
  glVertex3f(1.f, -1.f, 1.f);
  glVertex3f(-1.f, -1.f, 1.f);
  glVertex3f(1.f, -1.f, -1.f);
  glVertex3f(-1.f, -1.f, -1.f);
  glEnd();

  // top
  glBegin(GL_TRIANGLE_STRIP);
  glVertex3f(1.f, 1.f, 1.f);
  glVertex3f(1.f, 1.f, -1.f);
  glVertex3f(-1.f, 1.f, 1.f);
  glVertex3f(-1.f, 1.f, -1.f);
  glEnd();
}
