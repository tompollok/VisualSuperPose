#include "window.h"

#include <math.h>

#include <QElapsedTimer>
#include <QFile>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QTextStream>
#include <QVector3D>
#include <algorithm>
#include <iostream>

#include "util.h"

GLuint LoadShaders(const char* vertex_file_path,
                   const char* fragment_file_path) {
  // Create the shaders
  GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
  GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

  // Read the Vertex Shader code from the file
  QFile f(vertex_file_path);
  if (!f.open(QFile::ReadOnly)) {
    std::cerr << "Could not load file \"" << vertex_file_path << "\""
              << std::endl;
    return 1;
  }
  QTextStream t(&f);
  std::string VertexShaderCode = t.readAll().toStdString();

  // Read the Fragment Shader code from the file
  QFile f2(fragment_file_path);
  if (!f2.open(QFile::ReadOnly)) {
    std::cerr << "Could not load file \"" << fragment_file_path << "\""
              << std::endl;
    return 1;
  }
  QTextStream t2(&f2);
  std::string FragmentShaderCode = t2.readAll().toStdString();

  GLint Result = GL_FALSE;
  int InfoLogLength;

  // Compile Vertex Shader
  printf("Compiling shader : %s\n", vertex_file_path);
  char const* VertexSourcePointer = VertexShaderCode.c_str();
  glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
  glCompileShader(VertexShaderID);

  // Check Vertex Shader
  glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  if (InfoLogLength > 0) {
    std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
    glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL,
                       &VertexShaderErrorMessage[0]);
    printf("%s\n", &VertexShaderErrorMessage[0]);
  }

  // Compile Fragment Shader
  printf("Compiling shader : %s\n", fragment_file_path);
  char const* FragmentSourcePointer = FragmentShaderCode.c_str();
  glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
  glCompileShader(FragmentShaderID);

  // Check Fragment Shader
  glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  if (InfoLogLength > 0) {
    std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
    glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL,
                       &FragmentShaderErrorMessage[0]);
    printf("%s\n", &FragmentShaderErrorMessage[0]);
  }

  // Link the program
  printf("Linking program\n");
  GLuint ProgramID = glCreateProgram();
  glAttachShader(ProgramID, VertexShaderID);
  glAttachShader(ProgramID, FragmentShaderID);
  glLinkProgram(ProgramID);

  // Check the program
  glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
  glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  if (InfoLogLength > 0) {
    std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
    glGetProgramInfoLog(ProgramID, InfoLogLength, NULL,
                        &ProgramErrorMessage[0]);
    printf("%s\n", &ProgramErrorMessage[0]);
  }

  glDetachShader(ProgramID, VertexShaderID);
  glDetachShader(ProgramID, FragmentShaderID);

  glDeleteShader(VertexShaderID);
  glDeleteShader(FragmentShaderID);

  return ProgramID;
}

Window::Window() {
  QSurfaceFormat format;
  // format.setProfile(QSurfaceFormat::CoreProfile);
  // format.setVersion(4,1);
  format.setDepthBufferSize(24);
  format.setStencilBufferSize(8);
  format.setSamples(4);
  format.setSwapInterval(0);  // 1: vsync 0: no limit
  setFormat(format);
}

void Window::addPointcloud(const std::vector<WorldPoint>& pointcloud,
                           float pointSize) {
  mPointclouds.emplace_back(pointcloud, pointSize);
}

void Window::addFrustum(const Frustum& frustum) {
  mFrustums.push_back(frustum);
}

void Window::createScene() {
  for (auto& pc : mPointclouds) {
    pc.setupGL();
  }

  glGenBuffers(1, &frustumBufferVert);
  glGenBuffers(1, &frustumBufferColor);
  glGenBuffers(1, &frustumBufferIndex);

  size_t frustumBuffSize = mFrustums.size() * Frustum::vertices * 3;
  std::vector<GLfloat> vertData(frustumBuffSize);
  std::vector<GLfloat> colorData(frustumBuffSize);

  size_t frustumBuffSizeIndices = mFrustums.size() * Frustum::indices;
  std::vector<GLuint> indexData(frustumBuffSizeIndices);

  for (size_t i = 0; i < mFrustums.size(); ++i) {
    mFrustums[i].setupGL(vertData.data(), colorData.data(), indexData.data(),
                         i);
  }

  glBindBuffer(GL_ARRAY_BUFFER, frustumBufferVert);
  glBufferData(GL_ARRAY_BUFFER, frustumBuffSize * sizeof(GLfloat),
               vertData.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, frustumBufferColor);
  glBufferData(GL_ARRAY_BUFFER, frustumBuffSize * sizeof(GLfloat),
               colorData.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, frustumBufferIndex);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, frustumBuffSizeIndices * sizeof(GLuint),
               indexData.data(), GL_STATIC_DRAW);
}

void Window::initializeGL() {
  glewInit();

  GLuint VertexArrayID;
  glGenVertexArrays(1, &VertexArrayID);
  glBindVertexArray(VertexArrayID);

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClearDepth(1.0f);

  glEnable(GL_ALPHA_TEST);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

  glDepthFunc(GL_LESS);

  qDebug() << "version: "
           << QLatin1String(
                  reinterpret_cast<const char*>(glGetString(GL_VERSION)));
  qDebug() << "GSLS version: "
           << QLatin1String(reinterpret_cast<const char*>(
                  glGetString(GL_SHADING_LANGUAGE_VERSION)));

  programID = LoadShaders(":/vertex.glsl", ":/fragment.glsl");

  createScene();
}

void Window::resizeGL(int w, int h) { cam.resize(w, h); }

void Window::paintGL() {
  quint64 frameTime = timer.nsecsElapsed();
  timer.restart();
  frames++;
  frameTimeAcc += frameTime;
  if (frames > 100) {
    double fps = 1e9 * 100.0 / frameTimeAcc;
    frameTimeAcc = 0.0;
    frames = 0;
    std::cout << "fps: " << fps << std::endl;
  }

  cam.move(frameTime);
  cam.render(programID);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram(programID);

  QMatrix4x4 t;
  glUniformMatrix4fv(glGetUniformLocation(programID, "M"), 1, GL_FALSE,
                     t.constData());

  for (auto& pc : mPointclouds) {
    pc.draw();
  }

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, frustumBufferVert);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, frustumBufferColor);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, frustumBufferIndex);
  glDrawElements(GL_LINES, mFrustums.size() * Frustum::indices, GL_UNSIGNED_INT,
                 (void*)0);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);

  if (other != nullptr) {
    other->getCamera()->drawFrustum();
  }

  update();
}

void Window::keyEvent(int key, bool pressed) {
  int roll = 1;
  switch (key) {
    case Qt::Key_W:
    case Qt::Key_Up:
      cam.moveZ(!pressed);
      break;
    case Qt::Key_S:
    case Qt::Key_Down:
      cam.moveZ(pressed);
      break;
    case Qt::Key_A:
    case Qt::Key_Left:
      cam.moveX(!pressed);
      break;
    case Qt::Key_D:
    case Qt::Key_Right:
      cam.moveX(pressed);
      break;
    case Qt::Key_Space:
      cam.moveY(pressed);
      break;
    case Qt::Key_Shift:
      cam.moveY(!pressed);
      break;
    case Qt::Key_E:
      roll *= -1;
    case Qt::Key_Q:
      cam.setRoll(pressed ? roll : 0);
      break;
    case Qt::Key_Escape:
      if (pressed) {
        if (mouseGrabbed) {
          QGuiApplication::restoreOverrideCursor();
          mouseGrabbed = false;
        } else {
          QGuiApplication::setOverrideCursor(Qt::BlankCursor);
          QCursor::setPos(position().x() + width() / 2,
                          position().y() + height() / 2);
          mouseGrabbed = true;
        }
      }
  }
}

void Window::keyPressEvent(QKeyEvent* e) { keyEvent(e->key(), true); }

void Window::keyReleaseEvent(QKeyEvent* e) { keyEvent(e->key(), false); }

void Window::mouseMoveEvent(QMouseEvent* e) {
  if (skipMME) {
    skipMME = false;
  } else if (mouseGrabbed) {
    int w2 = width() / 2;
    int h2 = height() / 2;

    int deltaX = e->x() - w2;
    int deltaY = e->y() - h2;

    cam.rotate(deltaX, deltaY);

    QCursor::setPos(position().x() + width() / 2,
                    position().y() + height() / 2);
    skipMME = true;
  }
}
