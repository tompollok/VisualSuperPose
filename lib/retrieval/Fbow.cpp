#include "Fbow.h"

#include <QDebug>

QByteArray FBoW::toByteArray() const {
  std::ostringstream oss(std::ios::binary);
  fbow.toStream(oss);
  qDebug() << fbow.size();
  QByteArray bstr = QByteArray::fromStdString(oss.str());
  return bstr;
}

void FBoW::fromByteArray(QByteArray data) {
  std::istringstream iss(std::ios::binary);
  iss.str(data.toStdString());
  fbow.fromStream(iss);
  qDebug() << fbow.size();
}

const fbow::fBow& getBoW(const Image* img) {
  if (img->bow == nullptr) {
    throw std::runtime_error("No BoW computed");
  }

  return static_cast<FBoW*>(img->bow.get())->fbow;
}

void setBoW(Image* img, const fbow::fBow& bow) {
  auto fb = std::make_shared<FBoW>();
  fb->fbow = bow;
  img->bow = fb;
}
