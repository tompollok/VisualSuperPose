#pragma once

#include <fbow/fbow.h>
#include <fbow/vocabulary_creator.h>
#include <types/image.h>

#include <QByteArray>
/**
 * @brief The FBoW struct: FBoW-Map
 * Implements the abstract BoW class that is saved in the Image struct
 */
struct FBoW : BoW {
  virtual QByteArray toByteArray() const;
  virtual void fromByteArray(QByteArray data);

  fbow::fBow fbow;
  fbow::fBow2 fbow2;
};

/**
 * @brief extract the FBoW map from an image
 */
const fbow::fBow& getBoW(const Image* img);

/**
 * @brief save the FBoW map to an image
 */
void setBoW(Image* img, const fbow::fBow& bow);
