#include "image.h"

void Image::forgetImages()
{
  grayscaleImage = cv::Mat();
  colorImage = cv::Mat();
}

void Image::forgetAll() {
  this->forgetFbow();
  this->forgetImages();
  this->forgetSiftDescriptors();
  this->forgetSiftKeypoints();
}


