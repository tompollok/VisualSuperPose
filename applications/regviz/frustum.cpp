#include "frustum.h"

void Frustum::setup(Extrinsics pose, Intrinsics intr, cv::Vec3b color) {
  mExtrinsics = pose;
  mIntrinsics = intr;
  mColor = color;
}

void Frustum::setupGL(float* dataVert, float* dataColor, GLuint* dataIndex,
                      int offset) {
  int imageWidth = mIntrinsics.width();
  int imageHeight = mIntrinsics.height();

  float fx = static_cast<float>(mIntrinsics.Fx());
  float fy = static_cast<float>(mIntrinsics.Fy());

  float cx = static_cast<float>(mIntrinsics.Cx());
  float cy = static_cast<float>(mIntrinsics.Cy());

  if (imageWidth < 1) imageWidth = 1920;
  if (imageHeight < 1) imageHeight = 1080;

  fx = 1.0f / fx;
  fy = 1.0f / fy;
  cx = -1.0f * cx;
  cy = -1.0f * cy;

  float cx_fx = cx * fx;
  float cy_fy = cy * fy;
  float imageWidth_cx_fx = (imageWidth + cx) * fx;
  float imageHeight_cy_fy = (imageHeight + cy) * fy;

  cv::Mat extr32;
  cv::Mat(mExtrinsics.getRTMatrix(Extrinsics::Local2Ref))
      .convertTo(extr32, CV_32F);

  cv::Mat points = cv::Mat::ones(4, 5, CV_32F);

  // 0: top right
  points.at<float>(0, 0) = imageWidth_cx_fx;
  points.at<float>(1, 0) = cy_fy;
  points.at<float>(2, 0) = 1.0;

  // 1: top left
  points.at<float>(0, 1) = cx_fx;
  points.at<float>(1, 1) = cy_fy;
  points.at<float>(2, 1) = 1.0;

  // 2: bot left
  points.at<float>(0, 2) = cx_fx;
  points.at<float>(1, 2) = imageHeight_cy_fy;
  points.at<float>(2, 2) = 1.0;

  // 3: bot right
  points.at<float>(0, 3) = imageWidth_cx_fx;
  points.at<float>(1, 3) = imageHeight_cy_fy;
  points.at<float>(2, 3) = 1.0;

  // 4: origin
  points.at<float>(0, 4) = 0.0;
  points.at<float>(1, 4) = 0.0;
  points.at<float>(2, 4) = 0.0;

  points = extr32 * points;

  int vertexOffset = offset * vertices * 3;

  for (size_t v = 0; v < 5; ++v) {
    for (size_t i = 0; i < 3; ++i) {
      dataVert[vertexOffset + v * 3 + i] = points.at<float>(i, v);
      dataColor[vertexOffset + v * 3 + i] = mColor[i] / 255.f;
    }
  }

  int offsetIndex = offset * indices;

  GLuint v0 = offset * vertices;  // offsetVert / vertices / 3;

  dataIndex[offsetIndex + 0] = v0 + 4;
  dataIndex[offsetIndex + 1] = v0 + 0;
  dataIndex[offsetIndex + 2] = v0 + 4;
  dataIndex[offsetIndex + 3] = v0 + 1;
  dataIndex[offsetIndex + 4] = v0 + 4;
  dataIndex[offsetIndex + 5] = v0 + 2;
  dataIndex[offsetIndex + 6] = v0 + 4;
  dataIndex[offsetIndex + 7] = v0 + 3;

  dataIndex[offsetIndex + 8] = v0 + 0;
  dataIndex[offsetIndex + 9] = v0 + 1;
  dataIndex[offsetIndex + 10] = v0 + 0;
  dataIndex[offsetIndex + 11] = v0 + 3;

  dataIndex[offsetIndex + 12] = v0 + 2;
  dataIndex[offsetIndex + 13] = v0 + 1;
  dataIndex[offsetIndex + 14] = v0 + 2;
  dataIndex[offsetIndex + 15] = v0 + 3;
}
