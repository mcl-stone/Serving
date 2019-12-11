// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <vector>
#ifdef BCLOUD
#ifdef WITH_GPU
#include "paddle/paddle_inference_api.h"
#else
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#endif
#else
#include "paddle/fluid/inference/paddle_inference_api.h"  //NOLINT
#endif
#include "demo-serving/face_classification.pb.h"

#include <sys/time.h>

// opencv
#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"

namespace baidu {
namespace paddle_serving {
namespace serving {

static const char* FACE_CLASSIFY_MODEL_NAME = "face_classify_model";

class FaceClassifyOp
    : public baidu::paddle_serving::predictor::OpWithChannel<
          baidu::paddle_serving::predictor::face_classification::Response> {
 public:
  typedef std::vector<paddle::PaddleTensor> TensorVector;

  DECLARE_OP(FaceClassifyOp);
  int inference();

 private:
  cv::Mat _image_8u_tmp;
  cv::Mat _image_8u_rgb;
  std::vector<char> _image_vec_tmp;
};
}  // namespace serving
}  // namespace paddle_serving
}  // namespace baidu
