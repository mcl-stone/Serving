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

#include "demo-serving/op/face_classify_op.h"
#include <ctype.h>
#include <iostream>
#include <string>
#include "predictor/framework/infer.h"
#include "predictor/framework/memory.h"

namespace baidu {
namespace paddle_serving {
namespace serving {

using baidu::paddle_serving::predictor::MempoolWrapper;
using baidu::paddle_serving::predictor::face_classification::Request;
using baidu::paddle_serving::predictor::face_classification::Response;
using baidu::paddle_serving::predictor::face_classification::EmbVector;
using baidu::paddle_serving::predictor::InferManager;

bool is_base64(char c) { return (isalnum(c) || (c == '+') || (c == '/')); }

std::string base64_decode(std::string base64_string) {
  const std::string base64_chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz"
      "0123456789+/";
  int64_t string_len = base64_string.size();
  int string_index = 0;
  int counter = 0;
  unsigned char base64_array[4], ascii_array[3];

  std::string binary;

  while ((string_index < string_len) &&
         is_base64(base64_string[string_index])) {
    base64_array[counter] = base64_string[string_index];
    counter++;
    string_index++;
    if (counter == 4) {
      counter = 0;
      for (int i = 0; i < 4; i++) {
        base64_array[i] = base64_chars.find(base64_array[i]) & 0xff;
      }
      ascii_array[0] = (base64_array[0] << 2) + ((base64_array[1] & 0x30) >> 4);
      ascii_array[1] =
          ((base64_array[1] & 0xff) << 4) + ((base64_array[2] & 0x3c) >> 2);
      ascii_array[2] = ((base64_array[2] & 0x3) << 6) + base64_array[3];

      for (int i = 0; i < 3; i++) {
        binary += ascii_array[i];
      }
    }
  }
  if (counter > 0) {
    for (int i = 0; i < counter; i++) {
      base64_array[i] = base64_chars.find(base64_array[i]) & 0xff;
    }
    ascii_array[0] = (base64_array[0] << 2) + ((base64_array[1] & 0x30) >> 4);
    ascii_array[1] =
        ((base64_array[1] & 0xff) << 4) + ((base64_array[2] & 0x3c) >> 2);
  }
  for (int i = 0; i < counter - 1; i++) {
    binary += ascii_array[i];
  }

  return binary;
}

int FaceClassifyOp::inference() {
  // receive data
  const Request *req = dynamic_cast<const Request *>(get_request_message());
  Response *res = mutable_data<Response>();
  int64_t batch_size = req->base64_string_size();
  //int64_t emb_size = req->emb_size();

  // package tensor
  std::string image_binary;

  int64_t width = 112;
  int64_t height = 112;
  int64_t channels = 3;
  int64_t image_size = channels * height * width;
  float img_mean = 127.5;
  float img_std = 128.0;

  TensorVector *input = butil::get_object<TensorVector>();

  paddle::PaddleTensor image_tensor;
  image_tensor.name = "image";
  image_tensor.dtype = paddle::FLOAT32;

  image_tensor.shape.push_back(batch_size);
  image_tensor.shape.push_back(channels);
  image_tensor.shape.push_back(width);
  image_tensor.shape.push_back(height);
  image_tensor.data.Resize(image_size * batch_size * sizeof(float));

  float *image_data = reinterpret_cast<float *>(image_tensor.data.data());

  for (int64_t bi = 0; bi < batch_size; bi++) {
    std::string base64_string = req->base64_string(bi);
    image_binary = base64_decode(base64_string);
    const unsigned char *p =
        reinterpret_cast<const unsigned char *>(image_binary.c_str());

    int64_t length = image_binary.length();

    _image_vec_tmp.clear();
    _image_vec_tmp.assign(p, p + length);
    _image_8u_tmp = cv::imdecode(cv::Mat(_image_vec_tmp), CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(_image_8u_tmp, _image_8u_rgb, CV_BGR2RGB);

    if (_image_8u_rgb.rows != height || _image_8u_rgb.cols != width ||
        _image_8u_rgb.channels() != channels) {
      LOG(ERROR) << "Image " << bi << " has incompitable size";
      return -1;
    }
    if (bi == 0) {
      LOG(INFO) << "Receive image shape {" << _image_8u_rgb.channels() << ","
                << _image_8u_rgb.rows << "," << _image_8u_rgb.cols << "}";
    }
    float *image_p = image_data + image_size * bi;
    for (int h = 0; h < height; h++) {
      unsigned char *p = _image_8u_rgb.ptr<unsigned char>(h);
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channels; c++) {
          image_p[width * height * c + width * h + w] = (p[channels * w + c] - img_mean) / img_std;
        }
      }
    }
  }

  //show input data
  /*
  std::ostringstream oss;
  for (int i = 0; i < 112; i ++) {
    oss << std::to_string(image_data[i]) << " " ;
  }
  LOG(INFO) << "input data : " << oss.str();
  */

  input->push_back(image_tensor);

  // inference
  TensorVector *output = butil::get_object<TensorVector>();
  if (InferManager::instance().infer(
          FACE_CLASSIFY_MODEL_NAME, input, output, batch_size)) {
    LOG(ERROR) << "Failed do infer in fluid model: "
               << FACE_CLASSIFY_MODEL_NAME;
    return -1;
  }

  // output
  float *emb_data = reinterpret_cast<float *>((*output)[0].data.data());

  LOG(INFO) << "batch size " << (*output)[0].shape[0] << " embedding size "
            << (*output)[0].shape[1];
  int64_t emb_size = (*output)[0].shape[1];
  for (int64_t bi = 0; bi < batch_size; bi++) {
    EmbVector *res_instance = res->add_instance();
    for (int64_t ei = 0; ei < emb_size; ei++) {
      int64_t index = ei + bi * emb_size;
      res_instance->add_embedding(*(emb_data + index));
    }
  }
  for (size_t i = 0; i < input->size(); ++i) {
    (*input)[i].shape.clear();
  }
  input->clear();
  butil::return_object<TensorVector>(input);

  for (size_t i = 0; i < output->size(); ++i) {
    (*output)[i].shape.clear();
  }
  output->clear();
  butil::return_object<TensorVector>(output);

  return 0;
}

DEFINE_OP(FaceClassifyOp);

}  // namespace serving
}  // namespace paddle_serving
}  // namespace baidu
