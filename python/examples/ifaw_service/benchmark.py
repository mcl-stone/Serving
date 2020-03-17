# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from image_reader import ImageReader
from paddle_serving_client import Client
from paddle_serving_client.utils import MultiThreadRunner
from paddle_serving_client.utils import benchmark_args
import time
import os
from batch_test.ifaw_test_dir import *
import base64
import ujson
import requests
import numpy as np

args = benchmark_args()


def predict(image_path, server):
    image = (base64.b64encode(open(image_path).read()))
    req = ujson.dumps({"image": image, "fetch": ["score"]})
    r = requests.post(
        server, data=req, headers={"Content-Type": "application/json"})
    ifaw_ret = {}

    try:
        res = r.json()["score"]
        cls_id = np.argmax( res )
        class_name = cls_id_name_map[cls_id]
        if class_name in score_map and score_map[class_name] <= res[cls_id]:
            ifaw_ret["score"] = res[cls_id]
            ifaw_ret["class_name"] = class_name
        return cls_id, class_name, res[cls_id]
    except Exception as e:
        print( e )

def single_func(idx, resource):
    file_list = []
    image_name_pred_map = load_pred()
    for key in image_name_pred_map:
        image_path = "./batch_test/img1/{}".format( key )
        file_list.append(image_path)
    img_list = []
    for i in range(4000):
        img_list.append(open(file_list[i]).read())
    if args.request == "http":
        start = time.time()
        server = "http://"+resource["endpoint"][0]+"/image/prediction"
        for i in range(4000):
            cls_id, class_name, prob = predict(file_list[i], server)
            '''
            if idx == 3:
                res = {"cls_id":cls_id, "class_name": class_name, "top1_prob":prob}
                print("{}\t{}".format(file_list[i].split("/")[-1],ujson.dumps(res)))
            '''
        end = time.time()
    return [[end - start]]


if __name__ == "__main__":
    multi_thread_runner = MultiThreadRunner()
    endpoint_list = ["127.0.0.1:9393"]
    #card_num = 4
    #for i in range(args.thread):
    #    endpoint_list.append("127.0.0.1:{}".format(9295 + i % card_num))
    start = time.time()
    result = multi_thread_runner.run(single_func, args.thread,
                                     {"endpoint": endpoint_list})
    avg_cost = 0
    end = time.time()
    cost = end - start
    for i in range(args.thread):
        avg_cost += result[0][i]
    avg_cost = avg_cost / args.thread
    print("thread num {}".format(args.thread))
    print("average total cost {} s.".format(avg_cost))
    print("qps : {} ".format((4000*args.thread)/cost))
