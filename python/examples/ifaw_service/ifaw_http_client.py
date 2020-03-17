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

import requests
import base64
import json
import numpy as np

cls_id_name_map = {
        0: "others",
        1: "xiang",
        2: "xiang",
        3: "xiang",
        4: "hu",
        5: "hu",
        6: "hu",
        7: "chuanshanjia",
        8: "chuanshanjia",
        9: "chuanshanjia",
        }

score_map = {
        "chuanshanjia": 0.9303,
        "hu"          : 0.3510,
        "xiang"       : 0.8809,
        }

def predict(image_path, server):
    image = base64.b64encode(open(image_path).read())
    req = json.dumps({"image": image, "fetch": ["score"]})
    try :
        r = requests.post(
            server, data=req, headers={"Content-Type": "application/json"})
    except Exception as e:
        print(e)
    ifaw_ret = {}
    # print(r.json() )

    try:
        res = r.json()["score"]
        cls_id = np.argmax( res )
        class_name = cls_id_name_map[cls_id]
        print( cls_id, class_name, res[cls_id] )
        if class_name in score_map and score_map[class_name] <= res[cls_id]:
            ifaw_ret["score"] = res[cls_id]
            ifaw_ret["class_name"] = class_name
    except Exception as e:
        print( e )

    print( ifaw_ret )
    print("ok")


if __name__ == "__main__":
    server = "http://127.0.0.1:9393/image/prediction"
    # image_path = "./ifaw_image/1001434650,2504511999.JPG"
    # image_path = "./ifaw_image/1585156278,2268306162.jpg"
    image_path = "./ifaw_image/bec1da53d283f7adccbac37273bd6a71.jpg"
    predict(image_path, server)
