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
# pylint: disable=doc-string-missing

import sys
import time
import requests
from paddle_serving_app.reader import IMDBDataset
from paddle_serving_client import Client
from paddle_serving_client.utils import MultiThreadRunner
from paddle_serving_client.utils import benchmark_args

args = benchmark_args()


def single_func(idx, resource):
    imdb_dataset = IMDBDataset()
    imdb_dataset.load_resource("./imdb.vocab")
    dataset = []
    with open("./test_data/part-0") as fin:
        for line in fin:
            dataset.append(line.strip())
    start = time.time()
    if args.request == "rpc":
        client = Client()
        client.load_client_config(args.model)
        client.connect([args.endpoint])
        for i in range(1000):
            if args.batch_size >= 1:
                feed_batch = []
                for bi in range(args.batch_size):
                    word_ids, label = imdb_dataset.get_words_and_label(dataset[
                        bi])
                    feed_batch.append({"words": word_ids})
                result = client.predict(feed=feed_batch, fetch=["prediction"])
                if result is None:
                    raise ("predict failed.")
            else:
                print("unsupport batch size {}".format(args.batch_size))

    elif args.request == "http":
        if args.batch_size >= 1:
            feed_batch = []
            for bi in range(args.batch_size):
                feed_batch.append({"words": dataset[bi]})
            r = requests.post(
                "http://{}/imdb/prediction".format(args.endpoint),
                json={"feed": feed_batch,
                      "fetch": ["prediction"]})
            if r.status_code != 200:
                print('HTTP status code -ne 200')
                raise ("predict failed.")
        else:
            print("unsupport batch size {}".format(args.batch_size))
    end = time.time()
    return [[end - start]]


multi_thread_runner = MultiThreadRunner()
result = multi_thread_runner.run(single_func, args.thread, {})
avg_cost = 0
for cost in result[0]:
    avg_cost += cost
print("total cost {} s of each thread".format(avg_cost / args.thread))
