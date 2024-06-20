# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import logging
import argparse
import cv2
import numpy as np
import onnx
import re
import os
import collections
from PIL import Image
import onnxruntime as ort
from sklearn import metrics
from onnx_neural_compressor import data_reader
from onnx_neural_compressor import config
from onnx_neural_compressor import quantization
from onnx_neural_compressor.quantization import tuning

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

def _topk_shape_validate(preds, labels):
    # preds shape can be Nxclass_num or class_num(N=1 by default)
    # it's more suitable for 'Accuracy' with preds shape Nx1(or 1) output from argmax
    if isinstance(preds, int):
        preds = [preds]
        preds = np.array(preds)
    elif isinstance(preds, np.ndarray):
        preds = np.array(preds)
    elif isinstance(preds, list):
        preds = np.array(preds)
        preds = preds.reshape((-1, preds.shape[-1]))

    # consider labels just int value 1x1
    if isinstance(labels, int):
        labels = [labels]
        labels = np.array(labels)
    elif isinstance(labels, tuple):
        labels = np.array([labels])
        labels = labels.reshape((labels.shape[-1], -1))
    elif isinstance(labels, list):
        if isinstance(labels[0], int):
            labels = np.array(labels)
            labels = labels.reshape((labels.shape[0], 1))
        elif isinstance(labels[0], tuple):
            labels = np.array(labels)
            labels = labels.reshape((labels.shape[-1], -1))
        else:
            labels = np.array(labels)
    # labels most have 2 axis, 2 cases: N(or Nx1 sparse) or Nxclass_num(one-hot)
    # only support 2 dimension one-shot labels
    # or 1 dimension one-hot class_num will confuse with N

    if len(preds.shape) == 1:
        N = 1
        class_num = preds.shape[0]
        preds = preds.reshape([-1, class_num])
    elif len(preds.shape) >= 2:
        N = preds.shape[0]
        preds = preds.reshape([N, -1])
        class_num = preds.shape[1]

    label_N = labels.shape[0]
    assert label_N == N, 'labels batch size should same with preds'
    labels = labels.reshape([N, -1])
    # one-hot labels will have 2 dimension not equal 1
    if labels.shape[1] != 1:
        labels = labels.argsort()[..., -1:]
    return preds, labels

class TopK:
    def __init__(self, k=1):
        self.k = k
        self.num_correct = 0
        self.num_sample = 0

    def update(self, preds, labels, sample_weight=None):
        preds, labels = _topk_shape_validate(preds, labels)
        preds = preds.argsort()[..., -self.k:]
        if self.k == 1:
            correct = metrics.accuracy_score(preds, labels, normalize=False)
            self.num_correct += correct

        else:
            for p, l in zip(preds, labels):
                # get top-k labels with np.argpartition
                # p = np.argpartition(p, -self.k)[-self.k:]
                l = l.astype('int32')
                if l in p:
                    self.num_correct += 1

        self.num_sample += len(labels)

    def reset(self):
        self.num_correct = 0
        self.num_sample = 0

    def result(self):
        if self.num_sample == 0:
            logger.warning("Sample num during evaluation is 0.")
            return 0
        return self.num_correct / self.num_sample


class DataReader(data_reader.CalibrationDataReader):
    def __init__(self, model_path, dataset_location, image_list, batch_size=1, calibration_sampling_size=-1):
        self.batch_size = batch_size
        self.image_list = []
        self.label_list = []
        src_lst = []
        label_lst = []
        num = 0
        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(dataset_location, image_name)
                if not os.path.exists(src):
                    continue
                src_lst.append(src)
                label_lst.append(int(label))
                if len(src_lst) == batch_size:
                    self.image_list.append(src_lst)
                    self.label_list.append(label_lst)
                    num += batch_size
                    if calibration_sampling_size > 0 and num >= calibration_sampling_size:
                        break
                    src_lst = []
                    label_lst = []
        if len(src_lst) > 0:
            self.image_list.append(src_lst)
            self.label_list.append(label_lst)
        model = onnx.load(model_path, load_external_data=False)
        self.inputs_names = [input.name for input in model.graph.input]
        self.iter_next = iter(self.image_list)

    def _preprpcess(self, src):
        with Image.open(src) as image:
            image = np.array(image.convert('RGB')).astype(np.float32)
            image = image / 255.
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

            h, w = image.shape[0], image.shape[1]

            y0 = (h - 224) // 2
            x0 = (w - 224) // 2
            image = image[y0:y0 + 224, x0:x0 + 224, :]
            image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            image = image.transpose((2, 0, 1))
        return image.astype('float32')

    def get_next(self):
        lst = next(self.iter_next, None)
        if lst is not None:
            return {self.inputs_names[0]: np.stack([self._preprpcess(src) for src in lst])}
        else:
            return None

    def rewind(self):
        self.iter_next = iter(self.image_list)


def eval_func(model, dataloader, metric):
    metric.reset()
    sess = ort.InferenceSession(model, providers=ort.get_available_providers())
    labels = dataloader.label_list
    for idx, batch in enumerate(dataloader):
        output = sess.run(None, batch)
        metric.update(output, labels[idx])
    return metric.result()

if __name__ == "__main__":
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    parser = argparse.ArgumentParser(
        description="Resnet50 fine-tune examples for image classification tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help="Pre-trained model on onnx file"
    )
    parser.add_argument(
        '--dataset_location',
        type=str,
        help="Imagenet data path"
    )
    parser.add_argument(
        '--label_path',
        type=str,
        help="Imagenet label path"
    )
    parser.add_argument(
        '--benchmark',
        action='store_true', \
        default=False
    )
    parser.add_argument(
        '--tune',
        action='store_true', \
        default=False,
        help="whether quantize the model"
    )
    parser.add_argument(
        '--output_model',
        type=str,
        help="output model path"
    )
    parser.add_argument(
        '--mode',
        type=str,
        help="benchmark mode of performance or accuracy"
    )
    parser.add_argument(
        '--quant_format',
        type=str,
        default='QOperator',
        choices=['QDQ', 'QOperator'],
        help="quantization format"
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
    )
    args = parser.parse_args()

    model = onnx.load(args.model_path)
    top1 = TopK()
    dataloader = DataReader(args.model_path, args.dataset_location, args.label_path, args.batch_size)
    def eval(onnx_model):
        dataloader.rewind()
        return eval_func(onnx_model, dataloader, top1)

    if args.benchmark:
        if args.mode == 'performance':
            total_time = 0.0
            num_iter = 100
            num_warmup = 10

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = args.intra_op_num_threads
            session = onnxruntime.InferenceSession(model.SerializeToString(),
                                                   sess_options,
                                                   providers=onnxruntime.get_available_providers())
            ort_inputs = {}
            len_inputs = len(session.get_inputs())
            inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]
            
            for idx, batch in enumerate(dataloader):
                if idx + 1 > num_iter:
                    break
                tic = time.time()
                predictions = session.run(None, batch)
                toc = time.time()
                if idx >= num_warmup:
                    total_time += toc - tic

            print("\n", "-" * 10, "Summary:", "-" * 10)
            print(args)
            throughput = (num_iter - num_warmup) / total_time
            print("Throughput: {} samples/s".format(throughput))
        elif args.mode == 'accuracy':
            acc_result = eval_func(model, dataloader, top1)
            print("Batch size = %d" % dataloader.batch_size)
            print("Accuracy: %.5f" % acc_result)

    if args.tune:
        calibration_data_reader = DataReader(args.model_path, args.dataset_location, args.label_path, args.batch_size, calibration_sampling_size=100)

        custom_tune_config = tuning.TuningConfig(
            config_set=config.StaticQuantConfig.get_config_set_for_tuning(
                quant_format=quantization.QuantFormat.QOperator if args.quant_format == "QOperator" else quantization.QuantFormat.QDQ,
            )
        )
        best_model = tuning.autotune(
            model_input=args.model_path,
            tune_config=custom_tune_config,
            eval_fn=eval,
            calibration_data_reader=calibration_data_reader,
        )
        onnx.save(best_model, args.output_model)
