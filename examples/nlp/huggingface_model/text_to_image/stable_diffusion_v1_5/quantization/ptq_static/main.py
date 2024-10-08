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
import argparse
import inspect
import logging
import os
import time
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
import torch
from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline

from onnx_neural_compressor import data_reader
from onnx_neural_compressor.quantization import QuantType, config, quantize

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.WARN
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model_path",
    type=str,
    help="Folder path of ONNX Stable-diffusion model, it contains model_index.json and sub-model folders.",
)
parser.add_argument("--quantized_unet_path", type=str, default=None, help="Path of the quantized unet model.")
parser.add_argument("--benchmark", action="store_true", default=False)
parser.add_argument("--tune", action="store_true", default=False, help="whether quantize the model")
parser.add_argument("--output_model", type=str, default=None, help="output model path")
parser.add_argument("--image_path", type=str, default="image.png", help="generated image path")
parser.add_argument(
    "--batch_size",
    default=1,
    type=int,
)
parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars")
parser.add_argument("--alpha", type=float, default=0.7)
parser.add_argument("--seed", type=int, default=1234, help="random seed for generation")
parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
args = parser.parse_args()

ORT_TO_NP_TYPE = {
    "tensor(bool)": np.bool_,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(int32)": np.int32,
    "tensor(uint32)": np.uint32,
    "tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
}

np.random.seed(args.seed)

def benchmark(model):
    generator = None if args.seed is None else np.random.RandomState(args.seed)

    pipe = OnnxStableDiffusionPipeline.from_pretrained(args.model_path, provider=args.provider)
    if args.quantized_unet_path is not None:
        unet = OnnxRuntimeModel(model=ort.InferenceSession(args.quantized_unet_path, providers=[args.provider]))
        pipe.unet = unet

    image = None

    tic = time.time()
    image = pipe(prompt=args.prompt, generator=generator).images[0]
    toc = time.time()

    if image is not None:
        image.save(args.image_path)
        print("Generated image is saved as " + args.image_path)

    print("\n", "-" * 10, "Summary:", "-" * 10)
    throughput = 1 / (toc - tic)
    print("Throughput: {} samples/s".format(throughput))


class DataReader(data_reader.CalibrationDataReader):

    def __init__(self, model_path, batch_size=1):
        self.encoded_list = []
        self.batch_size = batch_size

        model = onnx.load(os.path.join(model_path, "unet/model.onnx"), load_external_data=False)
        inputs_names = [input.name for input in model.graph.input]

        generator = np.random
        pipe = OnnxStableDiffusionPipeline.from_pretrained(model_path, provider="CPUExecutionProvider")
        prompt = "A cat holding a sign that says hello world"
        self.batch_size = batch_size
        guidance_scale = 7.5
        do_classifier_free_guidance = guidance_scale > 1.0
        num_images_per_prompt = 1
        negative_prompt_embeds = None
        negative_prompt = None
        callback = None
        eta = 0.0
        latents = None
        prompt_embeds = None
        if prompt_embeds is None:
            # get prompt text embeddings
            text_inputs = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = pipe.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]

        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = pipe.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            negative_prompt_embeds = pipe.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

        # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, 512 // 8, 512 // 8)
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # set timesteps
        pipe.scheduler.set_timesteps(50)

        latents = latents * np.float64(pipe.scheduler.init_noise_sigma)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (input.type for input in pipe.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]
        for i, t in enumerate(pipe.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            ort_input = {}
            for name, inp in zip(inputs_names, [latent_model_input, timestep, prompt_embeds]):
                ort_input[name] = inp
            self.encoded_list.append(ort_input)
            noise_pred = pipe.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)
            noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = pipe.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % 1 == 0:
                step_idx = i // getattr(pipe.scheduler, "order", 1)
                callback(step_idx, t, latents)

        self.iter_next = iter(self.encoded_list)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.encoded_list)


if __name__ == "__main__":
    if args.benchmark:
        benchmark(args.model_path)

    if args.tune:
        data_reader = DataReader(args.model_path)
        cfg = config.StaticQuantConfig(
            data_reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=True,
            extra_options={
                "SmoothQuant": True,
                "SmoothQuantAlpha": args.alpha,
                "WeightSymmetric": True,
                "ActivationSymmetric": False,
                "OpTypesToExcludeOutputQuantization": ["MatMul", "Gemm"],
            },
        )
        input_path = os.path.join(args.model_path, "unet/model.onnx")
        quantize(input_path, args.output_model, cfg, optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
