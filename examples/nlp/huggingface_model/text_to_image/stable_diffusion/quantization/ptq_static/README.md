Step-by-Step
============

This example shows how to quantize the unet model of [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) with SmoothQuant and generate images with the quantized unet.

# Prerequisite

## 1. Environment
```shell
pip install onnx-neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/installation_guide.md#validated-software-environment).

## 2. Prepare Model


```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers/scripts
python convert_stable_diffusion_checkpoint_to_onnx.py --model_path runwayml/stable-diffusion-v1-5 --output_path stable-diffusion
```

# Run

## 1. Quantization

```bash
bash run_quant.sh --input_model=/path/to/stable-diffusion \ # folder path of stable-diffusion
                  --output_model=/path/to/save/unet_model \ # model path as *.onnx
                  --alpha=0.7 # optional
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/stable-diffusion \ # folder path of stable-diffusion
                      --quantized_unet_path=/path/to/quantized/unet \ # optional, run fp32 model if not provided
                      --prompt="a photo of an astronaut riding a horse on mars" \ # optional
                      --image_path=image.png # optional
```

Benchmark will print the throughput data and save the generated image.
Our test results with default parameters is:
<p float="left">
  <img src="./imgs/fp32.jpg" width = "300" height = "300" alt="fp32" align=center />
  <img src="./imgs/int8.jpg" width = "300" height = "300" alt="int8" align=center />
</p>
