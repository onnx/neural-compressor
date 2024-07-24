Quantization
===============

1. [Quantization Introduction](#quantization-introduction)
2. [Quantization Fundamentals](#quantization-fundamentals)
3. [Get Started](#get-started)
   3.1 [Post Training Quantization](#post-training-quantization)
   3.2 [Specify Quantization Rules](#specify-quantization-rules)
   3.3 [Specify Quantization Recipes](#specify-quantization-recipes)
   3.4 [Specify Quantization Backend and Device](#specify-quantization-backend-and-device)
4. [Examples](#examples)

## Quantization Introduction

Quantization is a very popular deep learning model optimization technique invented for improving the speed of inference. It minimizes the number of bits required by converting a set of real-valued numbers into the lower bit data representation, such as int8 and int4, mainly on inference phase with minimal to no loss in accuracy. This way reduces the memory requirement, cache miss rate, and computational cost of using neural networks and finally achieve the goal of higher inference performance. On Intel 3rd Gen Intel® Xeon® Scalable Processors, user could expect up to 4x theoretical performance speedup. We expect further performance improvement with [Intel® Advanced Matrix Extensions](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) on 4th Gen Intel® Xeon® Scalable Processors.

## Quantization Fundamentals

`Affine quantization` and `Scale quantization` are two common range mapping techniques used in tensor conversion between different data types.

The math equation is like: $$X_{int8} = round(Scale \times X_{fp32} + ZeroPoint)$$.

**Affine Quantization**

This is so-called `asymmetric quantization`, in which we map the min/max range in the float tensor to the integer range. Here int8 range is [-128, 127], uint8 range is [0, 255].

here:

If INT8 is specified, $Scale = (|X_{f_{max}} - X_{f_{min}}|) / 127$ and $ZeroPoint = -128 - X_{f_{min}} / Scale$.

or

If UINT8 is specified, $Scale = (|X_{f_{max}} - X_{f_{min}}|) / 255$ and $ZeroPoint = - X_{f_{min}} / Scale$.

**Scale Quantization**

This is so-called `Symmetric quantization`, in which we use the maximum absolute value in the float tensor as float range and map to the corresponding integer range.

The math equation is like:

here:

If INT8 is specified, $Scale = max(abs(X_{f_{max}}), abs(X_{f_{min}})) / 127$ and $ZeroPoint = 0$.

or

If UINT8 is specified, $Scale = max(abs(X_{f_{max}}), abs(X_{f_{min}})) / 255$ and $ZeroPoint = 128$.

*NOTE*

Sometimes the reduce_range feature, that's using 7 bit width (1 sign bit + 6 data bits) to represent int8 range, may be needed on some early Xeon platforms, it's because those platforms may have overflow issues due to fp16 intermediate calculation result when executing int8 dot product operation. After AVX512_VNNI instruction is introduced, this issue gets solved by supporting fp32 intermediate data.

### Quantization Support Matrix

| Framework | Backend Library |  Symmetric Quantization | Asymmetric Quantization |
| :-------------- |:---------------:| ---------------:|---------------:|
| ONNX Runtime | [MLAS](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/core/mlas) | Activation (int8/uint8), Weight (int8/uint8) | Activation (int8/uint8), Weight (int8/uint8) |

> ***Note***
>
> Activation (uint8) + Weight (int8) is recommended for performance on x86-64 machines with AVX2 and AVX512 extensions.


#### Quantization Scheme
+ Symmetric Quantization
    + int8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(int8) - min(int8) - 1); zero_point = 0
    + uint8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(uint8) - min(uint8)); zero_point = 0
+ Asymmetric Quantization
    + int8: scale = (rmax - rmin) / (max(int8) - min(int8)); zero_point = round(min(int8) - rmin / scale)
    + uint8: scale = (rmax - rmin) / (max(uint8) - min(uint8)); zero_point = round(min(uint8) - rmin / scale)

#### Reference
+ MLAS:  [MLAS Quantization](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/onnx_quantizer.py)

### Quantization Approaches

Quantization has two different approaches which belong to optimization on inference:
1) post training dynamic quantization
2) post training static  quantization

#### Post Training Dynamic Quantization

The weights of the neural network get quantized into 8 bits format from float32 format offline. The activations of the neural network is quantized as well with the min/max range collected during inference runtime.

This approach is widely used in dynamic length neural networks, like NLP model.

#### Post Training Static Quantization

Compared with `post training dynamic quantization`, the min/max range in weights and activations are collected offline on a so-called `calibration` dataset. This dataset should be able to represent the data distribution of those unseen inference dataset. The `calibration` process runs on the original fp32 model and dumps out all the tensor distributions for `Scale` and `ZeroPoint` calculations. Usually preparing 100 samples are enough for calibration.

This approach is major quantization approach people should try because it could provide the better performance comparing with `post training dynamic quantization`.

## Get Started

The design philosophy of the quantization interface of Neural Compressor is easy-of-use. It requests user to provide `model_input`, `model_output` and `quant_config`. Those parameters would be used to quantize and save the model.

`model_input` is the ONNX model location or the ONNX model object.

`model_output` is the path to save ONNX model.

`quant_config` is the configuration to do quantization.

User could leverage Neural Compressor to directly generate a fully quantized model without accuracy validation. Currently, Neural Compressor supports `Post Training Static Quantization` and `Post Training Dynamic Quantization`.

### Post Training Quantization

``` python
from onnx_neural_compressor.quantization import quantize, config
from onnx_neural_compressor import data_reader


class DataReader(data_reader.CalibrationDataReader):
    def get_next(self): ...

    def rewind(self): ...


calibration_data_reader = DataReader()  # only needed by StaticQuantConfig
qconfig = config.StaticQuantConfig(calibration_data_reader)  # or qconfig = DynamicQuantConfig()
quantize(model, q_model_path, qconfig)
```

### Specify Quantization Rules
Neural Compressor support specify quantization rules by operator name. Users can use `set_local` API of configs to achieve the above purpose by below code:

```python
fp32_config = config.GPTQConfig(weight_dtype="fp32")
quant_config = config.GPTQConfig(
    weight_bits=4,
    weight_dtype="int",
    weight_sym=False,
    weight_group_size=32,
)
quant_config.set_local("/h.4/mlp/fc_out/MatMul", fp32_config)
```


### Specify Quantization Backend and Device

Neural-Compressor will quantized models with user-specified backend or detecting the hardware and software status automatically to decide which backend should be used. The automatically selected priority is: GPU/NPU > CPU.


<table class="center">
    <thead>
        <tr>
            <th>Backend</th>
            <th>Backend Library</th>
            <th>Support Device(cpu as default)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="left">CPUExecutionProvider</td>
            <td align="left">MLAS</td>
            <td align="left">cpu</td>
        </tr>
        <tr>
            <td align="left">TensorrtExecutionProvider</td>
            <td align="left">TensorRT</td>
            <td align="left">gpu</td>
        </tr>
        <tr>
            <td align="left">CUDAExecutionProvider</td>
            <td align="left">CUDA</td>
            <td align="left">gpu</td>
        </tr>
        <tr>
            <td align="left">DnnlExecutionProvider</td>
            <td align="left">OneDNN</td>
            <td align="left">cpu</td>
        </tr>
        <tr>
            <td align="left">DmlExecutionProvider*</td>
            <td align="left">OneDNN</td>
            <td align="left">npu</td>
        </tr>
    </tbody>
</table>
<br>
<br>

> ***Note***
>
> DmlExecutionProvider support works as experimental, please expect exceptions.
>
> Known limitation: the batch size of onnx models has to be fixed to 1 for DmlExecutionProvider, no multi-batch and dynamic batch support yet.


## Examples

User could refer to [examples](../../examples) on how to quantize a new model.
