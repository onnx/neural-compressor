<div align="center">

Neural Compressor
===========================
<h3> An open-source Python library supporting popular model compression techniques for ONNX</h3>

[![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/onnx/neural-compressor)
[![version](https://img.shields.io/badge/release-1.0-green)](https://github.com/onnx/neural-compressor/releases)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/onnx/neural-compressor/blob/master/LICENSE)


---
<div align="left">

## About this fork

This is a fork of [onnx/neural-compressor](https://github.com/onnx/neural-compressor)
whose `diy` branch fixes and speeds up the **SmoothQuant static int8 path**. It is
used (vendored as a submodule) to build the SmoothQuant int8 Parakeet TDT 0.6B v3
ASR encoder published at
[Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx),
which serves the browser app
[parakeet_web](https://github.com/thiswillbeyourgithub/parakeet_web). The changes
are candidates for upstream PRs; until those exist, here is what `diy` changes
versus the upstream repo:

Correctness:

- **Auto-alpha search no longer runs on an exhausted dataloader.** Upstream's
  `alpha="auto"` consumed the calibration reader before the search, so every layer
  silently got `alpha_min`; the reader is now rewound and the per-node reference
  activations are cached (also fixes the large-model path).
- **The alpha grid includes both endpoints**: upstream's `np.arange` stopped one
  step short of `alpha_max`, so the maximum alpha was never evaluated.
- **The selected execution provider reaches the smoother calibration.** The
  smoother passed `execution_provider=` to the `Calibrator`, whose constructor
  parameter is `providers=`, so the argument fell into `**kwargs` and was
  silently dropped: every smoother-calibration forward ran on the
  `CPUExecutionProvider` even when the caller selected CUDA. The provider list is
  now forwarded, so `--ep cuda` actually calibrates on the GPU.

Memory and speed:

- **Streaming Entropy/Percentile calibration.** Static calibration used to buffer
  every dumped activation tensor of every calibration sample until the end of the
  loop (an in-code upstream TODO), so RAM grew linearly with samples x dumped
  tensors and large exports OOMed. Each sample is now folded straight into the
  per-tensor histogram; peak RAM no longer depends on the sample count.
- **The auto-alpha QDQ sub-graph session is built once per node** (weights fed as
  runtime inputs), not once per (node, alpha) and originally once per calibration
  sample; wide alpha grids no longer balloon ORT arena memory or rebuild time.
- **Variable-length calibration windows are accepted.** That once-per-node QDQ
  sub-graph baked the FIRST sample's concrete activation shape into its input/output
  value_infos, so the cached session rejected any later calibration window of a
  different sequence length (`Got invalid dimensions for input ... Got: N Expected:
  M`). Calibration was silently constrained to uniformly-sized windows. The activation
  input/output dims are now left dynamic (ORT resolves the real shape per `run()` from
  the fed array, so equal-length calibration is numerically unchanged), letting one
  calibration set mix short and long clips, e.g. per-language short utterances plus
  full-length speeches.
- **Per-node activation caching across the alpha grid** so each alpha evaluation
  does not re-harvest the same reference activations.
- **The large-model alpha search never touches the model proto.** Each (node, alpha)
  evaluation used to rewrite the node's weight initializer twice (scale then recover),
  and even plain `numpy_helper.to_array` reads materialize external weights into the
  proto. Under protobuf's upb backend every such write abandons the old bytes in the
  ModelProto's arena (freed only when the whole proto dies), so retained RAM grew with
  nodes x alphas and survived into static calibration: a 0.1-step grid OOMed a real
  encoder export where a 0.2-step grid fit. Candidate weights are now scaled in numpy
  and fed to the QDQ-loss session directly, and external weights are read via a
  throwaway TensorProto, so the search retains nothing.
- **Memory-conservative calibration sessions.** The augmented dump graphs return
  hundreds of activation tensors per forward and ORT's BFC arenas grow ahead of
  demand (power-of-two extends) and never release, so the calibration sessions
  carried ~1 GB of pure allocator slack on a 0.6B encoder and could abort with a
  BFCArena "Failed to allocate memory" on a loaded host. These sessions run only a
  handful of forwards, so the allocator-speed trade is free:
  `conservative_session_resources()` disables the CPU arena and pins the CUDA
  arena to `kSameAsRequested` growth (with the heuristic cuDNN algo search),
  applied to both the static-calibration session and the smoother's dump session.
- **Streaming per-channel percentile in the smoother calibration.** The smoother
  calibration stacked every collected sample's activations and called
  `np.percentile` once at the end, so RAM grew as samples x frames x channels
  (~6 GB per ~400 s window, hundreds of GB for a multi-window export) and
  swap-thrashed mid-pass. Because the high percentiles SmoothQuant uses (99.999 by
  default) depend only on the largest values per channel, each sample is now
  folded into a per-channel running top-K reducer (`StreamingChannelPercentile`)
  and freed; the reducer reproduces `np.percentile` bit-for-bit (its float64
  promotion and `t >= 0.5` lerp branch included) and re-checks that K was large
  enough for the actual row count, so a result can never be silently wrong.
- **Sliced static-calibration dump to bound VRAM.** Static int8 calibration
  augments the model so EVERY calibrated tensor becomes a graph output, and ORT
  keeps all graph outputs resident for the whole forward, so a long calibration
  window peaks every activation at once (the per-layer attention-score MatMuls are
  ~0.8 GB each near a FastConformer's ~400 s reach) and overflows GPU VRAM. With
  `dump_batch_size > 0` (the `CalibDumpBatch` extra_option) the calibrated tensors
  are dumped in slices of that many graph outputs, each slice its own augment +
  forward over the same (rewound) windows, so only one slice is resident at a
  time; the per-tensor calibration ranges stay bit-identical to the single-pass
  dump (it only trades extra forwards for a smaller peak).

Features:

- **Per-op-type alpha override** in the auto-alpha search (pin an op type to a
  fixed alpha, or give it its own `min:max:step` grid).
- **Per-layer alpha summary** logged after the search (histogram per op type plus
  one line per smoothed node).
- **Sensitivity-based mixed precision** (`extra_options["SmoothQuantExcludeWorst"]`):
  the auto-alpha search records every smoothed node's best achievable QDQ loss,
  normalized by that node's reference-output energy so it is comparable across
  nodes (`Smoother.auto_alpha_losses`). Setting the option to an int n (or a
  fraction in (0, 1)) keeps the n worst-quantizing nodes out of quantization
  entirely, trading a little file size for accuracy on the layers int8 hurts most.
- **Resumable, crash-safe exports.** Given a checkpoint directory
  (`extra_options["SmoothQuantCheckpointDir"]` plus
  `CalibParamsCheckpointFile`), the smoother persists its expensive intermediates
  as they are produced and reloads whatever already exists on the next run: the
  smoother calibration, the per-node alpha search (append-only `alphas.jsonl`, one
  line per completed node grid so an interrupted multi-hour search resumes from
  the last completed node), and the static-calibration quantization params. Both
  per-sample calibration loops also checkpoint MID-pass, time-gated by
  `CheckpointIntervalSec` (default 1200 s) so a fast pass pays no IO, so an OOM- or
  time-killed pass resumes from the last sample instead of restarting. All writes
  are atomic (temp + `os.replace`) so a crash mid-write never leaves a truncated
  file. Checkpoints carry no model fingerprint (the caller keys the directory by
  its full run configuration and owns invalidation) but DO record the slice
  partition that produced the static-calibration partial, so a resume with a
  changed `CalibDumpBatch` or tensor set is discarded rather than applied to the
  wrong slices.
- **Richer quantization statistics table**: fp32 ops are split into "quantizable"
  (weight-bearing, the quantizer could convert them directly) versus "needs an
  int8 input" (weightless/pass-through ops that only convert inside an int8
  region), and the Reshape/Transpose glue rows are always shown.
- tqdm progress bars over the alpha search and both calibration passes, and the
  per-evaluation ORT warning spam is silenced.

The changes are regression-tested from the model repo
([`scripts/test_quantize-int8-smoothquant.py`](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx/blob/main/scripts/test_quantize-int8-smoothquant.py)).
This fork was developed with [Claude Code](https://claude.com/claude-code).
The original upstream README follows.

---

Neural Compressor aims to provide popular model compression techniques inherited from [Intel Neural Compressor](https://github.com/intel/neural-compressor) yet focused on ONNX model quantization such as SmoothQuant, weight-only quantization through [ONNX Runtime](https://onnxruntime.ai/). In particular, the tool provides the key features, typical examples, and open collaborations as below:

* Support a wide range of Intel hardware such as [Intel Xeon Scalable Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html) and AIPC

* Validate popular LLMs such as [LLama2](./examples/nlp/huggingface_model/text_generation/), [Llama3](./examples/nlp/huggingface_model/text_generation/), [Qwen2](./examples/nlp/huggingface_model/text_generation/) and broad models such as [BERT-base](./examples/nlp/bert/quantization), and [ResNet50](./examples/image_recognition/resnet50/quantization/ptq_static) from popular model hubs such as [Hugging Face](https://huggingface.co/), [ONNX Model Zoo](https://github.com/onnx/models#models), by leveraging automatic [accuracy-driven](./docs/design.md#workflow) quantization strategies

* Collaborate with software platforms such as [Microsoft Olive](https://github.com/microsoft/Olive), and open AI ecosystem such as [Hugging Face](https://huggingface.co/blog/intel), [ONNX](https://github.com/onnx/models#models) and [ONNX Runtime](https://github.com/microsoft/onnxruntime)

## Installation

### Install from source
```Shell
git clone https://github.com/onnx/neural-compressor.git
cd neural-compressor
pip install -r requirements.txt
pip install .
```

> **Note**:
> Further installation methods can be found under [Installation Guide](./docs/installation_guide.md).

## Getting Started

Setting up the environment:
```bash
pip install onnx-neural-compressor "onnxruntime>=1.17.0" onnx
```
After successfully installing these packages, try your first quantization program.
> Notes: please install from source before the formal pypi release.

### Weight-Only Quantization (LLMs)
Following example code demonstrates Weight-Only Quantization on LLMs, device will be selected for efficiency automatically when multiple devices are available.

Run the example:
```python
from onnx_neural_compressor.quantization import matmul_nbits_quantizer

algo_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig()
quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
    model,
    n_bits=4,
    block_size=32,
    is_symmetric=True,
    algo_config=algo_config,
)
quant.process()
best_model = quant.model
```

### Static Quantization

```python
from onnx_neural_compressor.quantization import quantize, config
from onnx_neural_compressor import data_reader


class DataReader(data_reader.CalibrationDataReader):
    def __init__(self):
        self.encoded_list = []
        # append data into self.encoded_list

        self.iter_next = iter(self.encoded_list)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.encoded_list)


data_reader = DataReader()
qconfig = config.StaticQuantConfig(calibration_data_reader=data_reader)
quantize(model, output_model_path, qconfig)
```

## Documentation

<table class="docutils">
  <thead>
  <tr>
    <th colspan="8">Overview</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="3" align="center"><a href="./docs/design.md#architecture">Architecture</a></td>
      <td colspan="3" align="center"><a href="./docs/design.md#workflow">Workflow</a></td>
      <td colspan="3" align="center"><a href="./examples/">Examples</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="8">Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="4" align="center"><a href="./docs/quantization.md">Quantization</a></td>
          <td colspan="4" align="center"><a href="./docs/smooth_quant.md">SmoothQuant</td>
      <tr>
          <td colspan="4" align="center"><a href="./docs/quantization_weight_only.md">Weight-Only Quantization (INT8/INT4) </td>
           </td>
          <td colspan="4" align="center"><a href="./docs/quantization_layer_wise.md">Layer-Wise Quantization </td>
      </tr>
  </tbody>
</table>



## Additional Content

* [Contribution Guidelines](./docs/source/CONTRIBUTING.md)
* [Security Policy](SECURITY.md)

## Communication
- [GitHub Issues](https://github.com/onnx/neural-compressor/issues): mainly for bug reports, new feature requests, question asking, etc.
- [Email](mailto:inc.maintainers@intel.com): welcome to raise any interesting research ideas on model compression techniques by email for collaborations.
