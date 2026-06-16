Neural Compressor
===========================
<h3> An open-source Python library supporting popular model compression techniques for ONNX</h3>

[![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/onnx/neural-compressor)
[![version](https://img.shields.io/badge/release-1.0-green)](https://github.com/onnx/neural-compressor/releases)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/onnx/neural-compressor/blob/master/LICENSE)


---

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
  activations are cached (also fixes the large-model path). (commit [`d6745e4`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/d6745e4))
- **The alpha grid includes both endpoints**: upstream's `np.arange` stopped one
  step short of `alpha_max`, so the maximum alpha was never evaluated. (commit [`8d88f45`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/8d88f45))
- **The selected execution provider reaches the smoother calibration.** The
  smoother passed `execution_provider=` to the `Calibrator`, whose constructor
  parameter is `providers=`, so the argument fell into `**kwargs` and was
  silently dropped: every smoother-calibration forward ran on the
  `CPUExecutionProvider` even when the caller selected CUDA. The provider list is
  now forwarded, so `--ep cuda` actually calibrates on the GPU. (commit [`ad8c01b`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/ad8c01b))
- **SmoothQuant `extra_options` are no longer silently ignored.** `quantize()`
  routes a `StaticQuantConfig` (with `extra_options["SmoothQuant"]`) into
  `smooth_quant_entry`, which called
  `smoother.transform(**quant_config.get_model_params_dict())`. But
  `get_model_params_dict()` only surfaces the static-quant knobs, so NONE of the
  smooth knobs (`SmoothQuantAlpha`, `SmoothQuantOpTypes`, `AutoAlphaArgs`, ...)
  reached `transform()`: the smoother always ran with its hard-coded defaults
  (`alpha=0.5`, the `[0.3, 0.7]` auto grid, op types Conv+Gemm+MatMul+FusedConv)
  no matter what the caller set. A new `_smoothquant_transform_params()` maps the
  documented `extra_options` names to the `transform()` argument names so they
  take effect. (commit [`d6745e4`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/d6745e4))

Memory and speed:

- **Streaming Entropy/Percentile calibration.** Static calibration used to buffer
  every dumped activation tensor of every calibration sample until the end of the
  loop (an in-code upstream TODO), so RAM grew linearly with samples x dumped
  tensors and large exports OOMed. Each sample is now folded straight into the
  per-tensor histogram; peak RAM no longer depends on the sample count. (commit [`6f69d55`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/6f69d55))
- **The auto-alpha QDQ sub-graph session is built once per node** (weights fed as
  runtime inputs), not once per (node, alpha) and originally once per calibration
  sample; wide alpha grids no longer balloon ORT arena memory or rebuild time. (commits [`594f6f9`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/594f6f9), [`f2a102f`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/f2a102f))
- **Variable-length calibration windows are accepted.** That once-per-node QDQ
  sub-graph baked the FIRST sample's concrete activation shape into its input/output
  value_infos, so the cached session rejected any later calibration window of a
  different sequence length (`Got invalid dimensions for input ... Got: N Expected:
  M`). Calibration was silently constrained to uniformly-sized windows. The activation
  input/output dims are now left dynamic (ORT resolves the real shape per `run()` from
  the fed array, so equal-length calibration is numerically unchanged), letting one
  calibration set mix short and long clips, e.g. per-language short utterances plus
  full-length speeches. (commit [`0a22d36`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/0a22d36))
- **Per-node activation caching across the alpha grid** so each alpha evaluation
  does not re-harvest the same reference activations. (commit [`d6745e4`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/d6745e4))
- **The large-model alpha search never touches the model proto.** Each (node, alpha)
  evaluation used to rewrite the node's weight initializer twice (scale then recover),
  and even plain `numpy_helper.to_array` reads materialize external weights into the
  proto. Under protobuf's upb backend every such write abandons the old bytes in the
  ModelProto's arena (freed only when the whole proto dies), so retained RAM grew with
  nodes x alphas and survived into static calibration: a 0.1-step grid OOMed a real
  encoder export where a 0.2-step grid fit. Candidate weights are now scaled in numpy
  and fed to the QDQ-loss session directly, and external weights are read via a
  throwaway TensorProto, so the search retains nothing. (commit [`72de740`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/72de740))
- **Memory-conservative calibration sessions.** The augmented dump graphs return
  hundreds of activation tensors per forward and ORT's BFC arenas grow ahead of
  demand (power-of-two extends) and never release, so the calibration sessions
  carried ~1 GB of pure allocator slack on a 0.6B encoder and could abort with a
  BFCArena "Failed to allocate memory" on a loaded host. These sessions run only a
  handful of forwards, so the allocator-speed trade is free:
  `conservative_session_resources()` disables the CPU arena and pins the CUDA
  arena to `kSameAsRequested` growth (with the heuristic cuDNN algo search),
  applied to both the static-calibration session and the smoother's dump session. (commit [`6c6e0cf`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/6c6e0cf))
- **Streaming per-channel percentile in the smoother calibration.** The smoother
  calibration stacked every collected sample's activations and called
  `np.percentile` once at the end, so RAM grew as samples x frames x channels
  (~6 GB per ~400 s window, hundreds of GB for a multi-window export) and
  swap-thrashed mid-pass. Because the high percentiles SmoothQuant uses (99.999 by
  default) depend only on the largest values per channel, each sample is now
  folded into a per-channel running top-K reducer (`StreamingChannelPercentile`)
  and freed; the reducer reproduces `np.percentile` bit-for-bit (its float64
  promotion and `t >= 0.5` lerp branch included) and re-checks that K was large
  enough for the actual row count, so a result can never be silently wrong. (commit [`e61fb24`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/e61fb24))
- **Sliced static-calibration dump to bound VRAM.** Static int8 calibration
  augments the model so EVERY calibrated tensor becomes a graph output, and ORT
  keeps all graph outputs resident for the whole forward, so a long calibration
  window peaks every activation at once (the per-layer attention-score MatMuls are
  ~0.8 GB each near a FastConformer's ~400 s reach) and overflows GPU VRAM. With
  `dump_batch_size > 0` (the `CalibDumpBatch` extra_option) the calibrated tensors
  are dumped in slices of that many graph outputs, each slice its own augment +
  forward over the same (rewound) windows, so only one slice is resident at a
  time; the per-tensor calibration ranges stay bit-identical to the single-pass
  dump (it only trades extra forwards for a smaller peak). (commit [`a495323`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/a495323))

Features:

- **Per-op-type alpha override** in the auto-alpha search (pin an op type to a
  fixed alpha, or give it its own `min:max:step` grid). (commit [`280b123`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/280b123))
- **Per-layer alpha summary** logged after the search (histogram per op type plus
  one line per smoothed node). (commit [`0a532e1`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/0a532e1))
- **Sensitivity-based mixed precision** (`extra_options["SmoothQuantExcludeWorst"]`):
  the auto-alpha search records every smoothed node's best achievable QDQ loss,
  normalized by that node's reference-output energy so it is comparable across
  nodes (`Smoother.auto_alpha_losses`). Setting the option to an int n (or a
  fraction in (0, 1)) keeps the n worst-quantizing nodes out of quantization
  entirely, trading a little file size for accuracy on the layers int8 hurts most. (commit [`89b4363`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/89b4363))
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
  wrong slices. (commits [`55a3c75`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/55a3c75), [`628f57f`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/628f57f), [`9eb2d4e`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/9eb2d4e), [`0c58d7b`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/0c58d7b))
- **Richer quantization statistics table**: fp32 ops are split into "quantizable"
  (weight-bearing, the quantizer could convert them directly) versus "needs an
  int8 input" (weightless/pass-through ops that only convert inside an int8
  region), and the Reshape/Transpose glue rows are always shown. (commit [`e792411`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/e792411))
- tqdm progress bars over the alpha search and both calibration passes, and the
  per-evaluation ORT warning spam is silenced. (commits [`82e3451`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/82e3451), [`9f0e94b`](https://github.com/thiswillbeyourgithub/neural-compressor-fork/commit/9f0e94b))

The changes are regression-tested from the model repo
([`scripts/test_quantize-int8-smoothquant.py`](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx/blob/main/scripts/test_quantize-int8-smoothquant.py)).
This fork was developed with [Claude Code](https://claude.com/claude-code).
The original upstream README follows.
