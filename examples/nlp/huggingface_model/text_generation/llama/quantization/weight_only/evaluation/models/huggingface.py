# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import copy
import os
import tempfile
from typing import Literal

import accelerate
import huggingface_hub
import lm_eval.api.instance
import lm_eval.api.model
import lm_eval.models.utils
import lm_eval.utils
import onnxruntime
import optimum.onnxruntime
import optimum.version
import packaging.version
import torch
import torch.nn.functional as F  # noqa: N812
import tqdm
import transformers

eval_logger = lm_eval.utils.eval_logger


class HFLM(lm_eval.api.model.TemplateLM):
    """An abstracted Huggingface model class. Enables usage with both models of
    `optimum.onnxruntime.ORTModelForCausalLM` and
    `optimum.onnxruntime.ORTModelForSeq2SeqLM` classes.
    """  # noqa: D205

    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str | transformers.PreTrainedModel | None = "gpt2",
        backend: Literal["default", "causal", "seq2seq"] | None = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: str | None = "main",
        tokenizer: str | transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | None = None,
        truncation: bool | None = False,
        logits_cache: bool = True,
        max_length: int | None = None,
        provider: str | None = "CPUExecutionProvider",
        batch_size: int | str | None = 1,
        max_batch_size: int | None = 64,
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
        add_bos_token: bool | None = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()

        available_providers = onnxruntime.get_available_providers()
        assert provider in available_providers, f"{provider} is not available."
        self._provider = provider
        self._device = torch.device("cpu")  # use cpu to generate torch tensor

        # optionally: take in an already-initialized ORTModel
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. " + "Many other model arguments may be ignored. "  # noqa: G003
            )
            self._model = pretrained
            self._config = self._model.config
            self.model.providers  # noqa: B018

            if tokenizer:
                assert isinstance(tokenizer, (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast))
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self._config._name_or_path
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )
        else:
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))

            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )

        # determine which of 'causal' and 'seq2seq' backends to use
        self._get_backend(config=self.config, backend=backend, trust_remote_code=trust_remote_code)

        # if we passed `pretrained` as a string, initialize our model now
        if isinstance(pretrained, str):
            self._create_model(pretrained=pretrained)

        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
        )

        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        if self.tokenizer.pad_token:
            pass
        elif self.tokenizer.unk_token:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif self.tokenizer.eos_token:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        elif getattr(self.config, "model_type", None) == "qwen":
            # Qwen's trust_remote_code tokenizer does not allow for adding special tokens
            self.tokenizer.pad_token = "<|endoftext|>"
        elif self.tokenizer.__class__.__name__ in ("RWKVWorldTokenizer", "Rwkv5Tokenizer"):
            # The RWKV world tokenizer, does not allow for adding special tokens /
            # setting the pad token (which is set as 0)
            # The additional tokenizer name check is needed, as there exists rwkv4 models with neox tokenizer
            # ---
            # Note that the world tokenizer class name, might change in the future
            # for the final huggingface merge
            # https://github.com/huggingface/transformers/pull/26963
            assert self.tokenizer.pad_token_id == 0
        else:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # TODO: override this for Gemma
        self.add_bos_token = add_bos_token
        if getattr(self.config, "model_type", None) == "gemma":
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', "  # noqa: ISC003, G003
                + "a BOS token will be used as Gemma underperforms without it."
            )

        self._max_length = max_length

        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        if not isinstance(pretrained, str):
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`,"  # noqa: ISC003, G003
                + " assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def provider(self):
        return self._provider

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _get_backend(
        self,
        config: transformers.PretrainedConfig | transformers.AutoConfig,
        backend: Literal["default", "causal", "seq2seq"] | None = "default",
        trust_remote_code: bool | None = False,
    ) -> None:
        """Helper method during initialization.

        Determines the backend ("causal" (decoder-only) or "seq2seq" (encoder-decoder))
        model type to be used.
        """
        assert backend in ["default", "causal", "seq2seq"]
        if backend != "default":
            # if we've settled on non-default backend, use that manually
            if backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
            eval_logger.info(f"Overrode HF model backend type, and using type '{backend}'")  # noqa: G004
        elif config.model_type in transformers.models.auto.modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
            # first check if model type is listed under seq2seq models, since some
            # models like MBart are listed in both seq2seq and causal mistakenly in HF transformers.
            # these special cases should be treated as seq2seq models.
            self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
        elif self.config.model_type in transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
        else:
            if not trust_remote_code:
                eval_logger.warning(
                    "HF model type is neither marked as CausalLM or Seq2SeqLM. \
                This is expected if your model requires `trust_remote_code=True` but may be an error otherwise."
                )
            # if model type is neither in HF transformers causal or seq2seq model registries
            # then we default to AutoModelForCausalLM
            self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

        assert self.AUTO_MODEL_CLASS in [
            transformers.AutoModelForCausalLM,
            transformers.AutoModelForSeq2SeqLM,
        ]

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
    ) -> None:
        self._config = transformers.PretrainedConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def _create_model(
        self,
        pretrained: str,
    ) -> None:
        """Create ORTModelForCausalLM or ORTModelForSeq2SeqLM model."""
        if not os.path.exists(pretrained):
            eval_logger.warning("`{}` path does not exist. Will try to download it from huggingface.")
            try:
                local_dir = tempfile.TemporaryDirectory().name
                huggingface_hub.snapshot_download(pretrained, local_dir=local_dir)
                pretrained = local_dir
            except Exception:  # noqa: TRY302
                raise

        if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS:
            if (
                not os.path.exists(os.path.join(pretrained, "decoder_model.onnx"))
                and not os.path.exists(os.path.join(pretrained, "decoder_with_past_model.onnx"))
                and not os.path.exists(os.path.join(pretrained, "decoder_model_merged.onnx"))
                and not os.path.exists(os.path.join(pretrained, "model.onnx"))
            ):
                raise ValueError(
                    "Couldn't find any ONNX model name in " + "['decoder_model.onnx', 'decoder_with_past_model.onnx', "  # noqa: ISC003
                    f"'decoder_model_merged.onnx', 'model.onnx'] in {pretrained}."
                )

            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

            if packaging.version.Version(optimum.version.__version__) >= packaging.version.Version("1.14.0"):
                if os.path.exists(os.path.join(pretrained, "model.onnx")):
                    session = optimum.onnxruntime.ORTModelForCausalLM.load_model(
                        os.path.join(pretrained, "model.onnx"), provider=self.provider, session_options=sess_options
                    )
                    inputs_names = [input.name for input in session.get_inputs()]  # noqa: A001
                    key_value_input_names = [key for key in inputs_names if (".key" in key) or (".value" in key)]
                    use_cache = len(key_value_input_names) > 0

                    self._model = optimum.onnxruntime.ORTModelForCausalLM(
                        session,
                        self.config,
                        use_cache=bool(use_cache),
                        use_io_binding=bool(use_cache),
                    )
                elif os.path.exists(os.path.join(pretrained, "decoder_model_merged.onnx")):
                    session = optimum.onnxruntime.ORTModelForCausalLM.load_model(
                        os.path.join(pretrained, "decoder_model_merged.onnx"),
                        provider=self.provider,
                        session_options=sess_options,
                    )
                    self._model = optimum.onnxruntime.ORTModelForCausalLM(session, self.config, use_cache=True)
                elif os.path.exists(os.path.join(pretrained, "decoder_with_past_model.onnx")):
                    session = optimum.onnxruntime.ORTModelForCausalLM.load_model(
                        os.path.join(pretrained, "decoder_with_past_model.onnx"),
                        provider=self.provider,
                        session_options=sess_options,
                    )
                    self._model = optimum.onnxruntime.ORTModelForCausalLM(session, self.config, use_cache=True)
                elif os.path.exists(os.path.join(pretrained, "decoder_model.onnx")):
                    session = optimum.onnxruntime.ORTModelForCausalLM.load_model(
                        os.path.join(pretrained, "decoder_model.onnx"),
                        provider=self.provider,
                        session_options=sess_options,
                    )
                    self._model = optimum.onnxruntime.ORTModelForCausalLM(
                        session, self.config, use_cache=False, use_io_binding=False
                    )
            elif os.path.exists(os.path.join(pretrained, "model.onnx")):
                session = optimum.onnxruntime.ORTModelForCausalLM.load_model(
                    os.path.join(pretrained, "model.onnx"), provider=self.provider, session_options=sess_options
                )
                inputs_names = session.get_inputs()
                key_value_input_names = [key for key in inputs_names if (".key" in key) or (".value" in key)]
                use_cache = len(key_value_input_names) > 0

                self._model = optimum.onnxruntime.ORTModelForCausalLM(
                    session[0],
                    self.config,
                    pretrained,
                    use_cache=bool(use_cache),
                    use_io_binding=bool(use_cache),
                )
            elif os.path.exists(os.path.join(pretrained, "decoder_model_merged.onnx")):
                sessions = optimum.onnxruntime.ORTModelForCausalLM.load_model(
                    os.path.join(pretrained, "decoder_model_merged.onnx"),
                    provider=self.provider,
                    session_options=sess_options,
                )
                self._model = optimum.onnxruntime.ORTModelForCausalLM(
                    sessions[0], self.config, pretrained, use_cache=True
                )
            elif os.path.exists(os.path.join(pretrained, "decoder_with_past_model.onnx")):
                sessions = optimum.onnxruntime.ORTModelForCausalLM.load_model(
                    os.path.join(pretrained, "decoder_model.onnx"),
                    os.path.join(pretrained, "decoder_with_past_model.onnx"),
                    provider=self.provider,
                    session_options=sess_options,
                )
                self._model = optimum.onnxruntime.ORTModelForCausalLM(
                    sessions[0], self.config, pretrained, sessions[1], use_cache=True
                )
            else:
                sessions = optimum.onnxruntime.ORTModelForCausalLM.load_model(
                    os.path.join(pretrained, "decoder_model.onnx"),
                    provider=self.provider,
                    session_options=sess_options,
                )
                self._model = optimum.onnxruntime.ORTModelForCausalLM(
                    sessions[0], self.config, pretrained, use_cache=False, use_io_binding=False
                )
        elif transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS:
            if not os.path.exists(os.path.join(pretrained, "encoder_model.onnx")) or (
                not os.path.exists(os.path.join(pretrained, "decoder_model.onnx"))
                and not os.path.exists(os.path.join(pretrained, "decoder_model_merged.onnx"))
            ):
                raise ValueError(
                    "Please ensure encoder_model.onnx and " f"decoder_model(_merged).onnx are under {pretrained}."  # noqa: ISC001
                )

            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            if os.path.exists(os.path.join(pretrained, "decoder_model_merged.onnx")):
                sessions = optimum.onnxruntime.ORTModelForSeq2SeqLM.load_model(
                    os.path.join(pretrained, "encoder_model.onnx"),
                    os.path.join(pretrained, "decoder_model_merged.onnx"),
                )

                self._model = optimum.onnxruntime.ORTModelForSeq2SeqLM(
                    sessions[0],
                    sessions[1],
                    self.config,
                    pretrained,
                    use_cache=True,
                )

            elif os.path.exists(os.path.join(pretrained, "decoder_with_past_model.onnx")):
                sessions = optimum.onnxruntime.ORTModelForSeq2SeqLM.load_model(
                    os.path.join(pretrained, "encoder_model.onnx"),
                    os.path.join(pretrained, "decoder_model.onnx"),
                    os.path.join(pretrained, "decoder_with_past_model.onnx"),
                )

                self._model = optimum.onnxruntime.ORTModelForSeq2SeqLM(
                    sessions[0],
                    sessions[1],
                    self.config,
                    pretrained,
                    sessions[2],
                    use_cache=True,
                )
            else:
                sessions = optimum.onnxruntime.ORTModelForSeq2SeqLM.load_model(
                    os.path.join(pretrained, "encoder_model.onnx"),
                    os.path.join(pretrained, "decoder_model.onnx"),
                )

                self._model = optimum.onnxruntime.ORTModelForSeq2SeqLM(
                    sessions[0],
                    sessions[1],
                    self.config,
                    pretrained,
                    use_cache=False,
                    use_io_binding=False,
                )

    def _create_tokenizer(
        self,
        pretrained: str | transformers.PreTrainedModel,
        tokenizer: str | transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | None,
        revision: str | None = "main",
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
    ) -> None:
        """Helper method during initialization.

        Create a tokenizer object corresponding to the correct
        tokenizer for value of `pretrained`, or use the pre-initialized tokenizer passed.
        """
        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )
            else:
                assert isinstance(tokenizer, (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast))
                self.tokenizer = tokenizer
        else:
            # Get tokenizer based on 'pretrained'
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                # get the HF hub name via accessor on model
                model_name = self.config._name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                use_fast=use_fast_tokenizer,
            )

    def _detect_batch_size(self, requests=None, pos: int = 0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len((context_enc + continuation_enc)[-(self.max_length + 1) :][:-1])
            max_context_enc = len(context_enc[-(self.max_length + 1) :])
            max_cont_enc = len(continuation_enc[-(self.max_length + 1) :])
        else:
            max_length = self.max_length

        # if OOM, then halves batch_size and tries again
        @accelerate.find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size):
            if transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS:
                length = max(max_context_enc, max_cont_enc)
                batched_conts = torch.ones((batch_size, length), device=self._device).long()
                test_batch = torch.ones((batch_size, length), device=self._device).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones((batch_size, max_length), device=self._device).long()
            for _ in range(5):
                F.log_softmax(self._model_call(test_batch, **call_kwargs), dim=-1)

            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        if self.world_size > 1:
            # if multi-GPU, always take minimum over all selected batch sizes
            max_rnk_bs = torch.tensor([batch_size], device=self._device)
            gathered = self.accelerator.gather(max_rnk_bs).cpu().detach().numpy().tolist()
            batch_size = min(gathered)
            lm_eval.models.utils.clear_torch_cache()
            return batch_size

        lm_eval.models.utils.clear_torch_cache()
        return batch_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> list[int]:
        if add_special_tokens is None:
            if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS:
                add_special_tokens = False or self.add_bos_token
            elif transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS:
                # TODO: investigate best practices for enc-dec models + special tokens
                add_special_tokens = True

        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int | None = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS:
            add_special_tokens = False or self.add_bos_token
        elif transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS:
            add_special_tokens = True

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens, skip_special_tokens=True):
        if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS:
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        elif transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS:
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        """Call model to get logits results.

        Args:
            inps (torch.Tensor):
                A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
                [batch, sequence_ctx]. the size of sequence may vary from call to call
            attn_mask (torch.Tensor, optional):
                A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
                (and must be passed) if self.AUTO_MODEL_CLASS is intel_extension_for_transformers.
                Defaults to None.
            labels (torch.Tensor, optional):
                A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
                (and must be passed) if self.AUTO_MODEL_CLASS is intel_extension_for_transformers.
                .transformers.AutoModelForSeq2SeqLM. Defaults to None.

        Returns:
            torch tensor: A torch tensor of shape [batch, sequence, vocab] with the
            logits returned from the model's decoder
        """
        if attn_mask is not None or labels is not None:
            assert attn_mask is not None and labels is not None
            assert transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS
            decoder_start_token_id = self._config.decoder_start_token_id
            pad_token_id = self._config.pad_token_id
            shifted_input_ids = labels.new_zeros(labels.shape)
            shifted_input_ids[..., 1:] = labels[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
            shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
            return self.model(
                inps,
                attention_mask=attn_mask,
                decoder_input_ids=shifted_input_ids,
                labels=labels,
            ).logits
        else:
            assert transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS
            if (
                hasattr(self.model, "config")
                and hasattr(self.model.config, "auto_map")
                and "chatglm2" in self.model.config.auto_map["AutoConfig"]
            ):
                input_bs, input_len = inps.shape
                bos = torch.tensor([64790, 64792]).repeat(input_bs, 1)
                inps = torch.cat((bos, inps), 1)

            inputs_names = [input.name for input in self.model.model.get_inputs()]  # noqa: A001
            if "position_ids" in inputs_names:
                # model is exported with optimum >= 1.14.0 with new input 'position_ids'
                input_shape = inps.shape
                position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
                output = self.model(
                    inps,
                    torch.ones(inps.shape, dtype=torch.int64),
                    position_ids,
                ).logits
            else:
                output = self.model(inps, torch.ones(inps.shape, dtype=torch.int64)).logits
            return output

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = lm_eval.models.utils.stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int | None = None, inplen: int | None = None
    ) -> torch.Tensor:
        if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS:
            assert contlen and inplen, "Must pass input len and cont. len to select scored logits for causal LM"
            # discard right-padding.
            # also discard the input/context tokens. we'll only score continuations.
            logits = logits[inplen - contlen : inplen]
        elif transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS:
            assert contlen and not inplen, "Selecting scored logits for Seq2SeqLM requires only cont. len"
            # only discard right-padding.
            # the logits input to this fn only contain decoder-side tokens.
            logits = logits[:contlen]

        return logits

    def loglikelihood_rolling(
        self, requests: list[lm_eval.api.instance.Instance], disable_tqdm: bool = False
    ) -> list[float]:
        loglikelihoods = []

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for (string,) in tqdm.tqdm([req.args for req in requests], disable=(disable_tqdm or (self.rank != 0))):
            rolling_token_windows = list(
                map(
                    lm_eval.utils.make_disjoint_window,
                    lm_eval.utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now,
            # we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None, *x) for x in rolling_token_windows]

            pad_amnt = 0
            if self.world_size > 1:
                # We pad out the external document-level iterator so the inner iterator doesn't hang
                mytensor = torch.tensor(len(rolling_token_windows), device=self._device)
                gathered = self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()

                pad_amnt = max(gathered) - gathered[self.rank]
                if pad_amnt > 0:
                    rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

            string_nll = self._loglikelihood_tokens(
                requests=rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )

            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (self.batch_sizes[sched - 1] == self.max_batch_size):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        print(f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size")
        self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        print(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched]

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        # TODO:
        # implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: tuple[tuple[str, str], list[int], list[int]]):
            """Defines the key for the sorted method."""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: tuple[tuple[str, str], list[int], list[int]]):
            """Defines the key to group and lookup one-token continuations."""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = lm_eval.models.utils.Collator(
            requests,
            sort_fn=_collate,
            group_by=(
                "contexts" if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS and self.logits_cache else None
            ),
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = self.batch_size if self.batch_size != "auto" else override_bs if override_bs is not None else 0
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and n_reordered_requests > 0 and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm.tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS:
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self._device,
                    )
                    (inplen,) = inp.shape
                elif transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS:
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self._device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self._device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = max(padding_len_cont, contlen) if padding_len_cont is not None else contlen

                padding_len_inp = max(padding_len_inp, inplen) if padding_len_inp is not None else inplen

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS:
                batched_inps = lm_eval.models.utils.pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS:
                # TODO: left-pad encoder inps and mask?
                batched_inps = lm_eval.models.utils.pad_and_concat(padding_len_inp, inps)  # [batch, padding_len_inp]
                batched_conts = lm_eval.models.utils.pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = lm_eval.models.utils.pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }
            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)  # noqa: PLW2901
                logits = logits.unsqueeze(0)  # [1, seq, vocab]  # noqa: PLW2901

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(  # noqa: B020, PLW2901
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(cont_toks, dtype=torch.long, device=self._device).unsqueeze(0)  # [1, seq]  # noqa: PLW2901
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]  # noqa: PLW2901

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(self, requests: list[lm_eval.api.instance.Instance], disable_tqdm: bool = False) -> list[str]:
        res = []

        def _collate(req: tuple[str, dict]):
            """Defines the key for the sorted method."""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm.tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size if adaptive_batch_size is not None else 0
        )
        batch_fn = self._batch_scheduler if self.batch_size == "auto" and not adaptive_batch_size else None

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = lm_eval.models.utils.Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs:
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [kwargs]
                    elif not isinstance(until, list):
                        raise ValueError(f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}")
            else:
                raise ValueError(f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}")  # noqa: TRY004
            # add EOS token to stop sequences
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs:
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS:
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
            elif transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS:
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self._device)
            attn_masks = attn_masks.to(self._device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS:
                    cont_toks = cont_toks[context_enc.shape[1] :]  # noqa: PLW2901

                s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res
