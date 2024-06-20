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


import glob
import json
import logging
import os
import re
import sys
from pathlib import Path

import lm_eval.logging_utils
import lm_eval.tasks
import lm_eval.utils
import numpy as np
from evaluation import evaluator

DEFAULT_RESULTS_FILE = "results.json"


def _handle_non_serializable(o):
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def cli_evaluate(args) -> None:
    if args.wandb_args:
        wandb_logger = lm_eval.logging_utils.WandbLogger(**lm_eval.utils.simple_parse_args_string(args.wandb_args))

    eval_logger = lm_eval.utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info("Verbosity set to %s", args.verbosity)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")  # noqa: G004
    task_manager = lm_eval.tasks.TaskManager(args.verbosity, include_path=args.include_path)

    if args.limit:
        eval_logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING.REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format("\n - ".join(task_manager.all_tasks)))  # noqa: G001
        sys.exit()
    elif os.path.isdir(args.tasks):
        task_names = []
        yaml_path = os.path.join(args.tasks, "*.yaml")
        for yaml_file in glob.glob(yaml_path):
            config = lm_eval.utils.load_yaml_config(yaml_file)
            task_names.append(config)
    else:
        task_list = args.tasks.split(",")
        task_names = task_manager.match_tasks(task_list)
        for task in [task for task in task_list if task not in task_names]:
            if os.path.isfile(task):
                config = lm_eval.utils.load_yaml_config(task)
                task_names.append(config)
        task_missing = [
            task for task in task_list if task not in task_names and "*" not in task
        ]  # we don't want errors if a wildcard ("*") task name was used

        if task_missing:
            missing = ", ".join(task_missing)
            eval_logger.error(
                f"Tasks were not found: {missing}\n"  # noqa: G004
                f"{lm_eval.utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
            )
            raise ValueError(
                f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks,"  # noqa: ISC003
                + " or '--verbosity DEBUG' to troubleshoot task registration issues."
            )

    if args.output_path:
        path = Path(args.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file():
            raise FileExistsError(f"File already exists at {path}")
        output_path_file = path.joinpath(DEFAULT_RESULTS_FILE)
        if output_path_file.is_file():
            eval_logger.warning(f"File {output_path_file} already exists. Results will be overwritten.")  # noqa: G004
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)

    # Respect user's value passed in via CLI, otherwise default to True and add to comma-separated model args
    if args.trust_remote_code:
        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = str(args.trust_remote_code)
        args.model_args = args.model_args + f",trust_remote_code={os.environ['HF_DATASETS_TRUST_REMOTE_CODE']}"

    eval_logger.info(f"Selected Tasks: {task_names}")  # noqa: G004
    eval_logger.info("Loading selected tasks...")

    request_caching_args = evaluator.request_caching_arg_to_dict(cache_requests=args.cache_requests)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        provider=args.provider,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        user_model=args.user_model,  # to validate the model in memory,
        tokenizer=args.tokenizer,  # to use tokenizer in mem,
        **request_caching_args,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        # Add W&B logging
        if args.wandb_args:
            try:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if args.log_samples:
                    wandb_logger.log_eval_samples(samples)
            except Exception as e:  # noqa: BLE001
                eval_logger.info(f"Logging to Weights and Biases failed due to {e}")  # noqa: G004

        if args.output_path:
            output_path_file.open("w", encoding="utf-8").write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():  # noqa: B007
                    output_name = "{}_{}".format(re.sub("/|=", "__", args.model_args), task_name)
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.write_text(samples_dumped, encoding="utf-8")

        print(
            f"{args.model} ({args.model_args}),"
            f" gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(lm_eval.utils.make_table(results))
        if "groups" in results:
            print(lm_eval.utils.make_table(results, "groups"))

        if args.wandb_args:
            # Tear down wandb run once all the logging is done.
            wandb_logger.run.finish()

    return results
