"""Tests for general components."""

import unittest

from onnx_neural_compressor import constants, logger
from onnx_neural_compressor.quantization import config, tuning

from typing import Any, Callable, List, Optional, Tuple, Union  # isort: skip


PRIORITY_FAKE_ALGO = 100
FAKE_CONFIG_NAME = "fake"
PRIORITY_FAKE_ALGO_1 = 90
FAKE_CONFIG_NAME_1 = "fake_one"
DEFAULT_WEIGHT_BITS = [4, 6]

FAKE_MODEL_INFO = [("OP1_NAME", "OP_TYPE1"), ("OP2_NAME", "OP_TYPE1"), ("OP3_NAME", "OP_TYPE2")]


class FakeModel:

    def __init__(self) -> None:
        self.name = "fake_model"

    def __call__(self, x) -> Any:
        return x

    def __repr__(self) -> str:
        return "FakeModel"


@config.register_config(algo_name=FAKE_CONFIG_NAME, priority=PRIORITY_FAKE_ALGO)
class FakeAlgoConfig(config.BaseConfig):
    """Config class for fake algo."""

    supported_configs: List = []
    params_list = [
        "weight_dtype",
        "weight_bits",
        config.TuningParam("target_op_type_list", tunable_type=List[List[str]]),
    ]
    name = FAKE_CONFIG_NAME

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        target_op_type_list: List[str] = ["Conv", "Gemm"],
        white_list: Optional[List[Union[str, Callable]]] = constants.DEFAULT_WHITE_LIST,
    ):
        """Init fake config.

        Args:
            weight_dtype (str): Data type for weights, default is "int".
            weight_bits (int): Number of bits used to represent weights, default is 4.
        """
        super().__init__(white_list=white_list)
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.target_op_type_list = target_op_type_list
        self._post_init()

    def to_dict(self):
        return super().to_dict()

    @classmethod
    def from_dict(cls, config_dict):
        return super(FakeAlgoConfig, cls).from_dict(config_dict=config_dict)

    @classmethod
    def register_supported_configs(cls) -> List:
        pass

    @staticmethod
    def get_model_info(model: Any) -> List[Tuple[str, Any]]:
        return FAKE_MODEL_INFO

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "FakeAlgoConfig", List["FakeAlgoConfig"]]:
        return FakeAlgoConfig(weight_bits=DEFAULT_WEIGHT_BITS)


def get_default_fake_config() -> FakeAlgoConfig:
    """Generate the default fake config.

    Returns:
        the default fake config.
    """
    return FakeAlgoConfig()


@config.register_config(algo_name=FAKE_CONFIG_NAME_1, priority=PRIORITY_FAKE_ALGO_1)
class FakeAlgoOneConfig(config.BaseConfig):
    """Config class for fake algo."""

    supported_configs: List = []
    params_list = [
        "weight_dtype",
        "weight_bits",
        config.TuningParam("target_op_type_list", tunable_type=List[List[str]]),
    ]
    name = FAKE_CONFIG_NAME_1

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        target_op_type_list: List[str] = ["Conv", "Gemm"],
        white_list: Optional[List[Union[str, Callable]]] = constants.DEFAULT_WHITE_LIST,
    ):
        """Init fake config.

        Args:
            weight_dtype (str): Data type for weights, default is "int".
            weight_bits (int): Number of bits used to represent weights, default is 4.
        """
        super().__init__(white_list=white_list)
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.target_op_type_list = target_op_type_list
        self._post_init()

    def to_dict(self):
        return super().to_dict()

    @classmethod
    def from_dict(cls, config_dict):
        return super(FakeAlgoOneConfig, cls).from_dict(config_dict=config_dict)

    @classmethod
    def register_supported_configs(cls) -> List:
        pass

    @staticmethod
    def get_model_info(model: Any) -> List[Tuple[str, Any]]:
        return FAKE_MODEL_INFO

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "FakeAlgoOneConfig", List["FakeAlgoOneConfig"]]:
        return FakeAlgoOneConfig(weight_bits=DEFAULT_WEIGHT_BITS)


def get_all_config_set() -> Union[config.BaseConfig, List[config.BaseConfig]]:
    return config.get_all_config_set_from_config_registry()


config.register_supported_configs()


class TestEvaluator(unittest.TestCase):

    def test_single_eval_fn(self):

        def fake_eval_fn(model):
            return 1.0

        evaluator = tuning.Evaluator()
        evaluator.set_eval_fn_registry(fake_eval_fn)
        evaluator.self_check()
        self.assertEqual(evaluator.get_number_of_eval_functions(), 1)

    def test_single_eval_fn_dict(self):
        acc_data = iter([1.0, 0.8, 0.99, 1.0, 0.99, 0.99])

        def eval_acc_fn(model) -> float:
            return next(acc_data)

        eval_fns = {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"}

        evaluator = tuning.Evaluator()
        evaluator.set_eval_fn_registry(eval_fns)
        evaluator.self_check()
        self.assertEqual(evaluator.get_number_of_eval_functions(), 1)


class TestBaseConfig(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        # print the test name
        logger.info(f"Running TestBaseConfig test: {self.id()}")

    def test_api(self):
        fake_default_config = get_default_fake_config()
        self.assertEqual(fake_default_config.weight_dtype, "int")
        config_set = get_all_config_set()
        self.assertEqual(len(config_set), len(config.config_registry.get_all_config_cls()))
        self.assertEqual(
            [i for i in config_set if getattr(i, "name", "None") == FAKE_CONFIG_NAME][0].weight_bits,
            DEFAULT_WEIGHT_BITS,
        )

        model = FakeModel()
        fake_default_config.set_local("OP1_NAME", FakeAlgoConfig(weight_dtype="uint"))
        config_mapping = fake_default_config.to_config_mapping(model)
        self.assertEqual(config_mapping["OP1_NAME"]["weight_dtype"], "uint")
        self.assertEqual(config_mapping["OP2_NAME"]["weight_dtype"], "int")

    def test_config_expand_complex_tunable_type(self):
        target_op_type_list_options = [["Conv", "Gemm"], ["Conv", "Matmul"]]
        configs = FakeAlgoConfig(target_op_type_list=target_op_type_list_options)
        configs_list = configs.expand()
        self.assertEqual(len(configs_list), len(target_op_type_list_options))
        for i in range(len(configs_list)):
            self.assertEqual(configs_list[i].target_op_type_list, target_op_type_list_options[i])

    def test_mixed_two_algos(self):
        model = FakeModel()
        OP1_NAME = "OP1_NAME"
        OP2_NAME = "OP2_NAME"
        fake_config = FakeAlgoConfig(weight_bits=4, white_list=[OP1_NAME])
        fake1_config = FakeAlgoOneConfig(weight_bits=2, white_list=[OP2_NAME])
        mixed_config = fake_config + fake1_config
        config_mapping = mixed_config.to_config_mapping(model=model)
        self.assertIn(OP1_NAME, config_mapping)
        self.assertIn(OP2_NAME, config_mapping)

    def test_config_expand(self) -> None:
        cfg = config.RTNConfig(
            weight_bits=[4, 8], weight_sym=[True, False], layer_wise_quant=[True, False], providers=[["CPU"], ["CUDA"]]
        )
        expand_cfgs = cfg.expand()
        self.assertEqual(expand_cfgs[0].weight_bits, 4)
        self.assertEqual(expand_cfgs[0].weight_sym, True)
        self.assertEqual(expand_cfgs[0].layer_wise_quant, True)
        self.assertEqual(expand_cfgs[0].providers, ["CPU"])

        self.assertEqual(expand_cfgs[1].weight_bits, 8)
        self.assertEqual(expand_cfgs[1].weight_sym, True)
        self.assertEqual(expand_cfgs[1].layer_wise_quant, True)
        self.assertEqual(expand_cfgs[1].providers, ["CPU"])

        self.assertEqual(expand_cfgs[2].weight_bits, 4)
        self.assertEqual(expand_cfgs[2].weight_sym, False)
        self.assertEqual(expand_cfgs[2].layer_wise_quant, True)
        self.assertEqual(expand_cfgs[2].providers, ["CPU"])

        self.assertEqual(expand_cfgs[3].weight_bits, 8)
        self.assertEqual(expand_cfgs[3].weight_sym, False)
        self.assertEqual(expand_cfgs[3].layer_wise_quant, True)
        self.assertEqual(expand_cfgs[3].providers, ["CPU"])

        self.assertEqual(expand_cfgs[4].weight_bits, 4)
        self.assertEqual(expand_cfgs[4].weight_sym, True)
        self.assertEqual(expand_cfgs[4].layer_wise_quant, True)
        self.assertEqual(expand_cfgs[4].providers, ["CUDA"])

        self.assertEqual(expand_cfgs[5].weight_bits, 8)
        self.assertEqual(expand_cfgs[5].weight_sym, True)
        self.assertEqual(expand_cfgs[5].layer_wise_quant, True)
        self.assertEqual(expand_cfgs[5].providers, ["CUDA"])

        self.assertEqual(expand_cfgs[6].weight_bits, 4)
        self.assertEqual(expand_cfgs[6].weight_sym, False)
        self.assertEqual(expand_cfgs[6].layer_wise_quant, True)
        self.assertEqual(expand_cfgs[6].providers, ["CUDA"])

        self.assertEqual(expand_cfgs[7].weight_bits, 8)
        self.assertEqual(expand_cfgs[7].weight_sym, False)
        self.assertEqual(expand_cfgs[7].layer_wise_quant, True)
        self.assertEqual(expand_cfgs[7].providers, ["CUDA"])

        self.assertEqual(expand_cfgs[8].weight_bits, 4)
        self.assertEqual(expand_cfgs[8].weight_sym, True)
        self.assertEqual(expand_cfgs[8].layer_wise_quant, False)
        self.assertEqual(expand_cfgs[8].providers, ["CPU"])

        self.assertEqual(expand_cfgs[9].weight_bits, 8)
        self.assertEqual(expand_cfgs[9].weight_sym, True)
        self.assertEqual(expand_cfgs[9].layer_wise_quant, False)
        self.assertEqual(expand_cfgs[9].providers, ["CPU"])

        self.assertEqual(expand_cfgs[10].weight_bits, 4)
        self.assertEqual(expand_cfgs[10].weight_sym, False)
        self.assertEqual(expand_cfgs[10].layer_wise_quant, False)
        self.assertEqual(expand_cfgs[10].providers, ["CPU"])

        self.assertEqual(expand_cfgs[11].weight_bits, 8)
        self.assertEqual(expand_cfgs[11].weight_sym, False)
        self.assertEqual(expand_cfgs[11].layer_wise_quant, False)
        self.assertEqual(expand_cfgs[11].providers, ["CPU"])

        self.assertEqual(expand_cfgs[12].weight_bits, 4)
        self.assertEqual(expand_cfgs[12].weight_sym, True)
        self.assertEqual(expand_cfgs[12].layer_wise_quant, False)
        self.assertEqual(expand_cfgs[12].providers, ["CUDA"])

        self.assertEqual(expand_cfgs[13].weight_bits, 8)
        self.assertEqual(expand_cfgs[13].weight_sym, True)
        self.assertEqual(expand_cfgs[13].layer_wise_quant, False)
        self.assertEqual(expand_cfgs[13].providers, ["CUDA"])

        self.assertEqual(expand_cfgs[14].weight_bits, 4)
        self.assertEqual(expand_cfgs[14].weight_sym, False)
        self.assertEqual(expand_cfgs[14].layer_wise_quant, False)
        self.assertEqual(expand_cfgs[14].providers, ["CUDA"])

        self.assertEqual(expand_cfgs[15].weight_bits, 8)
        self.assertEqual(expand_cfgs[15].weight_sym, False)
        self.assertEqual(expand_cfgs[15].layer_wise_quant, False)
        self.assertEqual(expand_cfgs[15].providers, ["CUDA"])

    def test_config_expand_with_empty_options(self):
        configs = FakeAlgoConfig(weight_dtype=["int", "float32"], weight_bits=[])
        configs_list = configs.expand()
        self.assertEqual(len(configs_list), 2)


class TestConfigSet(unittest.TestCase):

    def setUp(self):
        self.config_set = [get_default_fake_config(), get_default_fake_config()]
        self.config_set_obj = tuning.ConfigSet.from_fwk_configs(self.config_set)

    def test_config_set(self) -> None:
        self.assertEqual(len(self.config_set_obj), len(self.config_set))
        self.assertEqual(self.config_set_obj[0].weight_bits, self.config_set[0].weight_bits)


class TestConfigSampler(unittest.TestCase):

    def setUp(self):
        self.config_set = [get_default_fake_config(), get_default_fake_config()]
        self.seq_sampler = tuning.SequentialSampler(self.config_set)

    def test_config_sampler(self) -> None:
        self.assertEqual(list(self.seq_sampler), list(range(len(self.config_set))))


class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        self.config_set = [FakeAlgoConfig(weight_bits=4), FakeAlgoConfig(weight_bits=8)]
        self.loader = tuning.ConfigLoader(self.config_set)

    def test_config_loader(self) -> None:
        self.assertEqual(len(list(self.loader)), len(self.config_set))
        for i, cfg in enumerate(self.loader):
            self.assertEqual(cfg, self.config_set[i])

    def test_config_loader_skip_verified_config(self) -> None:
        config_set = [FakeAlgoConfig(weight_bits=[4, 8]), FakeAlgoConfig(weight_bits=8)]
        config_loader = tuning.ConfigLoader(config_set)
        config_count = 0
        for i, _ in enumerate(config_loader):
            config_count += 1
        self.assertEqual(config_count, 2)


if __name__ == "__main__":
    unittest.main()
