import unittest

import numpy as np
import onnx
from onnx_neural_compressor.algorithms import utility as quant_utils


class TestQuantUtility(unittest.TestCase):

    def test_pad_tensor(self):
        data = np.random.random((100, 32))
        group_size = 32
        k_blocks = (100 - 1) // 32 + 1
        pad_data = quant_utils.pad_tensor(data, group_size, k_blocks)
        self.assertEqual(pad_data.shape, (k_blocks * group_size, 32))

    def test_4bit_quant_tensor(self):
        data = np.random.random((100, 32))
        q_data, scale, zp = quant_utils.quant_tensor(data)


    def test_quant_dequant_data(self):
        data = np.random.random((100, 32))
        qrange = quant_utils.get_qmin_qmax_for_qType(
            qType=onnx.TensorProto.UINT8,
            reduce_range=False,
            sym=True,
        )
        self.assertEqual(qrange[0], 0)
        self.assertEqual(qrange[1], 255)

        rmin = np.min(np.min(data), 0)
        rmax = np.max(np.max(data), 0)

        _, _, zero_point, scale, quantized_data = quant_utils.quantize_data(
            data=data,
            quantize_range=qrange,
            qType=onnx.TensorProto.UINT8,
            sym=True,
        )

        dq_data = quant_utils.dequantize_data(
            tensor_value=quantized_data,
            scale_value=scale,
            zo_value=zero_point,
        )
        self.assertLess(np.max(np.abs(dq_data - data)), 0.005)

        _, _, zero_point, scale, quantized_data = quant_utils.quantize_data_per_channel(
            data=data,
            quantize_range=qrange,
            qType=onnx.TensorProto.UINT8,
            sym=True,
            axis=1,
        )

        dq_data = quant_utils.dequantize_data(
            tensor_value=quantized_data,
            scale_value=scale,
            zo_value=zero_point,
            axis=1,
        )

        self.assertLess(np.max(np.abs(dq_data - data)), 0.005)


if __name__ == "__main__":
    unittest.main()