#!/usr/bin/env python3
"""CPU runtime contracts; no training data, checkpoints or GPU required."""
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

import torch

torch.set_num_threads(1)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "test_composition", "test_decoder", "test_gpu_worker", "test_japanese",
    "test_rescorer", "test_transducer", "test_schemes",
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
