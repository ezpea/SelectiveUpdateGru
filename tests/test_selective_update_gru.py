import unittest
import torch
import torch.nn as nn
from selective_update_gru import SelectiveUpdateGru


class TestSelectiveUpdateGru(unittest.TestCase):
    """
    Unit test for selective_update_gru
    """

    def setUp(self):
        self.input_size = 6
        self.hidden_size = 4
        self.num_layers = 3
        self.batch_size = 2
        self.seq_length = 4
        self.model = SelectiveUpdateGru(
            self.input_size, self.hidden_size, self.num_layers
        )

    def test_initialization(self):
        self.assertIsInstance(self.model.gru, nn.GRU)
        self.assertEqual(self.model.gru.input_size, self.input_size)
        self.assertEqual(self.model.gru.hidden_size, self.hidden_size)
        self.assertEqual(self.model.gru.num_layers, self.num_layers)

    def test_forward_pass(self):
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        out, h = self.model(x)
        self.assertEqual(
            out.shape, (self.batch_size, self.seq_length, self.hidden_size)
        )
        self.assertEqual(h.shape, (self.num_layers, self.batch_size, self.hidden_size))

    def test_selective_update(self):
        torch.manual_seed(0)
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        x[:, :, 0] = 1  # Set control signal to 1
        out, _ = self.model(x)
        # Check if the output matches the control signal
        self.assertTrue(torch.all(out[:, :, 0] == 1))

        x[:, :, 0] = 0  # Set control signal to 0
        out, _ = self.model(x)
        # Check if the output does not match the control signal
        self.assertTrue(torch.all(out[:, :, 0] == 0))

    def test_selective_update_2(self):
        torch.manual_seed(0)
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        freeze_mask = torch.randn(self.batch_size, self.seq_length) > 0
        x[:, :, 0] = torch.where(freeze_mask, 1, 0)
        out, _ = self.model(x)

        # Check if the control output matches the control signal
        self.assertTrue(torch.all(out[..., 0] == x[..., 0]))

        # Check that the network's output does not change when the control signal is
        # 1 and that it does change when the input is 0
        padded_out = torch.cat(
            [torch.zeros(self.batch_size, 1, self.hidden_size), out], dim=1
        )

        # outputs dont change when freeze mask is on
        any_of_non_control_output_changed = torch.any(
            padded_out[:, :, 1:].diff(dim=1), dim=-1
        )
        self.assertTrue(
            torch.all(
                torch.where(out[:, :, 0] != 0, ~any_of_non_control_output_changed, True)
            )
        )

        # outputs change when freeze mask is off
        self.assertTrue(
            torch.all(
                torch.where(out[:, :, 0] == 0, any_of_non_control_output_changed, True)
            )
        )


if __name__ == "__main__":
    unittest.main()
