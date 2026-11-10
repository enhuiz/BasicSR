import ast
import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class SlurmArgumentTest(unittest.TestCase):
    def test_nodelist_is_one_argument_and_failure_prevents_init(self):
        source = Path(__file__).resolve().parents[1] / "basicsr/utils/dist_util.py"
        node = next(n for n in ast.parse(source.read_text()).body if getattr(n, "name", "") == "_init_dist_slurm")
        cuda = SimpleNamespace(device_count=lambda: 1, set_device=Mock())
        dist = SimpleNamespace(init_process_group=Mock())
        run = Mock(return_value=SimpleNamespace(stdout="worker0\nworker1\n"))
        ns = dict(os=os, torch=SimpleNamespace(cuda=cuda), dist=dist,
                  subprocess=SimpleNamespace(run=run))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
        node_list = "node; printf example"
        with patch.dict(os.environ, {"SLURM_PROCID": "0", "SLURM_NTASKS": "1", "SLURM_NODELIST": node_list}):
            ns["_init_dist_slurm"]("nccl")
            self.assertEqual(run.call_args.args[0], ["scontrol", "show", "hostname", node_list])
            self.assertNotIn("shell", run.call_args.kwargs)
            self.assertEqual(os.environ["MASTER_ADDR"], "worker0")
            dist.init_process_group.reset_mock()
            run.side_effect = subprocess.CalledProcessError(1, "scontrol")
            with self.assertRaises(subprocess.CalledProcessError):
                ns["_init_dist_slurm"]("nccl")
            dist.init_process_group.assert_not_called()


if __name__ == "__main__":
    unittest.main()
