import os

import torch

from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner


class ResMLPTrackingRunner(MotionTrackingOnPolicyRunner):
    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        os.makedirs(path, exist_ok=True)

        # as_onnx() 返回的就是 _OnnxResMLPModel（obs + depth_camera + graping -> actions）
        model = self.alg.get_policy().as_onnx(verbose=verbose)
        model.to("cpu")
        model.eval()

        # dummy 输入尺寸直接取 _OnnxResMLPModel.get_dummy_inputs()
        obs, depth_camera, graping = model.get_dummy_inputs()

        torch.onnx.export(
            model,
            (obs, depth_camera, graping),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=model.input_names,      # ["obs", "depth_camera", "graping"]
            output_names=model.output_names,    # ["actions"]
            dynamic_axes={},
            dynamo=False,
        )