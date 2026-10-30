from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner,_OnnxMotionModel
import os
from typing import cast
import torch
import torch.nn as nn
from mjlab.tasks.tracking.mdp import MotionCommand
class MotionTrackingCameraOnPolicyRunner(MotionTrackingOnPolicyRunner):
  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
    model = _OnnxCameraMotionModel(self.alg.get_policy(), cmd.motion)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    depth_camera = torch.zeros(1, 1, 270, 480)
    is_grab = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, depth_camera,is_grab),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "depth_camera"],
      output_names=[
        "actions",
      ],
      dynamic_axes={},
      dynamo=False,
    )

class _OnnxCameraMotionModel(_OnnxMotionModel):
  """ONNX-exportable model that wraps the policy and bundles motion reference data."""
  
  def forward(self, x, depth_camera,is_grab):
    return (
      self.policy(x,depth_camera,is_grab)
    )