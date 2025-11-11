import torch

class StateEstimator:
    """
    Simple state estimator.
    For now, it's a placeholder that directly uses ground truth values.
    This class can be extended to include filters like Kalman Filters.
    """
    def __init__(self, device):
        self.device = device
        self.estimated_lin_vel = torch.zeros(3, device=self.device)

    def update(self, dt, base_quat, base_lin_vel_world, base_ang_vel_world, dof_pos, dof_vel):
        """
        Update the state estimate.

        Args:
            dt (float): Time step.
            base_quat (torch.Tensor): Base orientation quaternion.
            base_lin_vel_world (torch.Tensor): Ground truth linear velocity in world frame.
            base_ang_vel_world (torch.Tensor): Ground truth angular velocity in world frame.
            dof_pos (torch.Tensor): Joint positions.
            dof_vel (torch.Tensor): Joint velocities.
        """
        # For now, we use a "perfect" estimator that just returns the ground truth.
        # This is where a real estimation algorithm (e.g., from IMU data) would go.
        self.estimated_lin_vel = base_lin_vel_world

    @property
    def get_estimated_lin_vel(self):
        return self.estimated_lin_vel
