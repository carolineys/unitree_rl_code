import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from ppo_runner import MujocoOnPolicyRunner

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def print_mj_data(data):
    # Print joint positions (qpos)
    print("\nJoint positions (qpos):")
    print(data.qpos)

    # Print joint velocities (qvel)
    print("\nJoint velocities (qvel):")
    print(data.qvel)

    # Print joint accelerations (qacc)
    print("\nJoint accelerations (qacc):")
    print(data.qacc)

    # Print actuator forces (act)
    print("\nActuator forces (act):")
    print(data.act)



def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    train_cfg = {'algorithm': {'clip_param': 0.2, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95, 'learning_rate': 0.0005, 'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'schedule': 'adaptive', 'use_clipped_value_loss': True, 'value_loss_coef': 1.0}, 'init_member_classes': {}, 'policy': {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0}, 'runner': {'algorithm_class_name': 'PPO', 'checkpoint': -1, 'experiment_name': 'rough_a1', 'load_run': -1, 'max_iterations': 500, 'num_steps_per_env': 24, 'policy_class_name': 'ActorCritic', 'resume': True, 'resume_path': None, 'run_name': '', 'save_interval': 50}, 'runner_class_name': 'OnPolicyRunner', 'seed': 1}

    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_dof_pos = np.array(config["default_dof_pos"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_dof_pos.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    viewer = mujoco.viewer.launch_passive(m, d)
    m.opt.timestep = simulation_dt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_runner = MujocoOnPolicyRunner(
        m, d, viewer,
        train_cfg=train_cfg,
        log_dir='/home/caroline/Desktop/robotics/unitree_rl_gym/logs/a1', 
        device=device)
    ppo_runner.load("/home/caroline/Desktop/robotics/unitree_rl_gym/logs/rough_a1/Jan07_20-25-50_/model_1500.pt")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            # print_mj_data(d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                lin_vel = d.qvel[:3]
                ang_vel = d.qvel[3:6]

                qj = (qj - default_dof_pos) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)


                obs[:3] = lin_vel * lin_vel_scale
                obs[3:6] = ang_vel * ang_vel_scale
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = qj
                obs[24:36] = dqj
                obs[36:48] = action
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_dof_pos

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
