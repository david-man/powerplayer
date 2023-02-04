import numpy as np
import gym as gym
from ray.rllib import MultiAgentEnv
from ray.tune import register_env, tune
from ray.util.client import ray


class IrrigationEnv(MultiAgentEnv):
    def __init__(self, return_agent_actions = False, part=False):
        self.num_agents = 5
        self.observation_space = gym.spaces.Box(low=200, high=800, shape=(1,))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))

    def reset(self):
        obs = {}
        self.water = np.random.uniform(200,800)
        for i in range(self.num_agents):
            obs[i] = np.array([self.water])
        return obs

    def cal_rewards(self, action_dict):
        self.curr_water = self.water
        reward = 0
        for i in range(self.num_agents):
            water_demanded = self.water*action_dict[i][0]
            if self.curr_water == 0:
                # No water is left in stream
                reward -= water_demanded*100 # Penalty
            elif self.curr_water - water_demanded<0:
                # Water in stream is less than water demanded, withdraw all left
                water_needed = water_demanded - self.curr_water
                water_withdrawn = self.curr_water
                self.curr_water = 0
                reward += -water_withdrawn**2 + 200*water_withdrawn
                reward -= water_needed*100 # Penalty
            else:
                # Water in stream is more than water demanded, withdraw water demanded
                self.curr_water -= water_demanded
                water_withdrawn = water_demanded
                reward += -water_withdrawn**2 + 200*water_withdrawn

        return reward

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        reward = self.cal_rewards(action_dict)

        for i in range(self.num_agents):

            obs[i], rew[i], done[i], info[i] = np.array([self.curr_water]), reward, True, {}

        done["__all__"] = True
        return obs, rew, done, info
def env_creator(_):
    return IrrigationEnv()
if __name__ == '__main__':

    single_env = IrrigationEnv()
    env_name = "IrrigationEnv"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space
    num_agents = single_env.num_agents
    def gen_policy():
        return (None, obs_space, act_space, {})
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()
    def policy_mapping_fn(agent_id):
        return 'agent-' + str(agent_id)


    config = {
        "log_level": "WARN",
        "num_workers": 3,
        "num_cpus_for_driver": 1,
        "num_cpus_per_worker": 1,
        "lr": 5e-3,
        "model": {"fcnet_hiddens": [8, 8]},
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "env": "IrrigationEnv"
    }

    exp_name = 'more_corns_yey'
    exp_dict = {
        'name': exp_name,
        'run_or_experiment': 'PPO',
        "stop": {
            "training_iteration": 100
        },
        'checkpoint_freq': 20,
        "config": config,
    }
    ray.init()
    tune.run(**exp_dict)
