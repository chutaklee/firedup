import random
import time

import gym
import numpy as np
import torch
import torch.nn.functional as F

from fireup.algos.per_dqn import core
from fireup.utils.logx import EpochLogger
from stable_baselines.common.segment_tree import MinSegmentTree, SumSegmentTree
from stable_baselines.deepq.replay_buffer import ReplayBuffer


# ref: stable baselines, with slightly api modification
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def store(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample_batch(self, batch_size, beta=0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = list(
            encoded_sample
        )
        return dict(
            obs1=obs_batch,
            obs2=next_obs_batch,
            acts=act_batch.reshape(-1, 1),
            rews=rew_batch,
            done=np.array(done_mask, dtype=np.float32),
            weights=weights,
            idxes=idxes,
        )

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


def per_dqn(
    env_fn,
    dueling_dqn=False,
    double_dqn=False,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=5000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    min_replay_history=20000,
    prioritized_replay_alpha=0.6,
    beta_start=0.4,
    beta_frames=10000,
    epsilon_decay_period=250000,
    epsilon_train=0.01,
    epsilon_eval=0.001,
    lr=1e-3,
    max_ep_len=1000,
    update_period=4,
    target_update_period=8000,
    batch_size=100,
    logger_kwargs=dict(),
    save_freq=1,
):
    """
    Deep Q-Network w/ Prioritized Experience Replay from https://arxiv.org/abs/1511.05952

    positional arguments:

        --prioritized_replay_alpha

        --beta_start

        --beta_frames

    optional arguments:

        --dueling_dqn    enable Dueling DQN from https://arxiv.org/abs/1511.06581

        --double_dqn     enable Double DQN from https://arxiv.org/abs/1509.06461
    """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = 1  # env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs["action_space"] = env.action_space

    if dueling_dqn:
        dqnetwork = core.DuelingDQNetwork
    else:
        dqnetwork = core.DQNetwork

    # Main computation graph
    main = dqnetwork(in_features=obs_dim, **ac_kwargs)

    # Target network
    target = dqnetwork(in_features=obs_dim, **ac_kwargs)

    # Experience buffer
    replay_buffer = PrioritizedReplayBuffer(replay_size, prioritized_replay_alpha)

    # Count variables
    if dueling_dqn:
        var_counts = tuple(
            core.count_vars(module) for module in [main.enc, main.v, main.a, main]
        )
        print(
            (
                "\nNumber of parameters: \t encoder: %d, \t value head: %d \t advantage head: %d \t total: %d\n"
            )
            % var_counts
        )
    else:
        var_counts = tuple(core.count_vars(module) for module in [main.q, main])
        print(("\nNumber of parameters: \t q: %d, \t total: %d\n") % var_counts)

    # Value train op
    value_params = main.parameters()
    value_optimizer = torch.optim.Adam(value_params, lr=lr)

    # Initializing targets to match main variables
    target.load_state_dict(main.state_dict())

    def get_action(o, epsilon):
        """Select an action from the set of available actions.
        Chooses an action randomly with probability epsilon otherwise
        act greedily according to the current Q-value estimates.
        """
        if np.random.random() <= epsilon:
            return env.action_space.sample()
        else:
            q_values = main(torch.Tensor(o.reshape(1, -1)))
            # return the action with highest Q-value for this observation
            return torch.argmax(q_values, dim=1).item()

    def test_agent(n=10):
        for _ in range(n):
            o, r, done, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (done or (ep_len == max_ep_len)):
                # epsilon_eval used when evaluating the agent
                o, r, done, _ = test_env.step(get_action(o, epsilon_eval))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        main.eval()

        # the epsilon value used for exploration during training
        epsilon = core.linearly_decaying_epsilon(
            epsilon_decay_period, t, min_replay_history, epsilon_train
        )
        a = get_action(o, epsilon)

        # Step the env
        o2, r, done, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if ep_len == max_ep_len else done

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, done)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if done or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # train at the rate of update_period if enough training steps have been run
        if len(replay_buffer) > min_replay_history and t % update_period == 0:
            main.train()
            beta = core.beta_by_frame(t, beta_start, beta_frames)
            batch = replay_buffer.sample_batch(batch_size, beta)
            (obs1, obs2, acts, rews, done, weights, idxes) = (
                torch.Tensor(batch["obs1"]),
                torch.Tensor(batch["obs2"]),
                torch.Tensor(batch["acts"]),
                torch.Tensor(batch["rews"]),
                torch.Tensor(batch["done"]),
                torch.Tensor(batch["weights"]),
                batch["idxes"],
            )
            q_pi = main(obs1).gather(1, acts.long()).squeeze()
            if double_dqn:
                next_act_idx = main(obs2).argmax(-1, keepdim=True)
                q_pi_targ = target(obs2).gather(1, next_act_idx).squeeze()
            else:
                q_pi_targ, _ = target(obs2).max(1)

            # Bellman backup for Q function
            backup = (rews + gamma * (1 - done) * q_pi_targ).detach()

            # DQN loss
            td_error = F.smooth_l1_loss(q_pi, backup) * weights
            priorities = td_error.detach().numpy() + 1e-5
            value_loss = td_error.mean()

            # Q-learning update
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            logger.store(LossQ=value_loss.item(), QVals=q_pi.data.numpy())

            # replay buffer update
            replay_buffer.update_priorities(idxes, priorities)

        # syncs weights from online to target network
        if t % target_update_period == 0:
            target.load_state_dict(main.state_dict())

        # End of epoch wrap-up
        if len(replay_buffer) > min_replay_history and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({"env": env}, main, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("EpRet", with_min_and_max=True)
            logger.log_tabular("TestEpRet", with_min_and_max=True)
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("TestEpLen", average_only=True)
            logger.log_tabular("TotalEnvInteracts", t)
            logger.log_tabular("QVals", with_min_and_max=True)
            logger.log_tabular("LossQ", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()
