import numpy as np
import torch
import torch.nn.functional as F
import gym
import time
from fireup.algos.qr_dqn import core
from fireup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DQN agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs]
        )


"""

Quantile Regression Deep Q-Network from http://arxiv.org/abs/1806.06923

"""
def qr_dqn(
    env_fn,
    dqnetwork=core.DQNetwork,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=5000,
    epochs=100,
    replay_size=int(1e6),
    Vmin=-10.0,  # hyperparameters for not-atari env
    Vmax=10.0,  # hyperparameters for not-atari env
    num_quantiles=50,  # hyperparameters for not-atari env
    gamma=0.99,
    min_replay_history=20000,
    epsilon_decay_period=250000,
    epsilon_train=0.01,
    epsilon_eval=0.001,
    lr=1e-3,
    max_ep_len=1000,
    update_period=4,
    target_update_period=8000,
    batch_size=100,
    logger_kwargs=dict(),
    save_freq=1
):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = 1  # env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Main computation graph
    main = dqnetwork(in_features=obs_dim, **ac_kwargs)

    # Target network
    target = dqnetwork(in_features=obs_dim, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=replay_size
    )

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [main.q, main])
    print(('\nNumber of parameters: \t q: %d, \t total: %d\n')%var_counts)

    # Value train op
    value_params = main.q.parameters()
    value_optimizer = torch.optim.Adam(value_params, lr=lr)

    # Initializing targets to match main variables
    target.load_state_dict(main.state_dict())

    # Quantile regression stuffs
    k = 1.0

    def huber(x):
        return torch.where(x.abs() < k, x**2 / 2, k * (x.abs() - k / 2))

    tau = torch.Tensor(
        (2 * np.arange(num_quantiles) + 1) / (2.0 * num_quantiles)
    ).view(1, -1)

    def get_action(o, epsilon):
        """Select an action from the set of available actions.
        Chooses an action randomly with probability epsilon otherwise
        act greedily according to the current Q-value estimates.
        """
        if np.random.random() <= epsilon:
            return env.action_space.sample()
        else:
            return main.policy(torch.Tensor(o.reshape(1, -1))).item()

    def test_agent(n=10):
        for _ in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # epsilon_eval used when evaluating the agent
                o, r, d, _ = test_env.step(get_action(o, epsilon_eval))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def update():
        main.train()
        batch = replay_buffer.sample_batch(batch_size)
        (obs1, obs2, acts, rews, done) = (
            torch.Tensor(batch['obs1']),
            torch.Tensor(batch['obs2']),
            torch.LongTensor(batch['acts']),  # (bsz, 1)
            torch.Tensor(batch['rews']),  # (bsz)
            torch.Tensor(batch['done'])  # (bsz)
        )
        bsz = obs1.size(0)

        q_dist1 = main(obs1)  # (bsz, action_dim, num_quantiles)
        acts = acts.squeeze(1)  # (bsz)
        q_dist1 = q_dist1[range(bsz), acts]  # (bsz, num_atoms)

        q_dist2 = target(obs2).detach()
        act_idx2 = q_dist2.mean(-1).argmax(-1)  # (bsz)
        # act_idx2 = main(obs2).mean(-1).argmax(-1)  # double dqn
        q_dist2 = q_dist2[range(bsz), act_idx2]  # (bsz, num_quantiles)

        rews = rews.unsqueeze(1)  # (bsz, 1)
        done = done.unsqueeze(1)  # (bsz, 1)
        T_theta = rews + (1 - done) * gamma * q_dist2

        diff = T_theta.t().unsqueeze(-1) - q_dist1
        loss = huber(diff) * (tau - (diff.detach() < 0).float()).abs()
        loss = loss.mean()

        value_optimizer.zero_grad()
        loss.backward()
        value_optimizer.step()

        return loss.item(), q_dist2.detach().numpy()

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        main.eval()

        # the epsilon value used for exploration during training
        epsilon = core.linearly_decaying_epsilon(
            epsilon_decay_period,
            t,
            min_replay_history,
            epsilon_train
        )
        a = get_action(o, epsilon)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # train at the rate of update_period if enough training steps have been run
        if replay_buffer.size > min_replay_history and t % update_period == 0:
            loss, QDist = update()
            logger.store(LossQ=loss, QVals=QDist.mean(-1))

        # syncs weights from online to target network
        if t % target_update_period == 0:
            target.load_state_dict(main.state_dict())

        # End of epoch wrap-up
        if replay_buffer.size > min_replay_history and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, main, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()
