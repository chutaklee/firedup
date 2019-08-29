import numpy as np
import torch
import torch.nn.functional as F
import gym
import time
from fireup.algos.iqn import core
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
            done=self.done_buf[idxs],
        )


"""

Implicit Quantile Network from http://arxiv.org/abs/1806.06923

"""


def iqn(
    env_fn,
    dqnetwork=core.DQNetwork,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=5000,
    epochs=100,
    replay_size=int(1e6),
    quantile_embedding_dim=64,  # n in equation 4 in IQN paper
    num_tau_samples=16,  # N in equation 3 in IQN paper
    num_tau_prime_samples=8,  # N' in equation 3 in IQN paper
    num_quantile_samples=32,  # K in equation 3 in IQN paper
    kappa=1.0,  # kappa for Huber Loss in IQN
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
    save_freq=1,
):
    """
    quantile_embedding_dim :  # n in equation 4 in IQN paper
    num_tau_samples : N in equation 3 in IQN paper
    num_tau_prime_samples : N' in equation 3 in IQN paper
    num_quantile_samples : K in equation 3 in IQN paper
    kappa : kappa for Huber Loss in IQN
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
    ac_kwargs["quantile_embedding_dim"] = quantile_embedding_dim

    # Main computation graph
    main = dqnetwork(in_features=obs_dim, **ac_kwargs)

    # Target network
    target = dqnetwork(in_features=obs_dim, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [main.z, main])
    print(("\nNumber of parameters: \t z: %d, \t total: %d\n") % var_counts)

    # Value train op
    params = main.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)

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
            return main.policy(torch.Tensor(o.reshape(1, -1)), num_tau_samples).item()

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
        """ref: https://github.com/google/dopamine/blob/master/dopamine/agents/implicit_quantile/implicit_quantile_agent.py
        """
        main.train()
        batch = replay_buffer.sample_batch(batch_size)
        (obs1, obs2, acts1, rews, done) = (
            torch.Tensor(batch["obs1"]),
            torch.Tensor(batch["obs2"]),
            torch.LongTensor(batch["acts"]),  # (bsz, 1)
            torch.Tensor(batch["rews"]),  # (bsz)
            torch.Tensor(batch["done"]),  # (bsz)
        )

        action_dim = env.action_space.n
        bsz = obs1.size(0)
        with torch.no_grad():
            z2, _ = target(obs2, num_tau_prime_samples)

            assert z2.size() == (bsz, action_dim, num_tau_prime_samples)

            # acts2 = main(obs2, num_quantile_samples)[0].mean(dim=-1).argmax(dim=-1)  # double dqn
            acts2 = z2.mean(dim=-1).argmax(dim=-1)  # (bsz)

            rews = rews.unsqueeze(1)
            done = done.unsqueeze(1)
            backups = rews + (1 - done) * gamma * z2[range(bsz), acts2]

            assert backups.size() == (bsz, num_tau_prime_samples)

        z1, replay_tau = main(obs1, num_tau_samples)
        acts1 = acts1.squeeze(1)  # (bsz)
        z1 = z1[range(bsz), acts1]  # (bsz, num_tau_samples)

        bellman_errors = backups.unsqueeze(-1) - z1.unsqueeze(1)

        assert bellman_errors.size() == (bsz, num_tau_prime_samples, num_tau_samples)

        huber_loss1 = (abs(bellman_errors) <= kappa).float() * 0.5 * bellman_errors ** 2
        huber_loss2 = (
            (abs(bellman_errors) > kappa).float()
            * kappa
            * (abs(bellman_errors) - kappa / 2)
        )
        huber_loss = huber_loss1 + huber_loss2

        replay_tau = replay_tau.view(bsz, num_tau_samples).unsqueeze(
            1
        )  # (bsz, 1, num_tau_samples)
        replay_tau = replay_tau.repeat(1, num_tau_prime_samples, 1)

        assert replay_tau.size() == (bsz, num_tau_prime_samples, num_tau_samples)

        tau_huber_loss = abs(replay_tau - ((bellman_errors < 0).float()).detach())
        tau_huber_loss = tau_huber_loss * huber_loss / kappa

        loss = tau_huber_loss.sum(dim=2).mean(dim=1)  # (bsz)

        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), None

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
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
            logger.store(LossQ=loss)  # , QVals=QDist.mean(-1))

        # syncs weights from online to target network
        if t % target_update_period == 0:
            target.load_state_dict(main.state_dict())

        # End of epoch wrap-up
        if replay_buffer.size > min_replay_history and t % steps_per_epoch == 0:
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
            logger.log_tabular("LossQ", average_only=True)
            # logger.log_tabular("QVals", with_min_and_max=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()
