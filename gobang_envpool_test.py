import time
import unittest

import envpool
import numpy as np
import tqdm


class GobangTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        print(envpool.list_all_envs())

    def testSync(self):
        num_envs = 10
        num_threads = 2
        num_player_planes = 3
        num_search = 100
        env = envpool.make_gym(
            "GobangSelfPlay", num_envs=num_envs,
            num_threads=num_threads, num_player_planes=num_player_planes,
            num_search=num_search,
        )
        state = [[] for _ in range(num_envs)]
        probs = [[] for _ in range(num_envs)]
        done = [False for _ in range(num_envs)]
        collect_this_state = [False for _ in range(num_envs)]
        player_step_count = [0 for _ in range(num_envs)]
        winner = [-1 for _ in range(num_envs)]
        selected_action = np.zeros((num_envs, ), dtype=np.int32)

        env.async_reset()
        with tqdm.tqdm(total=num_envs) as pbar:
            while not all(done):
                obs, reward, terminated, truncated, info = env.recv()
                self.assertTrue(np.logical_not(np.all(truncated)))
                self.assertTrue(
                    np.all(info["env_id"][:-1] <= info["env_id"][1:]))
                is_player_done = info["is_player_done"]
                for i in range(num_envs):
                    if not done[i] and is_player_done[i]:
                        collect_this_state[i] = True
                        player_step_count[i] += 1
                        probs[i].append(obs.mcts_result[i])
                    if is_player_done[i]:
                        mcts_result = obs.mcts_result[i]
                        mcts_result[mcts_result < 0] = 0
                        selected_action[i] = np.random.choice(
                            np.arange(15 * 15), p=mcts_result / np.sum(mcts_result))

                for i in range(num_envs):
                    if not done[i] and terminated[i]:
                        pbar.update(1)
                        done[i] = True
                        self.assertEqual(
                            player_step_count[i], info["player_step_count"][i],
                            f"Wrong player_step_count at env {i}"
                        )
                        winner[i] = info["winner"][i]
                    if not done[i] and collect_this_state[i]:
                        state[i].append(obs.state[i])
                        collect_this_state[i] = False

                for i in range(num_envs):
                    if not done[i]:
                        self.assertEqual(
                            player_step_count[i], info["player_step_count"][i])

                env_id = info["env_id"]
                actions = {
                    "prior_probs": 0.1 * np.ones((num_envs, 15 * 15), dtype=np.float32),
                    "value": 0.1 * np.ones((num_envs, ), dtype=np.float32),
                    "selected_action": selected_action,
                }
                env.send(actions, env_id)

        self.assertGreaterEqual(np.array(player_step_count).std(), 5)
        for i in range(num_envs):
            self.assertTrue(
                winner[i] == -1 or 1 - winner[i] == player_step_count[i] % 2,
                f"Wrong winner at env {i}"
            )
            self.assertEqual(len(state[i]), len(probs[i]))
            self.assertEqual(len(state[i]), player_step_count[i])
            self.assertTrue(np.all(state[i][0] == 0))
            for j, s in enumerate(state[i][1:], start=1):
                self.assertTrue(np.all(s[-1] == j % 2),
                                f"State {j} of env {i} is wrong")
                player0_step = np.sum(s[0])
                player1_step = np.sum(s[num_player_planes])
                self.assertTrue(player0_step + player1_step == j)
                self.assertLessEqual(np.abs(player0_step - player1_step), 1)
                for k in range(1, num_player_planes):
                    self.assertTrue(
                        np.sum(s[k]) == max(0, player0_step - k),
                        f"Wrong at [{i}, {j}, {k}]"
                    )
                    self.assertTrue(
                        np.sum(s[k + num_player_planes]
                               ) == max(0, player1_step - k),
                        f"Wrong at [{i}, {j}, {k}]"
                    )

                res = state[i][j] - state[i][j - 1]
                for k in range(num_player_planes):
                    self.assertTrue(
                        np.all(res[k] >= 0)
                        and np.sum(res[k]) <= 1
                    )
                    self.assertTrue(
                        np.all(res[k + num_player_planes] >= 0)
                        and np.sum(res[k + num_player_planes]) <= 1
                    )

    def testAsync(self):
        num_envs = 20
        batch_size = 8
        num_threads = 4
        num_player_planes = 4
        num_search = 100
        env = envpool.make_gym(
            "GobangSelfPlay", num_envs=num_envs,
            batch_size=batch_size, num_threads=num_threads,
            num_player_planes=num_player_planes, num_search=num_search,
        )
        state = [[] for _ in range(num_envs)]
        probs = [[] for _ in range(num_envs)]
        done = [False for _ in range(num_envs)]
        collect_this_state = [False for _ in range(num_envs)]
        player_step_count = [0 for _ in range(num_envs)]
        winner = [-1 for _ in range(num_envs)]
        selected_action = np.zeros((num_envs, ), dtype=np.int32)

        env.async_reset()
        with tqdm.tqdm(total=num_envs) as pbar:
            while not all(done):
                obs, reward, terminated, truncated, info = env.recv()
                self.assertTrue(np.logical_not(np.all(truncated)))
                is_player_done = info["is_player_done"]
                for i, index in enumerate(info["env_id"]):
                    if not done[index] and is_player_done[i]:
                        collect_this_state[index] = True
                        player_step_count[index] += 1
                        probs[index].append(obs.mcts_result[i])
                    if is_player_done[i]:
                        mcts_result = obs.mcts_result[i]
                        mcts_result[mcts_result < 0] = 0
                        selected_action[index] = np.random.choice(
                            np.arange(15 * 15), p=mcts_result / np.sum(mcts_result))

                for i, index in enumerate(info["env_id"]):
                    if not done[index] and terminated[i]:
                        pbar.update(1)
                        done[index] = True
                        self.assertEqual(
                            player_step_count[index], info["player_step_count"][i],
                            f"Wrong player_step_count at env {index}"
                        )
                        winner[index] = info["winner"][i]
                    if not done[index] and collect_this_state[index]:
                        state[index].append(obs.state[i])
                        collect_this_state[index] = False

                for i, index in enumerate(info["env_id"]):
                    if not done[index]:
                        self.assertEqual(
                            player_step_count[index], info["player_step_count"][i])

                env_id = info["env_id"]
                actions = {
                    "prior_probs": 0.1 * np.ones((batch_size, 15 * 15), dtype=np.float32),
                    "value": 0.1 * np.ones((batch_size, ), dtype=np.float32),
                    "selected_action": selected_action[env_id]
                }
                env.send(actions, env_id)

        self.assertGreaterEqual(np.array(player_step_count).std(), 5)
        for i in range(num_envs):
            self.assertTrue(
                winner[i] == -1 or 1 - winner[i] == player_step_count[i] % 2,
                f"Wrong winner at env {i}"
            )
            self.assertEqual(len(state[i]), len(probs[i]))
            self.assertEqual(len(state[i]), player_step_count[i])
            self.assertTrue(np.all(state[i][0] == 0))
            for j, s in enumerate(state[i][1:], start=1):
                self.assertTrue(np.all(s[-1] == j % 2),
                                f"State {j} of env {i} is wrong")
                player0_step = np.sum(s[0])
                player1_step = np.sum(s[num_player_planes])
                self.assertEqual(player0_step + player1_step, j,
                                 f"Wrong at [{i}, {j}]")
                self.assertLessEqual(np.abs(player0_step - player1_step), 1)
                for k in range(1, num_player_planes):
                    self.assertTrue(
                        np.sum(s[k]) == max(0, player0_step - k),
                        f"Wrong at [{i}, {j}, {k}]"
                    )
                    self.assertTrue(
                        np.sum(s[k + num_player_planes]
                               ) == max(0, player1_step - k),
                        f"Wrong at [{i}, {j}, {k}]"
                    )

                res = state[i][j] - state[i][j - 1]
                for k in range(num_player_planes):
                    self.assertTrue(
                        np.all(res[k] >= 0)
                        and np.sum(res[k]) <= 1
                    )
                    self.assertTrue(
                        np.all(res[k + num_player_planes] >= 0)
                        and np.sum(res[k + num_player_planes]) <= 1
                    )

    @unittest.skip("Too slow")
    def testDelay(self):
        num_envs = 250
        batch_size = 100
        num_threads = 10
        num_search = 400
        estimated_len = 15 * 15
        num_explore = 5
        delay_epsilon = estimated_len * num_search / batch_size
        print("[INFO]: delay_epsilon", delay_epsilon)
        env = envpool.make_gym(
            "GobangSelfPlay", num_envs=num_envs,
            batch_size=batch_size, num_threads=num_threads,
            num_search=num_search, delay_epsilon=delay_epsilon
        )

        n_episodes = 1000
        episode_count = 0
        episode_steps = [0 for _ in range(n_episodes + 1)]
        step_count = 0

        player_step_count = [0 for _ in range(num_envs)]
        selected_action = np.zeros((num_envs, ), dtype=np.int32)

        env.async_reset()
        with tqdm.tqdm(total=n_episodes) as pbar:
            while episode_count < n_episodes:
                obs, reward, terminated, truncated, info = env.recv()
                self.assertTrue(np.logical_not(np.all(truncated)))
                is_player_done = info["is_player_done"]
                for i, index in enumerate(info["env_id"]):
                    if is_player_done[i]:
                        player_step_count[index] += 1
                        mcts_result = obs.mcts_result[i]
                        mcts_result[mcts_result < 0] = 0
                        selected_action[index] = np.argmax(mcts_result) \
                            if player_step_count[index] > num_explore \
                            else np.random.choice(np.arange(15 * 15), p=mcts_result / np.sum(mcts_result))

                for i, index in enumerate(info["env_id"]):
                    if terminated[i] and episode_count < n_episodes:
                        pbar.update(1)
                        episode_count += 1
                        episode_steps[episode_count] = step_count
                        self.assertEqual(
                            player_step_count[index], info["player_step_count"][i],
                            f"Wrong player_step_count at env {index}"
                        )
                        player_step_count[index] = 0

                env_id = info["env_id"]
                actions = {
                    "prior_probs": 0.1 * np.ones((batch_size, 15 * 15), dtype=np.float32),
                    "value": 0.1 * np.ones((batch_size, ), dtype=np.float32),
                    "selected_action": selected_action[env_id]
                }
                env.send(actions, env_id)
                step_count += 1

        episode_steps = np.array(episode_steps)
        import pickle
        with open("{}_{}_{}_{:.2f}.pkl".format(
                num_envs, batch_size, num_search, delay_epsilon), "wb") as f:
            pickle.dump(episode_steps, f)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(episode_steps)
        plt.title("delay_epsilon = {}".format(delay_epsilon))

        x = np.arange(n_episodes + 1)
        y = np.array(episode_steps)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        print("[INFO]: m, c", m, c)
        plt.plot(x, m * x + c, 'r')

        plt.subplot(1, 3, 2)
        n_episode_per_collect = 10
        episode_steps = np.diff(episode_steps)
        episode_steps_sum = episode_steps\
            .reshape(-1, n_episode_per_collect).sum(axis=1)

        plt.plot(episode_steps_sum)
        print(
            f"[INFO]: mean {np.mean(episode_steps_sum)}\n"
            f"[INFO]: std {np.std(episode_steps_sum)}\n"
            f"[INFO]: std / # search = {np.std(episode_steps_sum) / num_search}\n"
        )

        plt.subplot(1, 3, 3)
        plt.hist(episode_steps_sum, bins=20)

        plt.savefig("delay_epsilon_{:.2f}.png".format(delay_epsilon))

    def testThroughput(self):
        num_envs = 400
        batch_size = 128
        num_threads = 10
        num_search = 10
        env = envpool.make_gym(
            "GobangSelfPlay", num_envs=num_envs,
            batch_size=batch_size, num_threads=num_threads,
            num_search=num_search,
        )

        actions = {
            "prior_probs": 0.1 * np.ones((batch_size, 15 * 15), dtype=np.float32),
            "value": 0.1 * np.ones((batch_size, ), dtype=np.float32),
        }
        n_steps = 1000
        env.async_reset()
        start = time.time()
        for _ in tqdm.trange(n_steps):
            obs, reward, terminated, truncated, info = env.recv()
            actions["selected_action"] = np.argmax(
                obs["mcts_result"], axis=1).astype(np.int32)
            env.send(actions, info["env_id"])
        duration = time.time() - start
        # fmt: off
        print(f"Throughput: {n_steps * batch_size / duration} steps/s")
        print(f"Latency: {duration / n_steps / (batch_size / num_threads) * 1000} ms/step")
        # fmt: on


if __name__ == "__main__":
    unittest.main()
