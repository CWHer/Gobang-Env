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
        action = [0 for _ in range(num_envs)]

        env.async_reset()
        with tqdm.tqdm(total=num_envs) as pbar:
            while not all(done):
                obs, reward, terminated, truncated, info = env.recv()
                self.assertTrue(np.logical_not(np.all(truncated)))
                is_player_done = info["is_player_done"]
                for i in range(num_envs):
                    if not done[i] and is_player_done[i]:
                        collect_this_state[i] = True
                        player_step_count[i] += 1
                        probs[i].append(obs.mcts_result[i])
                        mcts_result = obs.mcts_result[i]
                        mcts_result[mcts_result < 0] = 0
                        action[i] = np.random.choice(
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
                    "action": np.array(action, dtype=np.int32)
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
            for j, s in enumerate(state[i]):
                self.assertTrue(np.all(s[-1] == j % 2),
                                f"State {j} of env {i} is wrong")
                player0_step = np.sum(s[0])
                player1_step = np.sum(s[num_player_planes])
                self.assertTrue(player0_step + player1_step == j)
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
        action = [0 for _ in range(num_envs)]

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
                        mcts_result = obs.mcts_result[i]
                        mcts_result[mcts_result < 0] = 0
                        action[i] = np.random.choice(
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
                    "prior_probs": 0.1 * np.ones((num_envs, 15 * 15), dtype=np.float32),
                    "value": 0.1 * np.ones((num_envs, ), dtype=np.float32),
                    "action": np.array(action, dtype=np.int32)
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
            for j, s in enumerate(state[i]):
                self.assertTrue(np.all(s[-1] == j % 2),
                                f"State {j} of env {i} is wrong")
                player0_step = np.sum(s[0])
                player1_step = np.sum(s[num_player_planes])
                self.assertTrue(player0_step + player1_step == j)
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

    def testThroughput(self):
        num_envs = 400
        batch_size = 128
        num_threads = 4
        env = envpool.make_gym(
            "GobangSelfPlay", num_envs=num_envs,
            batch_size=batch_size, num_threads=num_threads
        )

        actions = {
            "prior_probs": 0.1 * np.ones((num_envs, 15 * 15), dtype=np.float32),
            "value": 0.1 * np.ones((num_envs, ), dtype=np.float32),
        }
        n_steps = 1000
        env.async_reset()
        start = time.time()
        for _ in tqdm.trange(n_steps):
            obs, reward, terminated, truncated, info = env.recv()
            actions["action"] = np.argmax(
                obs["mcts_result"], axis=1).astype(np.int32)
            env.send(actions, info["env_id"])
        duration = time.time() - start
        # fmt: off
        print(f"Throughput: {n_steps * batch_size / duration} steps/s")
        print(f"Latency: {duration / n_steps / (batch_size / num_threads) * 1000} ms/step")
        # fmt: on


if __name__ == "__main__":
    unittest.main()
