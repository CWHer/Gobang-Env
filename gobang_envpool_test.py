import time
import unittest

import envpool
import numpy as np
import tqdm


class GobangTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(envpool.list_all_envs())

    def testSync(self):
        num_envs = 10
        num_threads = 2
        env = envpool.make_gym(
            "GobangSelfPlay", num_envs=num_envs,
            num_threads=num_threads
        )
        state = [[] for _ in range(num_envs)]
        probs = [[] for _ in range(num_envs)]
        done = [False for _ in range(num_envs)]
        collect_this_state = [True for _ in range(num_envs)]
        player_step_count = [0 for _ in range(num_envs)]
        winner = [-1 for _ in range(num_envs)]

        env.async_reset()
        with tqdm.tqdm(total=num_envs) as pbar:
            while not all(done):
                obs, reward, terminated, truncated, info = env.recv()
                player_step = np.sum(obs.mcts_result, axis=1) != -15 * 15
                for i in range(num_envs):
                    if not done[i] and player_step[i]:
                        collect_this_state[i] = True
                        player_step_count[i] += 1
                        probs[i].append(obs.mcts_result[i])

                for i in range(num_envs):
                    if not done[i] and (terminated[i] or truncated[i]):
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
                    "value": 0.1 * np.ones((num_envs, 1), dtype=np.float32),
                }
                env.send(actions, env_id)

        for i in range(num_envs):
            self.assertTrue(
                winner[i] == -1 or 1 - winner[i] == player_step_count[i] % 2,
                f"Wrong winner at env {i}"
            )
            self.assertEqual(len(state[i]), len(probs[i]))
            self.assertEqual(len(state[i]), player_step_count[i])
            for i, s in enumerate(state[i]):
                self.assertTrue(np.all(s[-1] == i % 2))
                self.assertTrue(np.sum(s[:-1]) == i)

    def testAsync(self):
        num_envs = 20
        batch_size = 8
        num_threads = 4
        env = envpool.make_gym(
            "GobangSelfPlay", num_envs=num_envs,
            batch_size=batch_size, num_threads=num_threads
        )
        state = [[] for _ in range(num_envs)]
        probs = [[] for _ in range(num_envs)]
        done = [False for _ in range(num_envs)]
        collect_this_state = [True for _ in range(num_envs)]
        player_step_count = [0 for _ in range(num_envs)]
        winner = [-1 for _ in range(num_envs)]

        env.async_reset()
        with tqdm.tqdm(total=num_envs) as pbar:
            while not all(done):
                obs, reward, terminated, truncated, info = env.recv()
                player_step = np.sum(obs.mcts_result, axis=1) != -15 * 15
                for i, index in enumerate(info["env_id"]):
                    if not done[index] and player_step[i]:
                        collect_this_state[index] = True
                        player_step_count[index] += 1
                        probs[index].append(obs.mcts_result[i])

                for i, index in enumerate(info["env_id"]):
                    if not done[index] and (terminated[i] or truncated[i]):
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
                    "value": 0.1 * np.ones((num_envs, 1), dtype=np.float32),
                }
                env.send(actions, env_id)

        for i in range(num_envs):
            self.assertTrue(
                winner[i] == -1 or 1 - winner[i] == player_step_count[i] % 2,
                f"Wrong winner at env {i}"
            )
            self.assertEqual(len(state[i]), len(probs[i]))
            self.assertEqual(len(state[i]), player_step_count[i])
            for i, s in enumerate(state[i]):
                self.assertTrue(np.all(s[-1] == i % 2))
                self.assertTrue(np.sum(s[:-1]) == i)

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
            "value": 0.1 * np.ones((num_envs, 1), dtype=np.float32),
        }
        n_steps = 1000
        env.async_reset()
        start = time.time()
        for _ in tqdm.trange(n_steps):
            obs, reward, terminated, truncated, info = env.recv()
            env.send(actions, info["env_id"])
        duration = time.time() - start
        # fmt: off
        print(f"Throughput: {n_steps * batch_size / duration} steps/s")
        print(f"Latency: {duration / n_steps / (batch_size / num_threads) * 1000} ms/step")
        # fmt: on


if __name__ == "__main__":
    unittest.main()
