#include "envpool/gobang_mcts/gobang_envpool.hpp"

#include <gtest/gtest.h>

using GobangAction = typename GobangSpace::GobangEnv::Action;
using GobangState = typename GobangSpace::GobangEnv::State;

TEST(GobangEnvPoolTest, Single)
{
    auto config = GobangSpace::GobangEnvSpec::kDefaultConfig;
    int num_envs = 1;
    int batch_size = 1;
    config["num_envs"_] = num_envs;
    config["batch_size"_] = batch_size;
    config["num_threads"_] = 1;
    config["board_size"_] = 3;
    config["win_length"_] = 3;
    config["c_puct"_] = 1.0;
    config["num_search"_] = 20000;
    config["temp"_] = 1e-5;

    GobangSpace::GobangEnvSpec spec(config);
    GobangSpace::GobangEnvPool envpool(spec);
    Array all_env_ids(Spec<int>({num_envs}));
    for (int i = 0; i < num_envs; ++i)
        all_env_ids[i] = i;
    envpool.Reset(all_env_ids);
    int step_count = 0;
    int player_step = 0;
    while (true)
    {
        auto state_vec = envpool.Recv();
        GobangState state(&state_vec);
        // auto state_keys = state.StaticKeys();
        int check_value = 0;
        for (int i = 0; i < 3 * 3; ++i)
            check_value += static_cast<int>(state["obs:mcts_result"_][0][i]);
        if (check_value + 3 * 3 > 0)
        {
            player_step++;
            std::cout << "Player step: " << player_step << std::endl;
        }

        if (state["done"_][0])
        {
            EXPECT_EQ(player_step, 3 * 3);
            EXPECT_EQ(static_cast<int>(state["info:winner"_][0]), -1);
            break;
        }
        // construct action
        std::vector<Array> raw_action({Array(Spec<int>({batch_size})),
                                       Array(Spec<int>({batch_size})),
                                       Array(Spec<float>({batch_size, 3 * 3})),
                                       Array(Spec<float>({batch_size}))});
        GobangAction action(&raw_action);
        // auto action_keys = action.StaticKeys();
        auto env_id = state["info:env_id"_];
        for (int i = 0; i < batch_size; ++i)
        {
            action["env_id"_][i] = env_id[i];
            for (int j = 0; j < 3 * 3; ++j)
                action["prior_probs"_][i][j] = .1f;
            action["value"_][i] = 0;
        }
        envpool.Send(action);

        step_count++;
        // std::cout << "Step count: " << step_count << std::endl;
    }

    std::cout << "Step: " << step_count << std::endl;
    EXPECT_LT(step_count, config["num_search"_] * 3 * 3);
}