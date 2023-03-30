#include "envpool/gobang_mcts/mcts.hpp"
#include "envpool/gobang_mcts/gobang_env.hpp"

#include <numeric>
#include <gtest/gtest.h>

using GobangMCTS = MCTS<GobangEnv, GobangBoard>;

TEST(MCTSTest, Encode)
{
    GobangEnv env(15, 5);
    env.reset();
    auto valid_actions = env.getActions();
    EXPECT_EQ(valid_actions.size(), 15 * 15);
    env.step(0);
    env.step(15);
    env.step(1);
    env.step(16);
    env.step(2);

    std::shared_ptr<GobangEnv> mcts_env = std::make_shared<GobangEnv>(env);
    GobangMCTS mcts(1.0, 1000, mcts_env);
    auto encoded_state = env.getState();
    auto mcst_encoded_state = mcts.getState();
    EXPECT_EQ(encoded_state.size(), mcst_encoded_state.size());
    for (size_t i = 0; i < encoded_state.size(); ++i)
        EXPECT_EQ(encoded_state[i], mcst_encoded_state[i]);
}

TEST(MCTSTest, Search)
{
    GobangEnv env(8, 5);
    env.reset();
    env.step(0);
    env.step(8);
    env.step(1);
    env.step(9);
    env.step(2);
    env.step(10);
    env.step(3);

    std::shared_ptr<GobangEnv> mcts_env = std::make_shared<GobangEnv>(env);
    int num_search = 1000;
    auto mcts = std::make_shared<GobangMCTS>(1.0, num_search, mcts_env);
    auto done = mcts->search({}, 0);
    while (!done)
    {
        std::vector<float> prior_probs(8 * 8, .1f);
        float value = 0.0;
        done = mcts->search(prior_probs, value);
    }
    mcts->display();
    auto result = mcts->getResult();
    int best_action, visit_count = 0;
    for (int i = 0; i < result.size(); i++)
    {
        if (result[i].second > visit_count)
        {
            best_action = result[i].first;
            visit_count = result[i].second;
        }
    }
    EXPECT_EQ(best_action, 4);

    mcts->step(best_action + 10);
    mcts->display();
    auto result_before = mcts->getResult(true);
    auto visit_count_before = std::accumulate(
        result_before.begin(), result_before.end(), 0,
        [](int sum, const std::pair<int, int> &p)
        { return sum + p.second; });
    EXPECT_EQ(result_before.size(), 8 * 8 - 8);
    done = mcts->search({}, 0);
    while (!done)
    {
        std::vector<float> prior_probs(8 * 8, .1f);
        float value = 0.0;
        done = mcts->search(prior_probs, value);
    }
    mcts->display();
    auto result_after = mcts->getResult();
    auto visit_count_after = std::accumulate(
        result_after.begin(), result_after.end(), 0,
        [](int sum, const std::pair<int, int> &p)
        { return sum + p.second; });
    EXPECT_EQ(visit_count_after - visit_count_before, num_search);

    // reset root
    mcts->step(-1);
    result = mcts->getResult(true);
    EXPECT_TRUE(result.empty());
}