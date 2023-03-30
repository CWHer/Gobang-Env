#include "envpool/gobang_mcts/gobang_env.hpp"

#include <gtest/gtest.h>

TEST(GobangEnvTest, Basic)
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
    auto encoded_state = env.getState();
    EXPECT_EQ(encoded_state.size(), 3 * 15 * 15);
    EXPECT_EQ(encoded_state.front(), 1);
    EXPECT_EQ(encoded_state.back(), 1);
    env.step(17);
    env.step(3);
    env.step(18);
    env.step(4);
    auto result = env.checkFinished();
    EXPECT_EQ(result.first, true);
    EXPECT_EQ(result.second, 0);
    env.display();

    env.reset();
    result = env.checkFinished();
    EXPECT_EQ(result.first, false);
    EXPECT_EQ(result.second, -1);
    auto all_actions = env.getActions();
    EXPECT_EQ(all_actions.size(), 15 * 15);
}

TEST(GobangEnvTest, Restore)
{
    GobangEnv env(15, 5);
    env.reset();
    env.step(0);
    env.step(15);
    env.step(1);
    env.step(16);
    env.step(2);
    env.step(17);
    env.step(3);
    auto stat = env.getStat();
    env.reset();
    env.setStat(stat);
    env.step(18);
    env.step(4);
    auto result = env.checkFinished();
    EXPECT_EQ(result.first, true);
    EXPECT_EQ(result.second, 0);

    env.reset();
    env.setStat(stat);
    env.step(18);
    env.step(4);
    result = env.checkFinished();
    EXPECT_EQ(result.first, true);
    EXPECT_EQ(result.second, 0);
    env.display();
}

TEST(GobangEnvTest, Draw)
{
    GobangEnv env(5, 10);
    for (int i = 0; i < 5 * 5; i++)
    {
        env.step(i);
        // env.display();
        // auto result = env.checkFinished();
        // EXPECT_EQ(result.first, false);
    }

    env.display();
    auto result = env.checkFinished();
    EXPECT_EQ(result.first, true);
    EXPECT_EQ(result.second, -1);
}
