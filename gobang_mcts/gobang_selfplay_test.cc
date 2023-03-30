#include "envpool/gobang_mcts/gobang_selfplay.hpp"

#include <numeric>
#include <gtest/gtest.h>

TEST(GobangSelfPlayTest, Small)
{
    int num_search = 20000;
    GobangSelfPlay game(3, 3, 1.0f, num_search, 1e-5);
    int player = 0;
    int step_count = 0, display_count = 0;
    game.reset();
    bool done = game.step({}, 0);
    EXPECT_FALSE(done);
    while (!done)
    {
        std::vector<float> prior_probs(3 * 3, .1f);
        float value = 0.0;
        done = game.step(prior_probs, value);

        auto mcts_result = game.getSearchResult();
        int check_value = std::accumulate(
            mcts_result.begin(), mcts_result.end(), 0);
        if (check_value + mcts_result.size() > 0) // not all -1
        {
            player ^= 1;
            game.display();
            display_count++;
        }
        step_count++;
        // std::cout << "step: " << step_count << std::endl;
    }

    EXPECT_EQ(display_count, 3 * 3);
    std::cout << "Step: " << step_count << std::endl;
    EXPECT_LT(step_count, num_search * 3 * 3);
    auto winner = game.getWinner();
    EXPECT_TRUE(winner == -1); // when num_search is large enough
}
