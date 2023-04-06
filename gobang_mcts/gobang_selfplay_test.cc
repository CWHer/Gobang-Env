#include "envpool/gobang_mcts/gobang_selfplay.hpp"

#include <numeric>
#include <gtest/gtest.h>

TEST(GobangSelfPlayTest, Small)
{
    int num_search = 20000;
    GobangSelfPlay game(3, 3, 3, 1.0f, num_search);
    int player = 0;
    int step_count = 0, display_count = 0;
    game.reset();
    bool done = game.step({}, 0, 0);
    EXPECT_FALSE(done);

    int best_action;
    bool display_next = false;
    while (!done)
    {
        std::vector<float> prior_probs(3 * 3, .1f);
        float value = 0.0;
        if (display_next)
            prior_probs.clear(); // no prior probs for the first search
        done = game.step(prior_probs, value, best_action);

        if (display_next)
        {
            game.display();
            display_count++;
            display_next = false;
            player ^= 1;
        }

        bool is_player_done = game.isPlayerDone();
        if (is_player_done)
        {
            auto mcts_result = game.getSearchResult();
            int visit_count = 0;
            for (int i = 0; i < mcts_result.size(); i++)
            {
                if (mcts_result[i] > visit_count)
                {
                    best_action = i;
                    visit_count = mcts_result[i];
                }
            }
            display_next = true;
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
