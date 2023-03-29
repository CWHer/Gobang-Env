#include "envpool/gobang_mcts/mcts.hpp"

#include <tuple>
#include <vector>
#include <unordered_map>

class SelfPlayGobang
{
private:
    static const int NUM_PLAYERS = 2;

    // specs
    int board_size, win_length;
    float c_puct;
    int num_search;

    // stat
    Gobang gobang_env;
    std::vector<std::shared_ptr<MCTS>> players;
    int current_player, winner;
    bool is_done;

    // episode data
    std::vector<int> historical_actions; // debug
    std::vector<std::pair<int, int>> actions_visits;

public:
    SelfPlayGobang(int board_size, int win_length,
                   float c_puct, int num_search)
        : board_size(board_size), win_length(win_length),
          c_puct(c_puct), num_search(num_search),
          gobang_env(board_size, win_length),
          current_player(0), winner(-1), is_done(false)
    {
    }

    void reset()
    {
        gobang_env.reset();
        players.clear();
        for (int i = 0; i < NUM_PLAYERS; ++i)
            players.push_back(std::make_shared<MCTS>(
                c_puct, num_search, std::make_shared<Gobang>(gobang_env)));
        current_player = 0;
        winner = -1;
        is_done = false;
    }

    int selectAction()
    {
        // greedy
        int best_action = 0;
        int visit_count = 0;
        for (const auto &action_visit : actions_visits)
            if (action_visit.second > visit_count)
            {
                best_action = action_visit.first;
                visit_count = action_visit.second;
            }
        return best_action;

        // TODO: fix this
        return actions_visits.front().first;
    }

    bool step(std::vector<float> prior_probs, float value)
    {
        actions_visits.clear();
        while (true)
        {
            auto player = players[current_player];
            auto done = player->search(prior_probs, value);
            if (!done)
                return false;
            prior_probs.clear();
            // player->display();

            // update game state
            // std::cout << "Update game state" << std::endl;
            actions_visits = player->getResult();
            auto action = selectAction();
            historical_actions.push_back(action);
            gobang_env.step(action);
            for (auto &player : players)
            {
                // HACK: reset root here
                //  avoid update gobang_env more than once in a single step
                player->step(action, true);
                // player.step(action);
            }
            std::tie(is_done, winner) = gobang_env.checkFinished();
            assert(winner == -1 || winner == current_player);
            if (is_done)
                return true;

            current_player ^= 1;
        }
    }

    int getWinner()
    {
        assert(is_done);
        return winner;
    }

    std::vector<int> getState()
    {
        return players[current_player]->getState();
    }

    std::vector<int> getSearchResult()
    {
        std::vector<int> visit_counts(board_size * board_size, -1);
        for (const auto &action_visit : actions_visits)
            visit_counts[action_visit.first] = action_visit.second;
        return visit_counts;
    }

    void display()
    {
        gobang_env.display();
        std::cout << "Current player: " << current_player << std::endl;
        std::cout << "Winner: " << winner << std::endl;
        std::cout << "Actions: ";
        for (const auto &action : historical_actions)
            std::cout << action << " ";
        std::cout << std::endl;
    }
};