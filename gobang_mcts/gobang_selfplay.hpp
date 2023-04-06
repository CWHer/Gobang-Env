#pragma once

#include "envpool/gobang_mcts/utils.hpp"
#include "envpool/gobang_mcts/mcts.hpp"
#include "envpool/gobang_mcts/gobang_env.hpp"

#include <tuple>
#include <vector>
#include <numeric>
#include <algorithm>
#include <unordered_map>

class GobangSelfPlay
{
private:
    using GobangMCTS = MCTS<GobangEnv, GobangBoard>;
    static const int NUM_PLAYERS = 2;

    // specs
    int board_size, win_length;
    int num_player_planes;
    float c_puct;
    int num_search;

    // stat
    GobangEnv gobang_env;
    std::vector<std::shared_ptr<GobangMCTS>> players;
    int current_player, winner;
    bool is_player_done, is_game_done;

    // episode data
    std::vector<std::pair<int, int>> actions_visits;

public:
    std::vector<int> historical_actions; // debug

public:
    GobangSelfPlay(int board_size, int win_length, int num_player_planes,
                   float c_puct, int num_search)
        : board_size(board_size), win_length(win_length),
          num_player_planes(num_player_planes),
          c_puct(c_puct), num_search(num_search),
          gobang_env(board_size, win_length),
          current_player(0), winner(-1),
          is_player_done(false), is_game_done(false)
    {
    }

    void reset()
    {
        gobang_env.reset();
        players.clear();
        for (int i = 0; i < NUM_PLAYERS; ++i)
            players.push_back(std::make_shared<GobangMCTS>(
                c_puct, num_search, std::make_shared<GobangEnv>(gobang_env)));
        current_player = 0;
        winner = -1;
        is_player_done = false;
        is_game_done = false;
    }

    bool step(std::vector<float> prior_probs, float value, int action)
    {
        while (true)
        {
            if (!is_player_done)
            {
                auto player = players[current_player];
                auto done = player->search(prior_probs, value);
                if (!done)
                    return false;
                // player->display();

                // update game state
                // std::cout << "Update game state" << std::endl;
                actions_visits = player->getResult();
                is_player_done = true;
                return false;
            }
            actions_visits.clear();
            is_player_done = false;
            historical_actions.push_back(action);
            gobang_env.step(action);
            for (auto &player : players)
            {
                // HACK: reset root here
                //  avoid update gobang_env more than once in a single step
                // player->step(action, true);

                // HACK: is_player_done ensures that gobang_env is updated only once
                player->step(action);
            }
            std::tie(is_game_done, winner) = gobang_env.checkFinished();
            assertMsg(winner == -1 || winner == current_player,
                      "Winner is not current player",
                      __FILE__, __LINE__);
            if (is_game_done)
                return true;

            current_player ^= 1;
        }
    }

    int getWinner()
    {
        assertMsg(is_game_done, "Game is not done yet",
                  __FILE__, __LINE__);
        return winner;
    }

    bool isPlayerDone()
    {
        return is_player_done;
    }

    std::vector<int> getState()
    {
        if (!is_player_done) // for inference
            return players[current_player]->getState(num_player_planes);
        return gobang_env.getState(num_player_planes); // for training
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