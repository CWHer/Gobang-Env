#pragma once

#include "envpool/gobang_mcts/mcts.hpp"
#include "envpool/gobang_mcts/gobang_env.hpp"

#include <tuple>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <unordered_map>

class GobangSelfPlay
{
private:
    using GobangMCTS = MCTS<GobangEnv, GobangBoard>;
    static const int NUM_PLAYERS = 2;
    std::mt19937 rng;

    // specs
    int board_size, win_length;
    float c_puct;
    int num_search;
    float inv_temp, dirichlet_alpha, dirichlet_eps;

    // stat
    GobangEnv gobang_env;
    std::vector<std::shared_ptr<GobangMCTS>> players;
    int current_player, winner;
    bool is_done;

    // episode data
    std::vector<std::pair<int, int>> actions_visits;

public:
    std::vector<int> historical_actions; // debug

private:
    std::vector<float> softMax(const std::vector<float> &logits)
    {
        float max_logit = std::numeric_limits<float>::lowest();
        std::for_each(logits.begin(), logits.end(),
                      [&](const float &value)
                      { max_logit = std::max(max_logit, value); });
        std::vector<float> probs(logits);
        std::for_each(probs.begin(), probs.end(),
                      [=](float &value)
                      { value = std::exp(value - max_logit); });
        float sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0f);
        std::for_each(probs.begin(), probs.end(),
                      [=](float &value)
                      { value /= sum_probs; });
        return probs;
    }

    int selectAction()
    {
        // greedy
        // int best_action = 0;
        // int visit_count = 0;
        // for (const auto &action_visit : actions_visits)
        //     if (action_visit.second > visit_count)
        //     {
        //         best_action = action_visit.first;
        //         visit_count = action_visit.second;
        //     }
        // return best_action;

        // stochastic
        // 1. softmax
        auto logits = std::vector<float>(actions_visits.size(), 0.0f);
        std::transform(actions_visits.begin(), actions_visits.end(), logits.begin(),
                       [=](const std::pair<int, int> &action_visit)
                       { return inv_temp * std::log(action_visit.second + 1e-10); });
        auto probs = softMax(logits);
        // 2. dirichlet noise
        // TODO
        auto dirichlet_noise = std::vector<float>(actions_visits.size(), 0.0f);

        auto probs_with_noise = probs;
        std::discrete_distribution<int> dist(probs_with_noise.begin(), probs_with_noise.end());
        auto index = dist(rng);
        auto selected_action = actions_visits[index].first;
        return selected_action;
    }

public:
    GobangSelfPlay(int board_size, int win_length,
                   float c_puct, int num_search,
                   float temp = 1.0, float dirichlet_alpha = 0.3, float dirichlet_eps = 0.25)
        : rng(std::random_device()()),
          board_size(board_size), win_length(win_length),
          c_puct(c_puct), num_search(num_search),
          inv_temp(1.0f / temp), dirichlet_alpha(dirichlet_alpha), dirichlet_eps(dirichlet_eps),
          gobang_env(board_size, win_length),
          current_player(0), winner(-1), is_done(false)
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
        is_done = false;
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