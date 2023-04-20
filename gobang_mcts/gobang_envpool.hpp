#pragma once

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

#include "envpool/gobang_mcts/utils.hpp"
#include "envpool/gobang_mcts/gobang_selfplay.hpp"

namespace GobangSpace
{
    class GobangEnvFns
    {
    public:
        static decltype(auto) DefaultConfig()
        {
            return MakeDict(
                "board_size"_.Bind(15), "win_length"_.Bind(5),
                "num_player_planes"_.Bind(4),
                "c_puct"_.Bind(1.0), "num_search"_.Bind(1000),
                "delay_epsilon"_.Bind(0.0),
                "verbose_output"_.Bind(false));
            // Why do we need delay_epsilon?
            // e.g., num_envs = 400, bs = 128, num_search = 100, fixed_len = 40
            //  the 1st policy need ~ 400 / 128 * (40 - 1) * 100 ~ 12,000 steps to collect the first 10 episode
            //  however, the 2nd policy only need ~ 400 / 128 * 1 * 100 ~ 300 steps to collect the next 10 episode
            //  thus, this would cause unbalanced # sample
        }

        template <typename Config>
        static decltype(auto) StateSpec(const Config &conf)
        {
            return MakeDict(
                "obs:state"_.Bind(Spec<int>({conf["num_player_planes"_] * 2 + 1,
                                             conf["board_size"_], conf["board_size"_]})),
                "obs:mcts_result"_.Bind(Spec<int>({conf["board_size"_] * conf["board_size"_]})),
                "info:is_player_done"_.Bind(Spec<bool>({})),
                "info:player_step_count"_.Bind(Spec<int>({})),
                "info:winner"_.Bind(Spec<int>({})));
        }

        template <typename Config>
        static decltype(auto) ActionSpec(const Config &conf)
        {
            return MakeDict(
                "prior_probs"_.Bind(Spec<float>({conf["board_size"_] * conf["board_size"_]})),
                "value"_.Bind(Spec<float>({})), "selected_action"_.Bind(Spec<int>({})));
        }
    };

    using GobangEnvSpec = EnvSpec<GobangEnvFns>;

    class GobangEnv : public Env<GobangEnvSpec>
    {
    protected:
        int board_size, win_length;
        int num_player_planes;
        float c_puct;
        int num_search;

        std::shared_ptr<GobangSelfPlay> game;
        bool done;

        // delay
        int delay_steps;

        // debug
        int player_step_count = 0;
        bool verbose_output;

    private:
        void writeState()
        {
            State state = Allocate();
            auto state_ = game->getState();
            int *state_data = reinterpret_cast<int *>(state["obs:state"_].Data());
            std::copy(state_.begin(), state_.end(), state_data);
            // for (int index = 0, k = 0; k < num_player_planes * 2 + 1; ++k)
            //     for (int i = 0; i < board_size; i++)
            //         for (int j = 0; j < board_size; j++, index++)
            //             state["obs:state"_](k, i, j) = state_[index];

            bool is_player_done = game->isPlayerDone();
            if (is_player_done)
            {
                auto mcts_result_ = game->getSearchResult();
                int *mcts_result_data = reinterpret_cast<int *>(state["obs:mcts_result"_].Data());
                std::copy(mcts_result_.begin(), mcts_result_.end(), mcts_result_data);
            }
            state["info:is_player_done"_] = is_player_done;
            state["info:winner"_] = done ? game->getWinner() : -1;

            // debug
            if (is_player_done)
                player_step_count++;
            state["info:player_step_count"_] = player_step_count;
            if (done)
            {
                assertMsg(player_step_count == game->historical_actions.size(),
                          "Player step count should be equal to historical actions size");
                if (verbose_output)
                {
                    std::cout << "Player step count: " << player_step_count << std::endl;
                    std::cout << "Env id: " << env_id_ << std::endl;
                    game->display();
                    std::cout << std::endl;
                }
            }
        }

    public:
        GobangEnv(const Spec &spec, int env_id)
            : Env<GobangEnvSpec>(spec, env_id),
              board_size(spec.config["board_size"_]),
              win_length(spec.config["win_length"_]),
              num_player_planes(spec.config["num_player_planes"_]),
              c_puct(spec.config["c_puct"_]),
              num_search(spec.config["num_search"_]),
              delay_steps(spec.config["delay_epsilon"_] * env_id),
              verbose_output(spec.config["verbose_output"_])
        {
            if (verbose_output)
            {
                std::cout << "Env: " << env_id_
                          << " delay steps: " << delay_steps << std::endl;
            }
        }

        bool IsDone() override
        {
            return done;
        }

        void Reset() override
        {
            game = std::make_shared<GobangSelfPlay>(
                board_size, win_length, num_player_planes,
                c_puct, num_search);
            game->reset();
            done = false;
            player_step_count = 0;
            writeState();
            if (verbose_output)
            {
                std::cout << "Env: " << env_id_ << " reset" << std::endl;
            }
        }

        void Step(const Action &action) override
        {
            if (delay_steps > 0)
            {
                delay_steps--;
                if (verbose_output)
                {
                    std::cout << "Env: " << env_id_
                              << " remaining delay steps: " << delay_steps << std::endl;
                }
                writeState();
                return;
            }

            if (verbose_output && game->isPlayerDone())
            {
                std::cout << "Env: " << env_id_
                          << " step: " << static_cast<int>(action["selected_action"_]) << std::endl;
            }
            std::vector<float> prior_probs;
            if (!game->isPlayerDone())
            {
                // NOTE: prior_probs & values are of no use when mcts is not done
                prior_probs.resize(board_size * board_size);
                float *prior_probs_data = reinterpret_cast<float *>(action["prior_probs"_].Data());
                std::copy(prior_probs_data, prior_probs_data + prior_probs.size(), prior_probs.begin());
            }
            done = game->step(prior_probs, action["value"_],
                              action["selected_action"_]);
            writeState();
        }
    };

    using GobangEnvPool = AsyncEnvPool<GobangEnv>;
} // namespace Gobang