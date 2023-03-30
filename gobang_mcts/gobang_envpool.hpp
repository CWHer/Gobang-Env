#pragma once

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

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
                "c_puct"_.Bind(1.0), "num_search"_.Bind(1000),
                "temp"_.Bind(1.0), "dirichlet_alpha"_.Bind(0.3), "dirichlet_eps"_.Bind(0.25));
        }

        template <typename Config>
        static decltype(auto) StateSpec(const Config &conf)
        {
            // TODO: update encoder
            return MakeDict(
                "obs:state"_.Bind(Spec<int>({3, conf["board_size"_], conf["board_size"_]})),
                "obs:mcts_result"_.Bind(Spec<int>({conf["board_size"_] * conf["board_size"_]})),
                "info:winner"_.Bind(Spec<int>({1})));
        }

        template <typename Config>
        static decltype(auto) ActionSpec(const Config &conf)
        {
            return MakeDict(
                "prior_probs"_.Bind(Spec<float>({conf["board_size"_] * conf["board_size"_]})),
                "value"_.Bind(Spec<float>({1})));
        }
    };

    using GobangEnvSpec = EnvSpec<GobangEnvFns>;

    class GobangEnv : public Env<GobangEnvSpec>
    {
    protected:
        int board_size, win_length;
        float c_puct;
        int num_search;
        float inv_temp, dirichlet_alpha, dirichlet_eps;

        std::shared_ptr<GobangSelfPlay> game;
        bool done;

    private:
        void writeState()
        {
            State state = Allocate();
            auto state_ = game->getState();
            for (int index = 0, k = 0; k < 3; ++k)
                for (int i = 0; i < board_size; i++)
                    for (int j = 0; j < board_size; j++, index++)
                        state["obs:state"_](k, i, j) = state_[index];
            auto mcts_result_ = game->getSearchResult();
            for (int i = 0; i < mcts_result_.size(); i++)
                state["obs:mcts_result"_][i] = mcts_result_[i];
            state["info:winner"_] = done ? game->getWinner() : -1;

            // debug
            // int check_value = std::accumulate(
            //     mcts_result_.begin(), mcts_result_.end(), 0);
            // if (check_value + mcts_result_.size() > 0)
            //     game->display();
        }

    public:
        GobangEnv(const Spec &spec, int env_id)
            : Env<GobangEnvSpec>(spec, env_id),
              board_size(spec.config["board_size"_]),
              win_length(spec.config["win_length"_]),
              c_puct(spec.config["c_puct"_]),
              num_search(spec.config["num_search"_]),
              inv_temp(spec.config["temp"_]),
              dirichlet_alpha(spec.config["dirichlet_alpha"_]),
              dirichlet_eps(spec.config["dirichlet_eps"_])
        {
        }

        bool IsDone() override
        {
            return done;
        }

        void Reset() override
        {
            game = std::make_shared<GobangSelfPlay>(
                board_size, win_length,
                c_puct, num_search,
                inv_temp, dirichlet_alpha, dirichlet_eps);
            game->reset();
            done = game->step({}, 0);
            assert(!done);
            writeState();
        }

        void Step(const Action &action) override
        {
            std::vector<float> prior_probs(board_size * board_size, 0.0f);
            for (int i = 0; i < prior_probs.size(); i++)
                prior_probs[i] = action["prior_probs"_][i];
            done = game->step(prior_probs, action["value"_]);
            writeState();
        }
    };

    using GobangEnvPool = AsyncEnvPool<GobangEnv>;
} // namespace Gobang