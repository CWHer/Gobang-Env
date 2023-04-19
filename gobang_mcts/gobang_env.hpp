#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include <utility>

#include "envpool/gobang_mcts/utils.hpp"

struct GobangBoard
{
    int board_size;
    std::vector<int> board;
    int player;
    std::vector<int> historical_actions;

    GobangBoard(int board_size)
        : board_size(board_size), player(0)
    {
        board.resize(board_size * board_size, -1);
        historical_actions.reserve(board_size * board_size);
    }

    void step(int index)
    {
        assertMsg(index >= 0 && index < board.size(),
                  "Invalid index " + std::to_string(index),
                  __FILE__, __LINE__);
        assertMsg(board[index] == -1,
                  "Invalid index " + std::to_string(index),
                  __FILE__, __LINE__);
        board[index] = player;
        player ^= 1;
        historical_actions.push_back(index);
    }

    std::vector<int> getActions()
    {
        std::vector<int> actions;
        for (int i = 0; i < board_size; i++)
            for (int j = 0; j < board_size; j++)
                if (board[i * board_size + j] == -1)
                    actions.push_back(i * board_size + j);
        return actions;
    }

    std::vector<int> encode(int num_player_planes)
    {
        auto flatten_size = board.size();
        std::vector<int> board_copy(board);
        std::vector<int> encoded_state((num_player_planes * 2 + 1) * flatten_size);
        int action_offset = 1; // historical_actions[-1]
        for (int i = 0; i < num_player_planes; ++i)
        {
            for (int j = 0; j < flatten_size; j++)
                encoded_state[i * flatten_size + j] = board_copy[j] == 0;
            for (int j = 0; j < flatten_size; j++)
                encoded_state[(i + num_player_planes) * flatten_size + j] = board_copy[j] == 1;

            for (int j = 0; j < 2; j++)
                if (action_offset <= historical_actions.size())
                {
                    int action = *(historical_actions.end() - action_offset);
                    board_copy[action] = -1;
                    action_offset++;
                }
        }

        for (int i = 0; i < flatten_size; i++)
            encoded_state[num_player_planes * 2 * flatten_size + i] = player;
        return encoded_state;
    }

    void display()
    {
        std::cout << "Player: " << player << std::endl;
        for (int i = 0; i < board_size; i++)
        {
            for (int j = 0; j < board_size; j++)
            {
                std::cout << (board[i * board_size + j] == -1
                                  ? " -"
                              : board[i * board_size + j] == 0 ? " O"
                                                               : " X");
            }
            std::cout << std::endl;
        }
        std::cout << "Actions: ";
        for (const auto &action : historical_actions)
            std::cout << action << " ";
        std::cout << std::endl;
    }
};

class GobangEnv
{
private:
    GobangBoard board;
    int win_length;
    int winner;

public:
    GobangEnv(int board_size, int win_length)
        : board(board_size), win_length(win_length), winner(-1)
    {
    }

    void reset()
    {
        board = GobangBoard(board.board_size);
        winner = -1;
    }

    void setStat(const GobangBoard &stat)
    {
        this->board = stat;
        this->winner = -1;
    }

    GobangBoard getStat()
    {
        return board;
    }

    void display()
    {
        board.display();
    }

    void step(int index)
    {
        board.step(index);
    }

    std::vector<int> getActions()
    {
        return board.getActions();
    }

    std::pair<bool, int> checkFinished()
    {
        assertMsg(winner == -1,
                  "Game has already finished",
                  __FILE__, __LINE__);
        static const int dx[] = {1, 1, 0, -1};
        static const int dy[] = {0, 1, 1, 1};
        int blank_count = 0;
        for (int i = 0; i < board.board_size; i++)
            for (int j = 0; j < board.board_size; j++)
            {
                if (board.board[i * board.board_size + j] == -1)
                {
                    blank_count++;
                    continue;
                }
                for (int k = 0; k < 4; k++)
                {
                    int x = i, y = j, count = 0;
                    int color = board.board[i * board.board_size + j];
                    while (x >= 0 && x < board.board_size &&
                           y >= 0 && y < board.board_size &&
                           board.board[x * board.board_size + y] == color)
                    {
                        count++;
                        x += dx[k];
                        y += dy[k];
                    }
                    if (count >= win_length)
                    {
                        winner = color;
                        return std::make_pair(true, winner);
                    }
                }
            }
        if (blank_count == 0)
            return std::make_pair(true, -1);
        return std::make_pair(false, -1);
    }

    std::vector<int> getState(int num_player_planes)
    {
        return board.encode(num_player_planes);
    }

    int actionShape() const
    {
        return board.board_size * board.board_size;
    }
};