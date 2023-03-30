#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include <cassert>
#include <limits>
#include <iostream>

struct PUCT
{
    float prior_prob;
    float q_value;
    int visit_count;
    float c_puct;

    PUCT(float prior_prob, float c_puct)
        : prior_prob(prior_prob), q_value(0), visit_count(0), c_puct(c_puct)
    {
    }

    void update(float v)
    {
        visit_count++;
        q_value += (v - q_value) / visit_count;
    }

    float value(int parent_visit_count) const
    {
        return q_value + c_puct * prior_prob * sqrt(parent_visit_count) / (1 + visit_count);
    }
};

struct TreeNode : public std::enable_shared_from_this<TreeNode>
{
    std::weak_ptr<TreeNode> parent;
    std::vector<std::shared_ptr<TreeNode>> children;

    int action;

    PUCT puct;

    TreeNode(std::weak_ptr<TreeNode> parent,
             int action, float prior_prob, float c_puct)
        : parent(parent), action(action), puct(prior_prob, c_puct)
    {
    }

    bool isRoot() const
    {
        return parent.expired();
    }

    bool isLeaf() const
    {
        return children.empty();
    }

    int getVisitCount() const
    {
        return puct.visit_count;
    }

    void update(float v)
    {
        puct.update(v);
    }

    float value(int parent_visit_count) const
    {
        return puct.value(parent_visit_count);
    }

    std::shared_ptr<TreeNode> select()
    {
        assert(!this->isLeaf());
        std::shared_ptr<TreeNode> selected_child = nullptr;
        auto visit_count = this->getVisitCount();
        float best_value = std::numeric_limits<float>::lowest();
        for (const auto &child : children)
        {
            float value = child->value(visit_count);
            if (value > best_value)
            {
                best_value = value;
                selected_child = child;
            }
        }
        return selected_child;
    }

    void expand(std::vector<std::pair<int, float>> &actions_probs, float c_puct)
    {
        for (const auto &action_prob : actions_probs)
        {
            children.push_back(std::make_shared<TreeNode>(
                shared_from_this(), action_prob.first, action_prob.second, c_puct));
        }
    }

    std::shared_ptr<TreeNode> step(int action)
    {
        std::shared_ptr<TreeNode> next_root = nullptr;
        for (const auto &child : children)
            if (child->action == action)
            {
                next_root = child;
                break;
            }
        if (next_root != nullptr)
        {
            next_root->parent = std::weak_ptr<TreeNode>();
            return next_root;
        }

        return nullptr;
    }

    void display()
    {
        std::cout << "Total visit count: "
                  << this->getVisitCount() << std::endl;
        for (const auto &child : children)
        {
            std::cout << "  Action: " << child->action << " ";
            std::cout << "Visit count: " << child->getVisitCount() << " ";
            std::cout << "Q value: " << child->puct.q_value << " ";
            std::cout << "Value: " << child->value(this->getVisitCount()) << std::endl;
        }
    }
};

template <typename Env, typename EnvStat>
class MCTS
{
private:
    const float c_puct;
    const int num_search;

    int current_search;
    std::shared_ptr<TreeNode> root;
    EnvStat stat;

    // resume from selected node
    std::shared_ptr<TreeNode> selected_node;
    std::shared_ptr<Env> env;
    int winner;

public:
    MCTS(float c_puct, int num_search, std::shared_ptr<Env> env)
        : c_puct(c_puct), num_search(num_search), current_search(0),
          root(std::make_shared<TreeNode>(std::weak_ptr<TreeNode>(), -1, 0, c_puct)),
          stat(env->getStat()), selected_node(nullptr), env(env)
    {
    }

    bool selectNode()
    {
        // MCTS: select
        selected_node = root;
        env->setStat(stat);
        while (!selected_node->isLeaf())
        {
            selected_node = selected_node->select();
            env->step(selected_node->action);
        }

        auto result = env->checkFinished();
        winner = result.second;
        return result.first;
    }

    void expandNode(std::vector<float> prior_probs)
    {
        // MCTS: expand
        auto valid_actions = env->getActions();
        std::vector<std::pair<int, float>> actions_probs;
        for (const auto &action : valid_actions)
            actions_probs.push_back(std::make_pair(action, prior_probs[action]));
        selected_node->expand(actions_probs, c_puct);
    }

    void backPropagate(float value)
    {
        // MCTS: back propagate
        while (true)
        {
            selected_node->update(value);
            if (selected_node->isRoot())
                break;
            value = -value;
            selected_node = selected_node->parent.lock();
        }
    }

    bool search(std::vector<float> prior_probs, float value)
    {
        if (!prior_probs.empty())
        {
            expandNode(prior_probs);
            backPropagate(value);
            current_search++;
        }

        while (current_search < num_search)
        {
            auto terminal = selectNode();
            if (!terminal)
                return false;
            auto value = winner == -1 ? 0.0f : 1.0f;
            backPropagate(value);
            current_search++;
        }
        return true;
    }

    std::vector<int> getState()
    {
        return env->getState();
    }

    std::vector<std::pair<int, int>> getResult(bool ignore_unfinished = false)
    {
        assert(ignore_unfinished || root->getVisitCount() >= num_search);
        std::vector<std::pair<int, int>> actions_visits;
        for (const auto &child : root->children)
            actions_visits.push_back(
                std::make_pair(child->action, child->getVisitCount()));
        return actions_visits;
    }

    void step(int action, bool reset_root = false)
    {
        env->setStat(stat);
        root = root->step(reset_root ? -1 : action);
        if (root == nullptr)
            root = std::make_shared<TreeNode>(
                std::weak_ptr<TreeNode>(), -1, 0, c_puct);
        env->step(action);
        stat = env->getStat();
        current_search = 0;
    }

    void display()
    {
        env->setStat(stat);
        env->display();
        root->display();
    }
};