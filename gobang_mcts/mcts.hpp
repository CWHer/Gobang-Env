#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include <cassert>
#include <limits>
#include <iostream>

#include "envpool/gobang_mcts/utils.hpp"

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

struct TreeNode;

class TreeNodePool : public std::enable_shared_from_this<TreeNodePool>
{
    // NOTE: HACK: why do we need TreeNodePool?
    // Try not to allocate and free TreeNode objects during search.
    // This would largely reduce the execution time of MCTS::step().
private:
    std::vector<TreeNode> nodes;
    int allocated_count;

public:
    class Reference
    {
        // NOTE: this is a reference to a TreeNode in TreeNodePool
        friend class TreeNodePool;

    private:
        std::weak_ptr<TreeNodePool> pool;
        int index;

    public:
        Reference() : index(-1) {}
        Reference(std::weak_ptr<TreeNodePool> pool, int index) : pool(pool), index(index) {}
        Reference(const Reference &ref) : pool(ref.pool), index(ref.index) {}

        void clear() { index = -1; }
        bool empty() const
        {
            return pool.expired() || index == -1;
        }

        TreeNode &operator*()
        {
            assertMsg(!empty(), "Cannot dereference an empty reference");
            return pool.lock()->nodes[index];
        }

        const TreeNode &operator*() const
        {
            assertMsg(!empty(), "Cannot dereference an empty reference");
            return pool.lock()->nodes[index];
        }
    };

    TreeNodePool() = default;

    template <typename... Args>
    void reserve(int size, Args &&...args)
    {
        assertMsg(nodes.empty() && size > 0,
                  "Cannot reserve space for TreeNodePool twice");
        nodes.reserve(size);
        for (int i = 0; i < size; ++i)
            nodes.emplace_back(shared_from_this(), i, std::forward<Args>(args)...);
    }

    Reference allocate()
    {
        assertMsg(allocated_count < nodes.size(),
                  "No more space to allocate");
        return Reference(shared_from_this(), allocated_count++);
    }

    void clear()
    {
        allocated_count = 0;
    }
};

class RefVectorPool : public std::enable_shared_from_this<RefVectorPool>
{
    // NOTE: HACK: why do we need RefVectorPool?
    // TreeNodePool::clear method can't free the memory of std::vector (owned by each TreeNode).
    // which would cause excessive memory usage. So we use RefVectorPool to manage the memory.
private:
    std::vector<std::vector<TreeNodePool::Reference>> ref_vectors;
    int allocated_count;

public:
    class Reference
    {
        // NOTE: this is a reference to a std::vector<TreeNodePool::Reference> in RefVectorPool
        friend class RefVectorPool;

    private:
        std::weak_ptr<RefVectorPool> pool;
        int index;

    public:
        Reference() : index(-1) {}
        Reference(std::weak_ptr<RefVectorPool> pool, int index) : pool(pool), index(index) {}
        Reference(const Reference &ref) : pool(ref.pool), index(ref.index) {}

        void clear() { index = -1; }
        bool empty() const
        {
            return pool.expired() || index == -1;
        }

        std::vector<TreeNodePool::Reference> &operator*()
        {
            assertMsg(!empty(), "Cannot dereference an empty reference");
            return pool.lock()->ref_vectors[index];
        }

        const std::vector<TreeNodePool::Reference> &operator*() const
        {
            assertMsg(!empty(), "Cannot dereference an empty reference");
            return pool.lock()->ref_vectors[index];
        }
    };

    RefVectorPool() = default;

    void reserve(int size)
    {
        assertMsg(ref_vectors.empty() && size > 0,
                  "Cannot reserve space for RefArrayPool twice");
        ref_vectors.resize(size);
    }

    Reference allocate()
    {
        assertMsg(allocated_count < ref_vectors.size(),
                  "No more space to allocate");
        return Reference(shared_from_this(), allocated_count++);
    }

    void clear()
    {
        allocated_count = 0;
    }
};

struct TreeNode
{
    std::weak_ptr<TreeNodePool> tree_node_pool;
    int index_of_this;

    std::weak_ptr<RefVectorPool> ref_array_pool;

    TreeNodePool::Reference parent_ref;
    RefVectorPool::Reference children_refs;

    int action;
    PUCT puct;

    TreeNode(std::weak_ptr<TreeNodePool> tree_node_pool, int index_of_this,
             std::weak_ptr<RefVectorPool> ref_array_pool)
        : tree_node_pool(tree_node_pool), index_of_this(index_of_this),
          ref_array_pool(ref_array_pool), action(-1), puct(0, 0) {}

    void setStat(const TreeNodePool::Reference &parent_ref,
                 int action, float prior_prob, float c_puct)
    {
        this->parent_ref = parent_ref;
        this->children_refs.clear();
        this->action = action;
        this->puct = PUCT(prior_prob, c_puct);
    }

    bool isRoot() const
    {
        return parent_ref.empty();
    }

    bool isLeaf() const
    {
        return children_refs.empty();
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

    TreeNodePool::Reference select()
    {
        assertMsg(!this->isLeaf(), "Leaf node has no child to select");
        TreeNodePool::Reference selected_child;
        auto visit_count = this->getVisitCount();
        float best_value = std::numeric_limits<float>::lowest();
        for (const auto &child_ref : *children_refs)
        {
            float value = (*child_ref).value(visit_count);
            if (value > best_value)
            {
                best_value = value;
                selected_child = child_ref;
            }
        }
        return selected_child;
    }

    void expand(const std::vector<std::pair<int, float>> &actions_probs, float c_puct)
    {
        assertMsg(!tree_node_pool.expired() && !ref_array_pool.expired(),
                  "Reference to TreeNodePool or RefVectorPool is expired");

        auto this_ref = TreeNodePool::Reference(tree_node_pool, index_of_this);
        children_refs = ref_array_pool.lock()->allocate();
        (*children_refs).resize(actions_probs.size());
        for (int i = 0; i < actions_probs.size(); ++i)
        {
            auto &child_ref = (*children_refs)[i];
            child_ref = tree_node_pool.lock()->allocate();
            (*child_ref).setStat(this_ref, actions_probs[i].first, actions_probs[i].second, c_puct);
        }
    }

    TreeNodePool::Reference step(int action)
    {
        TreeNodePool::Reference next_root;
        for (const auto &child_ref : *children_refs)
            if ((*child_ref).action == action)
            {
                next_root = child_ref;
                break;
            }
        if (!next_root.empty())
            (*next_root).parent_ref = TreeNodePool::Reference();

        return next_root;
    }

    void display()
    {
        std::cout << "Total visit count: "
                  << this->getVisitCount() << std::endl;
        for (const auto &child_ref : *children_refs)
        {
            std::cout << "  Action: " << (*child_ref).action << " ";
            std::cout << "Visit count: " << (*child_ref).getVisitCount() << " ";
            std::cout << "Q value: " << (*child_ref).puct.q_value << " ";
            std::cout << "Value: " << (*child_ref).value(this->getVisitCount()) << std::endl;
        }
    }
};

template <typename Env, typename EnvStat>
class MCTS
{
private:
    std::shared_ptr<TreeNodePool> tree_node_pool;
    std::shared_ptr<RefVectorPool> ref_array_pool;

    const float c_puct;
    const int num_search;

    int current_search;
    TreeNodePool::Reference root_ref;
    EnvStat stat;

    // resume from selected node
    TreeNodePool::Reference selected_node;
    std::shared_ptr<Env> env;
    int winner;

public:
    MCTS(float c_puct, int num_search, std::shared_ptr<Env> env)
        : tree_node_pool(std::make_shared<TreeNodePool>()),
          ref_array_pool(std::make_shared<RefVectorPool>()),
          c_puct(c_puct), num_search(num_search), current_search(0),
          stat(env->getStat()), env(env)
    {
        assertMsg(num_search > 0, "num_search must be positive");

        ref_array_pool->reserve(num_search);
        tree_node_pool->reserve(num_search * env->actionShape(), ref_array_pool);

        root_ref = tree_node_pool->allocate();
        (*root_ref).setStat(TreeNodePool::Reference(), -1, 0, c_puct);
    }

    bool selectNode()
    {
        // MCTS: select
        selected_node = root_ref;
        env->setStat(stat);
        while (!(*selected_node).isLeaf())
        {
            selected_node = (*selected_node).select();
            env->step((*selected_node).action);
        }

        auto result = env->checkFinished();
        winner = result.second;
        return result.first;
    }

    void expandNode(const std::vector<float> &prior_probs)
    {
        // MCTS: expand
        auto valid_actions = env->getActions();
        std::vector<std::pair<int, float>> actions_probs;
        actions_probs.reserve(valid_actions.size());
        for (const auto &action : valid_actions)
            actions_probs.push_back(std::make_pair(action, prior_probs[action]));
        (*selected_node).expand(actions_probs, c_puct);
    }

    void backPropagate(float value)
    {
        // MCTS: back propagate
        while (true)
        {
            (*selected_node).update(value);
            if ((*selected_node).isRoot())
                break;
            value = -value;
            selected_node = (*selected_node).parent_ref;
        }
    }

    bool search(const std::vector<float> &prior_probs, float value)
    {
        // NOTE: selectNode before expand
        //  would ignore prior_probs & value if selected_node is nullptr
        if (!selected_node.empty())
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

    std::vector<int> getState(int num_player_planes)
    {
        return env->getState(num_player_planes);
    }

    std::vector<std::pair<int, int>> getResult(bool ignore_unfinished = false)
    {
        assertMsg(ignore_unfinished || (*root_ref).getVisitCount() >= num_search,
                  "MCTS search not finished");
        std::vector<std::pair<int, int>> actions_visits;
        for (const auto &child_ref : (*(*root_ref).children_refs))
            actions_visits.push_back(
                std::make_pair((*child_ref).action, (*child_ref).getVisitCount()));
        return actions_visits;
    }

    void step(int action, bool reset_root = false)
    {
        assertMsg(reset_root, "reset_root = false not implemented");
        env->setStat(stat);
        env->step(action);
        stat = env->getStat();

        tree_node_pool->clear();
        ref_array_pool->clear();
        current_search = 0;
        selected_node.clear();
        root_ref = tree_node_pool->allocate();
        (*root_ref).setStat(TreeNodePool::Reference(), -1, 0, c_puct);

        // TODO: support reset_root = false
        // root_ref = (*root_ref).step(action);
        // if (root_ref.empty() || reset_root)
        //     (*root_ref).setStat(TreeNodePool::Reference(), -1, 0, c_puct);
    }

    void display()
    {
        env->setStat(stat);
        env->display();
        (*root_ref).display();
    }
};