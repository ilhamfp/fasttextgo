/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "loss.h"
#include "utils.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace fasttext {

constexpr int64_t SIGMOID_TABLE_SIZE = 512;
constexpr int64_t MAX_SIGMOID = 8;
constexpr int64_t LOG_TABLE_SIZE = 512;

bool comparePairs(
    const std::pair<real, int32_t>& l,
    const std::pair<real, int32_t>& r) {
  return l.first > r.first;
}

real std_log(real x) {
  return std::log(x + 1e-5);
}

Loss::Loss(std::shared_ptr<Matrix>& wo) : wo_(wo) {
  t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
  }

  t_log_.reserve(LOG_TABLE_SIZE + 1);
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log_.push_back(std::log(x));
  }
}

real Loss::log(real x) const {
  if (x > 1.0) {
    return 0.0;
  }
  int64_t i = int64_t(x * LOG_TABLE_SIZE);
  return t_log_[i];
}

real Loss::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i =
        int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid_[i];
  }
}

void Loss::predict(
    int32_t k,
    real threshold,
    Predictions& heap,
    Model::State& state) const {
  computeOutput(state);
  findKBest(k, threshold, heap, state.output);
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void Loss::findKBest(
    int32_t k,
    real threshold,
    Predictions& heap,
    const Vector& output) const {
  for (int32_t i = 0; i < output.size(); i++) {
    if (output[i] < threshold) {
      continue;
    }
    if (heap.size() == k && std_log(output[i]) < heap.front().first) {
      continue;
    }
    heap.push_back(std::make_pair(std_log(output[i]), i));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}

void Loss::maxPredict(
      IntentionPredictions& predictions,
      Model::State& state) const {
  return;
}

std::vector<std::pair<std::string, int64_t>> Loss::getLabelCounts() {
  return std::vector<std::pair<std::string, int64_t>> {};
}

void Loss::save(std::ostream&) const {
  return;
}

BinaryLogisticLoss::BinaryLogisticLoss(std::shared_ptr<Matrix>& wo)
    : Loss(wo) {}

real BinaryLogisticLoss::binaryLogistic(
    int32_t target,
    Model::State& state,
    bool labelIsPositive,
    real lr,
    bool backprop) const {
  real score = sigmoid(wo_->dotRow(state.hidden, target));
  if (backprop) {
    real alpha = lr * (real(labelIsPositive) - score);
    state.grad.addRow(*wo_, target, alpha);
    wo_->addVectorToRow(state.hidden, target, alpha);
  }
  if (labelIsPositive) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

void BinaryLogisticLoss::computeOutput(Model::State& state) const {
  Vector& output = state.output;
  output.mul(*wo_, state.hidden);
  int32_t osz = output.size();
  for (int32_t i = 0; i < osz; i++) {
    output[i] = sigmoid(output[i]);
  }
}

OneVsAllLoss::OneVsAllLoss(std::shared_ptr<Matrix>& wo)
    : BinaryLogisticLoss(wo) {}

real OneVsAllLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t /* we take all targets here */,
    Model::State& state,
    real lr,
    bool backprop) {
  real loss = 0.0;
  int32_t osz = state.output.size();
  for (int32_t i = 0; i < osz; i++) {
    bool isMatch = utils::contains(targets, i);
    loss += binaryLogistic(i, state, isMatch, lr, backprop);
  }

  return loss;
}

NegativeSamplingLoss::NegativeSamplingLoss(
    std::shared_ptr<Matrix>& wo,
    int neg,
    const std::vector<int64_t>& targetCounts)
    : BinaryLogisticLoss(wo), neg_(neg), negatives_(), uniform_() {
  real z = 0.0;
  for (size_t i = 0; i < targetCounts.size(); i++) {
    z += pow(targetCounts[i], 0.5);
  }
  for (size_t i = 0; i < targetCounts.size(); i++) {
    real c = pow(targetCounts[i], 0.5);
    for (size_t j = 0; j < c * NegativeSamplingLoss::NEGATIVE_TABLE_SIZE / z;
         j++) {
      negatives_.push_back(i);
    }
  }
  uniform_ = std::uniform_int_distribution<size_t>(0, negatives_.size() - 1);
}

real NegativeSamplingLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  assert(targetIndex >= 0);
  assert(targetIndex < targets.size());
  int32_t target = targets[targetIndex];
  real loss = binaryLogistic(target, state, true, lr, backprop);

  for (int32_t n = 0; n < neg_; n++) {
    auto negativeTarget = getNegative(target, state.rng);
    loss += binaryLogistic(negativeTarget, state, false, lr, backprop);
  }
  return loss;
}

int32_t NegativeSamplingLoss::getNegative(
    int32_t target,
    std::minstd_rand& rng) {
  int32_t negative;
  do {
    negative = negatives_[uniform_(rng)];
  } while (target == negative);
  return negative;
}

HierarchicalSoftmaxLoss::HierarchicalSoftmaxLoss(
    std::shared_ptr<Matrix>& wo,
    const std::vector<int64_t>& targetCounts)
    : BinaryLogisticLoss(wo),
      paths_(),
      codes_(),
      tree_(),
      osz_(targetCounts.size()) {
  buildTree(targetCounts);
}

void HierarchicalSoftmaxLoss::buildTree(const std::vector<int64_t>& counts) {
  tree_.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree_[i].parent = -1;
    tree_[i].left = -1;
    tree_[i].right = -1;
    tree_[i].count = 1e15;
    tree_[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree_[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2] = {0};
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree_[leaf].count < tree_[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    tree_[i].left = mini[0];
    tree_[i].right = mini[1];
    tree_[i].count = tree_[mini[0]].count + tree_[mini[1]].count;
    tree_[mini[0]].parent = i;
    tree_[mini[1]].parent = i;
    tree_[mini[1]].binary = true;
  }
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree_[j].parent != -1) {
      path.push_back(tree_[j].parent - osz_);
      code.push_back(tree_[j].binary);
      j = tree_[j].parent;
    }
    paths_.push_back(path);
    codes_.push_back(code);
  }
}

real HierarchicalSoftmaxLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  real loss = 0.0;
  int32_t target = targets[targetIndex];
  const std::vector<bool>& binaryCode = codes_[target];
  const std::vector<int32_t>& pathToRoot = paths_[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], state, binaryCode[i], lr, backprop);
  }
  return loss;
}

void HierarchicalSoftmaxLoss::predict(
    int32_t k,
    real threshold,
    Predictions& heap,
    Model::State& state) const {
  dfs(k, threshold, 2 * osz_ - 2, 0.0, heap, state.hidden);
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void HierarchicalSoftmaxLoss::dfs(
    int32_t k,
    real threshold,
    int32_t node,
    real score,
    Predictions& heap,
    const Vector& hidden) const {
  if (score < std_log(threshold)) {
    return;
  }
  if (heap.size() == k && score < heap.front().first) {
    return;
  }

  if (tree_[node].left == -1 && tree_[node].right == -1) {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  real f = wo_->dotRow(hidden, node - osz_);
  f = 1. / (1 + std::exp(-f));

  dfs(k, threshold, tree_[node].left, score + std_log(1.0 - f), heap, hidden);
  dfs(k, threshold, tree_[node].right, score + std_log(f), heap, hidden);
}

IntentionHierarchicalSoftmaxLoss::IntentionHierarchicalSoftmaxLoss(
    std::shared_ptr<Matrix>& wo,
    const std::vector<std::pair<std::string, int64_t>>& labelCounts,
    std::string hfiles)
    : BinaryLogisticLoss(wo),
      paths_(),
      codes_(),
      tree_(),
      level_(),
      osz_(labelCounts.size()) {
  buildTree(labelCounts, hfiles);
}

IntentionHierarchicalSoftmaxLoss::IntentionHierarchicalSoftmaxLoss(
    std::shared_ptr<Matrix>& wo,
    std::istream& in)
    : BinaryLogisticLoss(wo),
      paths_(),
      codes_(),
      tree_(),
      level_(),
      osz_() {
  loadTree(in);
}

IntentionHierarchicalSoftmaxLoss::Node IntentionHierarchicalSoftmaxLoss::buildSubTree(std::vector<Node*>& nodes,
        int32_t level) {
  if (nodes.size() == 1) {
    Node node = *nodes[0];
    return node;
  }
  int32_t shift = tree_.size();
  for (int32_t i = 0; i < nodes.size(); i++) {
    tree_.push_back(*nodes[i]);
  }
  for (int32_t i = shift; i < shift + nodes.size(); i++) {
    if (tree_[i].left != -1) {
      tree_[tree_[i].left].parent = i;
    }
    if (tree_[i].right != -1) {
      tree_[tree_[i].right].parent = i;
    }
  }
  int32_t leaf = nodes.size() + shift - 1;
  int32_t node = nodes.size() + shift;
  for (int32_t i = nodes.size() + shift; i < 2 * nodes.size() + shift - 1; i++) {
    int32_t mini[2] = {0};
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= shift && (node >= tree_.size() || tree_[leaf].count < tree_[node].count)) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    tree_.push_back(Node());
    tree_[i].parent = -1;
    tree_[i].slevel = level;
    tree_[i].elevel = level;
    tree_[i].left = mini[0];
    tree_[i].right = mini[1];
    tree_[i].count = tree_[mini[0]].count + tree_[mini[1]].count;
    tree_[mini[0]].parent = i;
    tree_[mini[1]].parent = i;
    tree_[mini[1]].binary = true;
  }
  Node lastNode = tree_.back();
  tree_.pop_back();
  return lastNode;
}

void IntentionHierarchicalSoftmaxLoss::buildTree(const std::vector<std::pair<std::string, int64_t>>& labelCounts,
        const std::string hfiles) {
  std::unordered_map<std::string, Node> prev_mapping;
  for (int32_t i = 0; i < osz_; i++) {
    const std::string& label = labelCounts[i].first;
    Node node = Node();
    node.slevel = -1;
    node.elevel = 0;
    node.parent = -1;
    node.left = -1;
    node.right = -1;
    node.count = labelCounts[i].second;
    node.isLabel = true;
    node.name.push_back(label);
    prev_mapping[label] = node;
  }
  std::stringstream ss(hfiles);
  std::string file;
  int32_t level = 0;
  std::vector<Node*> nodes;
  while(std::getline(ss, file, ',')) {
    std::unordered_map<std::string, Node> curr_mapping;
    std::ifstream ifs(file);
    if (!ifs.is_open()) {
      throw std::invalid_argument(
              file + " cannot be opened for building hierarchy tree!");
    }
    std::string line;
    std::string ignore;
    while (std::getline(ifs, line)) {
      std::stringstream lineStream(line);
      std::string parent, children;
      lineStream >> ignore;
      lineStream >> parent;
      lineStream >> children;
      std::string child;
      std::stringstream childrenStream(children);
      nodes.clear();
      while (std::getline(childrenStream, child, '|')) {
        if (prev_mapping.find(child) == prev_mapping.end()) {
          continue;
        }
        nodes.push_back(&prev_mapping.at(child));
      }
      if (nodes.size() == 0) {
        continue;
      }
      std::sort(nodes.begin(), nodes.end(), [](const Node* n1, const Node* n2) {
          return n1->count > n2->count;
      });
      Node parentNode = buildSubTree(nodes, level);
      parentNode.name.push_back(parent);
      parentNode.elevel = level + 1;
      parentNode.isLabel = true;
      curr_mapping[parent] = parentNode;
    }
    prev_mapping = curr_mapping;
    level++;
  }
  nodes.clear();
  for (auto pair = prev_mapping.begin(); pair != prev_mapping.end(); ++pair) {
    nodes.push_back(&pair->second);
  }
  std::sort(nodes.begin(), nodes.end(), [](const Node* n1, const Node* n2) {
      return n1->count > n2->count;
  });
  Node parentNode = buildSubTree(nodes, level);
  tree_.push_back(parentNode);
  int32_t nonLeaf = -1;
  for (int32_t i = 0; i < tree_.size(); i++) {
    if (tree_[i].left == -1 && tree_[i].right == -1) {
      if (nonLeaf != -1) {
        std::iter_swap(tree_.begin() + i, tree_.begin() + nonLeaf);
        if (tree_[i].left != -1) {
          tree_[tree_[i].left].parent = i;
        }
        if (tree_[i].right != -1) {
          tree_[tree_[i].right].parent = i;
        }
        if (tree_[i].binary) {
          tree_[tree_[i].parent].right = i;
        } else {
          tree_[tree_[i].parent].left = i;
        }

        if (tree_[nonLeaf].left != -1) {
          tree_[tree_[nonLeaf].left].parent = nonLeaf;
        }
        if (tree_[nonLeaf].right != -1) {
          tree_[tree_[nonLeaf].right].parent = nonLeaf;
        }
        if (tree_[nonLeaf].binary) {
          tree_[tree_[nonLeaf].parent].right = nonLeaf;
        } else {
          tree_[tree_[nonLeaf].parent].left = nonLeaf;
        }
        nonLeaf++;
      }
    } else if (nonLeaf == -1) {
      nonLeaf = i;
    }
  }
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree_[j].parent != -1) {
      path.push_back(tree_[j].parent - osz_);
      code.push_back(tree_[j].binary);
      j = tree_[j].parent;
    }
    paths_.push_back(path);
    codes_.push_back(code);
  }
  level_ = level;
}

real IntentionHierarchicalSoftmaxLoss::forward(
        const std::vector<int32_t>& targets,
        int32_t targetIndex,
        Model::State& state,
        real lr,
        bool backprop) {
  real loss = 0.0;
  int32_t target = targets[targetIndex];
  const std::vector<bool>& binaryCode = codes_[target];
  const std::vector<int32_t>& pathToRoot = paths_[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], state, binaryCode[i], lr, backprop);
  }
  return loss;
}

void IntentionHierarchicalSoftmaxLoss::predict(
        int32_t k,
        real threshold,
        Predictions& heap,
        Model::State& state) const {
  dfs(k, threshold, 2 * osz_ - 2, 0.0, heap, state.hidden);
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void IntentionHierarchicalSoftmaxLoss::max_dfs(
        int32_t nodeId,
        real score,
        int32_t level,
        Prediction* bestPred,
        const Vector& hidden,
        bool isRoot) const {
  Node node = tree_[nodeId];
  if (score < bestPred->second) {
    return;
  }

  if (!isRoot && node.isLabel) {
    bestPred->first = nodeId;
    bestPred->second = score;
    return;
  }

  real f = wo_->dotRow(hidden, nodeId - osz_);
  f = 1. / (1 + std::exp(-f));

  max_dfs(node.left, score + std_log(1.0 - f), level, bestPred, hidden, false);
  max_dfs(node.right, score + std_log(f), level, bestPred, hidden, false);
}

void IntentionHierarchicalSoftmaxLoss::maxPredict(
        IntentionPredictions& predictions,
        Model::State& state) const {
  int32_t level = level_;
  int32_t nodeId = 2*osz_-2;
  real score = 0;
  Prediction prediction = std::make_pair(-1, -1e15);
  while (level != -1) {
    prediction.second = -1e15;
    max_dfs(nodeId, std_log(1.0), level, &prediction, state.hidden, true);
    nodeId = prediction.first;
    score = score + prediction.second;
    for (int32_t i = tree_[nodeId].name.size()-1; i >= 0; i--) {
      predictions.push_back(std::make_pair(score, tree_[nodeId].name[i]));
    }
    level = std::min(level-1, tree_[nodeId].slevel);
  }
}

void IntentionHierarchicalSoftmaxLoss::dfs(
        int32_t k,
        real threshold,
        int32_t node,
        real score,
        Predictions& heap,
        const Vector& hidden) const {
  if (score < std_log(threshold)) {
    return;
  }
  if (heap.size() == k && score < heap.front().first) {
    return;
  }

  if (tree_[node].left == -1 && tree_[node].right == -1) {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  real f = wo_->dotRow(hidden, node - osz_);
  f = 1. / (1 + std::exp(-f));

  dfs(k, threshold, tree_[node].left, score + std_log(1.0 - f), heap, hidden);
  dfs(k, threshold, tree_[node].right, score + std_log(f), heap, hidden);
}

std::vector<std::pair<std::string, int64_t>> IntentionHierarchicalSoftmaxLoss::getLabelCounts() {
  std::vector<std::pair<std::string, int64_t>> labelCounts;
  for (int32_t i = 0; i < osz_; i++) {
    std::pair<std::string, int64_t> pair = std::make_pair(tree_[i].name[0], tree_[i].count);
    labelCounts.push_back(pair);
  }
  return labelCounts;
}

void IntentionHierarchicalSoftmaxLoss::save(std::ostream& out) const {
  out.write((char*)&level_, sizeof(int32_t));
  out.write((char*)&osz_, sizeof(int32_t));
  for (auto node: tree_) {
    out.write((char *) &node.slevel, sizeof(int32_t));
    out.write((char *) &node.elevel, sizeof(int32_t));
    out.write((char *) &node.parent, sizeof(int32_t));
    out.write((char *) &node.left, sizeof(int32_t));
    out.write((char *) &node.right, sizeof(int32_t));
    out.write((char *) &node.binary, sizeof(bool));
    out.write((char *) &node.isLabel, sizeof(bool));
    for (auto label: node.name) {
      out.write(label.data(), label.size() * sizeof(char));
      out.put(124);
    }
    out.put(0);
  }
}

void IntentionHierarchicalSoftmaxLoss::loadTree(std::istream& in) {
  in.read((char*)&level_, sizeof(int32_t));
  in.read((char*)&osz_, sizeof(int32_t));
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    Node n = Node();
    in.read((char*)&n.slevel, sizeof(int32_t));
    in.read((char*)&n.elevel, sizeof(int32_t));
    in.read((char*)&n.parent, sizeof(int32_t));
    in.read((char*)&n.left, sizeof(int32_t));
    in.read((char*)&n.right, sizeof(int32_t));
    in.read((char*)&n.binary, sizeof(bool));
    in.read((char*)&n.isLabel, sizeof(bool));
    char c;
    std::string label;
    while ((c = in.get()) != 0) {
      if (c == 124) {
        n.name.push_back(label);
        label.clear();
      } else {
        label.push_back(c);
      }
    }
    tree_.push_back(n);
  }
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree_[j].parent != -1) {
      path.push_back(tree_[j].parent - osz_);
      code.push_back(tree_[j].binary);
      j = tree_[j].parent;
    }
    paths_.push_back(path);
    codes_.push_back(code);
  }
}

SoftmaxLoss::SoftmaxLoss(std::shared_ptr<Matrix>& wo) : Loss(wo) {}

void SoftmaxLoss::computeOutput(Model::State& state) const {
  Vector& output = state.output;
  output.mul(*wo_, state.hidden);
  real max = output[0], z = 0.0;
  int32_t osz = output.size();
  for (int32_t i = 0; i < osz; i++) {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz; i++) {
    output[i] /= z;
  }
}

real SoftmaxLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  computeOutput(state);

  assert(targetIndex >= 0);
  assert(targetIndex < targets.size());
  int32_t target = targets[targetIndex];

  if (backprop) {
    int32_t osz = wo_->size(0);
    for (int32_t i = 0; i < osz; i++) {
      real label = (i == target) ? 1.0 : 0.0;
      real alpha = lr * (label - state.output[i]);
      state.grad.addRow(*wo_, i, alpha);
      wo_->addVectorToRow(state.hidden, i, alpha);
    }
  }
  return -log(state.output[target]);
};

} // namespace fasttext
