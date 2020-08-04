/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <random>
#include <vector>

#include "matrix.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class Loss {
 private:
  void findKBest(
      int32_t k,
      real threshold,
      Predictions& heap,
      const Vector& output) const;

 protected:
  std::vector<real> t_sigmoid_;
  std::vector<real> t_log_;
  std::shared_ptr<Matrix>& wo_;

  real log(real x) const;
  real sigmoid(real x) const;

 public:
  explicit Loss(std::shared_ptr<Matrix>& wo);
  virtual ~Loss() = default;

  virtual real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) = 0;
  virtual void computeOutput(Model::State& state) const = 0;

  virtual void predict(
      int32_t /*k*/,
      real /*threshold*/,
      Predictions& /*heap*/,
      Model::State& /*state*/) const;
  virtual std::vector<std::pair<std::string, int64_t>> getLabelCounts();
  virtual void save(std::ostream&) const;
  virtual void maxPredict(
      IntentionPredictions& predictions,
      Model::State& state) const;
};

class BinaryLogisticLoss : public Loss {
 protected:
  real binaryLogistic(
      int32_t target,
      Model::State& state,
      bool labelIsPositive,
      real lr,
      bool backprop) const;

  real binaryLogisticBalanced(
      int32_t target,
      Model::State& state,
      bool labelIsPositive,
      real lr,
      bool backprop,
      real weight) const;

 public:
  explicit BinaryLogisticLoss(std::shared_ptr<Matrix>& wo);
  virtual ~BinaryLogisticLoss() noexcept override = default;
  void computeOutput(Model::State& state) const override;
};

class OneVsAllLoss : public BinaryLogisticLoss {
 public:
  explicit OneVsAllLoss(std::shared_ptr<Matrix>& wo);
  ~OneVsAllLoss() noexcept override = default;
  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
};

class NegativeSamplingLoss : public BinaryLogisticLoss {
 protected:
  static const int32_t NEGATIVE_TABLE_SIZE = 10000000;

  int neg_;
  std::vector<int32_t> negatives_;
  std::uniform_int_distribution<size_t> uniform_;
  int32_t getNegative(int32_t target, std::minstd_rand& rng);

 public:
  explicit NegativeSamplingLoss(
      std::shared_ptr<Matrix>& wo,
      int neg,
      const std::vector<int64_t>& targetCounts);
  ~NegativeSamplingLoss() noexcept override = default;

  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
};

class HierarchicalSoftmaxLoss : public BinaryLogisticLoss {
 protected:
  struct Node {
    int32_t parent;
    int32_t left;
    int32_t right;
    int64_t count;
    bool binary;
  };

  std::vector<std::vector<int32_t>> paths_;
  std::vector<std::vector<bool>> codes_;
  std::vector<Node> tree_;
  int32_t osz_;
  void buildTree(const std::vector<int64_t>& counts);
  void dfs(
      int32_t k,
      real threshold,
      int32_t node,
      real score,
      Predictions& heap,
      const Vector& hidden) const;

 public:
  explicit HierarchicalSoftmaxLoss(
      std::shared_ptr<Matrix>& wo,
      const std::vector<int64_t>& counts);
  ~HierarchicalSoftmaxLoss() noexcept override = default;
  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
  void predict(
      int32_t k,
      real threshold,
      Predictions& heap,
      Model::State& state) const override;
};

class IntentionHierarchicalSoftmaxLoss : public BinaryLogisticLoss {
 protected:
  struct Node {
      int32_t slevel;
      int32_t elevel;
      int32_t parent;
      int32_t left;
      int32_t right;
      int64_t count;
      bool binary;
      bool isLabel;
      std::vector<std::string> name;
  };

  typedef std::pair<int32_t, real> Prediction;

  std::vector<std::vector<int32_t>> paths_;
  std::vector<real> weights_;
  std::vector<std::vector<bool>> codes_;
  std::vector<Node> tree_;
  int32_t osz_;
  int32_t level_;
  Node buildSubTree(std::vector<Node*>& nodes, int32_t level);
  void buildTree(const std::vector<std::pair<std::string, int64_t>>& labelCounts, const std::string hfiles);
  void dfs(
      int32_t k,
      real threshold,
      int32_t node,
      real score,
      Predictions& heap,
      const Vector& hidden) const;
  void max_dfs(
      int32_t nodeId,
      real score,
      int32_t level,
      Prediction* bestPred,
      const Vector& hidden,
      bool isRoot) const;
  void loadTree(std::istream&);

public:
  explicit IntentionHierarchicalSoftmaxLoss(
      std::shared_ptr<Matrix>& wo,
      const std::vector<std::pair<std::string, int64_t>>& labelCounts,
      std::string hfiles);
  explicit IntentionHierarchicalSoftmaxLoss(
      std::shared_ptr<Matrix>& wo,
      std::istream& in);
  ~IntentionHierarchicalSoftmaxLoss() noexcept override = default;
  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
  void predict(
      int32_t k,
      real threshold,
      Predictions& heap,
      Model::State& state) const override;
  std::vector<std::pair<std::string, int64_t>> getLabelCounts() override;
  void maxPredict(
      IntentionPredictions& predictions,
      Model::State& state) const override;
  void save(std::ostream&) const override;
};

class SoftmaxLoss : public Loss {
 public:
  explicit SoftmaxLoss(std::shared_ptr<Matrix>& wo);
  ~SoftmaxLoss() noexcept override = default;
  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
  void computeOutput(Model::State& state) const override;
};

} // namespace fasttext
