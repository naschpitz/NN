#include "test_helpers.hpp"

#include "CNN_Normalization.hpp"

#include <cmath>

//===================================================================================================================//

//
// BatchNorm-specific tests.
//
// The existing test_cpu_integration_batchnorm.cpp uses INSTANCENORM, so the
// BatchNorm code path (batch-wide statistics, running-stat inference) is
// effectively untested. These tests fill that gap.
//

//
// BatchNorm inference: MUST use running stats (not recompute from batch).
// This is the critical difference from InstanceNorm.
//
static void testBatchNormInferenceUsesRunningStats()
{
  TestScope _t("testBatchNormInferenceUsesRunningStats");

  // 2 samples, 1 channel, 2x2 spatial
  CNN::Shape3D shape{1, 2, 2};

  CNN::Tensor3D<double> sample0(shape);
  sample0.data = {1.0, 2.0, 3.0, 4.0};

  CNN::Tensor3D<double> sample1(shape);
  sample1.data = {5.0, 6.0, 7.0, 8.0};

  CNN::NormParameters<double> params;
  params.numChannels = 1;
  params.gamma = {2.0};
  params.beta = {1.0};

  // CRITICAL: Set running stats that DO NOT match the batch statistics.
  // Batch stats across both samples (8 elements): mean=4.5, var=5.25
  // If BatchNorm recomputes at inference (wrong), output would use 4.5/5.25.
  // Running stats: mean=2.0, var=4.0 → must be used instead.
  params.runningMean = {2.0};
  params.runningVar = {4.0};

  CNN::NormLayerConfig config;
  config.epsilon = 0.0;

  std::vector<CNN::Tensor3D<double>*> batch = {&sample0, &sample1};
  CNN::Normalization<double>::propagate(batch, shape, params, config, CNN::LayerType::BATCHNORM, false);

  // Expected: gamma * (x - runningMean) / sqrt(runningVar + eps) + beta
  // = 2.0 * (x - 2.0) / sqrt(4.0) + 1.0
  // = 2.0 * (x - 2.0) / 2.0 + 1.0
  // = (x - 2.0) + 1.0
  // = x - 1.0
  double invStd = 1.0 / std::sqrt(4.0);

  CHECK_NEAR(sample0.data[0], 2.0 * (1.0 - 2.0) * invStd + 1.0, 1e-9, "bn infer s0[0]");
  CHECK_NEAR(sample0.data[3], 2.0 * (4.0 - 2.0) * invStd + 1.0, 1e-9, "bn infer s0[3]");
  CHECK_NEAR(sample1.data[0], 2.0 * (5.0 - 2.0) * invStd + 1.0, 1e-9, "bn infer s1[0]");
  CHECK_NEAR(sample1.data[3], 2.0 * (8.0 - 2.0) * invStd + 1.0, 1e-9, "bn infer s1[3]");

  // Verify output is NOT computed from batch statistics (which would give mean=4.5)
  // If batch stats were used: sample0[0] = 2.0*(1.0-4.5)/sqrt(5.25) + 1.0 ≈ -2.05
  // With running stats:        sample0[0] = 2.0*(1.0-2.0)/sqrt(4.0) + 1.0 = 0.0
  CHECK_NEAR(sample0.data[0], 0.0, 1e-9, "bn infer uses running stats (not batch stats)");
}

//===================================================================================================================//

//
// BatchNorm training: computes batch-wide statistics and updates running stats.
//
static void testBatchNormTrainComputesBatchStats()
{
  TestScope _t("testBatchNormTrainComputesBatchStats");

  // 2 samples, 1 channel, 2x2 spatial
  CNN::Shape3D shape{1, 2, 2};

  CNN::Tensor3D<double> sample0(shape);
  sample0.data = {1.0, 2.0, 3.0, 4.0};

  CNN::Tensor3D<double> sample1(shape);
  sample1.data = {5.0, 6.0, 7.0, 8.0};

  CNN::NormParameters<double> params;
  params.numChannels = 1;
  params.gamma = {1.0};
  params.beta = {0.0};
  params.runningMean = {0.0};
  params.runningVar = {1.0};

  CNN::NormLayerConfig config;
  config.epsilon = 0.0;
  config.momentum = 0.1;

  std::vector<double> statsMean, statsVar;
  std::vector<CNN::Tensor3D<double>> xNorm;

  std::vector<CNN::Tensor3D<double>*> batch = {&sample0, &sample1};
  CNN::Normalization<double>::propagate(batch, shape, params, config, CNN::LayerType::BATCHNORM, true, &xNorm,
                                        &statsMean, &statsVar);

  // Batch stats across all 8 elements: mean = 4.5, var = 5.25
  CHECK_NEAR(statsMean[0], 4.5, 1e-9, "bn train batch mean");
  CHECK_NEAR(statsVar[0], 5.25, 1e-9, "bn train batch var");

  // Output: (x - 4.5) / sqrt(5.25)
  double invStd = 1.0 / std::sqrt(5.25);
  CHECK_NEAR(sample0.data[0], (1.0 - 4.5) * invStd, 1e-9, "bn train out s0[0]");
  CHECK_NEAR(sample1.data[3], (8.0 - 4.5) * invStd, 1e-9, "bn train out s1[3]");

  // Running stats updated: runningMean = momentum * batchMean + (1-momentum) * oldMean
  // = 0.1 * 4.5 + 0.9 * 0.0 = 0.45
  CHECK_NEAR(params.runningMean[0], 0.45, 1e-7, "bn train runningMean updated");
  // runningVar = 0.1 * 5.25 + 0.9 * 1.0 = 1.425
  CHECK_NEAR(params.runningVar[0], 1.425, 1e-7, "bn train runningVar updated");
}

//===================================================================================================================//

//
// BatchNorm numerical gradient check with NON-UNIFORM dOutput.
// The existing test (test_cpu_layers2.cpp:testBatchNormBackpropagate) uses
// uniform dOutput=1.0 which produces dInput=0 by construction — a degenerate
// case that can't catch sign errors or scaling bugs.
//
static void testBatchNormBackpropGradient()
{
  TestScope _t("testBatchNormBackpropGradient");

  // 2 samples, 1 channel, 1x3 spatial (small but non-degenerate)
  CNN::Shape3D shape{1, 1, 3};

  CNN::Tensor3D<double> sample0(shape);
  sample0.data = {1.0, 3.0, 5.0};

  CNN::Tensor3D<double> sample1(shape);
  sample1.data = {2.0, 4.0, 6.0};

  CNN::NormParameters<double> params;
  params.numChannels = 1;
  params.gamma = {1.5};
  params.beta = {0.5};
  params.runningMean = {0.0};
  params.runningVar = {1.0};

  CNN::NormLayerConfig config;
  config.epsilon = 1e-7;
  config.momentum = 0.1;

  // Forward pass (training mode to compute batch stats)
  std::vector<double> statsMean, statsVar;
  std::vector<CNN::Tensor3D<double>> xNorm;

  CNN::Tensor3D<double> input0(shape), input1(shape);
  input0.data = sample0.data;
  input1.data = sample1.data;

  std::vector<CNN::Tensor3D<double>*> fwdBatch = {&input0, &input1};
  CNN::Normalization<double>::propagate(fwdBatch, shape, params, config, CNN::LayerType::BATCHNORM, true, &xNorm,
                                        &statsMean, &statsVar);

  // Non-uniform dOutput — this is what makes the test non-degenerate
  CNN::Tensor3D<double> dOut0(shape), dOut1(shape);
  dOut0.data = {0.1, -0.2, 0.3};
  dOut1.data = {-0.1, 0.2, -0.3};

  std::vector<double> dGamma, dBeta;
  std::vector<CNN::Tensor3D<double>*> dBatch = {&dOut0, &dOut1};
  CNN::Normalization<double>::backpropagate(dBatch, shape, params, config, CNN::LayerType::BATCHNORM, statsMean,
                                            statsVar, xNorm, dGamma, dBeta);

  // Numerical gradient check for each input element
  double eps = 1e-5;
  double lossWeight0[] = {0.1, -0.2, 0.3};
  double lossWeight1[] = {-0.1, 0.2, -0.3};

  for (int s = 0; s < 2; ++s) {
    CNN::Tensor3D<double>& baseInput = (s == 0) ? sample0 : sample1;
    const double* lossWeights = (s == 0) ? lossWeight0 : lossWeight1;
    CNN::Tensor3D<double>& dOut = (s == 0) ? dOut0 : dOut1;

    for (ulong i = 0; i < 3; ++i) {
      // Perturb +eps
      CNN::Tensor3D<double> p0(shape), p1(shape);
      p0.data = sample0.data;
      p1.data = sample1.data;

      if (s == 0)
        p0.data[i] += eps;
      else
        p1.data[i] += eps;

      std::vector<double> smP, svP;
      std::vector<CNN::Tensor3D<double>> xnP;
      std::vector<CNN::Tensor3D<double>*> bp = {&p0, &p1};
      CNN::Normalization<double>::propagate(bp, shape, params, config, CNN::LayerType::BATCHNORM, true, &xnP, &smP,
                                            &svP);

      // Perturb -eps
      CNN::Tensor3D<double> m0(shape), m1(shape);
      m0.data = sample0.data;
      m1.data = sample1.data;

      if (s == 0)
        m0.data[i] -= eps;
      else
        m1.data[i] -= eps;

      std::vector<double> smM, svM;
      std::vector<CNN::Tensor3D<double>> xnM;
      std::vector<CNN::Tensor3D<double>*> bm = {&m0, &m1};
      CNN::Normalization<double>::propagate(bm, shape, params, config, CNN::LayerType::BATCHNORM, true, &xnM, &smM,
                                            &svM);

      // Loss = sum(dOut * output) for all samples
      double numGrad = 0.0;

      for (ulong j = 0; j < 3; ++j) {
        numGrad += lossWeight0[j] * (p0.data[j] - m0.data[j]) / (2.0 * eps);
        numGrad += lossWeight1[j] * (p1.data[j] - m1.data[j]) / (2.0 * eps);
      }

      CHECK_NEAR(dOut.data[i], numGrad, 1e-4,
                 "bn numerical gradient s" + std::to_string(s) + "[" + std::to_string(i) + "]");
    }
  }
}

//===================================================================================================================//

void runBatchNormTests()
{
  testBatchNormInferenceUsesRunningStats();
  testBatchNormTrainComputesBatchStats();
  testBatchNormBackpropGradient();
}
