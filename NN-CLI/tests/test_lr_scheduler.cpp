#include "test_helpers.hpp"

#include "Common/Common_LearningRateScheduler.hpp"

#include "NN-CLI_ANNLoader.hpp"

#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>
#include <CNN_Sample.hpp>

#include <cmath>
#include <vector>

//===================================================================================================================//
//
// Helper: mirror the Runner's caller pattern — step the scheduler, then publish the returned
// LR back into state.currentLearningRate (the function computes-and-returns; the caller owns currentLearningRate).
//
//===================================================================================================================//

static float stepAndUpdate(const Common::LearningRateSchedulerConfig& cfg, Common::LearningRateSchedulerState& state,
                           ulong epoch, ulong totalEpochs, float valLoss)
{
  const float lr = Common::stepLearningRateScheduler(cfg, state, epoch, totalEpochs, true, valLoss);
  state.currentLearningRate = lr;
  return lr;
}

//===================================================================================================================//

static QString writeTempJson(const QString& name, const char* json)
{
  QString path = tempDir() + "/" + name;
  QFile file(path);
  file.open(QIODevice::WriteOnly);
  file.write(json);
  file.close();
  return path;
}

//===================================================================================================================//

static void testNoneKeepsConstantLR()
{
  TestScope _t("testNoneKeepsConstantLR");

  Common::LearningRateSchedulerConfig cfg; // type defaults to NONE
  Common::LearningRateSchedulerState state;
  state.baseLearningRate = 0.5f;
  state.currentLearningRate = 0.5f;

  for (ulong epoch = 0; epoch < 100; ++epoch) {
    const float lr = Common::stepLearningRateScheduler(cfg, state, epoch, 100, false, 0.0f);
    CHECK_NEAR(lr, 0.5f, 1e-6f, "NONE keeps currentLearningRate constant across epochs");
  }

  CHECK_NEAR(state.currentLearningRate, 0.5f, 1e-6f, "NONE leaves state.currentLearningRate untouched");
}

//===================================================================================================================//

static void testStepSchedule()
{
  TestScope _t("testStepSchedule");

  Common::LearningRateSchedulerConfig cfg;
  cfg.type = Common::LearningRateSchedulerType::STEP;
  cfg.gamma = 0.1f;
  cfg.stepSize = 10;

  Common::LearningRateSchedulerState state;
  state.baseLearningRate = 1.0f;
  state.currentLearningRate = 1.0f;

  // floor(epoch/10) = number of decay steps applied
  CHECK_NEAR(stepAndUpdate(cfg, state, 9, 100, 0.0f), 1.0f, 1e-4f, "step: epoch 9 = base");
  CHECK_NEAR(stepAndUpdate(cfg, state, 10, 100, 0.0f), 0.1f, 1e-4f, "step: epoch 10 = base*gamma");
  CHECK_NEAR(stepAndUpdate(cfg, state, 19, 100, 0.0f), 0.1f, 1e-4f, "step: epoch 19 = base*gamma");
  CHECK_NEAR(stepAndUpdate(cfg, state, 20, 100, 0.0f), 0.01f, 1e-4f, "step: epoch 20 = base*gamma^2");
  CHECK_NEAR(stepAndUpdate(cfg, state, 29, 100, 0.0f), 0.01f, 1e-4f, "step: epoch 29 = base*gamma^2");
  CHECK_NEAR(stepAndUpdate(cfg, state, 30, 100, 0.0f), 0.001f, 1e-4f, "step: epoch 30 = base*gamma^3");
}

//===================================================================================================================//

static void testStepRespectsMinLR()
{
  TestScope _t("testStepRespectsMinLR");

  Common::LearningRateSchedulerConfig cfg;
  cfg.type = Common::LearningRateSchedulerType::STEP;
  cfg.gamma = 0.1f;
  cfg.stepSize = 1;
  cfg.minLearningRate = 0.05f;

  Common::LearningRateSchedulerState state;
  state.baseLearningRate = 1.0f;
  state.currentLearningRate = 1.0f;

  // gamma^0..gamma^many; once the decay crosses minLearningRate it must clamp.
  float prev = 1.0f;

  for (ulong epoch = 0; epoch < 30; ++epoch) {
    const float lr = stepAndUpdate(cfg, state, epoch, 100, 0.0f);
    CHECK(lr >= cfg.minLearningRate - 1e-6f, "step: never below minLearningRate");
    CHECK(lr <= prev + 1e-6f, "step: monotonically non-increasing");
    prev = lr;
  }

  CHECK_NEAR(state.currentLearningRate, 0.05f, 1e-6f, "step: floors at minLearningRate");
}

//===================================================================================================================//

static void testCosineSchedule()
{
  TestScope _t("testCosineSchedule");

  Common::LearningRateSchedulerConfig cfg;
  cfg.type = Common::LearningRateSchedulerType::COSINE;
  cfg.minLearningRate = 0.0f;

  Common::LearningRateSchedulerState state;
  state.baseLearningRate = 1.0f;
  state.currentLearningRate = 1.0f;

  CHECK_NEAR(stepAndUpdate(cfg, state, 0, 100, 0.0f), 1.0f, 1e-4f, "cosine: epoch 0 = baseLearningRate");
  CHECK_NEAR(stepAndUpdate(cfg, state, 50, 100, 0.0f), 0.5f, 1e-3f, "cosine: epoch 50 = midpoint");
  CHECK_NEAR(stepAndUpdate(cfg, state, 100, 100, 0.0f), 0.0f, 1e-3f, "cosine: epoch 100 = minLearningRate");
}

//===================================================================================================================//

static void testCosineMonotonicAndFloor()
{
  TestScope _t("testCosineMonotonicAndFloor");

  Common::LearningRateSchedulerConfig cfg;
  cfg.type = Common::LearningRateSchedulerType::COSINE;
  cfg.minLearningRate = 0.01f;

  Common::LearningRateSchedulerState state;
  state.baseLearningRate = 1.0f;
  state.currentLearningRate = 1.0f;

  float prev = 1.0f;

  for (ulong epoch = 0; epoch <= 150; ++epoch) {
    const float lr = stepAndUpdate(cfg, state, epoch, 100, 0.0f);
    CHECK(lr >= cfg.minLearningRate - 1e-6f, "cosine: never below minLearningRate");
    CHECK(lr <= prev + 1e-6f, "cosine: monotonically non-increasing");
    prev = lr;
  }

  CHECK_NEAR(state.currentLearningRate, 0.01f, 1e-6f, "cosine: final floors at minLearningRate");
}

//===================================================================================================================//

static void testPlateauReduces()
{
  TestScope _t("testPlateauReduces");

  Common::LearningRateSchedulerConfig cfg;
  cfg.type = Common::LearningRateSchedulerType::PLATEAU;
  cfg.gamma = 0.5f;
  cfg.patience = 3;
  cfg.minDelta = 1e-4f;
  cfg.minLearningRate = 0.0f;

  Common::LearningRateSchedulerState state;
  state.baseLearningRate = 1.0f;
  state.currentLearningRate = 1.0f;

  // epoch 0 (val=1.0): first call seeds bestValidationLoss, returns currentLearningRate unchanged
  CHECK_NEAR(stepAndUpdate(cfg, state, 0, 100, 1.0f), 1.0f, 1e-6f, "plateau: epoch 0 seeds best");
  CHECK(state.initialized == true, "plateau: initialized after first call");

  // epoch 1 (val=0.9): improvement → counter reset, no reduce
  CHECK_NEAR(stepAndUpdate(cfg, state, 1, 100, 0.9f), 1.0f, 1e-6f, "plateau: epoch 1 improvement, no reduce");
  CHECK(state.epochsSinceImprovement == 0, "plateau: counter reset on improvement");

  // epochs 2,3,4 (val=0.9): no improvement → counter 1,2,3; reduce fires at counter==patience
  CHECK_NEAR(stepAndUpdate(cfg, state, 2, 100, 0.9f), 1.0f, 1e-6f, "plateau: epoch 2 no reduce (counter 1)");
  CHECK_NEAR(stepAndUpdate(cfg, state, 3, 100, 0.9f), 1.0f, 1e-6f, "plateau: epoch 3 no reduce (counter 2)");
  CHECK_NEAR(stepAndUpdate(cfg, state, 4, 100, 0.9f), 0.5f, 1e-6f, "plateau: epoch 4 reduces (counter==patience)");
  CHECK(state.epochsSinceImprovement == 0, "plateau: counter reset after reduce");
}

//===================================================================================================================//

static void testPlateauCounterResetsOnImprovement()
{
  TestScope _t("testPlateauCounterResetsOnImprovement");

  Common::LearningRateSchedulerConfig cfg;
  cfg.type = Common::LearningRateSchedulerType::PLATEAU;
  cfg.gamma = 0.5f;
  cfg.patience = 3;
  cfg.minDelta = 1e-4f;
  cfg.minLearningRate = 0.0f;

  Common::LearningRateSchedulerState state;
  state.baseLearningRate = 1.0f;
  state.currentLearningRate = 1.0f;

  stepAndUpdate(cfg, state, 0, 100, 1.0f); // seed best=1.0
  stepAndUpdate(cfg, state, 1, 100, 1.0f); // counter=1
  stepAndUpdate(cfg, state, 2, 100, 1.0f); // counter=2
  // improvement before patience is reached → counter must reset, no reduce
  CHECK_NEAR(stepAndUpdate(cfg, state, 3, 100, 0.5f), 1.0f, 1e-6f, "plateau: improvement aborts reduce");
  CHECK(state.epochsSinceImprovement == 0, "plateau: counter reset by late improvement");
}

//===================================================================================================================//

static void testPlateauRespectsMinLR()
{
  TestScope _t("testPlateauRespectsMinLR");

  Common::LearningRateSchedulerConfig cfg;
  cfg.type = Common::LearningRateSchedulerType::PLATEAU;
  cfg.gamma = 0.5f;
  cfg.patience = 1;
  cfg.minDelta = 1e-4f;
  cfg.minLearningRate = 0.1f;

  Common::LearningRateSchedulerState state;
  state.baseLearningRate = 1.0f;
  state.currentLearningRate = 1.0f;

  stepAndUpdate(cfg, state, 0, 100, 1.0f); // seed
  // patience=1 → every stagnant epoch after the seed reduces by gamma*0.5, clamped at minLearningRate=0.1
  for (ulong epoch = 1; epoch <= 10; ++epoch) {
    const float lr = stepAndUpdate(cfg, state, epoch, 100, 1.0f);
    CHECK(lr >= cfg.minLearningRate - 1e-6f, "plateau: never below minLearningRate");
  }

  CHECK_NEAR(state.currentLearningRate, 0.1f, 1e-6f, "plateau: floors at minLearningRate");
}

//===================================================================================================================//
//
// Resume correctness: a scheduler state loaded from a checkpoint must produce the same LR
// sequence as an uninterrupted run. Plateau is the interesting case (bestValidationLoss/counter persist).
//
//===================================================================================================================//

static void testPlateauResumeMatchesContinuous()
{
  TestScope _t("testPlateauResumeMatchesContinuous");

  auto buildCfg = [] {
    Common::LearningRateSchedulerConfig cfg;
    cfg.type = Common::LearningRateSchedulerType::PLATEAU;
    cfg.gamma = 0.5f;
    cfg.patience = 2;
    cfg.minDelta = 1e-4f;
    cfg.minLearningRate = 0.0f;
    return cfg;
  };

  const auto cfg = buildCfg();
  const float losses[8] = {1.0f, 0.8f, 0.8f, 0.8f, 0.6f, 0.6f, 0.6f, 0.6f};

  // Continuous run, 8 epochs.
  Common::LearningRateSchedulerState cont;
  cont.baseLearningRate = 1.0f;
  cont.currentLearningRate = 1.0f;
  std::vector<float> continuousLearningRates;
  Common::LearningRateSchedulerState contAt4; // snapshot of cont after epoch index 3 (4 epochs done)

  for (ulong e = 0; e < 8; ++e) {
    continuousLearningRates.push_back(stepAndUpdate(cfg, cont, e, 100, losses[e]));

    if (e == 3)
      contAt4 = cont;
  }

  // Resumed run: 4 epochs, snapshot state, "load" it, continue 4 more.
  Common::LearningRateSchedulerState resumed;
  resumed.baseLearningRate = 1.0f;
  resumed.currentLearningRate = 1.0f;

  for (ulong e = 0; e < 4; ++e)
    stepAndUpdate(cfg, resumed, e, 100, losses[e]);

  // Simulate checkpoint: copy the live state (as the serializer would round-trip it).
  Common::LearningRateSchedulerState loaded = resumed;
  CHECK(loaded.bestValidationLoss == contAt4.bestValidationLoss, "resume: bestValidationLoss persisted");
  CHECK(loaded.epochsSinceImprovement == contAt4.epochsSinceImprovement, "resume: counter persisted");
  CHECK_NEAR(loaded.currentLearningRate, contAt4.currentLearningRate, 1e-6f, "resume: currentLearningRate persisted");

  for (ulong e = 4; e < 8; ++e) {
    const float lr = stepAndUpdate(cfg, loaded, e, 100, losses[e]);
    CHECK_NEAR(lr, continuousLearningRates[e], 1e-6f, "resume: LR matches continuous run at each resumed epoch");
  }
}

//===================================================================================================================//

static void testStepResumeMatchesContinuous()
{
  TestScope _t("testStepResumeMatchesContinuous");

  Common::LearningRateSchedulerConfig cfg;
  cfg.type = Common::LearningRateSchedulerType::STEP;
  cfg.gamma = 0.1f;
  cfg.stepSize = 3;

  Common::LearningRateSchedulerState cont;
  cont.baseLearningRate = 1.0f;
  cont.currentLearningRate = 1.0f;

  // Run 10 continuous epochs; step is purely epoch-index driven.
  std::vector<float> continuousLearningRates;

  for (ulong e = 0; e < 10; ++e)
    continuousLearningRates.push_back(stepAndUpdate(cfg, cont, e, 100, 0.0f));

  // "Resume" from epoch 5 with absEpoch continuing — must match continuous at each index.
  Common::LearningRateSchedulerState resumed;
  resumed.baseLearningRate = 1.0f;
  resumed.currentLearningRate = 1.0f;

  for (ulong e = 5; e < 10; ++e)
    CHECK_NEAR(stepAndUpdate(cfg, resumed, e, 100, 0.0f), continuousLearningRates[e], 1e-6f,
               "step: resumed LR matches continuous (absEpoch-driven)");
}

//===================================================================================================================//

//===================================================================================================================//
//
// Phase 2: setLearningRate must reach the CNN dense head. CPU workers (and their dense-head
// ANN core) are recreated fresh each epoch from coreConfig.trainConfig, so setLearningRate has
// to update coreConfig.trainConfig.learningRate, not just the live this->trainConfig. SGD makes
// the update linear in LR, so a net built at LR=0.5 must equal a net built at LR=1.0 then
// setLearningRate(0.5) — and both must differ from a plain LR=1.0 net.
//
//===================================================================================================================//

static void testSetLearningRatePropagatesToDenseHead()
{
  TestScope _t("testSetLearningRatePropagatesToDenseHead");

  auto buildConfig = [](float lr) {
    CNN::CoreConfig<double> config;
    config.modeType = Common::ModeType::TRAIN;
    config.deviceType = Common::DeviceType::CPU;
    config.inputShape = {1, 5, 5};
    config.logLevel = Common::LogLevel::ERROR;

    CNN::CNNLayerConfig convLayer;
    convLayer.type = CNN::LayerType::CONV;
    convLayer.config = CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};
    CNN::CNNLayerConfig reluLayer;
    reluLayer.type = CNN::LayerType::RELU;
    reluLayer.config = CNN::ReLULayerConfig{};
    CNN::CNNLayerConfig flattenLayer;
    flattenLayer.type = CNN::LayerType::FLATTEN;
    flattenLayer.config = CNN::FlattenLayerConfig{};
    config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
    config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

    CNN::ConvParameters<double> initConv;
    initConv.numFilters = 1;
    initConv.inputC = 1;
    initConv.filterH = 3;
    initConv.filterW = 3;
    initConv.filters.assign(9, 0.1);
    initConv.biases.assign(1, 0.0);
    config.parameters.convParams = {initConv};

    // Dense head: Flatten yields 9 inputs → 1 sigmoid neuron. weights[1] holds the layer
    // (index 0 is the empty input slot — same convention as CNN/tests/test_gpu_exact.cpp).
    // Preset deterministically so two separately-built cores start identically.
    ANN::Parameters<double> denseParams;
    denseParams.weights.resize(2);
    denseParams.biases.resize(2);
    denseParams.weights[0] = {};
    denseParams.biases[0] = {};
    denseParams.weights[1] = {{0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3}};
    denseParams.biases[1] = {0.0};
    config.parameters.denseParams = denseParams;

    config.trainConfig.numEpochs = 3;
    config.trainConfig.learningRate = lr;
    config.trainConfig.shuffleSeed = 42;
    config.progressReports = 0;
    return config;
  };

  CNN::Samples<double> samples(1);
  samples[0].input = CNN::Tensor3D<double>({1, 5, 5}, 0.5);
  samples[0].output = {1.0};

  // CoreA: built and trained at LR=0.5 (reference).
  auto coreA = CNN::Core<double>::makeCore(buildConfig(0.5f));
  coreA->train(samples.size(), CNN::makeSampleProvider(samples));

  // CoreB: built at LR=1.0, then setLearningRate(0.5) before training.
  auto coreB = CNN::Core<double>::makeCore(buildConfig(1.0f));
  coreB->setLearningRate(static_cast<double>(0.5f));
  CHECK_NEAR(coreB->getTrainConfig().learningRate, 0.5f, 1e-6f, "setLearningRate updates core trainConfig");
  coreB->train(samples.size(), CNN::makeSampleProvider(samples));

  // CoreC: built and trained at LR=1.0 (no setLearningRate) — must differ from CoreA.
  auto coreC = CNN::Core<double>::makeCore(buildConfig(1.0f));
  coreC->train(samples.size(), CNN::makeSampleProvider(samples));

  const auto& pa = coreA->getParameters();
  const auto& pb = coreB->getParameters();
  const auto& pc = coreC->getParameters();

  // First non-empty dense weight entry (robust to the [0]-empty / [1]-layer convention).
  auto firstDenseWeight = [](const ANN::Parameters<double>& p) -> const double* {
    for (const auto& layer : p.weights)

      for (const auto& neuron : layer)

        if (!neuron.empty())
          return &neuron[0];
    return nullptr;
  };

  const double* dwa = firstDenseWeight(pa.denseParams);
  const double* dwb = firstDenseWeight(pb.denseParams);
  const double* dwc = firstDenseWeight(pc.denseParams);
  CHECK(dwa != nullptr && dwb != nullptr && dwc != nullptr, "dense weights present after training");

  // Conv filter (consumer #3) and dense weight (#5) must match between A and B...
  CHECK_NEAR(pa.convParams[0].filters[0], pb.convParams[0].filters[0], 1e-9, "conv: setLR path matches built-at-LR");
  CHECK_NEAR(*dwa, *dwb, 1e-9, "dense: setLR path matches built-at-LR");
  // ...and differ from C (proves the LR change actually changed the update, not a no-op).
  CHECK(std::fabs(pa.convParams[0].filters[0] - pc.convParams[0].filters[0]) > 1e-6, "conv: LR=1 differs from LR=0.5");
  CHECK(std::fabs(*dwa - *dwc) > 1e-6, "dense: LR=1 differs from LR=0.5");
}

//===================================================================================================================//

static void testSchedulerConfigParsingCosine()
{
  TestScope _t("testSchedulerConfigParsingCosine");

  QString configPath = writeTempJson("scheduler_cosine_parse.json", R"({
    "mode": "train",
    "layers": [{"numNeurons": 4, "actvFunc": "relu"}],
    "train": {
      "numEpochs": 100,
      "learningRate": 0.01,
      "learningRateScheduler": { "type": "cosine", "minLearningRate": 0.0001 }
    }
  })");

  auto config = NN_CLI::ANNLoader::loadModel(configPath.toStdString());
  CHECK(config.trainConfig.learningRateScheduler.type == Common::LearningRateSchedulerType::COSINE,
        "cosine type parsed");
  CHECK_NEAR(config.trainConfig.learningRateScheduler.minLearningRate, 0.0001f, 1e-7f, "cosine minLearningRate parsed");
  CHECK_NEAR(config.trainConfig.learningRateScheduler.gamma, 0.1f, 1e-7f, "gamma retains default when omitted");
}

//===================================================================================================================//

static void testSchedulerConfigParsingStep()
{
  TestScope _t("testSchedulerConfigParsingStep");

  QString configPath = writeTempJson("scheduler_step_parse.json", R"({
    "mode": "train",
    "layers": [{"numNeurons": 4, "actvFunc": "relu"}],
    "train": {
      "numEpochs": 100,
      "learningRate": 0.01,
      "learningRateScheduler": { "type": "step", "gamma": 0.5, "stepSize": 20 }
    }
  })");

  auto config = NN_CLI::ANNLoader::loadModel(configPath.toStdString());
  CHECK(config.trainConfig.learningRateScheduler.type == Common::LearningRateSchedulerType::STEP, "step type parsed");
  CHECK_NEAR(config.trainConfig.learningRateScheduler.gamma, 0.5f, 1e-7f, "step gamma parsed");
  CHECK(config.trainConfig.learningRateScheduler.stepSize == 20, "step stepSize parsed");
}

//===================================================================================================================//

static void testSchedulerConfigDefaultsToNone()
{
  TestScope _t("testSchedulerConfigDefaultsToNone");

  QString configPath = writeTempJson("scheduler_none_parse.json", R"({
    "mode": "train",
    "layers": [{"numNeurons": 4, "actvFunc": "relu"}],
    "train": { "numEpochs": 10, "learningRate": 0.01 }
  })");

  auto config = NN_CLI::ANNLoader::loadModel(configPath.toStdString());
  CHECK(config.trainConfig.learningRateScheduler.type == Common::LearningRateSchedulerType::NONE,
        "no scheduler block → NONE");
}

//===================================================================================================================//

static void testSchedulerMissingTypeThrows()
{
  TestScope _t("testSchedulerMissingTypeThrows");

  QString configPath = writeTempJson("scheduler_missing_type.json", R"({
    "mode": "train",
    "layers": [{"numNeurons": 4, "actvFunc": "relu"}],
    "train": {
      "numEpochs": 10,
      "learningRate": 0.01,
      "learningRateScheduler": { "gamma": 0.5 }
    }
  })");

  CHECK_THROWS(NN_CLI::ANNLoader::loadModel(configPath.toStdString()), "scheduler without 'type' must throw");
}

//===================================================================================================================//

static void testSchedulerAppliedDuringTraining()
{
  TestScope _t("testSchedulerAppliedDuringTraining");

  // Step scheduler: gamma=0.1 every stepSize=3 epochs, baseLearningRate=0.5, 10 epochs. The epoch-boundary
  // callback fires after epochs 0..9; the last is at absEpoch=9 → floor(9/3)=3 decay steps →
  // 0.5 * 0.1^3 = 0.0005. The --output model is saved post-train, so it captures that final LR.
  // If applyLearningRateScheduler were never wired, the output LR would stay 0.5 and this assertion fails.
  QString configPath = writeTempJson("scheduler_applied_train.json", R"({
    "mode": "train",
    "device": "cpu",
    "numThreads": 1,
    "saveModelInterval": 0,
    "layers": [
      { "numNeurons": 2, "actvFunc": "relu" },
      { "numNeurons": 8, "actvFunc": "relu" },
      { "numNeurons": 2, "actvFunc": "sigmoid" }
    ],
    "train": {
      "numEpochs": 10,
      "learningRate": 0.5,
      "learningRateScheduler": { "type": "step", "gamma": 0.1, "stepSize": 3 }
    }
  })");

  QString outDir = tempDir() + "/scheduler_applied_out";
  QDir(outDir).removeRecursively();
  QDir().mkpath(outDir);
  auto result = runNNCLI({"--model", configPath, "--mode", "train", "--device", "cpu", "--samples",
                          fixturePath("ann_train_samples.json"), "--output", outDir, "--log-level", "quiet"});
  CHECK(result.exitCode == 0, "scheduler training run exited 0");

  QJsonObject modelJson = readModelJsonFromPackage(findTrainedModel(outDir));
  CHECK(!modelJson.isEmpty(), "output model.json readable");
  QJsonObject trainJson = modelJson.value("train").toObject();
  CHECK(!trainJson.isEmpty(), "model.json has train block");
  const double finalLearningRate = trainJson.value("learningRate").toDouble(-999.0);
  CHECK_NEAR(static_cast<float>(finalLearningRate), 0.0005f, 1e-5f, "step scheduler annealed final LR to base*gamma^3");

  QJsonObject metaJson = modelJson.value("trainMetadata").toObject();
  CHECK(metaJson.contains("learningRateSchedulerState"), "scheduler state persisted in model metadata");
  QJsonObject ssJson = metaJson.value("learningRateSchedulerState").toObject();
  CHECK(ssJson.value("initialized").toBool(false), "persisted scheduler state marked initialized");
  const double persistedCurrentLearningRate = ssJson.value("currentLearningRate").toDouble(-999.0);
  CHECK_NEAR(static_cast<float>(persistedCurrentLearningRate), 0.0005f, 1e-5f,
             "persisted currentLearningRate matches annealed LR");
}

//===================================================================================================================//

void runLearningRateSchedulerTests()
{
  testNoneKeepsConstantLR();
  testSetLearningRatePropagatesToDenseHead();
  testStepSchedule();
  testStepRespectsMinLR();
  testCosineSchedule();
  testCosineMonotonicAndFloor();
  testPlateauReduces();
  testPlateauCounterResetsOnImprovement();
  testPlateauRespectsMinLR();
  testPlateauResumeMatchesContinuous();
  testStepResumeMatchesContinuous();
  testSchedulerConfigParsingCosine();
  testSchedulerConfigParsingStep();
  testSchedulerConfigDefaultsToNone();
  testSchedulerMissingTypeThrows();
  testSchedulerAppliedDuringTraining();
}
