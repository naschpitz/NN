#include "test_helpers.hpp"

#include "Common/Common_LRScheduler.hpp"

#include <vector>

//===================================================================================================================//
//
// Helper: mirror the Runner's caller pattern — step the scheduler, then publish the returned
// LR back into state.currentLR (the function computes-and-returns; the caller owns currentLR).
//
//===================================================================================================================//

static float stepAndUpdate(const Common::LRSchedulerConfig& cfg, Common::LRSchedulerState& state, ulong epoch,
                           ulong totalEpochs, float valLoss)
{
  const float lr = Common::stepLRScheduler(cfg, state, epoch, totalEpochs, true, valLoss);
  state.currentLR = lr;
  return lr;
}

//===================================================================================================================//

static void testNoneKeepsConstantLR()
{
  TestScope _t("testNoneKeepsConstantLR");

  Common::LRSchedulerConfig cfg; // type defaults to NONE
  Common::LRSchedulerState state;
  state.baseLR = 0.5f;
  state.currentLR = 0.5f;

  for (ulong epoch = 0; epoch < 100; ++epoch) {
    const float lr = Common::stepLRScheduler(cfg, state, epoch, 100, false, 0.0f);
    CHECK_NEAR(lr, 0.5f, 1e-6f, "NONE keeps currentLR constant across epochs");
  }

  CHECK_NEAR(state.currentLR, 0.5f, 1e-6f, "NONE leaves state.currentLR untouched");
}

//===================================================================================================================//

static void testStepSchedule()
{
  TestScope _t("testStepSchedule");

  Common::LRSchedulerConfig cfg;
  cfg.type = Common::LRSchedulerType::STEP;
  cfg.gamma = 0.1f;
  cfg.stepSize = 10;

  Common::LRSchedulerState state;
  state.baseLR = 1.0f;
  state.currentLR = 1.0f;

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

  Common::LRSchedulerConfig cfg;
  cfg.type = Common::LRSchedulerType::STEP;
  cfg.gamma = 0.1f;
  cfg.stepSize = 1;
  cfg.minLR = 0.05f;

  Common::LRSchedulerState state;
  state.baseLR = 1.0f;
  state.currentLR = 1.0f;

  // gamma^0..gamma^many; once the decay crosses minLR it must clamp.
  float prev = 1.0f;

  for (ulong epoch = 0; epoch < 30; ++epoch) {
    const float lr = stepAndUpdate(cfg, state, epoch, 100, 0.0f);
    CHECK(lr >= cfg.minLR - 1e-6f, "step: never below minLR");
    CHECK(lr <= prev + 1e-6f, "step: monotonically non-increasing");
    prev = lr;
  }

  CHECK_NEAR(state.currentLR, 0.05f, 1e-6f, "step: floors at minLR");
}

//===================================================================================================================//

static void testCosineSchedule()
{
  TestScope _t("testCosineSchedule");

  Common::LRSchedulerConfig cfg;
  cfg.type = Common::LRSchedulerType::COSINE;
  cfg.minLR = 0.0f;

  Common::LRSchedulerState state;
  state.baseLR = 1.0f;
  state.currentLR = 1.0f;

  CHECK_NEAR(stepAndUpdate(cfg, state, 0, 100, 0.0f), 1.0f, 1e-4f, "cosine: epoch 0 = baseLR");
  CHECK_NEAR(stepAndUpdate(cfg, state, 50, 100, 0.0f), 0.5f, 1e-3f, "cosine: epoch 50 = midpoint");
  CHECK_NEAR(stepAndUpdate(cfg, state, 100, 100, 0.0f), 0.0f, 1e-3f, "cosine: epoch 100 = minLR");
}

//===================================================================================================================//

static void testCosineMonotonicAndFloor()
{
  TestScope _t("testCosineMonotonicAndFloor");

  Common::LRSchedulerConfig cfg;
  cfg.type = Common::LRSchedulerType::COSINE;
  cfg.minLR = 0.01f;

  Common::LRSchedulerState state;
  state.baseLR = 1.0f;
  state.currentLR = 1.0f;

  float prev = 1.0f;

  for (ulong epoch = 0; epoch <= 150; ++epoch) {
    const float lr = stepAndUpdate(cfg, state, epoch, 100, 0.0f);
    CHECK(lr >= cfg.minLR - 1e-6f, "cosine: never below minLR");
    CHECK(lr <= prev + 1e-6f, "cosine: monotonically non-increasing");
    prev = lr;
  }

  CHECK_NEAR(state.currentLR, 0.01f, 1e-6f, "cosine: final floors at minLR");
}

//===================================================================================================================//

static void testPlateauReduces()
{
  TestScope _t("testPlateauReduces");

  Common::LRSchedulerConfig cfg;
  cfg.type = Common::LRSchedulerType::PLATEAU;
  cfg.gamma = 0.5f;
  cfg.patience = 3;
  cfg.minDelta = 1e-4f;
  cfg.minLR = 0.0f;

  Common::LRSchedulerState state;
  state.baseLR = 1.0f;
  state.currentLR = 1.0f;

  // epoch 0 (val=1.0): first call seeds bestValLoss, returns currentLR unchanged
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

  Common::LRSchedulerConfig cfg;
  cfg.type = Common::LRSchedulerType::PLATEAU;
  cfg.gamma = 0.5f;
  cfg.patience = 3;
  cfg.minDelta = 1e-4f;
  cfg.minLR = 0.0f;

  Common::LRSchedulerState state;
  state.baseLR = 1.0f;
  state.currentLR = 1.0f;

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

  Common::LRSchedulerConfig cfg;
  cfg.type = Common::LRSchedulerType::PLATEAU;
  cfg.gamma = 0.5f;
  cfg.patience = 1;
  cfg.minDelta = 1e-4f;
  cfg.minLR = 0.1f;

  Common::LRSchedulerState state;
  state.baseLR = 1.0f;
  state.currentLR = 1.0f;

  stepAndUpdate(cfg, state, 0, 100, 1.0f); // seed
  // patience=1 → every stagnant epoch after the seed reduces by gamma*0.5, clamped at minLR=0.1
  for (ulong epoch = 1; epoch <= 10; ++epoch) {
    const float lr = stepAndUpdate(cfg, state, epoch, 100, 1.0f);
    CHECK(lr >= cfg.minLR - 1e-6f, "plateau: never below minLR");
  }

  CHECK_NEAR(state.currentLR, 0.1f, 1e-6f, "plateau: floors at minLR");
}

//===================================================================================================================//
//
// Resume correctness: a scheduler state loaded from a checkpoint must produce the same LR
// sequence as an uninterrupted run. Plateau is the interesting case (bestValLoss/counter persist).
//
//===================================================================================================================//

static void testPlateauResumeMatchesContinuous()
{
  TestScope _t("testPlateauResumeMatchesContinuous");

  auto buildCfg = [] {
    Common::LRSchedulerConfig cfg;
    cfg.type = Common::LRSchedulerType::PLATEAU;
    cfg.gamma = 0.5f;
    cfg.patience = 2;
    cfg.minDelta = 1e-4f;
    cfg.minLR = 0.0f;
    return cfg;
  };

  const auto cfg = buildCfg();
  const float losses[8] = {1.0f, 0.8f, 0.8f, 0.8f, 0.6f, 0.6f, 0.6f, 0.6f};

  // Continuous run, 8 epochs.
  Common::LRSchedulerState cont;
  cont.baseLR = 1.0f;
  cont.currentLR = 1.0f;
  std::vector<float> contLR;
  Common::LRSchedulerState contAt4; // snapshot of cont after epoch index 3 (4 epochs done)

  for (ulong e = 0; e < 8; ++e) {
    contLR.push_back(stepAndUpdate(cfg, cont, e, 100, losses[e]));

    if (e == 3)
      contAt4 = cont;
  }

  // Resumed run: 4 epochs, snapshot state, "load" it, continue 4 more.
  Common::LRSchedulerState resumed;
  resumed.baseLR = 1.0f;
  resumed.currentLR = 1.0f;

  for (ulong e = 0; e < 4; ++e)
    stepAndUpdate(cfg, resumed, e, 100, losses[e]);

  // Simulate checkpoint: copy the live state (as the serializer would round-trip it).
  Common::LRSchedulerState loaded = resumed;
  CHECK(loaded.bestValLoss == contAt4.bestValLoss, "resume: bestValLoss persisted");
  CHECK(loaded.epochsSinceImprovement == contAt4.epochsSinceImprovement, "resume: counter persisted");
  CHECK_NEAR(loaded.currentLR, contAt4.currentLR, 1e-6f, "resume: currentLR persisted");

  for (ulong e = 4; e < 8; ++e) {
    const float lr = stepAndUpdate(cfg, loaded, e, 100, losses[e]);
    CHECK_NEAR(lr, contLR[e], 1e-6f, "resume: LR matches continuous run at each resumed epoch");
  }
}

//===================================================================================================================//

static void testStepResumeMatchesContinuous()
{
  TestScope _t("testStepResumeMatchesContinuous");

  Common::LRSchedulerConfig cfg;
  cfg.type = Common::LRSchedulerType::STEP;
  cfg.gamma = 0.1f;
  cfg.stepSize = 3;

  Common::LRSchedulerState cont;
  cont.baseLR = 1.0f;
  cont.currentLR = 1.0f;

  // Run 10 continuous epochs; step is purely epoch-index driven.
  std::vector<float> contLR;

  for (ulong e = 0; e < 10; ++e)
    contLR.push_back(stepAndUpdate(cfg, cont, e, 100, 0.0f));

  // "Resume" from epoch 5 with absEpoch continuing — must match continuous at each index.
  Common::LRSchedulerState resumed;
  resumed.baseLR = 1.0f;
  resumed.currentLR = 1.0f;

  for (ulong e = 5; e < 10; ++e)
    CHECK_NEAR(stepAndUpdate(cfg, resumed, e, 100, 0.0f), contLR[e], 1e-6f,
               "step: resumed LR matches continuous (absEpoch-driven)");
}

//===================================================================================================================//

void runLRSchedulerTests()
{
  testNoneKeepsConstantLR();
  testStepSchedule();
  testStepRespectsMinLR();
  testCosineSchedule();
  testCosineMonotonicAndFloor();
  testPlateauReduces();
  testPlateauCounterResetsOnImprovement();
  testPlateauRespectsMinLR();
  testPlateauResumeMatchesContinuous();
  testStepResumeMatchesContinuous();
}
