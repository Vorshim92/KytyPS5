// The trigger effects as the game sees them (src/libs/controller.cpp): scePadSetTriggerEffect and
// scePadGetTriggerEffectState.
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "libs/controller.h"
#include "libs/padData.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Controller;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "PadInputTests: failed: %s\n", text);
    std::abort();
  }
}

// scePadSetTriggerEffect succeeds whether or not effects reach a pad (KYTY_PAD_TRIGGER_EFFECTS, 0: none), and the
// effect state follows the game's effect and the trigger's travel: a weapon effect on L2 from position 2, released at 5.
void CheckTriggerEffects() {
  PadTriggerEffectParam param{};
  param.trigger_mask = 0x01;
  param.command[0].mode = 2;
  param.command[0].data[0] = 2;
  param.command[0].data[1] = 5;
  param.command[0].data[2] = 8;
  Check(PadSetTriggerEffect(1, &param) == 0, "a weapon effect on L2 is accepted");
  PadTriggerEffectStateInformation info{};
  SetAxis(HOST_INPUT_CONTROLLER_ID, Axis::TriggerLeft, 255);
  Check(PadGetTriggerEffectState(1, &info) == 0 && info.state[0] == PAD_TRIGGER_STATE_WEAPON_FULLY_PRESSED,
        "a fully pulled trigger is the weapon effect's fully pressed state");
  SetAxis(HOST_INPUT_CONTROLLER_ID, Axis::TriggerLeft, 0);
  Check(PadGetTriggerEffectState(1, &info) == 0 && info.state[0] == PAD_TRIGGER_STATE_WEAPON_NOT_PRESSED,
        "a released trigger is the weapon effect's not pressed state");
  param.command[0].data[0] = 1;
  Check(PadSetTriggerEffect(1, &param) != 0, "a weapon effect starting before position 2 is still refused");
}

} // namespace

int main() {
  Common::InitializeThreads();
  Common::Subsystems subsystems;
  subsystems.Initialize<Config::Lifecycle>();
  Config::ConfigOptions options;
  options.printf_direction = Config::OutputDirection::Silent;
  Config::Load(options);
  subsystems.Initialize<Log::Lifecycle>();

  Libs::Controller::Initialize();
  CheckTriggerEffects();
  Libs::Controller::Shutdown();
  subsystems.Destroy();

  std::printf("PadInputTests: all cases passed\n");
  return 0;
}
