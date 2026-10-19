// Pad input and output as the game sees them (src/libs/controller.cpp): the state history scePadRead gives, L2/R2 as
// buttons, and the trigger effects.
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

// The unread states, oldest first (Demon's Souls reads 64 a tick).
int Read(PadData *data) { return PadRead(1, data, 64); }

// A tap, then far more stick events before the next read than the history holds: the read still has the press and
// the release, and its newest state the sticks' last values (KYTY_PAD_STATE_COALESCE).
void CheckTapAcrossStickBurst() {
  static PadData data[64];
  (void)Read(data);
  SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CROSS, true);
  for (int i = 0; i < 200; i++) {
    SetAxis(HOST_INPUT_CONTROLLER_ID, Axis::LeftX, 100 + i % 50);
  }
  SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CROSS, false);
  for (int i = 0; i < 200; i++) {
    SetAxis(HOST_INPUT_CONTROLLER_ID, Axis::LeftX, 150 - i % 50);
  }
  const int count = Read(data);
  bool pressed = false;
  bool released_after = false;
  for (int i = 0; i < count; i++) {
    if ((data[i].buttons & PAD_BUTTON_CROSS) != 0) {
      pressed = true;
    } else if (pressed) {
      released_after = true;
    }
  }
  Check(pressed && released_after, "a tap between two reads keeps its press and its release");
  Check(count > 0 && data[count - 1].left_stick_x == 150 - 199 % 50,
        "the newest state has the sticks' last values");
}

// L2 pressed from a travel of 8, held down to 5, released at 4; a trigger resting at 2 presses nothing
// (KYTY_PAD_TRIGGER_HYSTERESIS).
void CheckTriggerButtons() {
  static PadData data[64];
  const auto l2_after = [&](int travel) {
    SetAxis(HOST_INPUT_CONTROLLER_ID, Axis::TriggerLeft, travel);
    const int count = Read(data);
    return count > 0 && (data[count - 1].buttons & PAD_BUTTON_L2) != 0;
  };
  Check(!l2_after(2), "a trigger resting at 2 is no L2 press");
  Check(l2_after(8), "a travel of 8 presses L2");
  Check(l2_after(5), "L2 stays held down to a travel of 5");
  Check(!l2_after(4), "L2 is released at a travel of 4");
  Check(!l2_after(6), "a travel of 6 from released presses nothing");
  Check(l2_after(255), "a full travel presses L2");
  Check(!l2_after(0), "a released trigger releases L2");
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
  CheckTapAcrossStickBurst();
  CheckTriggerButtons();
  CheckTriggerEffects();
  Libs::Controller::Shutdown();
  subsystems.Destroy();

  std::printf("PadInputTests: all cases passed\n");
  return 0;
}
