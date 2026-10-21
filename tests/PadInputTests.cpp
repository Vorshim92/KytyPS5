// Pad input and output as the game sees them (src/libs/controller.cpp): the state history scePadRead gives, the
// trigger effects and the pad trace (src/local/pad-trace.h).
#include "common/emulatorConfig.h"
#include "common/hostException.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "libs/controller.h"
#include "libs/padData.h"
#include "pad-trace.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

void SetEnvironment(const char *name, const char *value) {
#if defined(_WIN32)
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

std::vector<std::string> Fields(const std::string &line) {
  std::vector<std::string> fields(1);
  for (const char c : line) {
    if (c == ',') {
      fields.emplace_back();
    } else {
      fields.back() += c;
    }
  }
  return fields;
}

// KYTY_PAD_TRACE (on from the start, main): a read of a Circle press with the left stick pushed forward, the stick's
// raw SDL values noted and a mark set, then a read with nothing new.
void CheckPadTraceReads() {
  static PadData data[64];
  (void)Read(data);
  PadTrace::NoteAxis(HOST_INPUT_CONTROLLER_ID, 0 /* SDL_CONTROLLER_AXIS_LEFTX */, 0);
  PadTrace::NoteAxis(HOST_INPUT_CONTROLLER_ID, 1 /* SDL_CONTROLLER_AXIS_LEFTY */, -29164);
  SetAxis(HOST_INPUT_CONTROLLER_ID, Axis::LeftY, 13);
  SetAxis(HOST_INPUT_CONTROLLER_ID, Axis::LeftX, 128);
  SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CIRCLE, true);
  PadTrace::Mark();
  Check(Read(data) == 2, "the stick state and the press are two states");
  (void)Read(data);
  SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CIRCLE, false);
}

// The trace's file once the controller is shut down: its columns, and the two reads' rows last.
void CheckPadTraceFile(const std::filesystem::path &path) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) {
    lines.push_back(line);
  }
  Check(!lines.empty() && lines[0] == "ms,flip,call,states,raw_lx,raw_ly,lx,ly,rx,ry,mag,mag_raw,buttons,circle,cross,l1,"
                                      "r1,l2,r2,circle_all,l2_analog,r2_analog,marker,x,y,z",
        "the trace has its columns");
  Check(lines.size() >= 3, "the trace has a row per read");
  const auto pressed = Fields(lines[lines.size() - 2]);
  const auto again = Fields(lines[lines.size() - 1]);
  Check(pressed.size() == 26 && again.size() == 26, "every row has every column");
  Check(pressed[2] == "read" && pressed[3] == "2", "the row has the read's call and its new states");
  Check(pressed[4] == "0" && pressed[5] == "-29164" && pressed[11] == "0.8900", "the row has the raw stick values");
  Check(pressed[6] == "128" && pressed[7] == "13" && pressed[10] == "0.8984", "the row has the stick bytes");
  Check(pressed[12] == "0x00002000" && pressed[13] == "1" && pressed[19] == "0",
        "Circle is held in the newest state and up in the read's first");
  Check(pressed[22] == "1" && again[22] == "0", "the mark is on the next row only");
  Check(pressed[23].empty() && pressed[24].empty() && pressed[25].empty(), "no position without the game");
  Check(again[3] == "0" && again[13] == "1" && again[19] == "1", "a read with nothing new repeats the state");
}

} // namespace

int main() {
  Common::InitializeThreads();
  // (The trace's position read is a guarded copy: the host exception handler resumes its fault.)
  Common::HostException::InstallHandler([](const Common::HostException::ExceptionInfo &) { return false; });
  const auto trace = std::filesystem::temp_directory_path() / "pad_input_tests-pad.csv";
  SetEnvironment("KYTY_PAD_TRACE", "1");
  SetEnvironment("KYTY_RUN_LOG", (std::filesystem::temp_directory_path() / "pad_input_tests.out.log").string().c_str());
  Common::Subsystems subsystems;
  subsystems.Initialize<Config::Lifecycle>();
  Config::ConfigOptions options;
  options.printf_direction = Config::OutputDirection::Silent;
  Config::Load(options);
  subsystems.Initialize<Log::Lifecycle>();

  Libs::Controller::Initialize();
  CheckTapAcrossStickBurst();
  CheckTriggerEffects();
  CheckPadTraceReads();
  Libs::Controller::Shutdown();
  subsystems.Destroy();
  CheckPadTraceFile(trace);
  std::filesystem::remove(trace);

  std::printf("PadInputTests: all cases passed\n");
  return 0;
}
