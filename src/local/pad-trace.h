#ifndef EMULATOR_SRC_LOCAL_PAD_TRACE_H_
#define EMULATOR_SRC_LOCAL_PAD_TRACE_H_
// KYTY_PAD_TRACE=1: the pad as the game reads it, a CSV row per scePadRead / scePadReadState, for a run that drops to
// a walk: did the input dip or not? Each row has the time, the guest flip, the host stick's raw values (SDL), the stick
// bytes, buttons and trigger bytes the game takes, a marker (Scroll Lock) and the player's position (Demon's Souls
// 01.007.000, Loader::DemonsSoulsWarp::PlayerPlace, read on the game's thread at its read). The file goes beside the
// run's log (KYTY_RUN_LOG <stamp>.out.log: <stamp>-pad.csv), else to the working directory (pad-trace-<date-time>.csv);
// a thread of its own writes the rows about once a second and at the end ("Pad trace: writing <path>" on stdout).
// Off (the default): one cached check per read.

namespace Libs::Controller {
struct PadData;
} // namespace Libs::Controller

namespace PadTrace {

[[nodiscard]] bool On();

// The controller's start-up: the file opened and the writer thread started (KYTY_PAD_TRACE=1 only).
void Open();
// The controller's shutdown, also an emergency one: the rows so far written and the file closed (once).
void Close();

// The SDL event loop: a host controller's axis as SDL gives it (-32768..32767; the left stick's are kept).
void NoteAxis(int id, int sdl_axis, int value);
// Scroll Lock (the SDL event loop): the next row marked.
void Mark();

// scePadRead (`state_call` false) and scePadReadState, on the game's thread: the `num` states it returns, oldest first,
// as the game gets them; `unread` of them were new (0: scePadRead repeated the current state), `active_id` is the pad
// they came from (the raw stick values are that pad's).
void NoteRead(const Libs::Controller::PadData* data, int num, int unread, int active_id, bool state_call);

} // namespace PadTrace

#endif // EMULATOR_SRC_LOCAL_PAD_TRACE_H_
