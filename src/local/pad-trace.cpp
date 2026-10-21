#include "pad-trace.h"

#include "SDL_gamecontroller.h"
#include "libs/controller.h"
#include "libs/padData.h"
#include "loader/demonsSoulsWarp.h"

#include "frame-gen.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace PadTrace {

namespace {

// Rows the writer has not taken yet, at most (over an hour of reads: only a disk that stopped fills it).
constexpr size_t RowsMax = 1u << 18;

struct Row {
	double   ms         = 0;
	uint64_t flip       = 0;
	bool     state_call = false;
	int      states     = 0;
	int      raw_x      = INT_MIN; // (INT_MIN: no SDL value of that pad yet)
	int      raw_y      = INT_MIN;
	uint8_t  lx = 0, ly = 0, rx = 0, ry = 0, l2 = 0, r2 = 0;
	uint32_t buttons     = 0;
	uint32_t buttons_all = 0; // held in every state of the read
	bool     marker      = false;
	bool     placed      = false;
	float    xyz[3] {};
};

std::chrono::steady_clock::time_point g_start;

// The left stick of the pad whose axis event came last (the SDL event loop).
std::atomic<int> g_raw_id {INT_MIN};
std::atomic<int> g_raw_x {INT_MIN};
std::atomic<int> g_raw_y {INT_MIN};
std::atomic_bool g_marker {false};

std::mutex              g_mutex; // g_rows, g_open, g_start, g_dropped
std::condition_variable g_wake;
std::vector<Row>        g_rows;
bool                    g_open    = false;
uint64_t                g_dropped = 0;

std::timed_mutex g_file_mutex; // g_file, g_writing, g_written: the writer's
std::FILE*       g_file = nullptr;
std::string      g_path;
std::vector<Row> g_writing;
uint64_t         g_written = 0;

// Beside the run's log, else in the working directory.
std::string TracePath() {
	if (const char* log = std::getenv("KYTY_RUN_LOG"); log != nullptr && *log != '\0') {
		std::string path = log;
		if (path.size() > 8 && path.ends_with(".out.log")) {
			path.resize(path.size() - 8);
		} else if (const auto dot = path.find_last_of('.'); dot != std::string::npos && dot > path.find_last_of("\\/") + 1) {
			path.resize(dot);
		}
		return path + "-pad.csv";
	}
	const auto time = std::time(nullptr);
	std::tm    local {};
#if defined(_WIN32)
	localtime_s(&local, &time);
#else
	localtime_r(&time, &local);
#endif
	char name[64] {};
	std::strftime(name, sizeof(name), "pad-trace-%Y%m%d-%H%M%S.csv", &local);
	std::error_code error;
	const auto      path = std::filesystem::absolute(name, error);
	return error ? std::string(name) : path.string();
}

double Magnitude(double x, double y) {
	return std::sqrt(x * x + y * y);
}

void WriteRow(const Row& row) {
	const bool raw = row.raw_x != INT_MIN && row.raw_y != INT_MIN;
	char       raw_x[16] {}, raw_y[16] {}, raw_magnitude[16] {}, place[96] = ",,";
	if (row.raw_x != INT_MIN) std::snprintf(raw_x, sizeof(raw_x), "%d", row.raw_x);
	if (row.raw_y != INT_MIN) std::snprintf(raw_y, sizeof(raw_y), "%d", row.raw_y);
	if (raw) std::snprintf(raw_magnitude, sizeof(raw_magnitude), "%.4f", Magnitude(row.raw_x, row.raw_y) / 32768.0);
	if (row.placed) std::snprintf(place, sizeof(place), "%.3f,%.3f,%.3f", row.xyz[0], row.xyz[1], row.xyz[2]);
	const auto held = [&](uint32_t buttons, uint32_t button) { return (buttons & button) != 0 ? 1 : 0; };
	using namespace Libs::Controller;
	std::fprintf(g_file, "%.3f,%llu,%s,%d,%s,%s,%u,%u,%u,%u,%.4f,%s,", row.ms, static_cast<unsigned long long>(row.flip),
	             row.state_call ? "state" : "read", row.states, raw_x, raw_y, row.lx, row.ly, row.rx, row.ry,
	             Magnitude(row.lx - 128.0, row.ly - 128.0) / 128.0, raw_magnitude);
	std::fprintf(g_file, "0x%08x,%d,%d,%d,%d,%d,%d,%d,%u,%u,%d,%s\n", row.buttons, held(row.buttons, PAD_BUTTON_CIRCLE),
	             held(row.buttons, PAD_BUTTON_CROSS), held(row.buttons, PAD_BUTTON_L1), held(row.buttons, PAD_BUTTON_R1),
	             held(row.buttons, PAD_BUTTON_L2), held(row.buttons, PAD_BUTTON_R2), held(row.buttons_all, PAD_BUTTON_CIRCLE),
	             row.l2, row.r2, row.marker ? 1 : 0, place);
}

// The rows so far to the file (the caller holds g_file_mutex).
void FlushLocked() {
	{
		std::lock_guard lock(g_mutex);
		g_writing.swap(g_rows); // (both keep their capacity: the game's thread allocates only while they grow)
	}
	for (const auto& row: g_writing) WriteRow(row);
	g_written += g_writing.size();
	g_writing.clear();
	std::fflush(g_file);
}

void WriterThread() {
	std::unique_lock lock(g_mutex);
	while (g_open) {
		g_wake.wait_for(lock, std::chrono::seconds(1), [] { return !g_open; });
		if (!g_open) break;
		lock.unlock();
		{
			std::lock_guard file(g_file_mutex);
			if (g_file != nullptr) FlushLocked();
		}
		lock.lock();
	}
}

} // namespace

bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_PAD_TRACE");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return on;
}

void Open() {
	if (!On()) return;
	std::lock_guard file(g_file_mutex);
	if (g_file != nullptr || !g_path.empty()) return;
	g_path = TracePath();
	g_file = std::fopen(g_path.c_str(), "w");
	if (g_file == nullptr) {
		std::printf("Pad trace: cannot write %s\n", g_path.c_str());
		std::fflush(stdout);
		return;
	}
	std::fputs("ms,flip,call,states,raw_lx,raw_ly,lx,ly,rx,ry,mag,mag_raw,buttons,circle,cross,l1,r1,l2,r2,circle_all,"
	           "l2_analog,r2_analog,marker,x,y,z\n",
	           g_file);
	g_writing.reserve(1024);
	{
		std::lock_guard lock(g_mutex);
		g_rows.reserve(1024);
		g_start = std::chrono::steady_clock::now();
		g_open  = true;
	}
	std::thread(WriterThread).detach();
	std::printf("Pad trace: writing %s\n", g_path.c_str());
	std::fflush(stdout);
}

void Close() {
	if (!On()) return;
	{
		std::lock_guard lock(g_mutex);
		if (!g_open) return;
		g_open = false;
	}
	g_wake.notify_all();
	// (Timed: an emergency shutdown may come while the writer is stopped in the middle of a write.)
	std::unique_lock file(g_file_mutex, std::chrono::seconds(2));
	if (!file.owns_lock() || g_file == nullptr) return;
	FlushLocked();
	std::fclose(g_file);
	g_file = nullptr;
	uint64_t dropped = 0;
	{
		std::lock_guard lock(g_mutex);
		dropped = g_dropped;
	}
	std::printf("Pad trace: %llu rows in %s", static_cast<unsigned long long>(g_written), g_path.c_str());
	if (dropped != 0) std::printf(" (%llu more left out: the writer fell behind)", static_cast<unsigned long long>(dropped));
	std::printf("\n");
	std::fflush(stdout);
}

void NoteAxis(int id, int sdl_axis, int value) {
	if (!On()) return;
	if (sdl_axis != SDL_CONTROLLER_AXIS_LEFTX && sdl_axis != SDL_CONTROLLER_AXIS_LEFTY) return;
	// (Another pad's event: its stick from now on, the other axis unknown until it moves.)
	if (g_raw_id.exchange(id, std::memory_order_relaxed) != id) {
		g_raw_x.store(INT_MIN, std::memory_order_relaxed);
		g_raw_y.store(INT_MIN, std::memory_order_relaxed);
	}
	(sdl_axis == SDL_CONTROLLER_AXIS_LEFTX ? g_raw_x : g_raw_y).store(value, std::memory_order_relaxed);
}

void Mark() {
	if (!On()) return;
	g_marker.store(true, std::memory_order_relaxed);
}

void NoteRead(const Libs::Controller::PadData* data, int num, int unread, int active_id, bool state_call) {
	if (!On() || data == nullptr || num < 1) return;
	const auto  now = std::chrono::steady_clock::now();
	Row         row;
	const auto& newest = data[num - 1];
	row.flip           = Libs::Graphics::FrameGen::GuestFlips();
	row.state_call     = state_call;
	row.states         = unread;
	if (g_raw_id.load(std::memory_order_relaxed) == active_id) {
		row.raw_x = g_raw_x.load(std::memory_order_relaxed);
		row.raw_y = g_raw_y.load(std::memory_order_relaxed);
	}
	row.lx          = newest.left_stick_x;
	row.ly          = newest.left_stick_y;
	row.rx          = newest.right_stick_x;
	row.ry          = newest.right_stick_y;
	row.l2          = newest.analog_buttons_l2;
	row.r2          = newest.analog_buttons_r2;
	row.buttons     = newest.buttons;
	row.buttons_all = ~0u;
	for (int i = 0; i < num; i++) row.buttons_all &= data[i].buttons;
	row.marker = g_marker.exchange(false, std::memory_order_relaxed);
	float yaw  = 0;
	row.placed = Loader::DemonsSoulsWarp::PlayerPlace(row.xyz, &yaw);

	std::lock_guard lock(g_mutex);
	if (!g_open) return;
	row.ms = std::chrono::duration<double, std::milli>(now - g_start).count();
	if (g_rows.size() >= RowsMax) {
		g_dropped++;
		return;
	}
	g_rows.push_back(row);
}

} // namespace PadTrace
