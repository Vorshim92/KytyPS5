#pragma once

#include "libatrac9.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace Libs::Audio::Ajm {

// ATRAC9's DualSense vibration channel layouts (the config's channel field 6: one channel, 7: two), which LibAtrac9
// lacks: it indexes its six-entry channel table with the 3-bit field, out of bounds, and InitFrame then corrupts the
// heap. Their blocks are those of an ordinary mono and dual mono stream, so the decoder gets those layouts, every other
// bit of the config unchanged (the guest's config stays as it is). Demon's Souls has no such stream (none in its 38,926
// AT9 files and 240 sound banks). KYTY_AT9_VIBRATION_AS_MONO=0: passed on as they are (the NGS2 waveform parser then
// refuses them, as before).
inline bool At9VibrationAsMono() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_AT9_VIBRATION_AS_MONO");
		return value == nullptr || std::string_view(value) != "0";
	}();
	return on;
}

// Atrac9InitDecoder with a vibration layout as its mono or dual mono one (At9VibrationAsMono).
inline int At9InitDecoder(void* handle, const uint8_t* config_data) {
	uint8_t config[ATRAC9_CONFIG_DATA_SIZE];
	std::memcpy(config, config_data, sizeof(config));
	if (At9VibrationAsMono() && ((config[1] >> 1u) & 7u) >= 6u) {
		config[1] = static_cast<uint8_t>(config[1] - (6u << 1u));
	}
	return Atrac9InitDecoder(handle, config);
}

} // namespace Libs::Audio::Ajm
