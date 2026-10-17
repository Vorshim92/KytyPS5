#include "frame-gen-upscale.h"

#include "frame-gen.h"

#include "common/logging/log.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

#ifdef KYTY_FRAMEGEN_STREAMLINE
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
// NVIDIA Streamline SDK headers (_Build/deps/streamline/sdk/include, see CMakeLists.txt).
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>
#include <sl_helpers.h>

#include <type_traits>
#endif

namespace Libs::Graphics::FrameGen {

bool g_upscale = false;

namespace {

bool g_dlss_wanted = false; // KYTY_UPSCALE=dlss

// A line in the log and on stdout (the run log).
// NOLINTNEXTLINE(cert-dcl50-cpp)
void Report(const char* format, ...) {
	char    line[512];
	va_list args;
	va_start(args, format);
	std::vsnprintf(line, sizeof(line), format, args);
	va_end(args);
	LOGF("%s", line);
	std::printf("%s", line);
	std::fflush(stdout);
}

#ifdef KYTY_FRAMEGEN_STREAMLINE
// Streamline's entry points, taken from the interposer FrameGen::Initialize loaded, and the DLSS plugin's functions.
struct Api {
	PFun_slIsFeatureSupported*     is_feature_supported  = nullptr;
	PFun_slIsFeatureLoaded*        is_feature_loaded     = nullptr;
	PFun_slGetFeatureFunction*     get_feature_function  = nullptr;
	PFun_slGetFeatureVersion*      get_feature_version   = nullptr;
	PFun_slGetNewFrameToken*       get_new_frame_token   = nullptr;
	PFun_slSetConstants*           set_constants         = nullptr;
	PFun_slSetTagForFrame*         set_tag_for_frame     = nullptr;
	PFun_slEvaluateFeature*        evaluate              = nullptr;
	PFun_slDLSSSetOptions*         dlss_set_options      = nullptr;
	PFun_slDLSSGetOptimalSettings* dlss_optimal_settings = nullptr;
	PFun_slDLSSGetState*           dlss_get_state        = nullptr;
};
Api g_api;

bool LoadApi() {
	HMODULE module = GetModuleHandleW(L"sl.interposer.dll");
	if (module == nullptr) {
		return false;
	}
	const auto load = [&](auto& function, const char* name) {
		function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
		    reinterpret_cast<void*>(GetProcAddress(module, name)));
		return function != nullptr;
	};
	const auto feature = [&](auto& function, const char* name) {
		void* pointer = nullptr;
		if (g_api.get_feature_function(sl::kFeatureDLSS, name, pointer) != sl::Result::eOk || pointer == nullptr) {
			return false;
		}
		function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(pointer);
		return true;
	};
	return load(g_api.is_feature_supported, "slIsFeatureSupported") &&
	       load(g_api.is_feature_loaded, "slIsFeatureLoaded") &&
	       load(g_api.get_feature_function, "slGetFeatureFunction") &&
	       load(g_api.get_feature_version, "slGetFeatureVersion") &&
	       load(g_api.get_new_frame_token, "slGetNewFrameToken") && load(g_api.set_constants, "slSetConstants") &&
	       load(g_api.set_tag_for_frame, "slSetTagForFrame") && load(g_api.evaluate, "slEvaluateFeature") &&
	       feature(g_api.dlss_set_options, "slDLSSSetOptions") &&
	       feature(g_api.dlss_optimal_settings, "slDLSSGetOptimalSettings") &&
	       feature(g_api.dlss_get_state, "slDLSSGetState");
}
#endif

} // namespace

void InitializeUpscale() {
	const char* upscale = std::getenv("KYTY_UPSCALE");
	g_dlss_wanted       = upscale != nullptr && std::string_view(upscale) == "dlss";
	if (upscale != nullptr && *upscale != '\0' && !g_dlss_wanted) {
		Report("Upscaling: KYTY_UPSCALE=%s is not supported (only dlss): off\n", upscale);
	}
}

bool UpscaleWantsStreamline() {
	return g_dlss_wanted;
}

void UpscaleOnDevice(vk::PhysicalDevice physical_device) {
	if (!g_dlss_wanted) {
		return;
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!StreamlineLoaded() || !LoadApi()) {
		Report("Upscaling: NVIDIA's Streamline runtime with its DLSS plugin did not load: DLSS super resolution off\n");
		return;
	}
	sl::AdapterInfo adapter {};
	adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
	bool       loaded        = false;
	const auto supported     = g_api.is_feature_supported(sl::kFeatureDLSS, adapter);
	if (supported != sl::Result::eOk || g_api.is_feature_loaded(sl::kFeatureDLSS, loaded) != sl::Result::eOk ||
	    !loaded) {
		Report("Upscaling: DLSS super resolution is not available on this device (%s): off\n",
		       supported != sl::Result::eOk ? sl::getResultAsStr(supported) : "its plugin is not loaded");
		return;
	}
	// (The NGX DLSS version decides the presets: L and M need 310.5 or newer.)
	sl::FeatureVersion version {};
	const bool         known = g_api.get_feature_version(sl::kFeatureDLSS, version) == sl::Result::eOk;
	g_upscale                = true;
	Report("Upscaling: DLSS super resolution ready (NGX DLSS %s)\n",
	       known ? version.versionNGX.toStr().c_str() : "of unknown version");
#else
	(void)physical_device;
	Report("Upscaling: this build has no Streamline support: DLSS super resolution off\n");
#endif
}

} // namespace Libs::Graphics::FrameGen
