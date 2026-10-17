#ifndef EMULATOR_SRC_LOCAL_FRAME_GEN_UPSCALE_H_
#define EMULATOR_SRC_LOCAL_FRAME_GEN_UPSCALE_H_
// DLSS super resolution through NVIDIA Streamline (KYTY_UPSCALE=dlss, off by default), for Demon's Souls. Streamline is
// loaded as for DLSS-G (frame-gen.h, FrameGen::Initialize), with its DLSS plugin too, and also without frame
// generation: Streamline then stays the process' Vulkan loader, DLSS-G off.
#include "graphics/host_gpu/graphicContext.h"

namespace Libs::Graphics::FrameGen {

// KYTY_UPSCALE=dlss with DLSS available on the device (UpscaleOnDevice).
extern bool g_upscale;
[[nodiscard]] inline bool UpscaleOn() noexcept {
	return g_upscale;
}

// Process start (FrameGen::Initialize, before Streamline is loaded): the switches, read once.
void InitializeUpscale();
// DLSS is asked for: Streamline is loaded with its DLSS plugin, with or without frame generation.
[[nodiscard]] bool UpscaleWantsStreamline();
// After the device exists (after FrameGen::OnDevice): DLSS available on it or not, logged.
void UpscaleOnDevice(vk::PhysicalDevice physical_device);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_FRAME_GEN_UPSCALE_H_
