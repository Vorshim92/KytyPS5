#ifndef EMULATOR_SRC_LOCAL_FRAME_GEN_UPSCALE_H_
#define EMULATOR_SRC_LOCAL_FRAME_GEN_UPSCALE_H_
// DLSS super resolution through NVIDIA Streamline (KYTY_UPSCALE=dlss, off by default), for Demon's Souls. The game
// draws its final image into the flip image with one draw, the composite (found by frame-gen-hudless.h), that samples
// the 3D scene (RGBA16F at the render size, KYTY_RENDER_HEIGHT) and the HUD layer (the flip image's size). Just before
// that draw's descriptors are written, DLSS upscales the scene into an image of ours, with the depth and motion vectors
// the TAA pass read (copied right after its dispatch), and the composite's binding is pointed at it: the draw samples
// it by coordinates, so the bigger image takes the render-size one's place and the HUD layer stays as it is.
//   KYTY_UPSCALE_HEIGHT=<720..2160>|auto: the output's height (default 1440; auto: the flip image's), at the scene's
//     aspect. The DLSS mode is the one whose render size range holds the render size (slDLSSGetOptimalSettings).
//   KYTY_DLSS_PRESET=J|K|L|M: DLSS's preset in every mode (default: NVIDIA's per mode).
//   KYTY_DLSS_ALPHA=1: the alpha upscaled too (a check: by default only the color is).
// With the game's TAA kept, DLSS gets its resolved image, so without jitter. Frames without the TAA pass (menus,
// movies, loading) are composited as they are, and DLSS's history starts again after them.
//   KYTY_TAA_REPLACE=upscaler ("sin TAA"): the TAA runs, then its output's color is replaced by the raw jittered
//     frame's (taa-bypass.comp; its alpha, a counter and flags later passes read, kept), and the post-processing that
//     follows and DLSS work on that, DLSS with the camera's jitter (KYTY_UPSCALE_JITTER_SIGN=-1 the other way round).
//     The TAA scales its history by a game constant, so the raw frame is scaled to the TAA output's mean luminance:
//     KYTY_TAA_MATCH=0 not, KYTY_TAA_MATCH_SMOOTH=<w> this frame's weight in the gain (default 0.1),
//     KYTY_TAA_BYPASS_GAIN=<f> a fixed factor on top. (The brightness still differs from the TAA's in places.)
//   KYTY_TAA_REPLACE=dlss (with or without KYTY_UPSCALE): DLSS (DLAA, at the render size) in the TAA's place, on the
//     TAA dispatch's raw jittered color with the camera's jitter (KYTY_TAA_JITTER_SIGN=-1 the other way round), its
//     result written over the TAA output's color as sin TAA's raw frame is. No upscaling: the composite then scales
//     the 3D image itself, or the upscaler does.
//
// Streamline is loaded as for DLSS-G (frame-gen.h, FrameGen::Initialize), with its DLSS plugin too, and also without
// frame generation: it is then the process' Vulkan loader, DLSS-G off. DLSS evaluates in the game's command buffer on
// the renderer thread, DLSS-G works on the present thread: StreamlineGuard.
#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"

#include <cstdint>
#include <mutex>
#include <span>

namespace Libs::Graphics {
class CommandBuffer;
struct RenderState;
struct TextureBinding;
namespace FrameGen {

// KYTY_UPSCALE=dlss with DLSS available on the device (UpscaleOnDevice).
extern bool g_upscale;
// The TAA pass's images are wanted right after its dispatch (AfterTaaDispatch): the upscaler's, DLAA's.
extern bool g_taa_hooked;
[[nodiscard]] inline bool UpscaleOn() noexcept {
	return g_upscale;
}
[[nodiscard]] inline bool UpscaleCapturing() noexcept {
	return g_taa_hooked;
}

// Process start (FrameGen::Initialize, before Streamline is loaded): the switches, read once.
void InitializeUpscale();
// DLSS is asked for: Streamline is loaded with its DLSS plugin, with or without frame generation.
[[nodiscard]] bool UpscaleWantsStreamline();
// After the device exists (after FrameGen::OnDevice): DLSS available on it or not, logged.
void UpscaleOnDevice(vk::PhysicalDevice physical_device);

// Renderer thread (Kyty.Gpu), normal path, while UpscaleCapturing(): right after the TAA pass's dispatch is recorded,
// chained or not (`images`: its bindings, as its descriptors have them), its images copied, its output replaced.
void AfterTaaDispatch(CommandBuffer& command, std::span<const TextureBinding> images);
// Renderer thread, a draw FlipSlot found (frame-gen-hudless.h), with everything bound but its descriptors, which are
// written next (`rendering`: its pass, `images`: its pixel shader's): the first such draw of the frame that samples
// the 3D image has it upscaled and its binding's view pointed at the result. Nothing without the upscaler.
void UpscaleComposite(CommandBuffer& command, const RenderState& rendering, std::span<TextureBinding> images);
// At shutdown, the GPU idle: the upscaler's images.
void DestroyUpscale(GraphicContext& graphics);

// Streamline's calls that are not thread safe (frame-gen-upscale.cpp tells which and why), one at a time on the present
// and the renderer threads: held from construction to Release or destruction while DLSS super resolution is on, nothing
// otherwise (only the present thread calls Streamline then).
class StreamlineGuard {
public:
	StreamlineGuard();
	~StreamlineGuard() { Release(); }
	KYTY_CLASS_NO_COPY(StreamlineGuard);

	void Release();

private:
	std::mutex* m_mutex = nullptr;
};
// Under a StreamlineGuard: the index of a new Streamline frame token (slGetNewFrameToken). DLSS-G's frames and DLSS's
// take them from one counter, each index once, in increasing order.
[[nodiscard]] uint32_t NextStreamlineFrame();

} // namespace FrameGen
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_FRAME_GEN_UPSCALE_H_
