#include "frame-gen-upscale.h"

#include "frame-gen-hudless.h"
#include "frame-gen-vk.h"
#include "frame-gen.h"
#include "speculation-state.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#endif

#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

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

// Streamline from two threads. With DLSS super resolution on, the present thread (Kyty.Present) drives DLSS-G: its
// options and state, the frame's token, tags, constants and markers, and vkQueuePresentKHR, in which Streamline makes
// the generated frames; the renderer thread (Kyty.Gpu) drives DLSS: its options and state, its own tokens, tags and
// constants, and slEvaluateFeature, recorded into the game's command buffer. Streamline documents slEvaluateFeature and
// the features' options and state calls (slDLSSSetOptions, slDLSSGetOptimalSettings, slDLSSGetState, slDLSSGSetOptions,
// slDLSSGGetState) as not thread safe, so those of both threads take one mutex (StreamlineGuard): none runs while
// another does. The frame tokens are taken under it from one counter (NextStreamlineFrame): each index is used once,
// and they reach slGetNewFrameToken in increasing order. (Streamline keys a frame's tags and constants by its index,
// and DLSS-G finds those of the frame it presents by its markers' index: counters of their own gave one index to a
// DLSS-G frame and to an upscaled one, the fork's design.) The mutex is never held across vkQueuePresentKHR or Reflex's
// sleep, which wait for the display (the renderer would wait with them), and the calls documented as thread safe (tags,
// constants, markers) need none. Without super resolution only the present thread calls Streamline: nothing is locked.

namespace Libs::Graphics::FrameGen {

bool g_upscale    = false;
bool g_taa_hooked = false;

namespace {

void Report(const char* format, ...) KYTY_FORMAT_PRINTF(1, 2);

// The switches (InitializeUpscale).
bool     g_dlss_wanted   = false; // KYTY_UPSCALE=dlss
uint32_t g_output_height = 1440;  // KYTY_UPSCALE_HEIGHT; 0: auto (the flip image's)
char     g_preset        = '\0';  // KYTY_DLSS_PRESET: J, K, L or M; none: NVIDIA's per mode
bool     g_dlss_alpha    = false; // KYTY_DLSS_ALPHA=1
float    g_mv_sign       = -1.0f; // KYTY_FRAMEGEN_MV_SIGN (as DLSS-G's)
bool     g_mv_pixels     = false; // KYTY_FRAMEGEN_MV=px (as DLSS-G's)

// Streamline from two threads (above).
bool                  g_streamline_shared = false; // DLSS super resolution on (set before the threads use Streamline)
std::mutex            g_streamline_mutex;
std::atomic<uint32_t> g_streamline_frames {0};

// Renderer thread (Kyty.Gpu).
// An image of ours with its one view.
struct OwnImage {
	VulkanImage   image;
	vk::ImageView view = nullptr;
};
// The TAA pass's depth and motion vectors, copied right after its dispatch (later passes may reuse its images before
// the composite), and the guest frame they are of (GuestFlips then).
struct Snapshot {
	std::unique_ptr<OwnImage> depth, motion;
	vk::Extent2D              extent {};
	uint64_t                  flip = UINT64_MAX;
};
Snapshot                  g_snapshot;
std::unique_ptr<OwnImage> g_output;                      // DLSS's output, which the composite samples
uint64_t                  g_composite_flip = UINT64_MAX; // the last frame whose composite was found

// Why a frame was not upscaled, each logged once.
enum class Refusal : uint32_t { Speculation, TaaImages, NoComposite, NoScene, SceneBinding, Smaller };
uint32_t g_refusals = 0;

bool FirstRefusal(Refusal refusal) {
	const uint32_t bit   = 1u << static_cast<uint32_t>(refusal);
	const bool     first = (g_refusals & bit) == 0;
	g_refusals |= bit;
	return first;
}

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

void Destroy(GraphicContext& graphics, OwnImage& image) {
	if (image.view != nullptr) {
		graphics.device.destroyImageView(image.view, nullptr);
		image.view = nullptr;
	}
	if (image.image.image != nullptr) {
		graphics.DeleteImage(image.image);
	}
}

// `slot` as an image of ours of that format, size and use: made anew when it is not one, the one it replaces deleted
// once the GPU is past the commands recorded so far (frames still in flight use it).
OwnImage& Reserve(CommandBuffer& command, std::unique_ptr<OwnImage>& slot, vk::Format format, vk::Extent2D extent,
                  vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect) {
	if (slot != nullptr && slot->image.format == format && slot->image.extent.width == extent.width &&
	    slot->image.extent.height == extent.height && slot->image.usage == usage) {
		return *slot;
	}
	auto& graphics = command.GetGraphics();
	if (slot != nullptr) {
		command.GetContext().GetCommandScheduler().DeferOperation(
		    [&graphics, retired = std::move(slot)]() mutable { Destroy(graphics, *retired); });
	}
	slot = std::make_unique<OwnImage>();
	CreateImage(graphics, slot->image, format, vk::Extent3D {extent.width, extent.height, 1}, usage, "Upscaling image");
	vk::ImageViewCreateInfo view {};
	view.image            = slot->image.image;
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = format;
	view.subresourceRange = {aspect, 0, 1, 0, 1};
	RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &slot->view), "upscaling image view");
	return *slot;
}

// The image's first level (`aspect` of it) into `copy` of its format and size, which is left for sampling; the image's
// barrier state put back.
void CopyInto(vk::CommandBuffer command, Image& image, OwnImage& copy, vk::ImageAspectFlags aspect) {
	const SavedState state(image);
	const auto       aspects = AspectsOf(copy.image.format);
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	Transition(command, copy.image, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	           vk::PipelineStageFlagBits2::eTransfer, aspects);
	vk::ImageCopy region {};
	region.srcSubresource = {aspect, 0, 0, 1};
	region.dstSubresource = {aspect, 0, 0, 1};
	region.extent         = copy.image.extent;
	command.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, copy.image.image,
	                  vk::ImageLayout::eTransferDstOptimal, region);
	Transition(command, copy.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
	           vk::PipelineStageFlagBits2::eAllCommands, aspects);
	state.Restore(image, command);
}

// The TAA pass's depth (binding 1) and motion vectors (3): docs/RE-DEMONS-SOULS.md.
void TakeSnapshot(CommandBuffer& command, std::span<const TextureBinding> images) {
	auto& cache  = command.GetContext().GetTextureCache();
	auto* depth  = images.size() > 3 ? cache.TryGetImage(images[1].image_id) : nullptr;
	auto* motion = images.size() > 3 ? cache.TryGetImage(images[3].image_id) : nullptr;
	if (depth == nullptr || motion == nullptr || depth->backing.extent.width != motion->backing.extent.width ||
	    depth->backing.extent.height != motion->backing.extent.height || motion->info.IsDepth()) {
		if (FirstRefusal(Refusal::TaaImages)) {
			Report("Upscaling: the TAA pass's depth and motion vectors are not the images expected: not upscaled\n");
		}
		return;
	}
	const vk::Extent2D extent {motion->backing.extent.width, motion->backing.extent.height};
	const auto         depth_aspect = depth->info.IsDepth() ? vk::ImageAspectFlags {vk::ImageAspectFlagBits::eDepth}
	                                                        : vk::ImageAspectFlags {vk::ImageAspectFlagBits::eColor};
	const auto         usage        = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	auto& depth_copy  = Reserve(command, g_snapshot.depth, depth->backing.format, extent, usage, depth_aspect);
	auto& motion_copy = Reserve(command, g_snapshot.motion, motion->backing.format, extent, usage,
	                            vk::ImageAspectFlagBits::eColor);
	// (The dispatch may be in a chain, with no barrier after it: the transitions from the images' states, which its
	// bindings set, wait for it.)
	const auto vk_command = command.Handle();
	CopyInto(vk_command, *depth, depth_copy, depth_aspect);
	CopyInto(vk_command, *motion, motion_copy, vk::ImageAspectFlagBits::eColor);
	g_snapshot.extent = extent;
	g_snapshot.flip   = GuestFlips();
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

const char* ModeName(sl::DLSSMode mode) {
	switch (mode) {
		case sl::DLSSMode::eDLAA: return "DLAA";
		case sl::DLSSMode::eMaxQuality: return "Quality";
		case sl::DLSSMode::eBalanced: return "Balanced";
		case sl::DLSSMode::eMaxPerformance: return "Performance";
		case sl::DLSSMode::eUltraPerformance: return "Ultra Performance";
		default: return "Off";
	}
}

sl::DLSSPreset Preset() {
	switch (g_preset) {
		case 'J': return sl::DLSSPreset::ePresetJ;
		case 'K': return sl::DLSSPreset::ePresetK;
		case 'L': return sl::DLSSPreset::ePresetL;
		case 'M': return sl::DLSSPreset::ePresetM;
		default: return sl::DLSSPreset::eDefault;
	}
}

// A DLSS viewport: Streamline keys its options, tags and constants by it (DLSS-G's is 0).
struct Viewport {
	uint32_t     id = 0;
	vk::Extent2D render {}, output {}; // the sizes of its options
	bool         ready    = false;     // ... which DLSS takes
	bool         changed  = false;     // ... since its last evaluate (its history starts again)
	bool         off      = false;     // its evaluates kept failing
	uint32_t     failures = 0;         // evaluates failed in a row
};
Viewport g_composite_port {1};
bool     g_reset         = true;       // DLSS's history starts again
uint64_t g_upscaled_flip = UINT64_MAX; // the last frame upscaled
uint64_t g_composites    = 0;          // composites found with the TAA pass's images
uint64_t g_upscaled      = 0;          // ... and upscaled

bool IsIdentity(const vk::ComponentMapping& mapping) {
	const auto is = [](vk::ComponentSwizzle swizzle, vk::ComponentSwizzle own) {
		return swizzle == vk::ComponentSwizzle::eIdentity || swizzle == own;
	};
	return is(mapping.r, vk::ComponentSwizzle::eR) && is(mapping.g, vk::ComponentSwizzle::eG) &&
	       is(mapping.b, vk::ComponentSwizzle::eB) && is(mapping.a, vk::ComponentSwizzle::eA);
}

// The DLSS mode for these sizes: of the modes NGX takes (it refuses Ultra Quality, 0xbad00010 at every evaluate: the
// fork's d467231d), the one whose render size range (slDLSSGetOptimalSettings) holds `render`, and of those the one
// whose optimal render height is nearest; eOff when none holds it. Without ranges from NGX, the fork's thresholds on
// the ratio. `settings`: the chosen mode's.
sl::DLSSMode ChooseMode(vk::Extent2D render, vk::Extent2D output, sl::DLSSOptimalSettings& settings) {
	constexpr std::array<sl::DLSSMode, 5> Modes {sl::DLSSMode::eDLAA, sl::DLSSMode::eMaxQuality,
	                                             sl::DLSSMode::eBalanced, sl::DLSSMode::eMaxPerformance,
	                                             sl::DLSSMode::eUltraPerformance};
	sl::DLSSMode chosen   = sl::DLSSMode::eOff;
	double       distance = 0.0;
	bool         ranged   = false;
	for (const auto mode: Modes) {
		sl::DLSSOptions options {};
		options.mode         = mode;
		options.outputWidth  = output.width;
		options.outputHeight = output.height;
		sl::DLSSOptimalSettings optimal {};
		if (g_api.dlss_optimal_settings(options, optimal) != sl::Result::eOk || optimal.optimalRenderHeight == 0 ||
		    optimal.renderHeightMax == 0) {
			continue;
		}
		ranged = true;
		if (render.width < optimal.renderWidthMin || render.width > optimal.renderWidthMax ||
		    render.height < optimal.renderHeightMin || render.height > optimal.renderHeightMax) {
			continue;
		}
		const double from = std::fabs(std::log(static_cast<double>(optimal.optimalRenderHeight) / render.height));
		if (chosen == sl::DLSSMode::eOff || from < distance) {
			chosen   = mode;
			distance = from;
			settings = optimal;
		}
	}
	if (ranged) {
		return chosen;
	}
	const double ratio = static_cast<double>(output.height) / render.height;
	return ratio < 1.05   ? sl::DLSSMode::eDLAA
	       : ratio < 1.6  ? sl::DLSSMode::eMaxQuality
	       : ratio < 1.85 ? sl::DLSSMode::eBalanced
	       : ratio < 2.5  ? sl::DLSSMode::eMaxPerformance
	                      : sl::DLSSMode::eUltraPerformance;
}

// `port`'s DLSS options for these sizes, set when they change. False: DLSS does not take them (logged once a size).
bool SetOptions(Viewport& port, vk::Extent2D render, vk::Extent2D output) {
	if (port.off) {
		return false;
	}
	if (port.render == render && port.output == output) {
		return port.ready;
	}
	port.render  = render;
	port.output  = output;
	port.ready   = false;
	port.changed = true;
	const StreamlineGuard   streamline;
	sl::DLSSOptimalSettings settings {};
	const auto              mode = ChooseMode(render, output, settings);
	if (mode == sl::DLSSMode::eOff) {
		Report("Upscaling: no DLSS mode takes %ux%u to %ux%u (outside NGX's render sizes): not upscaled\n",
		       render.width, render.height, output.width, output.height);
		return false;
	}
	sl::DLSSOptions options {};
	options.mode                  = mode;
	options.outputWidth           = output.width;
	options.outputHeight          = output.height;
	options.colorBuffersHDR       = sl::Boolean::eTrue; // (the scene's linear RGBA16F, before the composite's encoding)
	options.useAutoExposure       = sl::Boolean::eTrue;
	options.alphaUpscalingEnabled = g_dlss_alpha ? sl::Boolean::eTrue : sl::Boolean::eFalse;
	options.dlaaPreset = options.qualityPreset = options.balancedPreset = options.performancePreset =
	    options.ultraPerformancePreset = options.ultraQualityPreset = Preset();
	if (const auto result = g_api.dlss_set_options(sl::ViewportHandle {port.id}, options); result != sl::Result::eOk) {
		Report("Upscaling: slDLSSSetOptions failed (%s): not upscaled\n", sl::getResultAsStr(result));
		return false;
	}
	port.ready = true;
	char range[96] {"NGX gave no render size range"};
	if (settings.optimalRenderHeight != 0) {
		std::snprintf(range, sizeof(range), "NGX's best %ux%u, %ux%u to %ux%u", settings.optimalRenderWidth,
		              settings.optimalRenderHeight, settings.renderWidthMin, settings.renderHeightMin,
		              settings.renderWidthMax, settings.renderHeightMax);
	}
	Report("Upscaling: DLSS %ux%u -> %ux%u, %s mode (%s), preset %c%s\n", render.width, render.height, output.width,
	       output.height, ModeName(mode), range, g_preset != '\0' ? g_preset : '-',
	       g_preset != '\0' ? "" : " (NVIDIA's for the mode)");
	return true;
}

// What DLSS reads or writes: the image, its view and its layout then.
struct Resource {
	const VulkanImage* image  = nullptr;
	vk::ImageView      view   = nullptr;
	vk::ImageLayout    layout = vk::ImageLayout::eUndefined;
};

sl::Resource Tagged(const Resource& resource) {
	const auto&  image = *resource.image;
	sl::Resource tagged(sl::ResourceType::eTex2d, static_cast<VkImage>(image.image), nullptr,
	                    static_cast<VkImageView>(resource.view), static_cast<uint32_t>(resource.layout));
	tagged.width        = image.extent.width;
	tagged.height       = image.extent.height;
	tagged.nativeFormat = static_cast<uint32_t>(image.format);
	tagged.mipLevels    = image.mip_levels;
	tagged.arrayLayers  = image.layers;
	tagged.flags        = 0;
	tagged.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage));
	return tagged;
}

// DLSS's constants. It takes no camera matrices (DLSS-G's, frame-gen.cpp, are left identity here): the jitter, the
// motion vectors' scale, the depth's direction and the reset.
sl::Constants Constants(vk::Extent2D render, sl::float2 jitter, bool reset) {
	sl::float4x4 identity;
	for (uint32_t i = 0; i < 4; ++i) {
		identity[i] = sl::float4(i == 0, i == 1, i == 2, i == 3);
	}
	const float   width  = static_cast<float>(render.width);
	const float   height = static_cast<float>(render.height);
	sl::Constants constants {};
	constants.cameraViewToClip = identity;
	constants.clipToCameraView = identity;
	constants.clipToLensClip   = identity;
	constants.clipToPrevClip   = identity;
	constants.prevClipToClip   = identity;
	constants.jitterOffset     = jitter;
	// The game's vectors point along the motion, in UV units (its TAA reads the history at uv - mv): DLSS wants them
	// towards the previous position, so negated, as DLSS-G's.
	constants.mvecScale =
	    g_mv_pixels ? sl::float2(g_mv_sign / width, g_mv_sign / height) : sl::float2(g_mv_sign, g_mv_sign);
	constants.cameraPinholeOffset       = sl::float2(0.0f, 0.0f);
	constants.cameraPos                 = sl::float3(0, 0, 0);
	constants.cameraUp                  = sl::float3(0, 1, 0);
	constants.cameraRight               = sl::float3(1, 0, 0);
	constants.cameraFwd                 = sl::float3(0, 0, 1);
	constants.cameraNear                = 0.1f;
	constants.cameraFar                 = 10000.0f;
	constants.cameraFOV                 = 1.0f;
	constants.cameraAspectRatio         = width / height;
	constants.motionVectorsInvalidValue = sl::INVALID_FLOAT;
	constants.depthInverted             = sl::Boolean::eTrue; // (reversed Z)
	constants.cameraMotionIncluded      = sl::Boolean::eTrue;
	constants.motionVectors3D           = sl::Boolean::eFalse;
	constants.reset                     = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
	constants.orthographicProjection    = sl::Boolean::eFalse;
	constants.motionVectorsDilated      = sl::Boolean::eFalse;
	constants.motionVectorsJittered     = sl::Boolean::eFalse;
	return constants;
}

bool Failed(Viewport& port, const char* call, sl::Result result) {
	if (port.failures < 4) {
		Report("Upscaling: %s failed (%s)\n", call, sl::getResultAsStr(result));
	}
	// (A failed evaluate tries to make NGX's feature again, which is slow.)
	if (++port.failures == 30) {
		port.off = true;
		Report("Upscaling: DLSS keeps failing: off (the game scales its 3D image itself)\n");
	}
	return false;
}

// DLSS on `port` in `command`: `color` (the render size) into `output`, with the snapshot's depth and motion vectors.
bool Evaluate(Viewport& port, vk::CommandBuffer command, const Resource& color, const Resource& output,
              sl::float2 jitter, bool reset) {
	const sl::ViewportHandle viewport {port.id};
	const auto               read = vk::ImageLayout::eShaderReadOnlyOptimal;
	auto                     in_color  = Tagged(color);
	auto                     in_depth  = Tagged({&g_snapshot.depth->image, g_snapshot.depth->view, read});
	auto                     in_motion = Tagged({&g_snapshot.motion->image, g_snapshot.motion->view, read});
	auto                     out_color = Tagged(output);
	const sl::Extent render_extent {0, 0, port.render.width, port.render.height};
	const sl::Extent output_extent {0, 0, port.output.width, port.output.height};
	const sl::ResourceTag tags[] {
	    sl::ResourceTag(&in_color, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate,
	                    &render_extent),
	    sl::ResourceTag(&in_depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate, &render_extent),
	    sl::ResourceTag(&in_motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate,
	                    &render_extent),
	    sl::ResourceTag(&out_color, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate,
	                    &output_extent),
	};
	auto* cmd = reinterpret_cast<sl::CommandBuffer*>(static_cast<VkCommandBuffer>(command));
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	// Streamline records into the command buffer itself: what this thread queued so far goes first.
	LocalVulkanRecording::Drain();
#endif
	const StreamlineGuard streamline;
	uint32_t              index = NextStreamlineFrame();
	sl::FrameToken*       token = nullptr;
	if (const auto result = g_api.get_new_frame_token(token, &index); result != sl::Result::eOk || token == nullptr) {
		return Failed(port, "slGetNewFrameToken", result);
	}
	const auto constants = Constants(port.render, jitter, reset || port.changed);
	if (const auto result = g_api.set_constants(constants, *token, viewport); result != sl::Result::eOk) {
		return Failed(port, "slSetConstants", result);
	}
	if (const auto result =
	        g_api.set_tag_for_frame(*token, viewport, tags, static_cast<uint32_t>(std::size(tags)), cmd);
	    result != sl::Result::eOk) {
		return Failed(port, "slSetTagForFrame", result);
	}
	const sl::BaseStructure* inputs[] {&viewport};
	if (const auto result = g_api.evaluate(sl::kFeatureDLSS, *token, inputs, 1, cmd); result != sl::Result::eOk) {
		return Failed(port, "DLSS evaluate", result);
	}
	port.failures = 0;
	port.changed  = false;
	return true;
}

void LogVideoMemory(const Viewport& port) {
	sl::DLSSState         state {};
	const StreamlineGuard streamline;
	if (g_api.dlss_get_state(sl::ViewportHandle {port.id}, state) == sl::Result::eOk) {
		Report("Upscaling: DLSS uses about %" PRIu64 " MiB of video memory\n", state.estimatedVRAMUsageInBytes >> 20u);
	}
}
#endif

} // namespace

void InitializeUpscale() {
	const char* upscale = std::getenv("KYTY_UPSCALE");
	g_dlss_wanted       = upscale != nullptr && std::string_view(upscale) == "dlss";
	if (upscale != nullptr && *upscale != '\0' && !g_dlss_wanted) {
		Report("Upscaling: KYTY_UPSCALE=%s is not supported (only dlss): off\n", upscale);
	}
	if (!g_dlss_wanted) {
		return;
	}
	if (const char* height = std::getenv("KYTY_UPSCALE_HEIGHT"); height != nullptr && *height != '\0') {
		char*      end   = nullptr;
		const auto value = std::strtoul(height, &end, 10);
		if (std::string_view(height) == "auto") {
			g_output_height = 0;
		} else if (end != height) {
			g_output_height = std::clamp(static_cast<uint32_t>(value), 720u, 2160u);
		}
	}
	if (const char* preset = std::getenv("KYTY_DLSS_PRESET"); preset != nullptr && *preset != '\0') {
		const char letter = static_cast<char>(*preset & ~0x20); // (upper case)
		g_preset          = std::string_view("JKLM").find(letter) != std::string_view::npos ? letter : '\0';
		if (g_preset == '\0') {
			Report("Upscaling: KYTY_DLSS_PRESET=%s is not one of J, K, L, M: NVIDIA's presets\n", preset);
		}
	}
	const char* alpha = std::getenv("KYTY_DLSS_ALPHA");
	g_dlss_alpha      = alpha != nullptr && std::string_view(alpha) == "1";
	const char* sign  = std::getenv("KYTY_FRAMEGEN_MV_SIGN");
	g_mv_sign         = sign != nullptr && std::atof(sign) > 0 ? 1.0f : -1.0f;
	const char* mv    = std::getenv("KYTY_FRAMEGEN_MV");
	g_mv_pixels       = mv != nullptr && std::string_view(mv) == "px";
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
	g_taa_hooked             = true;
	g_streamline_shared      = true;
	char height[16] {"auto"};
	if (g_output_height != 0) {
		std::snprintf(height, sizeof(height), "%u", g_output_height);
	}
	Report("Upscaling: DLSS super resolution ready (NGX DLSS %s), output height %s\n",
	       known ? version.versionNGX.toStr().c_str() : "of unknown version", height);
#else
	(void)physical_device;
	Report("Upscaling: this build has no Streamline support: DLSS super resolution off\n");
#endif
}

void AfterTaaDispatch(CommandBuffer& command, std::span<const TextureBinding> images) {
	// (A speculation's thread dispatches nothing the normal way: not reached.)
	if (Spec::Current() != nullptr) {
		if (FirstRefusal(Refusal::Speculation)) {
			Report("Upscaling: a speculative translation reached the upscaler: refused\n");
		}
		return;
	}
	const uint64_t flip = GuestFlips();
	// The frame before ran the TAA pass, but none of its draws was found to be the composite.
	if (g_snapshot.flip != UINT64_MAX && g_snapshot.flip + 1 == flip && g_composite_flip != g_snapshot.flip &&
	    FirstRefusal(Refusal::NoComposite)) {
		Report("Upscaling: a frame with the TAA pass had no composite (a normal-path draw into a flip image sampling "
		       "its 3D image): not upscaled\n");
	}
	if (g_upscale) {
		TakeSnapshot(command, images);
	}
}

void UpscaleComposite(CommandBuffer& command, const RenderState& rendering, std::span<TextureBinding> images) {
	if (!g_upscale) {
		return;
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (Spec::Current() != nullptr) {
		if (FirstRefusal(Refusal::Speculation)) {
			Report("Upscaling: a speculative translation reached the upscaler: refused\n");
		}
		return;
	}
	// Without this frame's TAA pass (menus, movies, loading) the 3D image stays as it is: its history starts again.
	const uint64_t flip = GuestFlips();
	if (g_snapshot.flip != flip) {
		g_reset = true;
		return;
	}
	// The 3D image: of the images the draw samples, the RGBA16F one of the TAA pass's size, of one level (without one
	// the draw is not the composite: a later one into the flip image may be).
	const auto      render = g_snapshot.extent;
	auto&           cache  = command.GetContext().GetTextureCache();
	TextureBinding* scene  = nullptr;
	Image*          color  = nullptr;
	for (auto& binding: images) {
		auto* image = binding.image_view != nullptr && binding.mip_views.empty() ? cache.TryGetImage(binding.image_id)
		                                                                         : nullptr;
		if (image != nullptr && image->backing.format == vk::Format::eR16G16B16A16Sfloat &&
		    image->backing.extent.width == render.width && image->backing.extent.height == render.height) {
			scene = &binding;
			color = image;
			break;
		}
	}
	if (scene == nullptr) {
		if (FirstRefusal(Refusal::NoScene)) {
			Report("Upscaling: a draw into the flip image samples no %ux%u RGBA16F image (of %zu): not upscaled\n",
			       render.width, render.height, images.size());
		}
		return;
	}
	// This frame's composite, upscaled or not.
	NoteUpscaled();
	g_composite_flip = flip;
	++g_composites;
	// Its descriptor is written with the 3D image's layout (CommitBindings): shader read-only for a texture neither a
	// target nor kept general. Only then may the upscaled image, in that layout, take its place (its view as the 3D
	// image's, but for the size).
	const auto& view = scene->desc.view_info;
	if (scene->desc.type != TextureCache::BindingType::Texture || color->binding.is_target ||
	    color->binding.force_general || color->info.data.Empty() || view.format != color->backing.format ||
	    view.type != vk::ImageViewType::e2D || !IsIdentity(view.mapping)) {
		if (FirstRefusal(Refusal::SceneBinding)) {
			Report("Upscaling: the composite's 3D image is not bound as a plain texture: not upscaled\n");
		}
		g_reset = true;
		return;
	}
	const uint32_t height = g_output_height != 0 ? g_output_height : rendering.height;
	if (height < render.height) {
		if (FirstRefusal(Refusal::Smaller)) {
			Report("Upscaling: the output height %u is below the render height %u: not upscaled\n", height,
			       render.height);
		}
		g_reset = true;
		return;
	}
	// (At the 3D image's aspect, an even width.)
	const auto width =
	    static_cast<uint32_t>(std::lround(static_cast<double>(render.width) * height / render.height / 2.0)) * 2u;
	const vk::Extent2D output {width, height};
	if (!SetOptions(g_composite_port, render, output)) {
		g_reset = true;
		return;
	}
	auto&      target  = Reserve(command, g_output, color->backing.format, output,
	                             vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
	                             vk::ImageAspectFlagBits::eColor);
	const bool in_pass = command.RenderingWith(rendering);
	command.EndRendering();
	const auto vk_command = command.Handle();
	color->Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {}, vk_command);
	Transition(vk_command, target.image, vk::ImageLayout::eGeneral,
	           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
	           vk::PipelineStageFlagBits2::eComputeShader);
	// (After the game's TAA the 3D image is resolved: no jitter left in it.)
	const Resource in {&color->backing, scene->image_view, vk::ImageLayout::eShaderReadOnlyOptimal};
	const Resource out {&target.image, target.view, vk::ImageLayout::eGeneral};
	const bool     done = Evaluate(g_composite_port, vk_command, in, out, sl::float2(0.0f, 0.0f),
	                               g_reset || g_upscaled_flip + 1 != flip);
	// (DLSS bound pipelines and descriptors of its own: the draw binds its state again.)
	command.InvalidateGraphicsState();
	Transition(vk_command, target.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
	           vk::PipelineStageFlagBits2::eAllCommands);
	if (in_pass) {
		command.ResumeRendering(rendering);
	}
	if (!done) {
		g_reset = true;
		return;
	}
	scene->image_view = target.view;
	scene->layout     = vk::ImageLayout::eShaderReadOnlyOptimal;
	g_reset           = false;
	g_upscaled_flip   = flip;
	if (++g_upscaled == 1 || g_upscaled % 3600 == 0) {
		Report("Upscaling: %ux%u -> %ux%u (%" PRIu64 " frames; %" PRIu64 " composites not upscaled)\n", render.width,
		       render.height, output.width, output.height, g_upscaled, g_composites - g_upscaled);
		LogVideoMemory(g_composite_port);
	}
#else
	(void)command;
	(void)rendering;
	(void)images;
#endif
}

void DestroyUpscale(GraphicContext& graphics) {
	for (auto* image: {&g_snapshot.depth, &g_snapshot.motion, &g_output}) {
		if (*image != nullptr) {
			Destroy(graphics, **image);
			image->reset();
		}
	}
}

StreamlineGuard::StreamlineGuard() {
	if (g_streamline_shared) {
		g_streamline_mutex.lock();
		m_mutex = &g_streamline_mutex;
	}
}

void StreamlineGuard::Release() {
	if (m_mutex != nullptr) {
		std::exchange(m_mutex, nullptr)->unlock();
	}
}

uint32_t NextStreamlineFrame() {
	return g_streamline_frames.fetch_add(1, std::memory_order_relaxed) + 1;
}

} // namespace Libs::Graphics::FrameGen
