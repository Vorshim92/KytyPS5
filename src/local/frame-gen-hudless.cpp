#include "frame-gen-hudless.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::FrameGen {

bool g_hudless = false;

namespace {

// The game's flip images by guest address (two or three alternate; four kept): a draw into one is the composite's
// candidate. Written at the guest flip, read by any translating thread (IsFlipTarget).
std::array<std::atomic<uint64_t>, 4> g_flips {};
uint32_t                             g_flip_next = 0;
// This frame's composite was drawn without its HUD: later draws into a flip image are the game's alone.
std::atomic<bool> g_taken {false};

// Kyty.Gpu.
int                                    g_ui_override = -1;    // KYTY_FG_HUDLESS_UI
bool                                   g_ui_debug    = false; // KYTY_FG_HUDLESS_DEBUG
std::unordered_map<uint64_t, uint32_t> g_draws;               // this frame's normal-path draws per first target
// The composite without its HUD (the flip image's pixels, as the frame image has them) and the flip image it was
// drawn into; the flip image and the HUD texture as the composite found them.
std::unique_ptr<VulkanImage> g_capture;
uint64_t                     g_capture_flip = 0;
std::unique_ptr<VulkanImage> g_flip_backup, g_ui_backup;
uint32_t                     g_candidate_logs = 0;

// The [hudless] line, every 600 guest flips: the frames given the HUD-less image (of those with DLSS-G's inputs),
// the composite draws seen and why one gave none, the HUD texture taken last.
struct Stats {
	uint32_t     frames = 0, scene = 0, captured = 0, composites = 0, refused = 0, no_hud = 0;
	int          candidate  = -1;
	size_t       candidates = 0, binding = 0;
	vk::Extent3D extent {};
	vk::Format   format = vk::Format::eUndefined;
};
Stats g_stats;

// A sampled image of the composite that may be its HUD texture.
struct Candidate {
	size_t   binding = 0;
	Image*   image   = nullptr;
	uint32_t draws   = 0;     // this frame's normal-path draws into it
	bool     full    = false; // the flip image's size
};

uint64_t KeyOf(ImageId id) {
	return (uint64_t {id.index} << 32u) | id.generation;
}

bool IsFlipAddress(uint64_t address) {
	return address != 0 && std::any_of(g_flips.begin(), g_flips.end(), [&](const std::atomic<uint64_t>& flip) {
		       return flip.load(std::memory_order_relaxed) == address;
	       });
}

// The frame image's format (Presenter::PrepareFrame): the flip image's pixels, which a copy or a blit then keeps.
vk::Format UnormOf(vk::Format format) {
	switch (format) {
		case vk::Format::eR8G8B8A8Srgb: return vk::Format::eR8G8B8A8Unorm;
		case vk::Format::eB8G8R8A8Srgb: return vk::Format::eB8G8R8A8Unorm;
		default: return format;
	}
}

// An interface layer's usual formats: 8 or 10 bits a color, with alpha.
bool IsUiFormat(vk::Format format) {
	switch (format) {
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32: return true;
		default: return false;
	}
}

// A render target of one level and layer, single-sampled, that can be copied whole and back (and so cleared).
bool IsCopyable(const VulkanImage& image) {
	const vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
	                                  vk::ImageUsageFlagBits::eColorAttachment;
	return image.mip_levels == 1 && image.layers == 1 && image.samples == 1 && (image.usage & usage) == usage;
}

void Transition(vk::CommandBuffer command, VulkanImage& image, vk::ImageLayout layout, vk::AccessFlags2 access,
                vk::PipelineStageFlags2 stage) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = image.state.pl_stage;
	barrier.srcAccessMask       = image.state.access_mask;
	barrier.dstStageMask        = stage;
	barrier.dstAccessMask       = access;
	barrier.oldLayout           = image.state.layout;
	barrier.newLayout           = layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image.image;
	barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
}

void Create(GraphicContext& graphics, VulkanImage& image, vk::Format format, vk::Extent3D extent,
            vk::ImageUsageFlags usage) {
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = usage;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, image)) {
		EXIT("HUD-less image: cannot allocate a %ux%u image\n", extent.width, extent.height);
	}
}

// One of the capture's images, of that format and size: made anew when it is not one, the image it replaces deleted
// once the GPU is past the commands recorded so far (the renderer thread's, which used it).
VulkanImage& Reserve(CommandBuffer& command, std::unique_ptr<VulkanImage>& image, vk::Format format,
                     vk::Extent3D extent) {
	if (image != nullptr && image->format == format && image->extent == extent) {
		return *image;
	}
	auto& graphics = command.GetGraphics();
	if (image != nullptr) {
		command.GetContext().GetCommandScheduler().DeferOperation(
		    [&graphics, retired = std::move(image)]() mutable { graphics.DeleteImage(*retired); });
	}
	image = std::make_unique<VulkanImage>();
	Create(graphics, *image, format, extent,
	       vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
	return *image;
}

void Delete(GraphicContext& graphics, std::unique_ptr<VulkanImage>& image) {
	if (image != nullptr && image->image != nullptr) {
		graphics.DeleteImage(*image);
	}
	image.reset();
}

// A texture cache image's barrier state as it was, put back once its copies are done.
struct SavedState {
	explicit SavedState(const Image& image): whole(image.CurrentState()), parts(image.backing.subresource_states) {}
	void Restore(Image& image, vk::CommandBuffer command) const { image.RestoreState(whole, parts, command); }

	VulkanImageState              whole;
	std::vector<VulkanImageState> parts;
};

// The image into `copy`, which is left a transfer source.
void CopyOut(vk::CommandBuffer command, Image& image, VulkanImage& copy) {
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	Transition(command, copy, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	           vk::PipelineStageFlagBits2::eTransfer);
	vk::ImageCopy region {};
	region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.extent         = copy.extent;
	command.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, copy.image,
	                  vk::ImageLayout::eTransferDstOptimal, region);
	Transition(command, copy, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	           vk::PipelineStageFlagBits2::eTransfer);
}

// CopyOut's `copy` back into the image.
void CopyBack(vk::CommandBuffer command, const VulkanImage& copy, Image& image) {
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	vk::ImageCopy region {};
	region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.extent         = copy.extent;
	command.copyImage(copy.image, vk::ImageLayout::eTransferSrcOptimal, image.backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, region);
}

// The HUD texture: the composite's candidate KYTY_FG_HUDLESS_UI, else the guess (-1: none). Demon's Souls samples
// its scene at the render size (2560x1440 at 4K output) and its HUD layer at the flip image's (3840x2160, RGBA16F),
// drawn by the table path and so not counted: first the image of the flip image's size, then one drawn into on the
// normal path, in an interface format, with the most draws.
int ChooseHud(const std::vector<Candidate>& candidates) {
	if (g_ui_override >= 0) {
		return g_ui_override < static_cast<int>(candidates.size()) ? g_ui_override : -1;
	}
	const auto rank = [](const Candidate& candidate) {
		return std::tuple {candidate.full, candidate.draws > 0, IsUiFormat(candidate.image->backing.format),
		                   candidate.draws};
	};
	int chosen = -1;
	for (size_t i = 0; i < candidates.size(); ++i) {
		if ((candidates[i].full || candidates[i].draws > 0) &&
		    (chosen < 0 || rank(candidates[i]) > rank(candidates[chosen]))) {
			chosen = static_cast<int>(i);
		}
	}
	return chosen;
}

void LogCandidates(const Image& flip, size_t sampled, const std::vector<Candidate>& candidates, int chosen) {
	std::printf("HUD-less: the composite into %ux%u %s samples %zu images; candidates:", flip.backing.extent.width,
	            flip.backing.extent.height, vk::to_string(flip.backing.format).c_str(), sampled);
	for (size_t i = 0; i < candidates.size(); ++i) {
		const auto& image = candidates[i].image->backing;
		std::printf(" [%zu] binding %zu %ux%u %s draws %u%s", i, candidates[i].binding, image.extent.width,
		            image.extent.height, vk::to_string(image.format).c_str(), candidates[i].draws,
		            static_cast<int>(i) == chosen ? " (HUD)" : "");
	}
	std::printf(candidates.empty() ? " none\n" : "\n");
	std::fflush(stdout);
}

void LogStats() {
	const auto& stats = g_stats;
	char        hud[160] {"none yet"};
	if (stats.candidate >= 0) {
		std::snprintf(hud, sizeof(hud), "candidate %d of %zu, binding %zu, %ux%u %s", stats.candidate,
		              stats.candidates, stats.binding, stats.extent.width, stats.extent.height,
		              vk::to_string(stats.format).c_str());
	}
	char line[384];
	std::snprintf(line, sizeof(line),
	              "[hudless] %u frames: %u with the HUD-less image of %u with DLSS-G's inputs; composite draws %u "
	              "(pass refused %u, no HUD texture %u); HUD texture: %s\n",
	              stats.frames, stats.captured, stats.scene, stats.composites, stats.refused, stats.no_hud, hud);
	LOGF("%s", line);
	std::printf("%s", line);
	std::fflush(stdout);
}

} // namespace

void InitializeHudless() {
	const char* value = std::getenv("KYTY_FG_HUDLESS");
	g_hudless         = value != nullptr && std::string_view(value) == "1";
	if (!g_hudless) {
		return;
	}
	if (const char* ui = std::getenv("KYTY_FG_HUDLESS_UI"); ui != nullptr && *ui != '\0') {
		g_ui_override = std::max(std::atoi(ui), -1);
	}
	const char* debug = std::getenv("KYTY_FG_HUDLESS_DEBUG");
	g_ui_debug        = debug != nullptr && std::string_view(debug) == "1";
	const char* line  = g_ui_debug ? "Frame generation: the HUD-less image (KYTY_FG_HUDLESS=1), the HUD texture left "
	                                 "cleared on screen (KYTY_FG_HUDLESS_DEBUG=1)\n"
	                               : "Frame generation: the HUD-less image (KYTY_FG_HUDLESS=1)\n";
	LOGF("%s", line);
	std::printf("%s", line);
	std::fflush(stdout);
}

int FlipSlot(std::span<const RenderColorInfo> targets) {
	if (targets.empty()) {
		return -1;
	}
	++g_draws[KeyOf(targets[0].image_id)];
	if (g_taken.load(std::memory_order_relaxed)) {
		return -1;
	}
	for (size_t i = 0; i < targets.size(); ++i) {
		if (IsFlipAddress(targets[i].desc.info.data.address)) {
			return static_cast<int>(i);
		}
	}
	return -1;
}

bool IsFlipTarget(uint64_t address) {
	return !g_taken.load(std::memory_order_relaxed) && IsFlipAddress(address);
}

void BeforeFlipDraw(CommandBuffer& command, const RenderState& rendering, const RenderColorInfo& target, uint32_t slot,
                    std::span<const TextureBinding> images, const std::function<void()>& redraw) {
	auto& cache = command.GetContext().GetTextureCache();
	auto* flip  = cache.TryGetImage(target.image_id);
	++g_stats.composites;
	// Drawn twice, the pass may write nothing but the flip image, whose copy alone undoes the first draw.
	if (flip == nullptr || rendering.num_color_attachments != 1 || slot != 0 ||
	    rendering.depth_stencil_attachment.image_view != nullptr || !IsCopyable(flip->backing)) {
		++g_stats.refused;
		return;
	}
	const auto& extent = flip->backing.extent;
	// Of the images the composite samples, the color render targets at least half the flip image's size.
	std::vector<Candidate> candidates;
	for (size_t i = 0; i < images.size(); ++i) {
		auto* image = cache.TryGetImage(images[i].image_id);
		if (image == nullptr || image->backing.image == flip->backing.image ||
		    images[i].desc.type != TextureCache::BindingType::Texture || image->info.IsDepth() ||
		    !IsCopyable(image->backing) || image->backing.extent.width * 2 < extent.width ||
		    image->backing.extent.height * 2 < extent.height ||
		    std::any_of(candidates.begin(), candidates.end(), [&](const Candidate& c) { return c.image == image; })) {
			continue;
		}
		const auto draws = g_draws.find(KeyOf(images[i].image_id));
		candidates.push_back({i, image, draws != g_draws.end() ? draws->second : 0u,
		                      image->backing.extent.width == extent.width &&
		                          image->backing.extent.height == extent.height});
	}
	const int chosen = ChooseHud(candidates);
	if (g_candidate_logs < 4) {
		++g_candidate_logs;
		LogCandidates(*flip, images.size(), candidates, chosen);
	}
	if (chosen < 0) {
		++g_stats.no_hud;
		return;
	}
	auto& ui = *candidates[chosen].image;
	// The flip image as the draw finds it is needed again unless its pass begins by clearing it.
	const bool       in_pass    = command.RenderingWith(rendering);
	const bool       keep_flip  = in_pass || !rendering.color_attachments[slot].is_clear;
	const SavedState flip_state(*flip);
	const SavedState ui_state(ui);
	command.EndRendering();
	auto vk_command = command.Handle();
	// Both as they are now (KYTY_FG_HUDLESS_DEBUG: the HUD texture is left cleared, so the game's own draw shows what
	// was taken for the HUD: whatever vanishes from the screen is what DLSS-G gets without it).
	if (keep_flip) {
		CopyOut(vk_command, *flip, Reserve(command, g_flip_backup, flip->backing.format, extent));
	}
	if (!g_ui_debug) {
		CopyOut(vk_command, ui, Reserve(command, g_ui_backup, ui.backing.format, ui.backing.extent));
	}
	// No HUD: its texture transparent, back in the layout the composite samples it in, the flip image an attachment.
	ui.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, vk_command);
	vk::ClearColorValue transparent {};
	transparent.uint32 = std::array<uint32_t, 4> {0u, 0u, 0u, 0u};
	const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk_command.clearColorImage(ui.backing.image, vk::ImageLayout::eTransferDstOptimal, &transparent, 1, &whole);
	ui_state.Restore(ui, vk_command);
	if (keep_flip) {
		flip_state.Restore(*flip, vk_command);
	}
	// The draw without it, in its pass as the game's draw finds it (continued, or begun as the game's begins).
	if (in_pass) {
		command.ResumeRendering(rendering);
	} else {
		command.BeginRendering(rendering);
	}
	redraw();
	command.EndRendering();
	CopyOut(vk_command, *flip, Reserve(command, g_capture, UnormOf(flip->backing.format), extent));
	// Everything back, the pass again in progress for the game's draw if it was.
	if (keep_flip) {
		CopyBack(vk_command, *g_flip_backup, *flip);
	}
	if (!g_ui_debug) {
		CopyBack(vk_command, *g_ui_backup, ui);
	}
	flip_state.Restore(*flip, vk_command);
	ui_state.Restore(ui, vk_command);
	if (in_pass) {
		command.ResumeRendering(rendering);
	}
	g_capture_flip = flip->info.data.address;
	g_taken.store(true, std::memory_order_relaxed);
	g_stats.candidate  = chosen;
	g_stats.candidates = candidates.size();
	g_stats.binding    = candidates[chosen].binding;
	g_stats.extent     = ui.backing.extent;
	g_stats.format     = ui.backing.format;
}

void PrepareHudless(FrameInputs& inputs, const Image& flip) {
	// This flip image among the recent ones (a new one in the oldest's place).
	const uint64_t address = flip.info.data.address;
	if (!IsFlipAddress(address)) {
		g_flips[g_flip_next].store(address, std::memory_order_relaxed);
		g_flip_next = (g_flip_next + 1) % static_cast<uint32_t>(g_flips.size());
	}
	const bool taken = g_taken.exchange(false, std::memory_order_relaxed);
	g_draws.clear();
	// The composite drawn into this flip image goes with a frame DLSS-G generates from: the frame's image and the
	// capture's trade places (the frame's former one, idle since the frame was acquired, takes the next capture).
	inputs.has_hudless = taken && inputs.valid && g_capture != nullptr && g_capture_flip == address;
	if (inputs.has_hudless) {
		std::swap(inputs.hudless, g_capture);
	}
	++g_stats.frames;
	g_stats.scene += inputs.valid ? 1u : 0u;
	g_stats.captured += inputs.has_hudless ? 1u : 0u;
	if (g_stats.frames == 600) {
		LogStats();
		g_stats.frames = g_stats.scene = g_stats.captured = 0;
		g_stats.composites = g_stats.refused = g_stats.no_hud = 0;
	}
}

void DestroyHudlessInputs(GraphicContext& graphics, FrameInputs& inputs) {
	Delete(graphics, inputs.hudless);
	inputs.has_hudless = false;
}

void DestroyHudless(GraphicContext& graphics) {
	for (auto* image: {&g_capture, &g_flip_backup, &g_ui_backup}) {
		Delete(graphics, *image);
	}
}

} // namespace Libs::Graphics::FrameGen
