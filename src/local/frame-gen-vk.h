#ifndef EMULATOR_SRC_LOCAL_FRAME_GEN_VK_H_
#define EMULATOR_SRC_LOCAL_FRAME_GEN_VK_H_
// Vulkan helpers for the images of our own that the HUD-less capture (frame-gen-hudless.cpp) and the upscaler
// (frame-gen-upscale.cpp) keep: barriers from the state each keeps, allocation, and a texture cache image's barrier
// state saved and put back around copies of it. For their .cpp files only.
#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/image.h"

#include <vector>

namespace Libs::Graphics::FrameGen {

// Every aspect of `format`: a barrier on an image of depth and stencil covers both (their layouts are not separate).
inline vk::ImageAspectFlags AspectsOf(vk::Format format) {
	switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eX8D24UnormPack32:
		case vk::Format::eD32Sfloat: return vk::ImageAspectFlagBits::eDepth;
		case vk::Format::eS8Uint: return vk::ImageAspectFlagBits::eStencil;
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32SfloatS8Uint: return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		default: return vk::ImageAspectFlagBits::eColor;
	}
}

// `image` from the barrier state it keeps to `layout`, for `access` at `stage`.
inline void Transition(vk::CommandBuffer command, VulkanImage& image, vk::ImageLayout layout, vk::AccessFlags2 access,
                       vk::PipelineStageFlags2 stage, vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) {
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
	barrier.subresourceRange    = {aspect, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
}

// A 2D image of one level and layer (`what`: its name in the error that ends the emulator when it cannot be made).
inline void CreateImage(GraphicContext& graphics, VulkanImage& image, vk::Format format, vk::Extent3D extent,
                        vk::ImageUsageFlags usage, const char* what) {
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
		EXIT("%s: cannot allocate a %ux%u image\n", what, extent.width, extent.height);
	}
}

// A texture cache image's barrier state as it was, put back once its copies are done.
struct SavedState {
	explicit SavedState(const Image& image): whole(image.CurrentState()), parts(image.backing.subresource_states) {}
	void Restore(Image& image, vk::CommandBuffer command) const { image.RestoreState(whole, parts, command); }

	VulkanImageState              whole;
	std::vector<VulkanImageState> parts;
};

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_FRAME_GEN_VK_H_
