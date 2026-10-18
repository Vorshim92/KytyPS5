#ifndef EMULATOR_SRC_LOCAL_FRAME_GEN_HUDLESS_H_
#define EMULATOR_SRC_LOCAL_FRAME_GEN_HUDLESS_H_
// DLSS-G's HUD-less color (KYTY_FG_HUDLESS=1, off by default): the game image without its interface, so that the
// generated frames do not drag the HUD along with the scene behind it. Demon's Souls draws its final image into the
// flip image with one draw (the composite) that samples the 3D scene and the HUD layer from separate textures. On the
// renderer thread that draw is recorded once more just before it, with the HUD texture cleared to transparent; the
// result is copied aside, and the flip image and the HUD texture are put back as they were, so the game's own draw
// follows unchanged. The copy goes with the presented frame (FrameInputs::hudless); the present thread lays it out as
// the backbuffer (ScaleHudless), which BeforePresent tags as kBufferTypeHUDLessColor.
//
// The composite is found here for the upscaler too (KYTY_UPSCALE, frame-gen-upscale.h), which upscales its 3D image
// first: the HUD-less image then has the upscaled scene as well.
//
// KYTY_FG_HUDLESS_UI=n: the HUD texture is the composite's candidate n (the log lists them), not the guess.
// KYTY_FG_HUDLESS_DEBUG=1: the HUD texture is left cleared, so the screen shows what DLSS-G gets (the HUD gone).
#include "frame-gen.h"
#include "frame-gen-upscale.h"

#include <cstdint>
#include <functional>
#include <span>

namespace Libs::Graphics {
struct DrawRenderState;
struct RenderColorInfo;
struct RenderState;
struct TextureBinding;
namespace FrameGen {

// KYTY_FG_HUDLESS=1 with DLSS-G ready (InitializeHudless).
extern bool g_hudless;
[[nodiscard]] inline bool HudlessOn() noexcept {
	return g_hudless;
}
// The game's final composite is wanted: for the HUD-less image or for the upscaler.
[[nodiscard]] inline bool CompositeOn() noexcept {
	return g_hudless || UpscaleOn();
}
// ... and the current guest frame shows the game's 3D view (SceneNoted): its composite is looked for.
[[nodiscard]] inline bool CompositeCapturing() {
	return CompositeOn() && SceneNoted();
}

// OnDevice, once DLSS-G is ready: the switches, read once.
void InitializeHudless();

// Renderer thread (Kyty.Gpu), normal draw path, while CompositeCapturing(). FlipSlot: the draw counted for its first
// target (the HUD texture guess), and the slot of `targets` that is a recent flip image while this frame's composite
// is still wanted (-1: none): by the HUD-less image until it is drawn without its HUD, by the upscaler until it took it
// (NoteUpscaled). Such a draw is recorded directly and stored for neither the table nor the native XPR path.
// BeforeFlipDraw, with everything of the draw (`state`) bound and before its render pass begins: `redraw` (the draw)
// once more without the HUD texture (one of `images`), its result kept, then the flip image and the HUD texture as
// they were (nothing when the HUD-less image is off or this frame's is taken).
[[nodiscard]] int FlipSlot(std::span<const RenderColorInfo> targets);
void BeforeFlipDraw(CommandBuffer& command, const RenderState& rendering, const DrawRenderState& state, uint32_t slot,
                    std::span<const TextureBinding> images, const std::function<void()>& redraw);
// Renderer thread: the upscaler is done with this frame's composite (UpscaleComposite, upscaled or not).
void NoteUpscaled();
// Any translating thread (the table and native XPR paths), while CompositeCapturing(): `address` is a recent flip
// image's and this frame's composite is still wanted (the draw is left to the normal path).
[[nodiscard]] bool IsFlipTarget(uint64_t address);

// At the guest flip of `flip` (after PrepareInputs with DLSS-G), while CompositeOn(): the flip image noted, the next
// frame's composite wanted again; with the HUD-less image, this composite's goes with `inputs`.
void PrepareComposite(FrameInputs& inputs, const Image& flip);
// Present thread, in the present's command buffer after RecordPresentCommands: the frame's HUD-less image laid out as
// the backbuffer (`region`: where the game image is, black around it), for BeforePresent's tag.
void ScaleHudless(GraphicContext& graphics, vk::CommandBuffer command, FrameInputs& inputs, vk::Extent2D backbuffer,
                  vk::Rect2D region, vk::Format format);
// A frame's HUD-less images (DestroyInputs); at shutdown, the capture's own.
void DestroyHudlessInputs(GraphicContext& graphics, FrameInputs& inputs);
void DestroyHudless(GraphicContext& graphics);

} // namespace FrameGen
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_FRAME_GEN_HUDLESS_H_
