# ComfyUI-Krea2Edit

Instruction-based image editing for **Krea 2** in ComfyUI — the node pack that powers
the **Krea 2 Identity Edit** LoRA. Turns Krea 2 (Raw or Turbo) into an image editor with dual
conditioning: the source image is injected both as VAE latent tokens (appearance) and
into the Qwen3-VL text encoder (semantic grounding), matching how the LoRA was trained.

☕ **[Support on Ko-fi](https://ko-fi.com/conradlocke)** — all tips go straight to GPU compute for future versions.

🧰 **Training code is public:** [krea2edit-trainer](https://github.com/lbouaraba/krea2edit-trainer) — the ai-toolkit extension these LoRAs were trained with, geometry-matched to the nodes, with measured consumer-GPU VRAM requirements.

## Model versions

**V2 (recommended)** ships as two LoRAs — **Krea 2 Identity Edit V2**
(`krea2_identity_edit_v2.safetensors`, restage a person from one photo) and **Krea 2 General
Edit V2** (`krea2_general_edit_v2.safetensors`, instruction-based editing). Use one per task.
The V2 nodes add **reference placement** (`ref_position` / `ref_position_a`). Older weights
keep working with these nodes. See [CHANGELOG.md](CHANGELOG.md).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jalberty2018/comfyui-krea2edit
# restart ComfyUI
```

Requirements: a ComfyUI version with native Krea 2 support, the Krea 2 model
(Raw or Turbo), the Qwen3-VL 4B text encoder used by Krea 2, and the Krea 2 Identity Edit
LoRAs (`krea2_identity_edit_v2.safetensors` and/or `krea2_general_edit_v2.safetensors`).
No extra Python dependencies.

## Nodes

### Krea2 Edit (grounded encode — Phrase Weights)

This fork adds an experimental phrase-weighted variant of the grounded encoder.
Replace **Krea2 Edit (grounded encode)** with **Krea2 Edit (grounded encode — Phrase Weights)**.
Keep the same CLIP, image, optional image_b, grounding_px and system_prompt inputs.
Connect its first output to the sampler's conditioning input.

Use **Krea2 Edit (source patch)** from this updated package in the MODEL path.
The weighted encoder's hooks are consumed by that source patch; a native Krea2
MODEL without this patch will not apply the phrase weights. No Enhancer install
is required. Existing source-patch and grounded-encoder node IDs remain unchanged.

Example instruction:

```text
Change the jacket to (dark red:2) and keep the (face unchanged:1.3).
```

Only the marked occurrence receives a weight. 1 is neutral, above 1 increases
relative attention odds, between 0 and 1 reduces them, and 0 applies a -80 score
bias. A weight is not a guarantee of proportional visual change or preservation.
Use separate, non-nested whole words/phrases in the instruction. System-prompt
weighting, raw chat markers and batched reference images are unsupported.

Phrase parsing follows the approach in
[ComfyUI-Krea2T-Enhancer](https://github.com/jalberty2018/ComfyUI-Krea2T-Enhancer).
The edit integration validates the exact token sequence and maps phrase rows
after visual-token expansion and native prefix removal. It supports zero, one
or two grounding images. The bias affects target-image queries to text keys;
source-image queries and the existing ref_boost/ref_boost_a biases are preserved.
Positive and negative conditioning can carry separate weights.

The optional phrase_report output lists the clean prompt, final token rows and
whether a hook was attached. It does not confirm sampler execution.
Weights of 1 do not attach a hook. Unannotated prompts use the original encoder path.

**Experimental:** model-free mapping and mask tests pass; GPU image editing,
CFG batching, control/attention patches and quantized setups need validation.
The additive mask uses square sequence memory, which can increase VRAM usage
with large references. Start without other attention patches.

For a fixed-seed test, compare the original encoder with plain text against the
new node with all weights 1. Then try one phrase at 2 and 5, keeping reference
strength and all other settings fixed. Repeat with two images if your workflow
uses image_b. CPU checks: `python tests/test_phrase_weights.py` (requires NumPy).

Parser attribution: copyright (c) 2026 capitan01R, MIT; see
[LICENSE-phrase-weights-MIT](LICENSE-phrase-weights-MIT). The original project
remains Apache-2.0.

### `Krea2EditModelPatch`
Wraps the diffusion model so the VAE-encoded source image is prepended as clean
in-context tokens (RoPE frame 1). Inputs:
- `model` — Krea 2 (LoRA already applied)
- `source_latent` — VAEEncode of the image being edited
- `source_latent_b` *(optional)* — second reference (RoPE frame 2) for two-input
  edits (e.g. person + scene)
- `vae` + `source_image` *(optional, recommended)* — the blur-proof pixel path: give
  the raw image (and VAE) and the node fits it to the target grid in pixel space.
  Required for `fit_mode: fit`.
- `target_latent` *(optional, recommended whenever you use the pixel path)* — wire the
  **same** latent you feed `KSampler.latent_image`. It only tells the node the output
  resolution ahead of time, so the source can be VAE-encoded here instead of on the first
  sampling step. Skipping it can cost real speed — see [Pixel path and VRAM](#pixel-path-and-vram).
- `fit_mode` *(default `fit`)* — how a source fits a mismatched output aspect ratio.
  `fit` = training-matched resample at a centered offset (v1.2); `crop` = center-crop,
  the v1/v1.1-legacy geometry (use with older weights).
- `ref_boost` *(default 1.0)* — reference-fidelity dial; >1 pulls harder toward the
  reference's appearance, <1 loosens. `ref_boost_a` is the same dial for the scene ref in two-ref edits.
- `ref_position` *(default `aligned (trained)`)* — where the **last** reference (the person,
  or the only image) sits in position space. `aligned` pins it onto the output layout (how the
  models were trained); `beside (right of target)` places it next to the canvas, so identity
  still flows but the layout pin is gone and the person can be restaged more freely;
  `bag of features` is experimental. `ref_position_a` is the same choice for the **first**
  reference (the scene). For putting a person into a scene: scene `aligned`, person `beside`.

### `Krea2EditGroundedEncode`
Image-grounded instruction encoding — the text encoder *sees* the image while
reading your instruction, exactly as during training. Inputs:
- `clip` — the Krea 2 CLIP (Qwen3-VL, loaded with `type: krea2`)
- `prompt` — the edit instruction ("recolor the car to matte black")
- `image` — the same source image
- `image_b` *(optional)* — second reference for two-input edits
- `grounding_px` — grounding resolution (default 768; v1.2 trained range
  384–768, and 1024+ often still works nicely). This is a quality dial: lower =
  stronger edit adherence, higher = stronger identity/likeness. Try 1024 for
  people, 512 for stubborn scene changes. (v1's trained range was 512–1536.)

**Both nodes are required.** With a stock `CLIPTextEncode` the model never sees the
image semantically and quality drops sharply, especially for scene-referential
instructions ("the man on the left").

## Minimal wiring

```
LoadImage ─┬─ VAEEncode ── Krea2EditModelPatch.source_latent
           └─ Krea2EditGroundedEncode.image     (+ your prompt)
UNETLoader ── LoraLoaderModelOnly (krea2_identity_edit_v1_2 @1.0) ── Krea2EditModelPatch.model
Krea2EditModelPatch ── KSampler.model
Krea2EditGroundedEncode ── KSampler.positive
Krea2EditGroundedEncode (empty prompt, same image) ── KSampler.negative
EmptySD3LatentImage ─┬─ KSampler.latent_image
                     └─ Krea2EditModelPatch.target_latent   (when using vae + source_image)
```

Example workflow in `workflows/`: `krea2_identity_edit.json` — single-image editor by
default; enable group 2 (toggle its Bypass off) for two-image person-into-scene edits.

## Usage notes (read these — they matter)

1. **Aspect ratio.** With `fit_mode: fit` (default in v1.2) and `vae` + `source_image`
   connected, mismatched source/output aspect ratios are handled — the source is
   resampled to the target grid. On `crop`/legacy weights, still match the AR: a
   mismatched AR is out of distribution and degrades identity/preservation.
2. **Turbo, 8 steps, CFG 1** is the fast path (~1 min at 2MP) and works for most
   edits: recolor, add/insert, attribute changes, restyles, scene translation.
3. **Removals and other "delete salient content" edits need real guidance:**
   use the **Raw** model at **CFG 3, ~20 steps**. Distilled Turbo at CFG 1 will
   usually re-render the subject instead of removing it.
4. At CFG > 1, ground the negative too: a second `Krea2EditGroundedEncode` with an
   empty prompt and the same image (this is the trained unconditional).
5. Two-input edits: scene image → `source_latent`/`image`, subject image →
   `source_latent_b`/`image_b`. Leave the b-inputs unconnected for single-image use.
6. **Generate at ≤2MP.** Above the trained range, source content can bleed into
   the output or subjects duplicate.
7. **Two distinct people:** place both references in a single pass (scene/subject A on
   the main inputs, subject B on the `_b` inputs) rather than adding them one at a time —
   simultaneous placement is currently more reliable than chaining separate edits. Face
   separation is still imperfect and a focus for future versions.
8. **Wire `target_latent` if you use `vae` + `source_image`** — see below.

## Pixel path and VRAM

The pixel path has to VAE-encode the source at the output resolution. Without
`target_latent` the node doesn't know that resolution until sampling starts, so the
encode runs on the first step — and `vae.encode` asks ComfyUI for VRAM at a moment when
the diffusion model is already resident and mid-run. ComfyUI frees room by partially
offloading whatever is loaded, the sampler included, and nothing loads it back (it loads
once, before its loop). The rest of the run then streams weights from CPU on every step.

Wiring `target_latent` moves the encode to node-execution time, restoring the normal
`VAEEncode → KSampler` order where the sampler evicts the VAE rather than the reverse.
The encode is cached either way, so this is purely about *when* it happens.

If you have VRAM headroom for the model and the VAE at once, nothing gets offloaded and
neither wiring costs you anything — which is why this only bites some setups. Wire it
anyway; it's free.

The console tells you which path you got:

```
[krea2edit] pre-encoding sources at target 128x128 (before sampling, fit_mode=fit)   <- good
[krea2edit] NOTE: connect 'target_latent' ...                                        <- encode will land mid-sampling
```

**Measure over 20+ steps, not 1.** The source still has to be VAE-encoded either way —
`target_latent` only changes *when*. What it removes is a per-step penalty, so it can
only show up across many steps. A 1-step run is almost entirely fixed overhead (model
load, text encode, VAE encode/decode) and will show no difference at all.

A third option is to skip the pixel path entirely: resize your source image to exactly
the output resolution and feed only `source_latent`. At matched resolution the latent
path does no resampling and produces the same reference geometry.

## License / credits

Nodes: Apache-2.0. The **Krea 2 Identity Edit** weights ship separately under the
Krea 2 Community License Agreement (see the model card, `LICENSE.pdf`, and `NOTICE`
in the weights repo).
Built on Krea 2 by Krea AI; text encoder Qwen3-VL (Alibaba).

## Contributors & thanks

This is a solo project, made a lot better by the community. Thank you to:

- **[stablellama](https://huggingface.co/stablellama)** — the MIT-licensed head/face/eye/person
  swap dataset behind those capabilities in v1.2.
- **[CeciliaXCIX](https://huggingface.co/CeciliaXCIX)** — tireless, high-quality community
  support in the discussions.
- **[akashzeno](https://github.com/akashzeno)** — node engineering: diagnosing the ComfyUI
  compatibility break and contributing the regression test.
- **[SubtleShader](https://huggingface.co/SubtleShader)** — testing the training code and
  consumer-GPU feedback.
- **Mark ([sogni.ai](https://sogni.ai))** — support, a GPU-fund donation, and getting the word out.
- **[ethanfel](https://github.com/ethanfel)** — root-caused the pixel path's VRAM interaction with
  the sampler and contributed the `target_latent` pre-encode (#15), tests included.

Want to help? Contributions of training data and node/code work are welcome, see the discussions.

## Scope and responsible use

Krea 2 Identity Edit is an identity-preserving character restaging model, trained only on
SFW data. It is not trained on any NSFW concepts, and I have no plans to add or support NSFW
data in current or future versions.

I do not endorse or support using this model to produce non-consensual, harmful, or sexual
imagery of real people, including deepfakes. Please use it responsibly and respect the
consent and likeness of anyone you depict.
