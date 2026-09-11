# Multi-Modal AI: Generative Images and Video

> **Feasibility annotations.** Every claim marked "verified" in this document was executed on
> 2026-09-11 against the actual `.env` key and the actual workstation. Reproduce with
> [multimodal_generative_feasibility.ipynb](multimodal_generative_feasibility.ipynb)
> (kernel `Python (genvis: diffusers)`).

---

## The one finding that changes the course design

The course draft assumes image generation is the reliable part and video is the fragile part —
Session 3 was given a fallback path for exactly that reason.

**With this OpenAI project, it is the other way around.**

| Capability | API status |
|---|---|
| `gpt-image-1` | `model_not_found` — project has no access |
| `dall-e-3`, `dall-e-2` | "model does not exist" |
| `sora-2` text-to-video | works |
| `sora-2` image-to-video (`input_reference`) | works |
| `gpt-4o-mini` vision understanding | works |

The project is limited to `gpt-4.1-mini`, `gpt-4o-mini`, `gpt-5-mini`, `text-embedding-3-small`.
Note that `sora-2` does **not** appear in `/v1/models` yet functions — the model list is not a
reliable capability indicator for the video API.

Consequence: **Sessions 1, 2 and 4 cannot run on the API.** They run locally on SDXL instead.
The fallback the draft reserved for Session 3 is what Sessions 1 and 2 actually need.

---

## Platform

| Component | Status |
|---|---|
| GPU | RTX 5070 Ti, 16.6 GB, sm_120 (Blackwell) |
| torch | 2.11.0+cu130 — supports sm_120 |
| Environment | venv `genvis` with `--system-site-packages`, Jupyter kernel registered |
| ffmpeg | 8.0.1 |
| Disk | 770 GB free |

`diffusers` must **not** be installed into the `vllm` conda env — it upgrades `huggingface_hub`
from 1.21.0 to 1.31.0 underneath `transformers` and `sentence-transformers`, which other courses
depend on. The `genvis` venv shares torch but isolates `huggingface_hub`.

### Measured local performance

| Operation | Model | Time | Peak VRAM |
|---|---|---|---|
| Text-to-image 1344x768 | SDXL-base fp16 | 7.0 s | 11.2 GB |
| img2img | SDXL-base (same weights) | 4.4 s | 11.2 GB |
| Outpainting | SDXL-inpaint-0.1 fp16 | 4.6 s | 8.5 GB |
| Image-to-video, 25 frames | SVD-XT fp16 + CPU offload | 83 s | 11.6 GB |
| Ken Burns zoom | ffmpeg only | < 1 s | — |

One-time downloads: SDXL 7 GB, SDXL-inpaint 6.9 GB, SVD 4.5 GB.
Model load from cache adds 70-80 s per pipeline switch.

---

## Course Positioning

This course is a second multimodal AI track with a clear focus on generative visual media.

It does not repeat the previous course on multimodal retrieval, speech pipelines, agents, and
multimodal RAG. Instead, it focuses on how generative AI systems create, edit, animate, and
package visual content.

The core progression is:

```
Prompt -> Image -> Edited Image -> Short Video -> Repeatable Visual Generation Workflow
```

## Course Goal

Students learn how to use multimodal generative AI to create and control image and video outputs
in a practical workflow.

By the end of the course, students should understand:

- how text prompts influence generated images
- how image generation differs from image understanding and retrieval
- how existing images can be edited, reformatted, or transformed
- why video generation is harder than image generation
- how motion, camera movement, and temporal consistency affect video outputs
- how to combine prompting, image generation, editing, and video assembly into a repeatable workflow
- why human review is essential for generative visual media

## Scope

**In scope:** text-to-image generation, prompt design for images, image editing and visual control,
basic image formatting for banners/thumbnails/slides, image-to-video concepts, short video clips and
motion prompts, simple video composition as a robust fallback, visual asset workflows, review and export.

**Out of scope:** speech-to-text, text-to-speech, voice agents, audio embeddings, audio retrieval,
advanced diffusion internals, LoRA / fine-tuning, ControlNet or complex ComfyUI workflows, deep legal
analysis of copyright, full cinematic video production, long-form video generation.

## Recommended Session Format

| Segment | Time |
|---|---|
| Conceptual introduction | 15-20 min |
| Practical demo | 30-35 min |
| Review and discussion | 5-10 min |

---

# Session 1 — Generate Images

## Core Question

How can we create a useful image from a text prompt?

## Main Idea

Text-to-image generation is not an exact design command. It is an iterative process where the prompt
guides the model toward a plausible visual output.

## Key Topics

### 1. Generative Vision vs. Vision Understanding

| Task | Example |
|---|---|
| Vision understanding | "What is visible in this image?" |
| Image retrieval | "Find similar images." |
| Image generation | "Create a new image." |
| Image editing | "Change this existing image." |

Key message: vision understanding analyzes existing images; generative vision creates new content.

### 2. Text-to-Image Basics

```
prompt -> model interpretation -> visual representation -> generated image -> review and iteration
```

The model does not create *the one correct image*. It creates a plausible visual interpretation.

### 3. Prompt Anatomy

```
subject + scene + style + composition + lighting + camera angle + aspect ratio + constraints
```

### 4. Visual Constraints

no readable text · no logos · no human faces · minimal background · wide composition ·
empty space on the left for title text · consistent color palette

> **Verified.** SDXL honours these through the *negative prompt*, not the main prompt. The negative
> prompt `text, letters, watermark, logo, human face, people, clutter, busy background` suppressed
> all of them in testing. Teach negative prompts as a first-class tool in this session — the course
> draft does not mention them, and without them the constraint list above does not work locally.

### 5. Iteration

```
generate -> inspect -> identify issues -> refine prompt -> generate variants -> select best result
```

### 6. Failure Modes

wrong style · cluttered composition · unreadable or incorrect text in the image · unwanted people
or faces · logos or brand-like elements · too many visual details · poor aspect ratio ·
visually attractive but not useful for the intended asset

## Technical Feasibility

| Path | Status |
|---|---|
| OpenAI API | **Not available.** No image model in the project. |
| Local SDXL | **Verified.** 7.0 s per 1344x768 image, 11.2 GB VRAM. |

**This is an improvement, not a workaround.** At 7 seconds and zero marginal cost, students can
iterate freely. The API route would cost ~$0.04 per image, which quietly discourages the third and
fourth attempt — in a session whose core message is *"good image generation is iterative, not
one-shot,"* free iteration is pedagogically better.

**Observed in testing:** the draft's own hero prompt, run verbatim, produced a washed-out grey image
with no video-frame motif and no usable left-hand title space. It demonstrated three of the failure
modes listed above on the first try. Consider using this as the deliberate starting point of the
demo rather than hoping for a clean first result.

**Throughput warning:** one GPU serves one student at a time. For a live cohort, either pre-generate
a set of starting images, or have students run the notebook on their own machines.

## Live Demo — Text-to-Image Prompting Lab

Generate a course hero image for *Multi-Modal AI: Generative Images and Video*.

1. Start with a weak prompt: `Generate an image about multimodal AI.`
2. Generate, review, name what fails.
3. Improve with style, composition, constraints — and a negative prompt.
4. Generate one or two variants, select, save.

Improved prompt:

```
Create a clean 16:9 hero image for a course called Multi-Modal AI: Generative Images and Video.
Show abstract connections between images, video frames, and neural network patterns. Use a modern
educational style, minimal composition, soft lighting, and leave empty space on the left for title
text. No readable text, no logos, no human faces.
```

## Student Takeaway

Text-to-image generation is controlled through prompt structure, constraints, iteration, and visual
evaluation.

---

# Session 2 — Edit Images

## Core Question

How can we improve and control an existing image?

## Main Idea

Image editing is often more useful than generating a new image from scratch. We rarely accept the
first generated image; we adapt it into a usable design asset.

## Key Topics

### 1. Generation vs. Editing

| Mode | Input | Output |
|---|---|---|
| Image generation | text prompt | new image |
| Image editing | image + instruction | modified image |
| Image variation | image + style direction | alternative version |
| Image expansion | image + target format | extended image |

### 2. Practical Editing Goals

simplify the background · remove clutter · create space for title text · adjust the aspect ratio ·
make the image more professional · change lighting or mood · improve composition · remove unwanted
text or logos

> **Correction — this list needs splitting.** Testing showed these goals divide into two groups that
> require *different techniques and different model weights*:
>
> | Goal | Technique | Extra download |
> |---|---|---|
> | Style, palette, mood, cleanup | img2img | none (same SDXL weights) |
> | Space for title text, aspect change, expansion | outpainting | inpaint checkpoint, 6.9 GB |
>
> **img2img changes style, not layout.** Run at `strength=0.45` on the Session 1 hero image, it
> correctly applied the requested blue/orange palette and simplified the background — and left the
> composition completely untouched. The "leave empty space on the left" goal was not met and cannot
> be met this way, at any strength: raising strength discards the source image rather than
> restructuring it.
>
> Teaching this distinction explicitly is worth more than the edit itself. It is the concrete,
> demonstrable version of the "Locality" criterion in the review table below.

### 3. Local Editing and Expansion

| Concept | Explanation |
|---|---|
| Inpainting | change one part of an image while keeping the rest stable |
| Outpainting | extend an image beyond its original borders |

> **Verified — and it produces a seam that cannot be prompted away.** Outpainting the left 40 %
> created the title space in 4.6 s, with a hard vertical edge at the mask boundary. Two fixes were
> tested and **both failed**:
>
> - *Feathered mask.* `StableDiffusionXLInpaintPipeline` binarizes the mask at 0.5; the Gaussian
>   blur is discarded before it reaches the model.
> - *Larger mask overlap* (160 px, 300 px). The seam stayed at exactly 40 % and vertical banding
>   artifacts appeared. The canvas has a hard brightness step at the paste boundary, and the VAE
>   encodes it.
>
> **What works is not generative.** A gradient scrim composited in PIL — background colour opaque at
> the left, falling off non-linearly to transparent at 58 % — gives a seamless title area in 0.48 s
> with no GPU. This is how designers solve it.
>
> Teach it in that order: attempt the outpaint, show the seam, try the obvious fixes, watch them
> fail, then reach for compositing. It is the most honest demonstration in the course of where
> generative tools stop and craft begins.

### 4. Format Control

| Format | Use case |
|---|---|
| 16:9 | slide hero, landing page banner |
| 1:1 | course card, social thumbnail |
| 9:16 | mobile story or short-form video |

Session 2 should focus mainly on the 16:9 banner use case.

### 5. Consistency Issues

style changes too much · key objects disappear · colors drift · composition becomes unstable ·
the model changes more than requested · the output no longer fits the intended use case

### 6. Image Review Criteria

| Criterion | Question |
|---|---|
| Prompt adherence | Did the model follow the instruction? |
| Locality | Did it change only what was requested? |
| Visual quality | Does the image look clean and professional? |
| Format fit | Does it work as a banner or slide visual? |
| Usability | Could we actually use this in course material? |
| Safety | Are there problematic or misleading elements? |

## Technical Feasibility

| Path | Status |
|---|---|
| OpenAI API (`/v1/images/edits`) | **Not available.** Same `gpt-image-1` block. |
| Local img2img | **Verified.** 4.4 s, no additional download. |
| Local outpainting | **Verified, seam persists.** 4.6 s, 8.5 GB VRAM, 6.9 GB one-time download. |
| Gradient scrim (PIL) | **Verified.** 0.48 s, no GPU, seamless — the actual solution. |

## Live Demo — Course Hero Image to 16:9 Banner

1. Load the Session 1 hero image.
2. Apply img2img for palette and cleanup — observe that layout does not change.
3. Apply outpainting to create the left title space.
4. Show the seam; demonstrate that feathering and wider overlap do not fix it.
5. Composite a gradient scrim instead — seamless, instant, deterministic.
6. Compare original and edited output, save.

## Student Takeaway

Generative vision workflows are not only about creating images. They are about turning rough
generations into usable visual assets — and different goals need different techniques.

---

# Session 3 — Create Short Video

## Core Question

What makes video generation harder than image generation, and how can we create a short visual
motion asset?

## Main Idea

Video introduces time. A generated video must not only look good in one frame; it must remain
visually coherent across frames.

## Key Topics

### 1. Why Video Is Harder Than Images

An image needs visual plausibility. A video additionally needs temporal consistency, stable objects,
coherent motion, controlled camera movement, consistent style over time, and smooth transitions.

Key message: video is not just many images. Video is images plus time, motion, and continuity.

### 2. Text-to-Video vs. Image-to-Video

| Mode | Input | Output | Demo suitability |
|---|---|---|---|
| Text-to-video | text prompt | generated video | conceptually useful, less predictable |
| Image-to-video | image + motion prompt | animated video | more controlled and demo-friendly |
| Video composition | images + transitions | MP4 asset | most reliable fallback |

### 3. Motion Prompting

```
slow zoom-in · subtle camera pan · gentle parallax · smooth transition ·
stable composition · no sudden cuts · minimal movement
```

> **This section depends on which path you use — and the split is sharp.**
>
> | Path | Motion control |
> |---|---|
> | `sora-2` (API) | natural-language motion prompts — verified working |
> | SVD (local) | **no text prompt at all**; motion via `motion_bucket_id` (integer) |
> | LTX-Video (local) | text prompts, but ~15-18 GB additional download |
>
> Stable Video Diffusion takes an image and a motion *strength*, not a motion *description*.
> "Slow cinematic zoom-in with subtle parallax" has no local equivalent under SVD; the nearest
> control is `motion_bucket_id=90` and `noise_aug_strength=0.02`.
>
> **Decision required.** Either teach motion prompting on the API path (costs money, works as the
> draft describes), or reframe this section locally as *parametric* motion control and drop the
> prompt vocabulary. Teaching motion-prompt language and then demoing an integer parameter would
> undercut the section. LTX-Video restores text control if the download is acceptable — untested here.

### 4. Temporal Consistency

Objects, style, and composition remain stable across time. Common failures: objects change shape,
details flicker, text appears and disappears, camera movement feels unstable, style changes mid-clip,
scene drifts from the prompt.

### 5. Robust Video Strategy

```
generated image -> simple motion plan -> zoom or pan -> short MP4 teaser
```

## Technical Feasibility

| Path | Status |
|---|---|
| `sora-2` API, text-to-video | **Verified.** 4 s clip, ~60 s wall clock, 1.7 MB MP4. |
| `sora-2` API, image-to-video | **Verified.** `input_reference` multipart upload, 1.9 MB MP4. |
| Local SVD-XT image-to-video | **Verified.** 25 frames in 83 s, 11.6 GB VRAM. |
| ffmpeg Ken Burns fallback | **Verified.** 150 frames, 5 s, 1280x720, sub-second. |

Input image must match target resolution: 1280x720 for `sora-2`, 1024x576 for SVD.

**Cost and throughput — the real constraint.** `sora-2` runs ~$0.40 and ~60 s per 4-second clip.
A cohort of 20 students at three attempts each is roughly $24 and, if concurrency is limited, far
more than the 30-35 minute demo window. Local SVD is free but takes 83 s per clip on a single shared
GPU, which is the same throughput problem without the invoice.

**Untested:** `sora-2` concurrency limits. This is the open question that decides whether Session 3
can be hands-on or must be instructor-demo-only. Measure it before scheduling the session.

## Live Demo — From Static Image to Short Motion Clip

1. Load the edited 16:9 banner.
2. Define a motion plan.
3. Generate the clip — API with a motion prompt, or local SVD with a motion bucket.
4. Fall back to ffmpeg zoom/pan if access or time is short.
5. Export MP4, review motion quality.

Motion prompt (API path only):

```
Animate this course hero image with a slow cinematic zoom-in. Add subtle motion to the abstract
neural network lines and video frame shapes. Keep the composition stable. No readable text, no
sudden camera movement, no scene change.
```

Key distinction for students: video *generation* creates new motion; video *composition* assembles
visual assets into a moving output. Both are useful; they are not the same thing.

## Student Takeaway

Generative video requires motion control and temporal consistency. For live demos, short controlled
clips beat ambitious cinematic scenes.

---

# Session 4 — Build a Visual Generation Workflow

## Core Question

How can we turn image generation, image editing, and short video creation into a repeatable visual
workflow?

## Main Idea

A useful generative visual application is not a single prompt. It is a controlled workflow that
starts with a brief, produces visual assets, reviews quality, and exports usable files.

## Key Topics

### 1. Creative Brief First

The workflow starts with a brief, not with a model call. A good brief includes topic, target
audience, message, visual style, output formats, constraints, and review criteria.

```
Course: Multi-Modal AI: Generative Images and Video
Audience: beginner to intermediate AI learners
Style: modern, clean, educational
Outputs: 16:9 hero image, 1:1 thumbnail, short teaser video
Constraints: no readable text, no logos, no human faces
```

### 2. Prompt Generation Pipeline

```
brief -> visual direction -> image prompt -> generated image -> edit prompt ->
formatted image assets -> motion prompt -> short video asset -> review notes
```

### 3. Multi-Format Visual Assets

| Output | Use case |
|---|---|
| `hero_16x9.png` | slide or landing page hero |
| `thumbnail_1x1.png` | course card or social preview |
| `teaser_video.mp4` | short motion asset |
| `prompts.json` | saved prompts and parameters |

> **Verified end-to-end.** The notebook produces all four from the brief, using the local video
> rather than the fallback. The 1:1 thumbnail is a centre-weighted crop of the 16:9 hero — note that
> this crops *away* the title space created in Session 2, which is correct for a thumbnail but worth
> saying out loud, since it shows that one asset's improvement can be another asset's loss.
>
> `prompts.json` should record model IDs and seeds, not just prompts. Without the seed, "regenerate
> the hero" is not reproducible. The notebook writes fixed seeds (42, 7, 11, 3) for exactly this reason.

### 4. Visual Consistency

similar style · similar colors · same visual theme · consistent level of detail · consistent mood ·
no conflicting visual messages

Deriving every asset from one generated source image, as the notebook does, gives this for free.
Generating each asset independently does not.

### 5. Human Review

```
generate -> review -> edit -> approve -> export
```

### 6. Evaluation

| Area | Question |
|---|---|
| Prompt fit | Does the output match the brief? |
| Visual quality | Does it look professional? |
| Format fit | Does each asset fit its intended format? |
| Consistency | Do the assets belong together? |
| Motion quality | Is the video stable and usable? |
| Usability | Could this be used in real course material? |
| Safety | Are there problematic, misleading, or copyrighted elements? |
| Cost and latency | Is the workflow practical? |

## Technical Feasibility

| Path | Status |
|---|---|
| Full local pipeline | **Verified.** Brief to four assets, no API calls. |
| API-based pipeline | **Not possible.** Image steps unavailable. |

End-to-end cost on a warm cache: ~7 s image + ~4 s img2img + ~5 s outpaint + ~83 s video ≈ 100 s of
compute, plus ~70-80 s per pipeline load when switching models. Budget roughly 6-7 minutes for a
cold full run.

## Live Demo — Generative Visual Kit Generator

```
outputs_genvis/
  kit_hero_16x9.png
  kit_thumbnail_1x1.png
  kit_teaser.mp4
  kit_prompts.json
```

## Student Takeaway

A generative visual application is a repeatable production workflow: brief, prompt, generate, edit,
animate, review, and export.

---

# Compact Course Overview

| Session | Title | Focus | Demo | Path |
|---|---|---|---|---|
| 1 | Generate Images | text-to-image, prompt anatomy, constraints | course hero image | local SDXL |
| 2 | Edit Images | editing, formatting, composition control | 16:9 banner | local SDXL + inpaint |
| 3 | Create Short Video | image-to-video, motion, consistency | teaser clip | API or local SVD |
| 4 | Build Visual Workflow | brief-to-assets pipeline, review, export | visual kit generator | fully local |

# Recommended Narrative

- Session 1: from prompt to image
- Session 2: from image to usable design asset
- Session 3: from static image to short motion clip
- Session 4: from one-off generation to repeatable visual workflow

# Key Teaching Decisions

**Keep the course visual.** Audio is not a topic here. (It is also unavailable on this key — all
audio models return 403.)

**Keep Session 3 robust.** Support both paths. Note that the reason has changed: not "video model
access may be unavailable" but "video is the only generative capability the API key actually has,
and it is the slowest and only paid step."

**Teach negative prompts in Session 1.** Not in the original draft; required for the constraint list
to work locally.

**Split the Session 2 editing goals.** Style edits and layout edits are different operations —
and layout space is a compositing problem, not a generative one.

**Decide the Session 3 motion story before teaching it.** Text-driven motion (API) and parametric
motion (local SVD) are genuinely different; pick one.

**Avoid overloading.** Diffusion architecture, training, LoRA, ControlNet, ComfyUI, long-form video,
character consistency, copyright law, audio generation — mention as optional next steps only.

# Open Items

1. **`sora-2` concurrency limits — untested.** Decides whether Session 3 is hands-on or demo-only.
2. **LTX-Video — untested.** ~15-18 GB; would restore text-driven motion prompts locally.
3. **`gpt-image-1` project access.** If it can be unlocked, Sessions 1, 2 and 4 gain an API path —
   but the local path is already faster and free, so this is optional rather than blocking.

# Final Course Description

Multi-Modal AI: Generative Images and Video is a four-session course that teaches students how to
create, control, and package visual media with generative AI. Students start by generating images
from structured prompts, then learn how to edit and format those images into usable design assets.
They then explore the basics of generative video and motion prompting before building a small visual
workflow that produces a hero image, thumbnail, and short teaser video from a course brief.

The course focuses on practical workflows, not deep model training. Students learn how to plan visual
outputs, write better prompts, evaluate generated media, handle failure modes, and turn one-off
generations into repeatable visual asset pipelines.
