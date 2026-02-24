# ACEtricks

Utility ComfyUI nodes for ACE-Step (1.0 and 1.5) music models.

## Notes

- Some of this stuff is pretty niche/experimental, especially the LM sampling bits, and may be changed/break workflows.
- You may need a fairly recent Python version (3.12+ should be fine).
- This repo has no affiliation with the official ACE-Step project.

---

## Nodes

All nodes have the `ACETricks` prefix for easy searching.
See the node descriptions and input/output tooltips for more information.

#### Audio

- `AudioAsLatent` - Rearranges `AUDIO` to look like a `LATENT`. Can be useful for using stuff like latent-only blend nodes. For conversion in the other direction, see `LatentAsAudio`.
- `AudioBlend` - Applies a blend function to `AUDIO`.
- `AudioFromBatch` - Extracts items from batches of `AUDIO`.
- `AudioLevels` - Can be used to normalize the values in `AUDIO`.
- `LatentAsAudio` - See `AudioAsLatent`.
- `MonoToStereo` - Converts mono `AUDIO` to stereo. Safe to use if it's already.
- `SetAudioDtype`- Sets the dtype for `AUDIO` tensors.
- `WaveForm` - Generates a waveform image from `AUDIO`.

#### Conditioning

- `CondJoinLyrics` - Joins split lyrics/conditioning into `CONDITIONING`.
- `CondSplitOutLyrics` - Can split out lyrics (and some other metadata like audio codes) from `CONDITIONING`.

#### Latent

- `SilentLatent` - Can generate latents full of silence for ACE-Step 1.0 and 1.5 (if given a 1.5 latent as reference). Can be interesting for initial generations if you set denoise to something lower than 1.0 (or multiply the `SIGMAS`). You can also use ComfyUI's built-in `LatentMultiply` node to multiply by -1.0 and make stuff louder!
- `VisualizeLatent` - Can output an `IMAGE` representation of ACE-Step 1.0 and 1.5 latents. Extended version of the previewing in `ComfyUI-bleh` mentioned in the Integrations section below.

#### ACE-Step 1.0

- `EncodeLyrics` - ACE 1.0-specific node for encoding conditioning with extended features.
- `Mask` - Can be used to generate masks for 1.0 latents. Does not currently support 1.5.

#### ACE-Step 1.5

- `Ace15CompressDuplicateAudioCodes` - Can collapse sequences of duplicate audio codes in `CONDITIONING`.
- `Ace15LLMInference` - **(experimental)** Advanced node for doing LLM inference (primarily) with ACE-Step 1.5's LLMs - used for generating audio codes. It is configured with YAML, see the default YAML text for descriptions of parameters.
- `EmptyAce15LatentFromConditioning` - Creates an empty `LATENT` to match the time covered by the audio codes in `CONDITIONING`.
- `ModelPatchAce15Use4dLatent` - **(experimental)** Patches an ACE-Step 1.5 model with a wrapper to handle 4D latent inputs. This is a horrible hack and may not be compatible with everything, I use it personally for all my generations and it works pretty well currently. See the description for `SqueezeUnsqueezeLatentDimension` node which you will need to use as well.
- `RawTextEncodeAce15` - **(experimental)** Advanced node for encoding ACE-Step 1.5 `CONDITIONING` which allows you complete control of the exact text used for the following items: DiT prompt (actual sampling), lyrics, LLM positive, LLM negative and audio codes. It is configured with YAML, see the default YAML text for descriptions of parameters.
- `SqueezeUnsqueezeLatentDimension` - Mostly useful with ACE-Step 1.5 since it uses 3D latents while a lot of nodes expects 4+. Unsqueezing adds an empty dimension, squeezing removes an empty dimension. For ACE 1.5 - leave dimension on the default of `2`. Unsqueeze to add the dimension, toggle unsqueeze mode off to remove it. Normal usage would look something like: Empty latent node → unsqueeze → sampler → squeeze. **Important**: You will also need to patch the model to deal with 4D latents. See the `ModelPatchAce15Use4dLatent` node.
- `TextEncodeAce15` - Simpler conditioning node for ACE-Step 1.5. Has some extended features but is mostly superceded by the `RawTextEncodeAce15` and `Ace15LLMInference` nodes.

#### Misc

- `TimeOffset` - Can be used to calculate offsets into the latent time dimension for 1.0 and 1.5 latents.

---

## Integrations

These can integrate with some of my other projects:

- [ComfyUI-sonar](https://github.com/blepping/ComfyUI-sonar) - custom noise types.
- [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) - Many additional blend functions. Not precisely an integration but the `ComfyUI-bleh` node pack can also show you graphical previews while sampling.

## Usage Tips

Think these are all 1.5-specific. In no specific order:

- Being able to handle the DiT prompt and codes (or another way to think of it is the LM prompts) seperately is very powerful. It's possible to adjust lyrics (nothing too extreme) if you're having trouble with conformance, while leaving audio codes the same. You can get interesting effects by using one signature/time signature/BPM for audio codes and a different one when you sample.
- The `Ace15LLMInference` node integrates with exotic noise types from `ComfyUI-sonar` so if you want to do something like temperature sampling with noise from the Cauchy distribution, you can! It can also use blend modes from `ComfyUI-bleh`, and fun fact: CFG is just a blend mode.
- Use the `SqueezeUnsqueezeLatentDimension` and `ModelPatchAce15Use4dLatent` nodes to make ACE-Step 1.5 work with nodes that don't support 3D latents (I.E. `ComfyUI-sonar`).
