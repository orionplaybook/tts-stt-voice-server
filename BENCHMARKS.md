# Benchmarks

Measured 2026-11-02 on an Apple M5 Pro Mac mini with 64 GB unified memory (64 GiB = 68.7 GB;
figures below are GB = 10⁹ bytes), macOS 27.2, mlx 0.32.4, mlx-audio 0.6.3. Every model ran
through the service's own engine code, one model per process. Memory is the process's physical
footprint (what Activity Monitor calls Memory, GPU buffers included), sampled every 20 ms.
Reproduce, or benchmark new models, with `voicectl bench stt|tts|memory|report` (see README).

## Summary

| | Choice | Target | Measured |
|---|---|---|---|
| STT (`whisper-1`) | Whisper large-v3 turbo, 6-bit (MIT) | 10 s voice note < 1.5 s | **0.29–0.30 s** (Ogg/Opus, decode included) |
| TTS (`tts-1` and `tts-1-hd`) | Kokoro-82M (Apache-2.0), EN/FR; picked by ear in a blind test | first audio < 2 s (quality), < 1 s (fast) | **0.19 s** per sentence |
| TTS for other languages (German) | Chatterbox multilingual, 8-bit (MIT), loaded on demand | first audio < 2 s | **1.1–1.2 s** per sentence warm; ~3.6 s on first use (load) |
| Whole service | Whisper + Kokoro loaded and warm | headroom next to the local LLMs | **1.7 GB** resident, **4.1 GB** peak under load (+1.9 GB while Chatterbox is loaded) |

The original plan had two tiers, a quality model for `tts-1-hd` and Kokoro as the fast fallback.
In the blind listening test Kokoro (sample B) was preferred over every larger model, so it now
serves both names, and Chatterbox stays configured as the model for German and as an alternative
voice. Which model each OpenAI name means is one line of configuration.

## Speech to text

Test audio: Google FLEURS dev set (CC-BY-4.0, read speech, varied speakers), downloaded by
`bench/fetch_data.py`. WER over 40 utterances per language (about 4 minutes and 750 words each),
language given unless noted; text lower-cased and stripped of punctuation before scoring.
"10 s voice note" is a FLEURS utterance re-encoded as Ogg/Opus 48 kHz 32 kb/s, as Slack sends.
"60 s clip" joins 5–8 utterances from different speakers with short pauses. Latency is the median
of 3 runs and includes ffmpeg decoding.

| Model | Licence | WER EN | WER FR | WER DE | WER EN / FR, no language hint | 10 s voice note EN / FR | 60 s clip EN / FR: latency, WER | Peak memory |
|---|---|---|---|---|---|---|---|---|
| Whisper large-v3 turbo (fp16) | MIT | 3.3% | 4.2% | 3.8% | 3.3% / 4.2% | 0.42 s / 0.30 s | 3.0 s, 9.3% / 2.1 s, 8.4% | 3.1 GB |
| **Whisper large-v3 turbo (6-bit)**, default | MIT | 3.6% | 4.2% | | | 0.29 s (FR) | | 2.3 GB |
| Whisper large-v3 turbo (4-bit) | MIT | 3.6% | 4.1% | | | 0.29 s (FR) | | 2.2 GB |
| Parakeet TDT 0.6B v3 | CC-BY-4.0 | 3.7% | 3.6% | 4.2% | 3.7% / 3.6% | 0.10 s / 0.09 s | 0.37 s, 7.8% / 0.40 s, 8.4% | 4.7 GB |
| Nemotron 3.5 ASR streaming 0.6B | OpenMDW-1.1 (MLX copy says NVIDIA Open Model License) | 7.4% | 9.1% | 7.4% | 7.5% / 9.2% | 0.64 s / 0.57 s | 3.3 s, 13.2% / 3.4 s, 12.9% | 2.8 GB |
| Qwen3-ASR 1.7B (8-bit) | Apache-2.0 | **2.6%** | **3.0%** | **3.2%** | 2.6% / 3.2% | 0.63 s / 0.40 s | 2.9 s, **7.0%** / 4.1 s, **5.8%** | 5.1 GB |
| Qwen3-ASR 0.6B (8-bit) | Apache-2.0 | 3.1% | 5.6% | 5.4% | 3.1% / 5.7% | 0.19 s / 0.21 s | 0.6 s, 24.0% / 1.1 s, 9.7% | 3.6 GB |

Findings:

- **Every model meets the 1.5 s target for a 10 s voice note** by a wide margin.
- **Qwen3-ASR 1.7B is the most accurate** on all three languages, by about a fifth to a quarter
  relative to Whisper turbo (EN 2.6% vs 3.3–3.6%, FR 3.0% vs 4.2%), and clearly better on long
  audio: clearly better in both English and French. It ships as an option rather than the
  default because it needs 2.8 GB resident instead of 1.7 GB, and on short voice notes (the
  typical assistant request) both models make the same one or two word errors per clip. One
  config line makes it the default (see config.toml). It is configured on demand as `qwen3-asr`
  (loaded on first use, unloaded after 10 idle minutes), to compare on real recordings.
- **Whisper turbo on long audio** (its weak point): its own long-form decoding skipped whole
  sentences (21–34% WER on the 60 s clips). The service feeds it ~10 s windows cut at pauses,
  which brings that to 8–9% (figures in the table). Short clips are unaffected.
- **Quantizing Whisper costs nothing measurable**: 6-bit and 4-bit match fp16 within one word per
  750 and run at the same speed, for 0.65–0.8 GB less memory. 6-bit is the default (4-bit saves
  only 0.15 GB more).
- **Parakeet** is 3–4× faster and good in French, but takes no prompt, drops sentences on long
  audio unless chunked (the service chunks at 30 s), and uses more memory. A fine alternative if
  latency ever matters more than accuracy.
- **Nemotron 3.5 ASR** is clearly behind on this set.
- **No language hint** (auto-detect) costs nothing on 10 s+ clips. Some clients force a language
  by default (Hermes Agent sends `language=en`); see the README.
- Accents: FLEURS has varied but mostly native speakers per language; no accented-speaker set was
  available offline. Your own recordings are the real test.

Per-clip figures (latency, real-time factor, WER for every clip and model): `bench/results/stt.json`.

## Text to speech

Passages (`bench/passages.json`): one sentence and one ~85-word paragraph, in English, French and
German, written for this test. Timings go through the service's speech path: the text is split
into sentences, each synthesized in turn (and streamed within the sentence by models that can),
so "first audio" is what a client waits for. Warm models, one run each.

| Sample | Model | Licence | Languages | Native streaming | First audio, sentence EN / FR / DE | Paragraph EN: first audio / total | RTF | Peak memory (alone) | Resident |
|---|---|---|---|---|---|---|---|---|---|
| A | VoxCPM2 2B (bf16) | Apache-2.0 | 30 incl. EN/FR/DE | no | 5.9 / 6.5 / 7.8 s | 2.9 / 28.1 s | 1.2 | > 50 GB | – |
| B | **Kokoro-82M** (bf16), fast tier | Apache-2.0 | EN, FR (no DE) | no | **0.19 / 0.19 / –** | 0.10 / 0.84 s | 0.03 | 4.1 GB | 1.0 GB |
| C | Qwen3-TTS 1.7B CustomVoice (bf16) | Apache-2.0 | 10 incl. EN/FR/DE | yes | 0.23 / 0.22 / 0.24 s | 0.23 / 10.4 s | 0.40 | 6.1 GB | 3.8 GB (8-bit) |
| D | Chatterbox multilingual 0.5B (fp16) | MIT | 23 incl. EN/FR/DE | no | 1.74 / 1.85 / 2.05 s | 1.08 / 8.1 s | 0.42 | 6.0 GB | 3.8 GB |
| E | Chatterbox Turbo 350M (fp16) | MIT | EN only | yes | 0.22 / – / – | 0.23 / 10.4 s | 0.46 | 6.1 GB | – |
| F | **Chatterbox multilingual (8-bit)**, quality tier | MIT | 23 incl. EN/FR/DE | no | **1.24 / 1.13 / 1.20 s** | 0.70 / 5.0 s | 0.25 | 4.2 GB | 1.9 GB |

RTF = generation time / audio length (below 1 = faster than real time). "Peak memory (alone)"
is one model in its own process, Python libraries included; "Resident" is what the model adds
once loaded and warm, which is what adds up inside the service.

Intelligibility (Whisper transcribing each sample, WER against the script): every candidate is
3–6% on the paragraphs in all its languages, i.e. a word or two, mostly numbers written as
digits ("9h30" for "neuf heures et demie"). No candidate dropped or garbled sentences.
Chatterbox samples with temperature, so two renderings of the same text differ slightly.

Findings and trade-offs:

- **Quality tier, Chatterbox 8-bit (F)**: Western (Resemble AI), MIT, English/French/German, meets
  the 2 s target with margin, half the memory of fp16 and faster. It cannot stream within a
  sentence, so a long first sentence waits for the whole sentence (about 0.25 s per second of
  speech). Whether 8-bit sounds as good as fp16 (D) is for your ears: compare D and F.
- **Qwen3-TTS (C)** streams, so first audio is ~0.2 s whatever the sentence length, and it takes
  style instructions (`instructions`). Its built-in speakers are Chinese, English (two male),
  Japanese and Korean natives; no native French or English female voice (the one used here is
  "aiden", American male). Alibaba, like Qwen3-ASR. 3.8 GB resident at 8-bit, twice Chatterbox 8-bit.
- **Chatterbox Turbo (E)** streams and is quick to first audio, but English only.
- **VoxCPM2 (A)** has rich 48 kHz output but is slower than real time and its memory grew past
  50 GB on the first run, which pushed this Mac deep into swap. Dropped. Its samples stay in the
  listening test.
- **Kokoro (B)** is the obvious fast tier: 0.2 s, tiny, clear, but flatter prosody, one French
  voice (ff_siwis), no German.
- Not tested: Higgs Audio v2/v3 (v3 is research/non-commercial only), Voxtral TTS (CC-BY-NC,
  excluded), Phonon French (CC-BY-NC, gated), CSM-1B (English only, gated), ZONOS2 and
  KugelAudio 7B (no ready MLX build or needs a reference voice; KugelAudio is 7B).

## Listening test

Result (2026-11-02): **B (Kokoro) preferred**; A (VoxCPM2) clearly last.

`samples/` holds the same passages from every candidate under letters A–F, no model names:
`<letter>_<language>_<sentence|paragraph>.mp3`. Listen first, pick by ear, then open
`samples/KEY.md`. Kokoro (B) and Chatterbox Turbo (E) have no German; E has no French.
Chatterbox (D, E, F) and Kokoro use their built-in default voices; Qwen3-TTS uses its "aiden"
speaker; VoxCPM2 clones a voice designed once from a text description (`bench/voices/`), not a
real person.

## Memory alongside LM Studio

The whole service, measured from outside the server process while it handled voice notes, a 60 s
clip and English and French paragraphs (`bench/memory_bench.py`):

| | fp16 Whisper + Chatterbox + Kokoro, 1 GB MLX cache | Quantized, all three preloaded | **As configured**: Whisper 6-bit + Kokoro; Chatterbox on demand |
|---|---|---|---|
| Resident after start | 5.9 GB | 3.1 GB | **1.7 GB** |
| Peak under load | 9.2 GB | 5.5 GB | **4.1 GB** |
| Back after the load | 5.9 GB | 3.1 GB | **1.6 GB** |

While Chatterbox is loaded (German; it unloads 10 idle minutes after its last use), add 1.9 GB.

Example budget on a 64 GB Mac that also serves two local LLMs needing about 30 GB and 17 GB at
peak (a 26B and a 4B model): 47 + 4.1 = 51.1 GB at the voice service's peak (53 GB with
Chatterbox loaded), leaving 15.5–17.5 GB of the 68.7 GB for macOS, Docker Desktop's VM (~3 GB)
and everything else; 20 GB when the voice service is idle. Computed from separate measurements, not measured with the LLMs loaded; run
`bench/memory_bench.py --llm-gb 30 --llm-gb 17` while they are loaded for the real figure.

Memory controls in the service: `mlx_cache_limit_gb` (0.25 GB) and `release_after_request`
return GPU buffers to macOS after every request; `idle_unload_s` per model unloads a model after
an idle period if memory gets tighter (it then reloads in a few seconds on the next request).
