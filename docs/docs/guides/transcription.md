---
title: Transcription
sidebar_label: Transcription
description: "Turn speech into text, on your machine or through an OpenAI-compatible endpoint, for voice notes and recordings"
---

# Transcription

A `Transcriber` turns audio into text: a voice note, a recording, a clip from a microphone. It is the piece the voice features are built on, and it can be used on its own:

```python
from domyn.agents.audio import LocalTranscriber

transcriber = LocalTranscriber()
transcript = await transcriber.transcribe("meeting.m4a")

print(transcript.text)
for segment in transcript.segments:
    print(f"{segment.start:6.1f}s  {segment.text}")
```

There are two, chosen by where the speech model runs:

| type | runs | needs |
|---|---|---|
| `local` | on this machine; no audio leaves it | the `audio` extra |
| `openai` | on a server speaking OpenAI's `/v1/audio/transcriptions`: OpenAI, vLLM serving a speech model, a gateway | the server's URL, and a key if it asks for one |

## On your machine

```sh
pip install 'domyn[audio]'     # or: uv add 'domyn[audio]'
```

The extra installs [onnx-asr](https://github.com/istupakov/onnx-asr), which runs speech models with ONNX Runtime on the CPU. It is available on Python 3.11 to 3.14: ONNX Runtime has no release for 3.15 yet.

The default model is NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): 25 European languages, Italian and English among them, recognized without being told which, with punctuation. Before it, [Silero](https://github.com/snakers4/silero-vad) finds the pauses in the audio (voice activity detection): that is what lets an hour-long recording through, and what gives each segment its start and end.

**Nothing is downloaded until it is needed.** Installing the extra downloads no model, and creating a `LocalTranscriber` loads nothing. The first transcription downloads the model, about 670 MB, into `~/.domyn/models/`, once; later ones read it from there. To download it ahead, for example before working offline, call `prepare()`, which also tells you when there is something to download:

```python
await transcriber.prepare(on_download=print)
# Downloading the speech model nemo-parakeet-tdt-0.6b-v3 (about 670 MB) to /Users/you/.domyn/models: only this once.
```

A download goes into a temporary directory that is renamed into place when complete, so an interrupted one is started again next time rather than mistaken for a model.

The options:

| field | default | |
|---|---|---|
| `model` | `nemo-parakeet-tdt-0.6b-v3` | any model onnx-asr knows, or a Hugging Face repository, such as `onnx-community/whisper-large-v3-turbo` |
| `quantization` | `int8` | `None` for full precision: about 2.5 GB for the default model, for little difference in practice |
| `vad` | `true` | `false` transcribes the audio in one piece, without segments: only for short clips |
| `language` | detected | an ISO 639-1 code such as `it`. Parakeet detects the language itself and ignores it; Whisper and Canary models use it |
| `models_dir` | `~/.domyn/models` | where models are downloaded |

WAV files are read as they are. Any other format (mp3, m4a, Ogg/Opus voice notes, webm…) is converted first with [ffmpeg](https://ffmpeg.org/), which has to be installed (`brew install ffmpeg`, `apt install ffmpeg`); without it, those formats fail with an error that says so.

## Through an endpoint

```python
from domyn.agents.audio import OpenAICompatibleTranscriber

transcriber = OpenAICompatibleTranscriber(
    url="http://localhost:8000/v1/audio/transcriptions",
    model="openai/whisper-large-v3-turbo",
)
```

The audio is sent as it is, in any format the server accepts, so no ffmpeg is needed. The fields:

| field | default | |
|---|---|---|
| `url` | `https://api.openai.com/v1/audio/transcriptions` | the full transcriptions URL |
| `model` | `whisper-1` | the model name the server expects |
| `api_key` | none | sent as a Bearer token |
| `api_key_env` | none | an environment variable holding the key, read when a request is sent. With neither, no key is sent: a vLLM started without `--api-key` |
| `language` | detected | sent to the server when set |
| `response_format` | `verbose_json` | also returns the language, the duration and the segments; `json` for a server or model that only returns the text |
| `timeout` | `300` | seconds |

## What comes back

`transcribe()` takes a path or the audio's bytes (with `filename=`, whose extension says the format) and returns a `Transcript`:

| field | |
|---|---|
| `text` | the whole text |
| `segments` | `start`, `end` (seconds) and `text` of each stretch of speech; empty when the engine gives none |
| `language` | the language set, or the one the server detected |
| `duration` | the length of the audio, in seconds, when known |

Audio that cannot be transcribed raises `TranscriptionError`, with a message saying what to do: install the extra or ffmpeg, check the file, check the server. The audio and what was said in it are logged only at debug level.

## In configuration

A transcriber is declared by its `type`, with the fields above:

```python
from domyn.agents.audio import resolve_transcriber

transcriber = resolve_transcriber({"type": "openai", "url": "...", "api_key_env": "ASR_KEY"})
```

A transcriber of your own is a subclass of `Transcriber` with its own `type` and a `transcribe` method; it is found by that `type` like the built-in ones.

## Voice notes in channels

The Telegram, Slack and WhatsApp channels take a `transcriber` (an instance or its configuration). A voice note or audio file a user sends is downloaded, transcribed, and given to the agent as text right after the audio, marked as a transcription: `[Voice message, transcribed] …`. The audio stays attached. When there is no transcriber, or the audio cannot be transcribed, the text says so instead, so the agent can tell the user rather than answer as if nothing had been said.

```python
from domyn.agents.channels import TelegramChannel

channel = TelegramChannel(token=token, transcriber={"type": "local"})
```

With the local transcriber, the first voice note downloads the model, so that one answer takes longer.

## Licenses

onnx-asr is MIT. The Parakeet model is released by NVIDIA under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), which allows commercial use with attribution; Silero VAD is MIT.
