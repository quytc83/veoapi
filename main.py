import os
import time
from datetime import datetime
import base64
import tempfile
import shutil
import threading
import uuid
import requests
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from google import genai
from google.genai import types
import ffmpeg
import imghdr



# ---------- Config ----------
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)
MERGE_CHUNK_SIZE = 5
MERGE_DOWNLOAD_TIMEOUT = 600
MERGE_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=2)
MERGE_JOBS: Dict[str, Dict[str, Any]] = {}
MERGE_JOBS_LOCK = threading.Lock()

app = FastAPI(title="Veo3 Image-to-Video API (i2v+TTS)")
app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")


# ---------- Schemas ----------
class I2VBody(BaseModel):
    image_url: Optional[HttpUrl] = None          # 1) Ảnh từ URL public (Drive 'uc?export=view&id=...' OK)
    image_base64: Optional[str] = None           # 2) Hoặc ảnh base64 (JPEG/PNG)
    prompt_video: str                             # Mô tả chuyển động/cảnh quay (cinematic...)
    negative_prompt: Optional[str] = None
    duration_seconds: int = 8                     # 4/6/8 (8s cho 1080p và interpolation)
    aspect_ratio: str = "16:9"                    # "16:9" | "9:16"
    resolution: str = "720p"                      # "720p" | "1080p" (8s)
    model: str = "veo-3.1-fast-generate-preview"  # hoặc "veo-3.1-generate-preview" | "veo-3.0-generate-001"
    # i2v yêu cầu "allow_adult" (theo docs Veo 3.x)
    person_generation: str = "allow_adult"        # không đổi nếu là i2v

    # Voiceover (tuỳ chọn) – nếu set thì sẽ tạo TTS và ghép vào video, thay thế audio của Veo
    voiceover_text: Optional[str] = None
    tts_voice_name: Optional[str] = "Kore"        # tên voice TTS (xem danh sách prebuilt voices)
    tts_sample_rate: int = 24000                  # PCM 24kHz theo hướng dẫn

    gemini_api_key: Optional[str] = None          # cho phép truyền API key mỗi request


class AudioSlideshowBody(BaseModel):
    audio_url: HttpUrl
    img_urls: List[HttpUrl]
    seconds_per_image: float = 3.0
    frame_rate: int = 30


class VideoVoiceoverBody(BaseModel):
    video_url: HttpUrl
    audio_url: HttpUrl
    original_audio_volume: float = 0.2
    voiceover_volume: float = 1.0


# ---------- Helpers ----------
def _fetch_image_bytes(image_url: str) -> bytes:
    resp = requests.get(image_url, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(400, f"Cannot download image: HTTP {resp.status_code}")
    return resp.content

def _image_part_from_input(body: I2VBody) -> types.Part:
    # Ưu tiên base64 nếu có
    if body.image_base64:
        try:
            data = base64.b64decode(body.image_base64)
        except Exception:
            raise HTTPException(400, "image_base64 không hợp lệ (không decode được).")
        # đoán mime từ bytes khi là base64
        kind = imghdr.what(None, data)  # 'png', 'jpeg', ...
        mime = "image/png" if kind == "png" else "image/jpeg"
        return types.Part.from_bytes(data=data, mime_type=mime)

    # Hoặc tải từ URL
    if body.image_url:
        url = str(body.image_url)  # HttpUrl -> str
        try:
            resp = requests.get(url, timeout=60)
        except Exception as e:
            raise HTTPException(400, f"Không tải được image_url: {e}")
        if resp.status_code != 200 or not resp.content:
            raise HTTPException(400, f"Không tải được image_url (HTTP {resp.status_code}).")

        data = resp.content
        ct = (resp.headers.get("Content-Type") or "").lower()

        # Ưu tiên Content-Type; fallback sang nhận diện bytes
        if "png" in ct:
            mime = "image/png"
        elif "jpeg" in ct or "jpg" in ct:
            mime = "image/jpeg"
        else:
            kind = imghdr.what(None, data)  # 'png', 'jpeg', ...
            mime = "image/png" if kind == "png" else "image/jpeg"

        return types.Part.from_bytes(data=data, mime_type=mime)

    raise HTTPException(400, "Bạn phải truyền image_url hoặc image_base64")


def _get_genai_client(api_key_override: Optional[str]) -> genai.Client:
    api_key = api_key_override or DEFAULT_GEMINI_API_KEY
    if not api_key:
        raise HTTPException(400, "GEMINI_API_KEY is required (set env or pass gemini_api_key).")
    return genai.Client(api_key=api_key)


def _model_supports_resolution(model_name: Optional[str]) -> bool:
    """
    Veo 2.x không hỗ trợ tham số resolution, nếu truyền sẽ bị 400.
    Veo 3.x (veo-3.*) hỗ trợ nên vẫn giữ để khách chọn 720p/1080p.
    """
    if not model_name:
        return True
    return not model_name.startswith("veo-2.")


def _poll_operation(client: genai.Client, op, sleep_sec=10):
    # Long-running operation for Veo video generation
    while not op.done:
        time.sleep(sleep_sec)
        op = client.operations.get(op)
    return op

def _download_genai_file_to_bytes(client: genai.Client, file_ref) -> bytes:
    dl = client.files.download(file=file_ref)
    # Một số phiên bản trả bytes; một số trả stream-like
    if isinstance(dl, (bytes, bytearray)):
        return bytes(dl)
    if hasattr(dl, "read"):
        return dl.read()
    # fallback phòng hờ vài kiểu wrapper lạ
    try:
        return bytes(dl)
    except Exception:
        raise RuntimeError("Unexpected download type from client.files.download()")


def _save_bytes(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def _persist_local_copy(source_path: str, target_name: str, target_dir: Optional[str] = None) -> str:
    base_name = os.path.basename(target_name)
    dest_dir = target_dir or SCRIPT_DIR
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, base_name)
    shutil.copyfile(source_path, dest_path)
    return dest_path


def _parse_resolution(resolution: str) -> tuple[int, int]:
    try:
        width_str, height_str = resolution.lower().split("x")
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            raise ValueError
        return width, height
    except ValueError:
        raise HTTPException(400, "resolution phải có dạng WIDTHxHEIGHT, ví dụ 1280x720")


def _download_to_path(url: str, dest_path: str, timeout: int = 120):
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        raise HTTPException(400, f"Không tải được {url}: {e}")
    if resp.status_code != 200:
        raise HTTPException(400, f"Không tải được {url} (status={resp.status_code})")
    with open(dest_path, "wb") as f:
        f.write(resp.content)


def _audio_duration_seconds(audio_path: str, require_mp3_wav: bool = False) -> float:
    try:
        probe = ffmpeg.probe(audio_path)
    except FileNotFoundError:
        raise HTTPException(
            500,
            "Không tìm thấy ffprobe. Cài đặt ffmpeg/ffprobe (ví dụ: brew install ffmpeg) rồi chạy lại.",
        )
    except ffmpeg.Error as e:
        raise HTTPException(400, f"Không đọc được metadata audio: {e}")

    if require_mp3_wav:
        format_name = (probe.get("format", {}).get("format_name") or "").lower()
        if not any(fmt in format_name for fmt in ("mp3", "wav")):
            raise HTTPException(400, "Audio chỉ hỗ trợ đuôi mp3 hoặc wav.")

    duration_candidates = []
    if "format" in probe and "duration" in probe["format"]:
        duration_candidates.append(probe["format"]["duration"])
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio" and "duration" in stream:
            duration_candidates.append(stream["duration"])

    for duration_str in duration_candidates:
        try:
            duration = float(duration_str)
            if duration > 0:
                return duration
        except (TypeError, ValueError):
            continue

    raise HTTPException(400, "Không xác định được độ dài file audio.")


def _build_slideshow_cycle(image_paths: List[str], seconds_per_image: float, frame_rate: int,
                           resolution: str, dest_path: str):
    if not image_paths:
        raise HTTPException(400, "img_urls không được rỗng.")
    if seconds_per_image <= 0:
        raise HTTPException(400, "seconds_per_image phải > 0.")
    if frame_rate <= 0:
        raise HTTPException(400, "frame_rate phải > 0.")

    width, height = _parse_resolution(resolution)
    streams = []
    for img_path in image_paths:
        stream = (
            ffmpeg
            .input(img_path, loop=1, framerate=frame_rate, t=str(seconds_per_image))
            .filter("scale", width, height, force_original_aspect_ratio="decrease")
            .filter("pad", width, height, "(ow-iw)/2", "(oh-ih)/2", color="black")
            .filter("setsar", "1")
        )
        streams.append(stream)

    concatenated = ffmpeg.concat(*streams, v=1, a=0)
    (
        ffmpeg
        .output(concatenated, dest_path, r=frame_rate, pix_fmt="yuv420p", vcodec="libx264")
        .overwrite_output()
        .run(quiet=True)
    )

def _aspect_ratio_from_resolution(resolution: str) -> str:
    width, height = _parse_resolution(resolution)
    if width * 9 == height * 16:
        return "16:9"
    if width * 16 == height * 9:
        return "9:16"
    raise HTTPException(400, "resolution phải theo tỷ lệ 16:9 hoặc 9:16.")


def _concat_videos(input_paths: List[str], output_path: str):
    if not input_paths:
        raise HTTPException(400, "Không có video để merge.")
    if len(input_paths) == 1:
        shutil.copyfile(input_paths[0], output_path)
        return

    list_path = f"{output_path}.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in input_paths:
            f.write(f"file '{p}'\n")

    (
        ffmpeg
        .input(list_path, f="concat", safe=0)
        .output(output_path, vcodec="libx264", acodec="aac", audio_bitrate="192k")
        .overwrite_output()
        .run(quiet=True)
    )
    os.remove(list_path)


def _prepare_clip_for_merge(src_path: str, dest_path: str):
    duration, has_audio, _ = _video_metadata(src_path)
    if duration <= 0:
        raise HTTPException(400, f"Video {os.path.basename(src_path)} không có độ dài hợp lệ.")

    inp = ffmpeg.input(src_path)
    video_stream = (
        inp.video
        .filter("trim", duration=duration)
        .filter("setpts", "PTS-STARTPTS")
    )

    if has_audio:
        audio_stream = (
            inp.audio
            .filter("apad")
            .filter("atrim", duration=duration)
            .filter("asetpts", "PTS-STARTPTS")
        )
    else:
        audio_stream = (
            ffmpeg
            .input("anullsrc=r=48000:cl=stereo", f="lavfi")
            .filter("atrim", duration=duration)
            .filter("asetpts", "PTS-STARTPTS")
        )

    (
        ffmpeg
        .output(
            video_stream,
            audio_stream,
            dest_path,
            vcodec="libx264",
            acodec="aac",
            audio_bitrate="192k",
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run(quiet=True)
    )


def _find_existing_media_path(url_str: str) -> Optional[str]:
    parsed = urlparse(url_str)
    base_name = os.path.basename(parsed.path)
    if not base_name:
        return None
    candidate = os.path.join(VIDEOS_DIR, base_name)
    if os.path.isfile(candidate):
        return candidate
    return None


def _video_metadata(video_path: str) -> tuple[float, bool, Optional[float]]:
    try:
        probe = ffmpeg.probe(video_path)
    except FileNotFoundError:
        raise HTTPException(
            500,
            "Không tìm thấy ffprobe. Cài đặt ffmpeg/ffprobe (ví dụ: brew install ffmpeg) rồi chạy lại.",
        )
    except ffmpeg.Error as e:
        raise HTTPException(400, f"Không đọc được metadata video: {e}")

    has_audio = False

    duration_candidates = []
    audio_duration_candidates = []
    if "format" in probe and "duration" in probe["format"]:
        duration_candidates.append(probe["format"]["duration"])
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            has_audio = True
            if "duration" in stream:
                audio_duration_candidates.append(stream["duration"])
        if "duration" in stream:
            duration_candidates.append(stream["duration"])

    duration = None
    for duration_str in duration_candidates:
        try:
            duration = float(duration_str)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    if duration is None or duration <= 0:
        raise HTTPException(400, "Không xác định được độ dài video.")

    audio_duration = None
    for duration_str in audio_duration_candidates:
        try:
            candidate = float(duration_str)
            if candidate > 0:
                if audio_duration is None or candidate > audio_duration:
                    audio_duration = candidate
        except (TypeError, ValueError):
            continue

    return duration, has_audio, audio_duration


def _apply_atempo_chain(stream, factor: float):
    if factor <= 0:
        raise HTTPException(400, "atempo factor phải > 0.")
    if abs(factor - 1.0) < 1e-6:
        return stream

    factors = []
    remaining = factor
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)

    for f in factors:
        if abs(f - 1.0) < 1e-6:
            continue
        stream = stream.filter("atempo", f)
    return stream


def _apply_transition_effect(src_path: str, dest_path: str, fade_duration: float,
                             is_first: bool, is_last: bool, clip_duration: float,
                             has_audio: bool, apply_blur: bool):
    if fade_duration <= 0 and not apply_blur:
        shutil.copyfile(src_path, dest_path)
        return

    safe_fade = min(fade_duration, max(clip_duration / 2.0 - 0.01, 0))
    if fade_duration > 0 and safe_fade <= 0 and not apply_blur:
        shutil.copyfile(src_path, dest_path)
        return

    inp = ffmpeg.input(src_path)
    video_stream = inp.video
    if fade_duration > 0 and safe_fade > 0:
        if not is_first:
            video_stream = video_stream.filter("fade", type="in", start_time=0, duration=safe_fade)
        if not is_last:
            start = max(clip_duration - safe_fade, 0)
            video_stream = video_stream.filter("fade", type="out", start_time=start, duration=safe_fade)

    if apply_blur:
        video_stream = video_stream.filter("tblend", all_mode="average").filter("tblend", all_mode="average")

    if has_audio:
        audio_stream = inp.audio
        if fade_duration > 0 and safe_fade > 0:
            if not is_first:
                audio_stream = audio_stream.filter("afade", type="in", start_time=0, duration=safe_fade)
            if not is_last:
                start = max(clip_duration - safe_fade, 0)
                audio_stream = audio_stream.filter("afade", type="out", start_time=start, duration=safe_fade)
    else:
        audio_stream = None

    out = ffmpeg.output(
        *( [video_stream, audio_stream] if audio_stream else [video_stream] ),
        dest_path,
        vcodec="libx264",
        pix_fmt="yuv420p",
        **({"acodec": "aac", "audio_bitrate": "192k"} if audio_stream else {})
    )
    out = out.overwrite_output()
    out.run(quiet=True)


def _build_audio_stream(input_handle, has_audio: bool, duration: float):
    if has_audio:
        return input_handle.audio
    return ffmpeg.input(
        "anullsrc=r=48000:cl=stereo",
        f="lavfi",
        t=max(duration, 0.1)
    ).audio


def _crossfade_two_videos(path_a: str, meta_a: tuple[float, bool],
                          path_b: str, meta_b: tuple[float, bool],
                          transition_seconds: float, output_path: str):
    duration_a, has_audio_a = meta_a
    duration_b, has_audio_b = meta_b
    if duration_a <= 0 or duration_b <= 0:
        _concat_videos([path_a, path_b], output_path)
        return

    fade_dur = min(transition_seconds, duration_a, duration_b)
    if fade_dur <= 0:
        _concat_videos([path_a, path_b], output_path)
        return

    offset = max(duration_a - fade_dur, 0)
    in_a = ffmpeg.input(path_a)
    in_b = ffmpeg.input(path_b)

    video_stream = ffmpeg.filter(
        [in_a.video, in_b.video],
        "xfade",
        transition="fade",
        duration=fade_dur,
        offset=offset,
    )

    audio_stream = ffmpeg.filter(
        [
            _build_audio_stream(in_a, has_audio_a, duration_a),
            _build_audio_stream(in_b, has_audio_b, duration_b),
        ],
        "acrossfade",
        d=fade_dur,
    )

    (
        ffmpeg
        .output(
            video_stream,
            audio_stream,
            output_path,
            vcodec="libx264",
            acodec="aac",
            audio_bitrate="192k",
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run(quiet=True)
    )


def _merge_with_crossfade(part_paths: List[str], clip_meta: List[tuple[float, bool]],
                          transition_seconds: float, tmp_dir: str) -> str:
    if len(part_paths) == 1:
        return part_paths[0]

    current_path = part_paths[0]
    current_meta = clip_meta[0]

    for idx in range(1, len(part_paths)):
        next_path = part_paths[idx]
        next_meta = clip_meta[idx]
        output_path = os.path.join(tmp_dir, f"crossfade_{idx:03d}.mp4")
        _crossfade_two_videos(current_path, current_meta, next_path, next_meta, transition_seconds, output_path)
        fade_dur = min(transition_seconds, current_meta[0], next_meta[0])
        current_duration = current_meta[0] + next_meta[0] - fade_dur
        current_has_audio = current_meta[1] or next_meta[1]
        current_meta = (current_duration, current_has_audio)
        current_path = output_path

    return current_path


# ---------- Core routes ----------
@app.post("/veo/i2v")
def create_video(body: I2VBody, request: Request):
    client = _get_genai_client(body.gemini_api_key)
    # 1) Chuẩn bị ảnh input (Part)
    image_part = _image_part_from_input(body)

    # 2) Gọi Veo generate_videos (image-to-video)
    config_kwargs = dict(
        aspect_ratio=body.aspect_ratio,
        duration_seconds=str(body.duration_seconds),  # SDK mong "4"|"6"|"8" (string)
        negative_prompt=body.negative_prompt if body.negative_prompt else None,
        person_generation=body.person_generation,     # i2v -> "allow_adult"
    )
    if _model_supports_resolution(body.model):
        config_kwargs["resolution"] = body.resolution
    config = types.GenerateVideosConfig(**config_kwargs)

    operation = client.models.generate_videos(
        model=body.model,
        prompt=body.prompt_video,
        image=image_part.as_image(),
        config=config
    )

    # 3) Poll tới khi xong
    operation = _poll_operation(client, operation, sleep_sec=8)

    resp = getattr(operation, "response", None)
    if not resp or not getattr(resp, "generated_videos", None):
        # Trả về thông tin để debug nhanh
        raise HTTPException(502, f"Veo did not return generated_videos. raw={getattr(operation, 'response', None)}")

    # 4) Tải video Veo (MP4 có audio native từ Veo – nếu không dùng TTS)
    gen_video = operation.response.generated_videos[0]
    video_bytes = _download_genai_file_to_bytes(client, gen_video.video)

    with tempfile.TemporaryDirectory() as td:
        raw_video_path = os.path.join(td, "veo_raw.mp4")
        _save_bytes(raw_video_path, video_bytes)

        final_path = raw_video_path  # mặc định

        # 5) Nếu có voiceover_text: tạo TTS và ghép audio vào video bằng ffmpeg
        if body.voiceover_text:
            # TTS bằng Gemini 2.5 TTS
            tts_resp = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=body.voiceover_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=body.tts_voice_name or "Kore"
                            )
                        )
                    ),
                )
            )
            pcm_b64 = tts_resp.candidates[0].content.parts[0].inline_data.data
            pcm_bytes = base64.b64decode(pcm_b64)

            # Lưu PCM thành WAV tạm để ffmpeg dễ xử lý
            wav_path = os.path.join(td, "vo.wav")
            # ffmpeg đọc raw PCM 24kHz mono s16le -> wav
            (
                ffmpeg
                .input('pipe:', format='s16le', ar=str(body.tts_sample_rate), ac='1')
                .output(wav_path, format='wav', ar=str(body.tts_sample_rate), ac=1)
                .overwrite_output()
                .run(input=pcm_bytes, quiet=True)
            )

            # Mux: thay track audio của video bằng TTS, giữ nguyên video
            final_path = os.path.join(td, "veo_tts.mp4")
            (
                ffmpeg
                .input(raw_video_path)
                .output(wav_path, format='wav')  # đảm bảo wav sẵn sàng
                .overwrite_output()
                .run(quiet=True)
            )
            (
                ffmpeg
                .input(raw_video_path)
                .input(wav_path)
                .output(final_path, vcodec='copy', acodec='aac', shortest=None, audio_bitrate='192k')
                .overwrite_output()
                .run(quiet=True)
            )

        # 6) Upload GCS + trả JSON
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_filename = f"{timestamp_str}.mp4"
        local_file_path = _persist_local_copy(final_path, local_filename, target_dir=VIDEOS_DIR)
        local_video_url = str(request.url_for("videos", path=local_filename))

        return {
            "status": "ok",
            "model": body.model,
            "duration": body.duration_seconds,
            "aspect_ratio": body.aspect_ratio,
            "resolution": body.resolution,
            "person_generation": body.person_generation,
            "negative_prompt": body.negative_prompt,
            "voiceover_injected": bool(body.voiceover_text),
            "video_url": local_video_url,
            "local_file": local_file_path,
        }


@app.post("/audio-slideshow")
def create_audio_slideshow(
    body: AudioSlideshowBody,
    request: Request,
    resolution: str = "1280x720",
):
    if not body.img_urls:
        raise HTTPException(400, "img_urls phải có ít nhất 1 phần tử.")

    with tempfile.TemporaryDirectory() as td:
        aspect_ratio = _aspect_ratio_from_resolution(resolution)
        # Download audio
        parsed_audio = urlparse(str(body.audio_url))
        audio_ext = os.path.splitext(parsed_audio.path)[1] or ".audio"
        audio_path = os.path.join(td, f"audio{audio_ext}")
        _download_to_path(str(body.audio_url), audio_path, timeout=300)
        audio_duration = _audio_duration_seconds(audio_path, require_mp3_wav=True)

        # Download images
        image_paths: List[str] = []
        for idx, img_url in enumerate(body.img_urls):
            parsed_img = urlparse(str(img_url))
            img_ext = os.path.splitext(parsed_img.path)[1] or ".png"
            img_path = os.path.join(td, f"img_{idx:03d}{img_ext}")
            _download_to_path(str(img_url), img_path, timeout=120)
            image_paths.append(img_path)

        # Build one slideshow cycle video
        cycle_path = os.path.join(td, "cycle.mp4")
        _build_slideshow_cycle(
            image_paths=image_paths,
            seconds_per_image=body.seconds_per_image,
            frame_rate=body.frame_rate,
            resolution=resolution,
            dest_path=cycle_path,
        )

        # Loop slideshow video and align with audio length
        final_temp_path = os.path.join(td, "slideshow.mp4")
        slideshow_video = ffmpeg.input(cycle_path, stream_loop=-1)
        audio_stream = ffmpeg.input(audio_path)
        (
            ffmpeg
            .output(
                slideshow_video.video,
                audio_stream.audio,
                final_temp_path,
                vcodec="libx264",
                acodec="aac",
                audio_bitrate="192k",
                pix_fmt="yuv420p",
                shortest=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_filename = f"audio_slideshow_{timestamp_str}.mp4"
        local_file_path = _persist_local_copy(final_temp_path, local_filename, target_dir=VIDEOS_DIR)
        local_video_url = str(request.url_for("videos", path=local_filename))

        return {
            "status": "ok",
            "audio_duration": audio_duration,
            "image_count": len(body.img_urls),
            "seconds_per_image": body.seconds_per_image,
            "frame_rate": body.frame_rate,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "video_url": local_video_url,
            "local_file": local_file_path,
        }


@app.post("/video/voiceover")
def add_voiceover(body: VideoVoiceoverBody, request: Request):
    with tempfile.TemporaryDirectory() as td:
        video_path = _find_existing_media_path(str(body.video_url))
        if not video_path:
            video_path = os.path.join(td, "input_video.mp4")
            _download_to_path(str(body.video_url), video_path, timeout=600)

        audio_path = _find_existing_media_path(str(body.audio_url))
        if not audio_path:
            audio_ext = os.path.splitext(urlparse(str(body.audio_url)).path)[1] or ".audio"
            audio_path = os.path.join(td, f"voiceover{audio_ext}")
            _download_to_path(str(body.audio_url), audio_path, timeout=300)

        audio_duration = _audio_duration_seconds(audio_path, require_mp3_wav=True)
        video_duration, has_original_audio, _ = _video_metadata(video_path)
        final_duration = max(audio_duration, video_duration)
        if video_duration <= 0:
            raise HTTPException(400, "Video không có độ dài hợp lệ.")

        video_input = ffmpeg.input(video_path)
        voice_input = ffmpeg.input(audio_path)

        video_stream = video_input.video
        if audio_duration > video_duration:
            stretch_ratio = audio_duration / video_duration
            video_stream = video_stream.filter("setpts", f"{stretch_ratio}*PTS")
        elif video_duration < final_duration:
            pad_dur = final_duration - video_duration
            video_stream = video_stream.filter("tpad", start_duration=0, stop_mode="clone", stop_duration=pad_dur)

        voice_audio = voice_input.audio
        if audio_duration < final_duration:
            pad_dur = final_duration - audio_duration
            voice_audio = voice_audio.filter("apad", pad_dur=pad_dur)
        voice_audio = voice_audio.filter("volume", body.voiceover_volume)

        audio_streams = [voice_audio]
        if has_original_audio:
            original_audio = video_input.audio
            if audio_duration > video_duration:
                tempo_factor = video_duration / audio_duration
                original_audio = _apply_atempo_chain(original_audio, tempo_factor)
            elif video_duration < final_duration:
                pad_dur = final_duration - video_duration
                original_audio = original_audio.filter("apad", pad_dur=pad_dur)
            original_audio = original_audio.filter("volume", body.original_audio_volume)
            audio_streams.append(original_audio)

        if len(audio_streams) == 1:
            mixed_audio = audio_streams[0]
        else:
            mixed_audio = ffmpeg.filter(audio_streams, "amix", inputs=len(audio_streams), dropout_transition=0)

        final_temp_path = os.path.join(td, "voiceover_video.mp4")
        (
            ffmpeg
            .output(
                video_stream,
                mixed_audio,
                final_temp_path,
                vcodec="libx264",
                acodec="aac",
                audio_bitrate="192k",
                pix_fmt="yuv420p",
                shortest=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_filename = f"voiceover_{timestamp_str}.mp4"
        local_file_path = _persist_local_copy(final_temp_path, local_filename, target_dir=VIDEOS_DIR)
        local_video_url = str(request.url_for("videos", path=local_filename))

        return {
            "status": "ok",
            "video_duration": video_duration,
            "audio_duration": audio_duration,
            "final_duration": final_duration,
            "video_url": local_video_url,
            "local_file": local_file_path,
            "original_audio_included": has_original_audio,
            "original_audio_volume": body.original_audio_volume,
            "voiceover_volume": body.voiceover_volume,
        }

# ---------- Merge multiple videos by URL ----------
class MergeVideosBody(BaseModel):
    video_urls: list[HttpUrl]
    transition_seconds: float = 0.0
    motion_blur: bool = False


@app.post("/merge_videos_job")
def create_merge_videos_job(body: MergeVideosBody, request: Request):
    if not body.video_urls:
        raise HTTPException(400, "Bạn phải truyền ít nhất 1 phần tử trong video_urls")

    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    with MERGE_JOBS_LOCK:
        MERGE_JOBS[job_id] = {
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }

    MERGE_JOB_EXECUTOR.submit(
        _run_merge_video_job,
        job_id,
        [str(url) for url in body.video_urls],
        body.transition_seconds,
        body.motion_blur,
        str(request.base_url),
    )

    return {"job_id": job_id, "status": "pending"}


@app.get("/merge_videos_job/{job_id}")
def get_merge_videos_job(job_id: str):
    return _serialize_job(job_id)


def _execute_merge_videos(video_urls: List[str], transition_seconds: float, motion_blur: bool) -> tuple[str, str, int]:
    if not video_urls:
        raise HTTPException(400, "Bạn phải truyền ít nhất 1 phần tử trong video_urls")

    # transition_seconds/motion_blur are intentionally ignored to keep the merge logic simple/stable
    _ = (transition_seconds, motion_blur)

    with tempfile.TemporaryDirectory() as td:
        max_workers = min(4, len(video_urls))

        def _download_job(idx: int, url_str: str):
            local_path = _find_existing_media_path(url_str)
            if local_path:
                return idx, local_path
            part_path = os.path.join(td, f"part_{idx:03d}.mp4")
            _download_to_path(url_str, part_path, timeout=MERGE_DOWNLOAD_TIMEOUT)
            return idx, part_path

        ordered_parts: List[Optional[str]] = [None] * len(video_urls)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_download_job, idx, url_str)
                for idx, url_str in enumerate(video_urls)
            ]
            for future in as_completed(futures):
                idx, path = future.result()
                ordered_parts[idx] = path

        part_paths = [p for p in ordered_parts if p]
        if not part_paths:
            raise HTTPException(400, "Không tải được video nào.")

        normalized_paths: List[str] = []
        for idx, path in enumerate(part_paths):
            normalized_path = os.path.join(td, f"normalized_{idx:03d}.mp4")
            _prepare_clip_for_merge(path, normalized_path)
            normalized_paths.append(normalized_path)
        part_paths = normalized_paths

        current_paths = part_paths
        level = 0
        while len(current_paths) > 1:
            next_paths: List[str] = []
            for chunk_idx in range(0, len(current_paths), MERGE_CHUNK_SIZE):
                chunk = current_paths[chunk_idx: chunk_idx + MERGE_CHUNK_SIZE]
                if len(chunk) == 1:
                    next_paths.append(chunk[0])
                    continue
                chunk_output = os.path.join(
                    td, f"merged_l{level}_{chunk_idx // MERGE_CHUNK_SIZE:03d}.mp4"
                )
                _concat_videos(chunk, chunk_output)
                next_paths.append(chunk_output)
            current_paths = next_paths
            level += 1

        merged_path = current_paths[0]

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_filename = f"merged_{timestamp_str}.mp4"
        local_file_path = _persist_local_copy(merged_path, local_filename, target_dir=VIDEOS_DIR)

    return local_file_path, local_filename, len(video_urls)


def _build_video_url(filename: str, base_url: str) -> str:
    relative_path = app.url_path_for("videos", path=filename)
    prefix = base_url.rstrip("/")
    if not prefix:
        return relative_path
    if not relative_path.startswith("/"):
        relative_path = f"/{relative_path}"
    return f"{prefix}{relative_path}"


def _set_job_state(job_id: str, status: str, *, result: Optional[Dict[str, Any]] = None,
                   error: Optional[Dict[str, Any]] = None):
    with MERGE_JOBS_LOCK:
        job = MERGE_JOBS.get(job_id)
        if not job:
            return
        job["status"] = status
        job["updated_at"] = datetime.utcnow()
        job["result"] = result
        job["error"] = error


def _run_merge_video_job(job_id: str, video_urls: List[str], transition_seconds: float,
                         motion_blur: bool, base_url: str):
    try:
        _set_job_state(job_id, "running")
        local_file_path, local_filename, input_count = _execute_merge_videos(
            video_urls, transition_seconds, motion_blur
        )
        result_payload = {
            "status": "ok",
            "input_count": input_count,
            "local_file": local_file_path,
            "video_url": _build_video_url(local_filename, base_url),
        }
        _set_job_state(job_id, "completed", result=result_payload, error=None)
    except HTTPException as exc:
        error_payload = {"status_code": exc.status_code, "detail": exc.detail}
        _set_job_state(job_id, "failed", error=error_payload)
    except Exception as exc:
        error_payload = {"status_code": 500, "detail": str(exc)}
        _set_job_state(job_id, "failed", error=error_payload)


def _serialize_job(job_id: str) -> Dict[str, Any]:
    with MERGE_JOBS_LOCK:
        job = MERGE_JOBS.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} không tồn tại")
        response: Dict[str, Any] = {
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"].isoformat(),
            "updated_at": job["updated_at"].isoformat(),
        }
        if job.get("result"):
            response["result"] = job["result"]
        if job.get("error"):
            response["error"] = job["error"]
        return response


@app.post("/merge_videos")
def merge_videos(body: MergeVideosBody, request: Request):
    local_file_path, local_filename, input_count = _execute_merge_videos(
        [str(url) for url in body.video_urls],
        body.transition_seconds,
        body.motion_blur,
    )
    local_video_url = str(request.url_for("videos", path=local_filename))

    return {
        "status": "ok",
        "input_count": input_count,
        "video_url": local_video_url,
        "local_file": local_file_path,
    }


@app.get("/")
def root():
    return {"hello": "Veo3 i2v API – ready"}
