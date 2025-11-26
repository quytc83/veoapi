import os
import io
import time
from datetime import datetime
import base64
import tempfile
import shutil
import requests
from urllib.parse import urlparse
from typing import Optional, List
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


# ---------- Core routes ----------
@app.post("/veo/i2v")
def create_video(body: I2VBody, request: Request):
    client = _get_genai_client(body.gemini_api_key)
    # 1) Chuẩn bị ảnh input (Part)
    image_part = _image_part_from_input(body)

    # 2) Gọi Veo generate_videos (image-to-video)
    config = types.GenerateVideosConfig(
        aspect_ratio=body.aspect_ratio,
        resolution=body.resolution,
        duration_seconds=str(body.duration_seconds),  # SDK mong "4"|"6"|"8" (string)
        negative_prompt=body.negative_prompt if body.negative_prompt else None,
        person_generation=body.person_generation,     # i2v -> "allow_adult"
    )

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

# ---------- Merge multiple videos by URL ----------
class MergeVideosBody(BaseModel):
    video_urls: list[HttpUrl]

@app.post("/merge_videos")
def merge_videos(body: MergeVideosBody, request: Request):
    if not body.video_urls:
        raise HTTPException(400, "Bạn phải truyền ít nhất 1 phần tử trong video_urls")

    with tempfile.TemporaryDirectory() as td:
        max_workers = min(4, len(body.video_urls))

        def _download_job(idx: int, url_str: str):
            part_path = os.path.join(td, f"part_{idx:03d}.mp4")
            _download_to_path(url_str, part_path, timeout=300)
            return idx, part_path

        ordered_parts: List[Optional[str]] = [None] * len(body.video_urls)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_download_job, idx, str(url))
                for idx, url in enumerate(body.video_urls)
            ]
            for future in as_completed(futures):
                idx, path = future.result()
                ordered_parts[idx] = path

        part_paths = [p for p in ordered_parts if p]
        if not part_paths:
            raise HTTPException(400, "Không tải được video nào.")

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
        local_video_url = str(request.url_for("videos", path=local_filename))

        return {
            "status": "ok",
            "input_count": len(body.video_urls),
            "video_url": local_video_url,
            "local_file": local_file_path,
        }


@app.get("/")
def root():
    return {"hello": "Veo3 i2v API – ready"}
