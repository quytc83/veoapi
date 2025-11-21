import os
import io
import time
import base64
import tempfile
import requests
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from google import genai
from google.genai import types
from google.cloud import storage
import ffmpeg
import imghdr



# ---------- Config ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

GCS_BUCKET = os.getenv("GCS_BUCKET", "tts_poc")
PROJECT_ID = os.getenv("GCP_PROJECT")

client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI(title="Veo3 Image-to-Video API (i2v+TTS)")


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

    # Upload
    gcs_bucket: Optional[str] = None              # override bucket nếu muốn
    gcs_prefix: Optional[str] = "veo_i2v"         # folder prefix


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


def _poll_operation(op, sleep_sec=10):
    # Long-running operation for Veo video generation
    while not op.done:
        time.sleep(sleep_sec)
        op = client.operations.get(op)
    return op

def _download_genai_file_to_bytes(file_ref) -> bytes:
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

def _gcs_upload_and_signed_url(local_path: str, dest_name: str, bucket_name: str) -> str:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_name)
    blob.upload_from_filename(local_path, content_type="video/mp4" if local_path.endswith(".mp4") else None)
    # Tạo signed url 7 ngày
    url = blob.generate_signed_url(expiration=7*24*3600)
    return url


# ---------- Core routes ----------
@app.post("/veo/i2v")
def create_video(body: I2VBody):
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
    operation = _poll_operation(operation, sleep_sec=8)

    resp = getattr(operation, "response", None)
    if not resp or not getattr(resp, "generated_videos", None):
        # Trả về thông tin để debug nhanh
        raise HTTPException(502, f"Veo did not return generated_videos. raw={getattr(operation, 'response', None)}")

    # 4) Tải video Veo (MP4 có audio native từ Veo – nếu không dùng TTS)
    gen_video = operation.response.generated_videos[0]
    video_bytes = _download_genai_file_to_bytes(gen_video.video)

    with tempfile.TemporaryDirectory() as td:
        raw_video_path = os.path.join(td, "veo_raw.mp4")
        _save_bytes(raw_video_path, video_bytes)

        final_path = raw_video_path  # mặc định

        # 5) Nếu có voiceover_text: tạo TTS và ghép audio vào video bằng ffmpeg
        voice_url = None
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
        bucket = body.gcs_bucket or GCS_BUCKET
        object_name = f"{body.gcs_prefix or 'veo_i2v'}/veo_{int(time.time())}.mp4"
        signed_url = _gcs_upload_and_signed_url(final_path, object_name, bucket)

        return {
            "status": "ok",
            "model": body.model,
            "duration": body.duration_seconds,
            "aspect_ratio": body.aspect_ratio,
            "resolution": body.resolution,
            "person_generation": body.person_generation,
            "negative_prompt": body.negative_prompt,
            "voiceover_injected": bool(body.voiceover_text),
            "gcs_bucket": bucket,
            "gcs_object": object_name,
            "video_url": signed_url
        }

# ---------- Merge multiple videos by URL ----------
class MergeVideosBody(BaseModel):
    video_urls: list[HttpUrl]
    gcs_bucket: Optional[str] = None              # override default bucket nếu muốn
    gcs_prefix: Optional[str] = "merged_videos"   # folder prefix trên GCS

@app.post("/merge_videos")
def merge_videos(body: MergeVideosBody):
    if not body.video_urls:
        raise HTTPException(400, "Bạn phải truyền ít nhất 1 phần tử trong video_urls")

    with tempfile.TemporaryDirectory() as td:
        part_paths = []
        for idx, url in enumerate(body.video_urls):
            url_str = str(url)
            try:
                resp = requests.get(url_str, timeout=300)
            except Exception as e:
                raise HTTPException(400, f"Không tải được video_url={url_str}: {e}")
            if resp.status_code != 200:
                raise HTTPException(400, f"Không tải được video_url={url_str}, status={resp.status_code}")

            part_path = os.path.join(td, f"part_{idx:03d}.mp4")
            with open(part_path, "wb") as f:
                f.write(resp.content)
            part_paths.append(part_path)

        if len(part_paths) == 1:
            # chỉ 1 video thì khỏi merge, upload luôn
            merged_path = part_paths[0]
        else:
            # Tạo file list cho ffmpeg concat demuxer
            list_path = os.path.join(td, "inputs.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for p in part_paths:
                    # đường dẫn tạm không có khoảng trắng nên đơn giản dùng ''
                    f.write(f"file '{p}'\n")

            merged_path = os.path.join(td, "merged.mp4")
            (
                ffmpeg
                .input(list_path, f="concat", safe=0)
                # re-encode về H.264 + AAC để hạn chế lỗi khác codec/khung hình
                .output(merged_path, vcodec="libx264", acodec="aac", audio_bitrate="192k")
                .overwrite_output()
                .run(quiet=True)
            )

        bucket = body.gcs_bucket or GCS_BUCKET
        object_name = f"{body.gcs_prefix or 'merged_videos'}/merged_{int(time.time())}.mp4"
        signed_url = _gcs_upload_and_signed_url(merged_path, object_name, bucket)

        return {
            "status": "ok",
            "input_count": len(body.video_urls),
            "gcs_bucket": bucket,
            "gcs_object": object_name,
            "video_url": signed_url,
        }


@app.get("/")
def root():
    return {"hello": "Veo3 i2v API – ready"}

