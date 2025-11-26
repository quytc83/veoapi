# Veo API

## Run with Docker

```bash
docker compose up --build
```

## Run locally

```bash
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## API Examples

### 1. Image-to-Video (`/veo/i2v`)

```bash
curl -X POST "http://0.0.0.0:8080/veo/i2v" \
 -H "Content-Type: application/json" \
 --data-binary @- <<'JSON'
{
  "image_url": "https://storage.googleapis.com/dulichveo3/scenario_1.png",
  "gemini_api_key": "AIzaS.....",
  "model": "veo-3.1-fast-generate-preview",
  "aspect_ratio": "9:16",
  "resolution": "720p",
  "duration_seconds": 8,
  "person_generation": "allow_adult",
  "negative_prompt": "cartoon, anime, low quality, artifacts, overexposed, oversaturated",
  "prompt_video": "A beautiful Vietnamese female tour guide wearing a traditional red áo dài stands in front of the main gate of the Temple of Literature (Văn Miếu) in Hanoi.\nMorning sunlight gently shines on the ancient red roofs and stone walls.\nShe smiles warmly and gestures as if welcoming visitors.\nThe camera slowly pans forward, focusing softly on her face, then reveals the gate and green trees around.\nStyle: cinematic, ultra-realistic, 4K, natural lighting, shallow depth of field, soft background bokeh, smooth motion.\nDuration: 8 seconds.\n\nVoiceover (Vietnamese): \"Xin chào quý du khách! Hôm nay, tôi xin mời mọi người cùng tham quan Quốc Tử Giám – trường đại học đầu tiên của Việt Nam, biểu tượng của tinh hoa tri thức và truyền thống hiếu học dân tộc.\""
}
JSON
```

### 2. Audio Slideshow (`/audio-slideshow`)

Tạo video từ audio và danh sách ảnh, hỗ trợ tham số `resolution=WIDTHxHEIGHT`.

```bash
curl -X POST "http://0.0.0.0:8080/audio-slideshow?resolution=1280x720" \
 -H "Content-Type: application/json" \
 -d '{
       "audio_url": "https://example.com/audio/voiceover.mp3",
       "img_urls": [
         "https://example.com/images/frame1.png",
         "https://example.com/images/frame2.jpg",
         "https://example.com/images/frame3.png"
       ],
       "seconds_per_image": 3,
       "frame_rate": 30
     }'
```

### 3. Video Voiceover (`/video/voiceover`)

Giữ lại tiếng gốc nhỏ rồi lồng tiếng mới từ `audio_url`. Nếu video/audio đã tồn tại trong thư mục `videos/` thì API tái sử dụng thay vì tải lại.

```bash
curl -X POST "http://0.0.0.0:8080/video/voiceover" \
 -H "Content-Type: application/json" \
 -d '{
       "video_url": "https://example.com/videos/input.mp4",
       "audio_url": "https://example.com/audio/voiceover.wav",
       "original_audio_volume": 0.25,
       "voiceover_volume": 1.0
     }'
```

### 4. Merge Videos (`/merge_videos`)

Ghép nhiều video URL; có thể bật `transition_seconds` (transition cross-fade) và `motion_blur` để làm chuyển cảnh mượt hơn. API cũng tự động bỏ qua download nếu file đã nằm trong `videos/`.

```bash
curl -X POST "http://0.0.0.0:8080/merge_videos" \
 -H "Content-Type: application/json" \
 -d '{
       "video_urls": [
         "https://example.com/videos/scene1.mp4",
         "https://example.com/videos/scene2.mp4",
         "https://example.com/videos/scene3.mp4"
       ],
       "transition_seconds": 1.5,
       "motion_blur": true
     }'
```
