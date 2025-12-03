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

**Chạy với Veo 2:** chọn `model` là `veo-2.0-generate-001` (hoặc các biến thể Veo 2.x). API tự động loại bỏ
tham số `resolution` vì Veo 2 không hỗ trợ, nhưng bạn vẫn có thể giữ trường này để đồng bộ payload giữa Veo 2 và Veo 3.

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

API này GHÉP NỐI THẲNG CÁC VIDEO THEO ĐÚNG THỨ TỰ TRONG `video_urls`. Không còn các hiệu ứng cross-fade hay motion blur để đảm bảo ổn định khi chạy trên host thật. Nếu file đã có trong thư mục `videos/` thì API tái sử dụng mà không tải lại. Trước khi ghép, mỗi clip sẽ được đồng bộ lại thời lượng hình/tiếng để không bị hiện tượng tiếng “chạy trước” khung hình.

```bash
curl -X POST "http://0.0.0.0:8080/merge_videos" \
 -H "Content-Type: application/json" \
 -d '{
       "video_urls": [
         "https://example.com/videos/scene1.mp4",
         "https://example.com/videos/scene2.mp4",
         "https://example.com/videos/scene3.mp4"
       ]
     }'
```

### 5. Merge Videos Job (`/merge_videos_job`)

Tạo job chạy nền để ghép video theo THỨ TỰ MẢNG, kết quả cuối cùng cũng chỉ đơn thuần là concat không hiệu ứng (đã đồng bộ audio để bám sát từng cảnh). POST để tạo job (trả về `job_id`), sau đó GET `/merge_videos_job/{job_id}` để poll trạng thái và lấy `video_url` khi hoàn tất.

```bash
# Tạo job ghép 2 video mẫu có sẵn sau khi chạy server (các file nằm trong thư mục videos/).
curl -X POST "http://0.0.0.0:8080/merge_videos_job" \
 -H "Content-Type: application/json" \
 -d '{
       "video_urls": [
         "http://0.0.0.0:8080/videos/20251127_213239.mp4",
         "http://0.0.0.0:8080/videos/20251127_213526.mp4"
       ]
     }'

# Poll kết quả (thay <JOB_ID> bằng job_id nhận được ở trên)
curl "http://0.0.0.0:8080/merge_videos_job/<JOB_ID>"
```
