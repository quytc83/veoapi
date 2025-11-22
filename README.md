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

## Example Request

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
