# Gemini API: YouTube Video Understanding Developer Guide

This guide provides a detailed overview of how to use the YouTube video understanding features of the Gemini API.

**Preview:** The YouTube URL feature is in preview and is available at no charge. Pricing and rate limits are likely to change.

## Including a YouTube URL

The Gemini API and AI Studio support YouTube URLs as a file data `Part`. You can include a YouTube URL with a prompt asking the model to summarize, translate, or otherwise interact with the video content.

**Limitations:**

*   For the free tier, you can't upload more than 8 hours of YouTube video per day.
*   For the paid tier, there is no limit based on video length.
*   For models before 2.5, you can upload only 1 video per request.
*   For models after 2.5, you can upload a maximum of 10 videos per request.
*   You can only upload public videos (not private or unlisted videos).

The following example shows how to include a YouTube URL with a prompt:

**Python**
```python
response = client.models.generate_content(
    model='models/gemini-2.5-flash',
    contents=types.Content(
        parts=[
            types.Part(
                file_data=types.FileData(file_uri='https://www.youtube.com/watch?v=9hE5-98ZeCg')
            ),
            types.Part(text='Please summarize the video in 3 sentences.')
        ]
    )
)
```

**JavaScript**
```javascript
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });

const result = await model.generateContent([
  "Please summarize the video in 3 sentences.",
  {
    fileData: {
      fileUri: "https://www.youtube.com/watch?v=9hE5-98ZeCg",
    },
  },
]);
console.log(result.response.text());
```

**Go**
```go
package main

import (
    "context"
    "fmt"
    "os"

    "google.golang.org/genai"
)

func main() {
    ctx := context.Background()
    client, err := genai.NewClient(ctx, nil)
    if err != nil {
        log.Fatal(err)
    }
    parts := []*genai.Part{
        genai.NewPartFromText("Please summarize the video in 3 sentences."),
        genai.NewPartFromURI("https://www.youtube.com/watch?v=9hE5-98ZeCg","video/mp4"),
    }
    contents := []*genai.Content{
        genai.NewContentFromParts(parts, genai.RoleUser),
    }
    result, _ := client.Models.GenerateContent(
        ctx,
        "gemini-2.5-flash",
        contents,
        nil,
    )
    fmt.Println(result.Text())
}
```

**REST**
```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[
          {"text": "Please summarize the video in 3 sentences."},
          {
            "file_data": {
              "file_uri": "https://www.youtube.com/watch?v=9hE5-98ZeCg"
            }
          }
        ]
      }]
    }' 2> /dev/null
```

## Referring to Timestamps in the Content

You can ask questions about specific points in time within the video using timestamps of the form `MM:SS`.

**Python**
```python
prompt = "What are the examples given at 00:05 and 00:10 supposed to show us?"
```

**JavaScript**
```javascript
const prompt = "What are the examples given at 00:05 and 00:10 supposed to show us?";
```

**Go**
```go
prompt := []*genai.Part{
    genai.NewPartFromURI(currentVideoFile.URI, currentVideoFile.MIMEType),
    // Adjusted timestamps for the NASA video
    genai.NewPartFromText("What are the examples given at 00:05 and " + "00:10 supposed to show us?"),
}
```

**REST**
```bash
PROMPT="What are the examples given at 00:05 and 00:10 supposed to show us?"
```

## Transcribing Video and Providing Visual Descriptions

The Gemini models can transcribe and provide visual descriptions of video content by processing both the audio track and visual frames. For visual descriptions, the model samples the video at a rate of 1 frame per second. This sampling rate may affect the level of detail in the descriptions, particularly for videos with rapidly changing visuals.

**Python**
```python
prompt = "Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions."
```

**JavaScript**
```javascript
const prompt = "Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions.";
```

**Go**
```go
prompt := []*genai.Part{
    genai.NewPartFromURI(currentVideoFile.URI, currentVideoFile.MIMEType),
    genai.NewPartFromText("Transcribe the audio from this video, giving timestamps for salient events in the video. Also " + "provide visual descriptions."),
}
```

**REST**
```bash
PROMPT="Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions."
```

## Customizing Video Processing

You can customize video processing in the Gemini API by setting clipping intervals or providing custom frame rate sampling.

### Set Clipping Intervals

You can clip video by specifying `videoMetadata` with start and end offsets.

**Python**
```python
response = client.models.generate_content(
    model='models/gemini-2.5-flash',
    contents=types.Content(
        parts=[
            types.Part(
                file_data=types.FileData(file_uri='https://www.youtube.com/watch?v=XEzRZ35urlk'),
                video_metadata=types.VideoMetadata(
                    start_offset='1250s',
                    end_offset='1570s'
                )
            ),
            types.Part(text='Please summarize the video in 3 sentences.')
        ]
    )
)
```

## Supported Video Formats

Gemini supports the following video format MIME types:
*   `video/mp4`
*   `video/mpeg`
*   `video/mov`
*   `video/avi`
*   `video/x-flv`
*   `video/mpg`
*   `video/webm`
*   `video/wmv`
*   `video/3gpp`

## Technical Details About Videos

*   **Supported models & context**: All Gemini 2.0 and 2.5 models can process video data.
*   Models with a 2M context window can process videos up to 2 hours long at default media resolution or 6 hours long at low media resolution, while models with a 1M context window can process videos up to 1 hour long at default media resolution or 3 hours long at low media resolution.
*   **File API processing**: When using the File API, videos are sampled at 1 frame per second (FPS) and audio is processed at 1Kbps (single channel). Timestamps are added every second. These rates are subject to change in the future for improvements in inference.
*   **Token calculation**: Each second of video is tokenized as follows:
    *   Individual frames (sampled at 1 FPS):
        *   If `mediaResolution` is set to `low`, frames are tokenized at 66 tokens per frame.
        *   Otherwise, frames are tokenized at 258 tokens per frame.
    *   Audio: 32 tokens per second.
    *   Metadata is also included.
    *   Total: Approximately 300 tokens per second of video at default media resolution, or 100 tokens per second of video at low media resolution.
*   **Timestamp format**: When referring to specific moments in a video within your prompt, use the `MM:SS` format (e.g., `01:15` for 1 minute and 15 seconds).
*   **Best practices**:
    *   Use only one video per prompt request for optimal results.
    *   If combining text and a single video, place the text prompt after the video part in the `contents` array.
    *   Be aware that fast action sequences might lose detail due to the 1 FPS sampling rate. Consider slowing down such clips if necessary.