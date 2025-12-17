# Media Files

## Adding Demo Videos/Images

The generated videos are too large for GitHub. Consider:

1. **Screenshots**: Extract key frames from videos
   ```bash
   ffmpeg -i rosbag2_2025_12_09-11_50_31_video.mp4 -ss 00:00:10 -frames:v 1 demo.png
   ```

2. **GIFs**: Create short animated demos (< 10MB)
   ```bash
   ffmpeg -i video.mp4 -ss 00:00:05 -t 5 -vf "fps=10,scale=640:-1:flags=lanczos" demo.gif
   ```

3. **YouTube/Google Drive**: Upload full videos and link in README

## Example Output

Place demo images here:
- `demo.png` - Main demonstration screenshot
- `detection_example.gif` - Short animation of detection
- `output_example.png` - Sample output frame
