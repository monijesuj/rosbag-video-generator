# ROS2 Bag Video Generator

Tools for processing ROS2 bag files into annotated videos with object detection overlays, pose estimation, and depth visualization. Designed for HRI (Human-Robot Interaction) research publications.

## Features

- **Detection Overlays**: GroundingDINO zero-shot object detection on recorded data
- **Pose Estimation**: MediaPipe pose with 8-direction orientation estimation
- **Depth Visualization**: Heatmap overlay from RealSense depth data
- **Split View Layout**: Side-by-side RGB + Depth with info panel
- **CSV Export**: Extract pose, velocity, and Vicon ground truth to CSV
- **Memory Efficient**: Streaming processing for large bags with `--max-frames` option

## Output Format

The generated video includes:
- **Left panel**: RGB image with detection bounding boxes and pose skeleton
- **Right panel**: Depth heatmap visualization
- **Bottom panel**: Frame info, drone position, human position (Vicon), detection counts

## Installation

### Prerequisites
- Python 3.8+
- ROS2 Humble (for bag recording, not required for processing)
- CUDA-capable GPU (recommended)

### Install dependencies
```bash
pip install rosbags opencv-python numpy torch mediapipe
pip install pillow matplotlib

# Install GroundingDINO (for detection features)
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e .
```

## Usage

### Generate video with detection overlays
```bash
python generate_video_from_bag_v2.py /path/to/rosbag \
    --fps 15 \
    --human-prompt "person" \
    --object-prompt "orange box"
```

### Generate video without detection (faster)
```bash
python generate_video_from_bag.py /path/to/rosbag --fps 15 --no-detection
```

### Limit frames for testing
```bash
python generate_video_from_bag_v2.py /path/to/rosbag --max-frames 500
```

### Extract data to CSV
```bash
python extract_bag_to_csv.py /path/to/rosbag --output-dir ./csv_output
```

## Command-line Arguments

### generate_video_from_bag_v2.py

| Argument | Default | Description |
|----------|---------|-------------|
| `bag_path` | (required) | Path to ROS2 bag directory |
| `--output` | `{bag_name}_video.mp4` | Output video path |
| `--fps` | `10.0` | Output video frame rate |
| `--human-prompt` | `"human"` | GroundingDINO prompt for humans |
| `--object-prompt` | `"orange colored cuboid"` | GroundingDINO prompt for objects |
| `--max-frames` | `None` | Limit number of frames to process |
| `--no-detection` | `False` | Skip detection (faster processing) |

### extract_bag_to_csv.py

| Argument | Default | Description |
|----------|---------|-------------|
| `bag_path` | (required) | Path to ROS2 bag directory |
| `--output-dir` | `./` | Output directory for CSV files |

## Expected Bag Topics

The scripts automatically detect camera topic prefixes (`/camera/camera_down/` or `/camera/camera/`).

| Topic | Type | Description |
|-------|------|-------------|
| `*/color/image_raw` | `sensor_msgs/Image` | RGB camera image |
| `*/depth/image_rect_raw` | `sensor_msgs/Image` | Depth image (16UC1) |
| `/spatial_drone/mavros/local_position/pose` | `geometry_msgs/PoseStamped` | Drone pose |
| `/spatial_drone/vicon/*/` | `geometry_msgs/TransformStamped` | Vicon ground truth |

## Output Files

### Video Output
- `{bag_name}_video.mp4` - Annotated video with detection overlays

### CSV Output
- `drone_pose.csv` - Drone position and orientation over time
- `drone_velocity_body.csv` - Drone velocity in body frame
- `vicon_drone.csv` - Vicon ground truth for drone
- `vicon_human.csv` - Vicon ground truth for human

## Example Workflow

```bash
# 1. Record a bag during experiment
ros2 bag record -a -o my_experiment

# 2. Extract CSV data for analysis
python extract_bag_to_csv.py my_experiment --output-dir ./data

# 3. Generate annotated video for publication
python generate_video_from_bag_v2.py my_experiment \
    --fps 15 \
    --human-prompt "person wearing black" \
    --object-prompt "white drone" \
    --max-frames 2000

# 4. Video ready at my_experiment_video.mp4
```

## Handling Corrupted Bags

If your bag has a missing or corrupted `metadata.yaml`:

```bash
# Reindex the bag
ros2 bag reindex /path/to/bag

# Fix storage_identifier if needed
sed -i 's/storage_identifier: ""/storage_identifier: sqlite3/' /path/to/bag/metadata.yaml
```

## Performance

| Bag Size | Max Frames | Detection | Processing Time |
|----------|------------|-----------|-----------------|
| 1000 frames | All | Enabled | ~3 min |
| 8000 frames | 2000 | Enabled | ~5 min |
| 8000 frames | All | Disabled | ~2 min |

*Times measured on RTX 4090*

## Files

| File | Description |
|------|-------------|
| `generate_video_from_bag_v2.py` | Full-featured video generator with detection + pose |
| `generate_video_from_bag.py` | Basic video generator (no pose estimation) |
| `extract_bag_to_csv.py` | Extract pose/velocity/Vicon data to CSV |

<!-- ## Citation

If you use this tool in your research, please cite:

```bibtex
@software{rosbag_video_generator,
  title = {ROS2 Bag Video Generator},
  author = {James},
  year = {2025},
  url = {https://github.com/monijesuj/rosbag-video-generator}
}
``` -->

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

<!-- - [rosbags](https://github.com/rpng/rosbags) - Pure Python ROS2 bag reader -->
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Zero-shot detection
- [MediaPipe](https://mediapipe.dev/) - Pose estimation
