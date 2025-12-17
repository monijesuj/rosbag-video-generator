#!/usr/bin/env python3
"""
Generate MP4 video from ROS2 bag with:
- RGB camera feed with detection bounding boxes
- Depth heatmap overlay
- Position annotations (X, Y, Z)
- Split view layout

Usage:
    python3 generate_video_from_bag.py <bag_path> [--output video.mp4]
"""

import argparse
import os
import sys
import cv2
import numpy as np
from collections import deque
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# Add GroundingDINO path
possible_gdino_paths = [
    '/home/imit-learn/James/catkin_ws/Object-Detection-and-Distance-Measurement/GroundingDINO',
    '/home/imit-learn/James/Object-Detection-and-Distance-Measurement/GroundingDINO',
]
for path in possible_gdino_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        break

# Try to import GroundingDINO
GDINO_AVAILABLE = False
try:
    import torch
    from PIL import Image as PILImage
    import groundingdino.datasets.transforms as T
    from groundingdino.util.inference import load_model, predict
    GDINO_AVAILABLE = True
    print("GroundingDINO loaded successfully")
except ImportError as e:
    print(f"GroundingDINO not available: {e}")
    print("Will generate video without detections")


class BagVideoGenerator:
    def __init__(self, bag_path: str, output_path: str, 
                 human_prompt: str = "human . person",
                 object_prompt: str = "orange colored cuboid . red bag",
                 fps: float = 10.0,
                 show_preview: bool = False):
        self.bag_path = bag_path
        self.output_path = output_path
        self.human_prompt = human_prompt
        self.object_prompt = object_prompt
        self.fps = fps
        self.show_preview = show_preview
        
        # Typestore for deserializing messages
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)
        
        # Detection settings
        self.box_threshold = 0.35
        self.text_threshold = 0.25
        
        # GroundingDINO model
        self.model = None
        self.device = None
        if GDINO_AVAILABLE:
            self._load_gdino_model()
        
        # Data buffers for synchronization
        self.color_buffer = {}  # timestamp -> image
        self.depth_buffer = {}  # timestamp -> image
        self.pose_buffer = deque(maxlen=1000)  # Recent poses
        self.human_vicon_buffer = deque(maxlen=1000)
        
    def _load_gdino_model(self):
        """Load GroundingDINO model."""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            # Find model files
            gdino_path = None
            for path in possible_gdino_paths:
                if os.path.exists(path):
                    gdino_path = path
                    break
            
            if gdino_path:
                config_path = os.path.join(gdino_path, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
                weights_path = os.path.join(gdino_path, "weights/groundingdino_swint_ogc.pth")
                
                if os.path.exists(config_path) and os.path.exists(weights_path):
                    self.model = load_model(config_path, weights_path)
                    self.model = self.model.to(self.device)
                    print("GroundingDINO model loaded")
                else:
                    print(f"Model files not found at {gdino_path}")
        except Exception as e:
            print(f"Failed to load GroundingDINO: {e}")
            self.model = None
    
    def detect_objects(self, image: np.ndarray, prompt: str):
        """Run GroundingDINO detection."""
        if self.model is None:
            return [], [], []
        
        try:
            # Preprocess image
            image_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_transformed, _ = transform(image_pil, None)
            
            # Run detection
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image_transformed,
                    caption=prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device
                )
            
            boxes_np = boxes.cpu().numpy() if len(boxes) > 0 else []
            confidences_np = logits.cpu().numpy() if len(logits) > 0 else []
            
            return boxes_np, confidences_np, phrases
        except Exception as e:
            print(f"Detection error: {e}")
            return [], [], []
    
    def draw_detections(self, image: np.ndarray, boxes, confidences, phrases, depth_image=None):
        """Draw bounding boxes and labels on image."""
        h, w = image.shape[:2]
        annotated = image.copy()
        
        for box, conf, phrase in zip(boxes, confidences, phrases):
            # GroundingDINO returns boxes in [cx, cy, w, h] format (normalized 0-1)
            # Convert to [x1, y1, x2, y2] pixel coordinates
            cx, cy, bw, bh = box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            # Determine color based on detection type
            phrase_lower = phrase.lower()
            if 'human' in phrase_lower or 'person' in phrase_lower:
                color = (0, 255, 0)  # Green for humans
                label = "HUMAN"
            else:
                color = (0, 165, 255)  # Orange for objects
                label = "OBJECT"
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Get depth at center if available
            depth_str = ""
            if depth_image is not None:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                    depth_val = depth_image[cy, cx]
                    if depth_val > 0:
                        depth_m = depth_val / 1000.0  # Convert mm to m
                        depth_str = f" Z={depth_m:.2f}m"
            
            # Draw label
            label_text = f"{label}: {conf:.2f}{depth_str}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def create_depth_heatmap(self, depth_image: np.ndarray, max_depth: float = 5.0) -> np.ndarray:
        """Convert depth image to colorful heatmap."""
        # Normalize depth
        depth_float = depth_image.astype(np.float32) / 1000.0  # Convert to meters
        depth_normalized = np.clip(depth_float / max_depth, 0, 1)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Make invalid depth (0) black
        depth_colored[depth_image == 0] = [0, 0, 0]
        
        return depth_colored
    
    def create_composite_frame(self, color_image: np.ndarray, depth_heatmap: np.ndarray,
                               drone_pose: dict = None, human_pose: dict = None,
                               frame_num: int = 0, timestamp: float = 0) -> np.ndarray:
        """Create a composite frame with RGB, depth, and info overlay."""
        h, w = color_image.shape[:2]
        
        # Resize depth to match color if needed
        if depth_heatmap.shape[:2] != (h, w):
            depth_heatmap = cv2.resize(depth_heatmap, (w, h))
        
        # Create side-by-side view
        composite = np.hstack([color_image, depth_heatmap])
        comp_h, comp_w = composite.shape[:2]
        
        # Add info panel at bottom
        info_panel_height = 80
        info_panel = np.zeros((info_panel_height, comp_w, 3), dtype=np.uint8)
        
        # Add text info
        y_offset = 25
        
        # Frame info
        cv2.putText(info_panel, f"Frame: {frame_num} | Time: {timestamp:.2f}s", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Drone pose
        if drone_pose:
            pose_text = f"Drone: X={drone_pose['x']:.2f} Y={drone_pose['y']:.2f} Z={drone_pose['z']:.2f}"
            cv2.putText(info_panel, pose_text, (10, y_offset + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Human pose (from Vicon)
        if human_pose:
            human_text = f"Human: X={human_pose['x']:.2f} Y={human_pose['y']:.2f} Z={human_pose['z']:.2f}"
            cv2.putText(info_panel, human_text, (10, y_offset + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Labels
        cv2.putText(info_panel, "RGB + Detections", (w//2 - 80, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, "Depth Heatmap", (w + w//2 - 60, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine
        final = np.vstack([composite, info_panel])
        
        return final
    
    def find_nearest_pose(self, timestamp: float, pose_list: list) -> dict:
        """Find the pose nearest to the given timestamp."""
        if not pose_list:
            return None
        
        best_pose = None
        best_diff = float('inf')
        
        for pose in pose_list:
            diff = abs(pose['timestamp'] - timestamp)
            if diff < best_diff:
                best_diff = diff
                best_pose = pose
        
        # Only return if within 0.5 seconds
        if best_diff < 0.5:
            return best_pose
        return None
    
    def generate(self):
        """Generate the video from the bag."""
        print(f"Processing bag: {self.bag_path}")
        print(f"Output: {self.output_path}")
        
        # First pass: collect all data with timestamps
        color_frames = []
        depth_frames = {}
        drone_poses = []
        human_poses = []
        
        print("\nReading bag data...")
        
        with Reader(self.bag_path) as reader:
            # Get connections
            color_conn = [c for c in reader.connections if c.topic == '/camera/camera_down/color/image_raw']
            depth_conn = [c for c in reader.connections if c.topic == '/camera/camera_down/depth/image_rect_raw']
            pose_conn = [c for c in reader.connections if c.topic == '/spatial_drone/mavros/local_position/pose']
            human_conn = [c for c in reader.connections if c.topic == '/spatial_drone/vicon/spatial_drone_human/spatial_drone_human']
            
            all_connections = color_conn + depth_conn + pose_conn + human_conn
            
            try:
                for connection, timestamp, rawdata in reader.messages(connections=all_connections):
                    ts_sec = timestamp / 1e9  # Convert nanoseconds to seconds
                    
                    if connection.topic == '/camera/camera_down/color/image_raw':
                        msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                        # Convert to numpy array
                        if msg.encoding == 'rgb8':
                            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        elif msg.encoding == 'bgr8':
                            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                        else:
                            continue
                        color_frames.append({'timestamp': ts_sec, 'image': img})
                        
                    elif connection.topic == '/camera/camera_down/depth/image_rect_raw':
                        msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                        # Depth is typically 16UC1
                        if msg.encoding == '16UC1':
                            img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                        else:
                            continue
                        depth_frames[ts_sec] = img
                        
                    elif connection.topic == '/spatial_drone/mavros/local_position/pose':
                        msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                        drone_poses.append({
                            'timestamp': ts_sec,
                            'x': msg.pose.position.x,
                            'y': msg.pose.position.y,
                            'z': msg.pose.position.z
                        })
                        
                    elif connection.topic == '/spatial_drone/vicon/spatial_drone_human/spatial_drone_human':
                        msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                        human_poses.append({
                            'timestamp': ts_sec,
                            'x': msg.pose.position.x,
                            'y': msg.pose.position.y,
                            'z': msg.pose.position.z
                        })
                    
                    if len(color_frames) % 500 == 0 and len(color_frames) > 0:
                        print(f"  Read {len(color_frames)} color frames...")
                        
            except Exception as e:
                print(f"  Warning: Stopped reading due to: {e}")
        
        print(f"\nLoaded: {len(color_frames)} color frames, {len(depth_frames)} depth frames")
        print(f"        {len(drone_poses)} drone poses, {len(human_poses)} human poses")
        
        if not color_frames:
            print("No color frames found!")
            return
        
        # Sort color frames by timestamp
        color_frames.sort(key=lambda x: x['timestamp'])
        start_time = color_frames[0]['timestamp']
        
        # Initialize video writer
        sample_frame = color_frames[0]['image']
        h, w = sample_frame.shape[:2]
        # Composite will be 2x width + info panel
        output_size = (w * 2, h + 80)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, output_size)
        
        if not writer.isOpened():
            print(f"Failed to open video writer for {self.output_path}")
            return
        
        print(f"\nGenerating video at {self.fps} FPS...")
        print(f"Output resolution: {output_size[0]}x{output_size[1]}")
        
        # Combined prompt for detection
        detection_prompt = f"{self.human_prompt} . {self.object_prompt}"
        
        # Process each frame
        for i, frame_data in enumerate(color_frames):
            color_image = frame_data['image']
            timestamp = frame_data['timestamp']
            relative_time = timestamp - start_time
            
            # Find nearest depth frame
            depth_image = None
            best_depth_diff = float('inf')
            for depth_ts, depth_img in depth_frames.items():
                diff = abs(depth_ts - timestamp)
                if diff < best_depth_diff:
                    best_depth_diff = diff
                    if diff < 0.1:  # Within 100ms
                        depth_image = depth_img
            
            # Run detection if model available
            if self.model is not None:
                boxes, confs, phrases = self.detect_objects(color_image, detection_prompt)
                annotated = self.draw_detections(color_image, boxes, confs, phrases, depth_image)
            else:
                annotated = color_image.copy()
            
            # Create depth heatmap
            if depth_image is not None:
                depth_heatmap = self.create_depth_heatmap(depth_image)
            else:
                depth_heatmap = np.zeros_like(annotated)
            
            # Find nearest poses
            drone_pose = self.find_nearest_pose(timestamp, drone_poses)
            human_pose = self.find_nearest_pose(timestamp, human_poses)
            
            # Create composite frame
            composite = self.create_composite_frame(
                annotated, depth_heatmap,
                drone_pose, human_pose,
                i, relative_time
            )
            
            # Write frame
            writer.write(composite)
            
            # Show preview
            if self.show_preview:
                preview = cv2.resize(composite, (composite.shape[1]//2, composite.shape[0]//2))
                cv2.imshow('Preview', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(color_frames)} frames ({100*(i+1)/len(color_frames):.1f}%)")
        
        writer.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        
        print(f"\nVideo saved to: {self.output_path}")
        print(f"Total frames: {len(color_frames)}")
        print(f"Duration: {relative_time:.1f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Generate MP4 video from ROS2 bag with detections')
    parser.add_argument('bag_path', type=str, help='Path to the ROS2 bag directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output video path (default: <bag_name>.mp4)')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Output video FPS (default: 10)')
    parser.add_argument('--human-prompt', type=str, default='human . person',
                        help='GroundingDINO prompt for humans')
    parser.add_argument('--object-prompt', type=str, default='orange colored cuboid . red bag',
                        help='GroundingDINO prompt for objects')
    parser.add_argument('--preview', action='store_true',
                        help='Show preview while generating')
    parser.add_argument('--no-detection', action='store_true',
                        help='Skip detection (faster, just RGB + depth)')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        bag_name = os.path.basename(args.bag_path.rstrip('/'))
        args.output = os.path.join(os.path.dirname(args.bag_path), f"{bag_name}.mp4")
    
    # Create generator
    generator = BagVideoGenerator(
        bag_path=args.bag_path,
        output_path=args.output,
        human_prompt=args.human_prompt,
        object_prompt=args.object_prompt,
        fps=args.fps,
        show_preview=args.preview
    )
    
    if args.no_detection:
        generator.model = None
    
    # Generate video
    generator.generate()


if __name__ == '__main__':
    main()
