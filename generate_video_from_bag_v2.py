#!/usr/bin/env python3
"""
Generate MP4 video from ROS2 bag with:
- RGB camera feed with detection bounding boxes (GroundingDINO)
- MediaPipe Pose estimation with orientation
- Depth heatmap overlay
- Position annotations (X, Y, Z)
- Split view layout

Based on combined_pose_gdino_realsense.py logic.

Usage:
    python3 generate_video_from_bag_v2.py <bag_path> [--output video.mp4]
"""

import argparse
import os
import sys
import cv2
import numpy as np
from collections import deque
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# Force Matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

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

# Try to import MediaPipe
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe loaded successfully")
except ImportError as e:
    print(f"MediaPipe not available: {e}")


class BagVideoGenerator:
    def __init__(self, bag_path: str, output_path: str, 
                 human_prompt: str = "human",
                 object_prompt: str = "orange colored cuboid",
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
        
        # Detection settings (same as combined script)
        self.box_threshold = 0.30
        self.text_threshold = 0.20
        
        # GroundingDINO model
        self.model = None
        self.device = None
        if GDINO_AVAILABLE:
            self._load_gdino_model()
        
        # MediaPipe Pose
        self.pose = None
        if MEDIAPIPE_AVAILABLE:
            self.pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
        
        # Camera intrinsics (will be updated from bag if available)
        self.intrinsics = {
            'fx': 615.0, 'fy': 615.0,
            'cx': 320.0, 'cy': 240.0
        }
        
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
    
    def preprocess_image_gdino(self, image_np):
        """Preprocess image for Grounding DINO."""
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_pil = PILImage.fromarray(image_np)
        image_transformed, _ = transform(image_pil, None)
        return image_transformed
    
    def detect_objects(self, image: np.ndarray, prompt: str):
        """Run GroundingDINO detection."""
        if self.model is None:
            return [], [], []
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = self.preprocess_image_gdino(image_rgb)
            
            # Run detection
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image_tensor,
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
    
    def get_xyz_from_depth(self, x_pixel, y_pixel, depth_m):
        """Convert pixel coordinates + depth to 3D coordinates."""
        X = (x_pixel - self.intrinsics['cx']) * depth_m / self.intrinsics['fx']
        Y = (y_pixel - self.intrinsics['cy']) * depth_m / self.intrinsics['fy']
        Z = depth_m
        return X, Y, Z
    
    def get_depth_at_box(self, depth_map, box, method='median'):
        """Get depth value at bounding box location."""
        h, w = depth_map.shape
        
        # Box is [cx, cy, w, h] normalized
        cx, cy, bw, bh = box
        cx_px, cy_px = cx * w, cy * h
        bw_px, bh_px = bw * w, bh * h
        
        x1 = int(max(0, cx_px - bw_px / 2))
        y1 = int(max(0, cy_px - bh_px / 2))
        x2 = int(min(w, cx_px + bw_px / 2))
        y2 = int(min(h, cy_px + bh_px / 2))
        
        # Extract depth region
        depth_region = depth_map[y1:y2, x1:x2]
        
        if method == 'median':
            valid_depths = depth_region[depth_region > 0]
            if len(valid_depths) > 0:
                return np.median(valid_depths), int(cx_px), int(cy_px)
        
        return depth_map[int(cy_px), int(cx_px)], int(cx_px), int(cy_px)
    
    def calculate_orientation(self, landmarks):
        """Calculate human orientation based on shoulder and hip positions using 3D cross product."""
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate shoulder midpoint
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_mid_z = (left_shoulder.z + right_shoulder.z) / 2
        
        # Calculate hip midpoint
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2
        hip_mid_z = (left_hip.z + right_hip.z) / 2
        
        # Vector from shoulders to hips (torso direction)
        torso_vector = np.array([
            hip_mid_x - shoulder_mid_x,
            hip_mid_y - shoulder_mid_y,
            hip_mid_z - shoulder_mid_z
        ])
        
        # Vector across shoulders (left to right)
        shoulder_vector = np.array([
            right_shoulder.x - left_shoulder.x,
            right_shoulder.y - left_shoulder.y,
            right_shoulder.z - left_shoulder.z
        ])
        
        # Normal vector to the body plane (3D cross product)
        normal_vector = np.cross(shoulder_vector, torso_vector)
        
        # Project normal vector onto XZ plane (horizontal plane)
        normal_xz = np.array([normal_vector[0], normal_vector[2]])
        
        # Calculate angle
        if np.linalg.norm(normal_xz) > 0:
            normal_xz_normalized = normal_xz / np.linalg.norm(normal_xz)
            angle_rad = np.arctan2(normal_xz_normalized[0], normal_xz_normalized[1])
            angle_deg = np.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
        else:
            angle_deg = 0
        
        # Confidence based on visibility of key landmarks
        confidence = min(
            left_shoulder.visibility, right_shoulder.visibility,
            left_hip.visibility, right_hip.visibility
        )
        
        # Orientation label
        orientation = self.angle_to_orientation(angle_deg)
        
        return orientation, angle_deg, confidence
    
    def angle_to_orientation(self, angle):
        """Convert angle to 8-direction orientation label."""
        if 337.5 <= angle or angle < 22.5:
            return "FRONT"
        elif 22.5 <= angle < 67.5:
            return "FRONT-RIGHT"
        elif 67.5 <= angle < 112.5:
            return "RIGHT"
        elif 112.5 <= angle < 157.5:
            return "BACK-RIGHT"
        elif 157.5 <= angle < 202.5:
            return "BACK"
        elif 202.5 <= angle < 247.5:
            return "BACK-LEFT"
        elif 247.5 <= angle < 292.5:
            return "LEFT"
        elif 292.5 <= angle < 337.5:
            return "FRONT-LEFT"
        return "UNKNOWN"
    
    def draw_detections(self, image: np.ndarray, boxes, confidences, phrases, depth_image=None):
        """Draw bounding boxes and labels on image (same logic as combined script)."""
        h, w = image.shape[:2]
        annotated = image.copy()
        
        for box, conf, phrase in zip(boxes, confidences, phrases):
            # GroundingDINO returns boxes in [cx, cy, w, h] format (normalized 0-1)
            cx, cy, bw, bh = box
            cx_px, cy_px = cx * w, cy * h
            bw_px, bh_px = bw * w, bh * h
            
            x1 = int(cx_px - bw_px / 2)
            y1 = int(cy_px - bh_px / 2)
            x2 = int(cx_px + bw_px / 2)
            y2 = int(cy_px + bh_px / 2)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            # Get depth and calculate 3D position
            depth_str = ""
            if depth_image is not None:
                depth_m, px_x, px_y = self.get_depth_at_box(depth_image, box)
                if depth_m > 0 and not np.isnan(depth_m):
                    X, Y, Z = self.get_xyz_from_depth(px_x, px_y, depth_m)
                    depth_str = f"X={X:.2f}m Y={Y:.2f}m Z={Z:.2f}m"
            
            # Determine color based on detection type
            phrase_lower = phrase.lower()
            if 'human' in phrase_lower or 'person' in phrase_lower:
                color = (0, 255, 0)  # Green for humans
            else:
                color = (255, 0, 255)  # Magenta for objects (same as combined script)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{phrase}: {conf:.2f}"
            label_y = max(20, y1 - 30)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated, (x1, label_y - 20), 
                        (x1 + label_size[0] + 10, label_y), color, -1)
            cv2.putText(annotated, label, (x1 + 5, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw distance overlay
            if depth_str:
                text_y = max(25, y1 - 10)
                text_size = cv2.getTextSize(depth_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated, (x1, text_y - text_size[1] - 5), 
                            (x1 + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(annotated, depth_str, (x1, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return annotated
    
    def process_pose(self, image, annotated):
        """Process MediaPipe pose and draw landmarks."""
        if self.pose is None:
            return annotated, None
        
        h, w = image.shape[:2]
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = self.pose.process(image_rgb)
        
        pose_data = None
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Calculate orientation
            orientation, angle, confidence = self.calculate_orientation(
                results.pose_landmarks.landmark
            )
            
            pose_data = {
                'orientation': orientation,
                'angle': angle,
                'confidence': confidence
            }
            
            # Display pose information
            text = f"Pose: {orientation} | Angle: {angle:.1f} | Conf: {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated, (10, 60), (20 + text_size[0], 95), (0, 0, 0), -1)
            cv2.rectangle(annotated, (10, 60), (20 + text_size[0], 95), (255, 255, 255), 2)
            cv2.putText(annotated, text, (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw orientation arrow
            center_x = w // 2
            center_y = h - 80
            arrow_length = 60
            end_x = int(center_x + arrow_length * np.sin(np.radians(angle)))
            end_y = int(center_y - arrow_length * np.cos(np.radians(angle)))
            
            cv2.circle(annotated, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.arrowedLine(annotated, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
            cv2.putText(annotated, "Body Direction", (center_x - 70, center_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated, pose_data
    
    def create_depth_heatmap(self, depth_image: np.ndarray, max_depth: float = 5.0) -> np.ndarray:
        """Convert depth image to colorful heatmap."""
        # Depth should already be in meters
        depth_normalized = np.clip(depth_image / max_depth, 0, 1)
        depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth_colored[depth_image == 0] = [0, 0, 0]
        return depth_colored
    
    def create_composite_frame(self, color_image: np.ndarray, depth_heatmap: np.ndarray,
                               drone_pose: dict = None, human_pose: dict = None,
                               pose_data: dict = None, num_detections: int = 0,
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
        info_panel_height = 100
        info_panel = np.zeros((info_panel_height, comp_w, 3), dtype=np.uint8)
        
        # Add text info
        y_offset = 25
        
        # Frame info
        info_text = f"Frame: {frame_num} | Time: {timestamp:.2f}s | Objects: {num_detections}"
        if pose_data:
            info_text += f" | Pose: {pose_data['orientation']}"
        cv2.putText(info_panel, info_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Drone pose
        if drone_pose:
            pose_text = f"Drone: X={drone_pose['x']:.2f} Y={drone_pose['y']:.2f} Z={drone_pose['z']:.2f}"
            cv2.putText(info_panel, pose_text, (10, y_offset + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Human pose (from Vicon)
        if human_pose:
            human_text = f"Human (Vicon): X={human_pose['x']:.2f} Y={human_pose['y']:.2f} Z={human_pose['z']:.2f}"
            cv2.putText(info_panel, human_text, (10, y_offset + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Labels
        cv2.putText(info_panel, "RGB + Detections + Pose", (w//2 - 100, y_offset + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, "Depth Heatmap", (w + w//2 - 60, y_offset + 75), 
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
        
        if best_diff < 0.5:
            return best_pose
        return None
    
    def generate(self, max_frames=None):
        """Generate the video from the bag."""
        print(f"Processing bag: {self.bag_path}")
        print(f"Output: {self.output_path}")
        if max_frames:
            print(f"Max frames: {max_frames}")
        
        # Detect camera topic prefix
        camera_prefix = None
        with Reader(self.bag_path) as reader:
            for conn in reader.connections:
                if 'color/image_raw' in conn.topic:
                    # Extract prefix like /camera/camera_down or /camera/camera
                    camera_prefix = conn.topic.replace('/color/image_raw', '')
                    break
        
        if camera_prefix is None:
            print("No camera topics found!")
            return
        
        print(f"Using camera prefix: {camera_prefix}")
        
        color_topic = f"{camera_prefix}/color/image_raw"
        depth_topic = f"{camera_prefix}/depth/image_rect_raw"
        
        # First pass: collect all data with timestamps
        color_frames = []
        depth_frames = {}
        drone_poses = []
        human_poses = []
        
        print("\nReading bag data...")
        
        with Reader(self.bag_path) as reader:
            # Get connections
            color_conn = [c for c in reader.connections if c.topic == color_topic]
            depth_conn = [c for c in reader.connections if c.topic == depth_topic]
            pose_conn = [c for c in reader.connections if c.topic == '/spatial_drone/mavros/local_position/pose']
            human_conn = [c for c in reader.connections if c.topic == '/spatial_drone/vicon/spatial_drone_human/spatial_drone_human']
            
            all_connections = color_conn + depth_conn + pose_conn + human_conn
            
            try:
                for connection, timestamp, rawdata in reader.messages(connections=all_connections):
                    ts_sec = timestamp / 1e9
                    
                    if connection.topic == color_topic:
                        # Check if we've reached max frames
                        if max_frames and len(color_frames) >= max_frames:
                            continue
                        msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                        if msg.encoding == 'rgb8':
                            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        elif msg.encoding == 'bgr8':
                            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                        else:
                            continue
                        color_frames.append({'timestamp': ts_sec, 'image': img})
                        
                    elif connection.topic == depth_topic:
                        msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                        if msg.encoding == '16UC1':
                            img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                            # Convert to meters
                            img = img.astype(np.float32) / 1000.0
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
                    
                    if len(color_frames) % 100 == 0 and len(color_frames) > 0:
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
        output_size = (w * 2, h + 100)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, output_size)
        
        if not writer.isOpened():
            print(f"Failed to open video writer for {self.output_path}")
            return
        
        print(f"\nGenerating video at {self.fps} FPS...")
        print(f"Output resolution: {output_size[0]}x{output_size[1]}")
        
        # Combined prompt for detection
        detection_prompt = self.human_prompt
        if self.object_prompt:
            detection_prompt = f"{self.human_prompt} . {self.object_prompt}"
        
        print(f"Detection prompt: {detection_prompt}")
        
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
                    if diff < 0.1:
                        depth_image = depth_img
            
            # Start with copy of color image
            annotated = color_image.copy()
            
            # Run detection if model available
            num_detections = 0
            if self.model is not None:
                boxes, confs, phrases = self.detect_objects(color_image, detection_prompt)
                num_detections = len(boxes)
                annotated = self.draw_detections(annotated, boxes, confs, phrases, depth_image)
            
            # Process pose
            annotated, pose_data = self.process_pose(color_image, annotated)
            
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
                drone_pose, human_pose, pose_data, num_detections,
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
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(color_frames)} frames ({100*(i+1)/len(color_frames):.1f}%)")
        
        writer.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        
        if self.pose:
            self.pose.close()
        
        print(f"\nVideo saved to: {self.output_path}")
        print(f"Total frames: {len(color_frames)}")
        print(f"Duration: {relative_time:.1f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Generate MP4 video from ROS2 bag with detections and pose')
    parser.add_argument('bag_path', type=str, help='Path to the ROS2 bag directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output video path (default: <bag_name>_video.mp4)')
    parser.add_argument('--fps', type=float, default=15.0,
                        help='Output video FPS (default: 15)')
    parser.add_argument('--human-prompt', type=str, default='human',
                        help='GroundingDINO prompt for humans')
    parser.add_argument('--object-prompt', type=str, default='orange colored cuboid',
                        help='GroundingDINO prompt for objects')
    parser.add_argument('--preview', action='store_true',
                        help='Show preview while generating')
    parser.add_argument('--no-detection', action='store_true',
                        help='Skip detection (faster, just RGB + depth)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process (default: all)')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        bag_name = os.path.basename(args.bag_path.rstrip('/'))
        args.output = os.path.join(os.path.dirname(args.bag_path), f"{bag_name}_video.mp4")
    
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
    generator.generate(max_frames=args.max_frames)


if __name__ == '__main__':
    main()
