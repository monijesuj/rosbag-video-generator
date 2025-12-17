#!/usr/bin/env python3
"""
Extract ROS2 bag topics to CSV files.
Extracts:
- /spatial_drone/mavros/local_position/pose -> drone_pose.csv
- /spatial_drone/mavros/local_position/velocity_body -> drone_velocity_body.csv
- Vicon topics (optional)
"""

import argparse
import csv
import os
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def extract_pose_to_csv(bag_path: str, output_dir: str, typestore):
    """Extract local_position/pose to CSV."""
    topic = '/spatial_drone/mavros/local_position/pose'
    output_file = os.path.join(output_dir, 'drone_pose.csv')
    
    print(f"Extracting {topic} to {output_file}...")
    
    with Reader(bag_path) as reader:
        # Find the connection for this topic
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"  Topic {topic} not found in bag!")
            return 0
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            writer.writerow([
                'timestamp_sec', 'timestamp_nanosec', 'timestamp_float',
                'frame_id',
                'position_x', 'position_y', 'position_z',
                'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'
            ])
            
            count = 0
            try:
                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    
                    # Convert timestamp to seconds
                    ts_sec = msg.header.stamp.sec
                    ts_nanosec = msg.header.stamp.nanosec
                    ts_float = ts_sec + ts_nanosec * 1e-9
                    
                    writer.writerow([
                        ts_sec, ts_nanosec, f"{ts_float:.9f}",
                        msg.header.frame_id,
                        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                        msg.pose.orientation.x, msg.pose.orientation.y, 
                        msg.pose.orientation.z, msg.pose.orientation.w
                    ])
                    count += 1
                    
                    if count % 5000 == 0:
                        print(f"  Processed {count} messages...")
            except Exception as e:
                print(f"  Warning: Stopped early due to: {e}")
    
    print(f"  Extracted {count} pose messages")
    return count


def extract_velocity_to_csv(bag_path: str, output_dir: str, typestore):
    """Extract local_position/velocity_body to CSV."""
    topic = '/spatial_drone/mavros/local_position/velocity_body'
    output_file = os.path.join(output_dir, 'drone_velocity_body.csv')
    
    print(f"Extracting {topic} to {output_file}...")
    
    with Reader(bag_path) as reader:
        # Find the connection for this topic
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"  Topic {topic} not found in bag!")
            return 0
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            writer.writerow([
                'timestamp_sec', 'timestamp_nanosec', 'timestamp_float',
                'frame_id',
                'linear_x', 'linear_y', 'linear_z',
                'angular_x', 'angular_y', 'angular_z'
            ])
            
            count = 0
            try:
                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    
                    # Convert timestamp to seconds
                    ts_sec = msg.header.stamp.sec
                    ts_nanosec = msg.header.stamp.nanosec
                    ts_float = ts_sec + ts_nanosec * 1e-9
                    
                    writer.writerow([
                        ts_sec, ts_nanosec, f"{ts_float:.9f}",
                        msg.header.frame_id,
                        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
                    ])
                    count += 1
                    
                    if count % 5000 == 0:
                        print(f"  Processed {count} messages...")
            except Exception as e:
                print(f"  Warning: Stopped early due to: {e}")
    
    print(f"  Extracted {count} velocity messages")
    return count


def extract_vicon_to_csv(bag_path: str, output_dir: str, topic: str, output_name: str, typestore):
    """Extract Vicon pose topic to CSV."""
    output_file = os.path.join(output_dir, output_name)
    
    print(f"Extracting {topic} to {output_file}...")
    
    with Reader(bag_path) as reader:
        # Find the connection for this topic
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"  Topic {topic} not found in bag!")
            return 0
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            writer.writerow([
                'timestamp_sec', 'timestamp_nanosec', 'timestamp_float',
                'frame_id',
                'position_x', 'position_y', 'position_z',
                'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'
            ])
            
            count = 0
            try:
                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    
                    # Convert timestamp to seconds
                    ts_sec = msg.header.stamp.sec
                    ts_nanosec = msg.header.stamp.nanosec
                    ts_float = ts_sec + ts_nanosec * 1e-9
                    
                    writer.writerow([
                        ts_sec, ts_nanosec, f"{ts_float:.9f}",
                        msg.header.frame_id,
                        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                        msg.pose.orientation.x, msg.pose.orientation.y, 
                        msg.pose.orientation.z, msg.pose.orientation.w
                    ])
                    count += 1
                    
                    if count % 10000 == 0:
                        print(f"  Processed {count} messages...")
            except Exception as e:
                print(f"  Warning: Stopped early due to: {e}")
    
    print(f"  Extracted {count} messages")
    return count


def main():
    parser = argparse.ArgumentParser(description='Extract ROS2 bag topics to CSV files')
    parser.add_argument('bag_path', type=str, help='Path to the ROS2 bag directory')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory for CSV files (default: same as bag)')
    parser.add_argument('--include-vicon', '-v', action='store_true',
                        help='Also extract Vicon ground truth topics')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.bag_path) if os.path.isfile(args.bag_path) else args.bag_path
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get the typestore for ROS2 Humble
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    print(f"Extracting from: {args.bag_path}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Extract pose
    extract_pose_to_csv(args.bag_path, args.output_dir, typestore)
    print()
    
    # Extract velocity
    extract_velocity_to_csv(args.bag_path, args.output_dir, typestore)
    print()
    
    # Optionally extract Vicon data
    if args.include_vicon:
        extract_vicon_to_csv(
            args.bag_path, args.output_dir,
            '/spatial_drone/vicon/spatial_drone_object/spatial_drone_object',
            'vicon_drone.csv',
            typestore
        )
        print()
        
        extract_vicon_to_csv(
            args.bag_path, args.output_dir,
            '/spatial_drone/vicon/spatial_drone_human/spatial_drone_human',
            'vicon_human.csv',
            typestore
        )
        print()
    
    print("Done!")
    print(f"\nCSV files saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
