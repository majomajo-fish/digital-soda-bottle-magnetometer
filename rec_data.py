#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rec_data.py - Camera Image Capture and Laser Spot Analysis Program

This program captures images from a camera and analyzes the center position
of laser spots in real-time. Supports multiple spot analysis and automatic
c_y value adjustment.

Usage:
    python rec_data.py <parameter_file> [--realtime]
    
Author: Kyoto University Geophysics Laboratory
"""

import os
import sys
import shutil
import datetime
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import traceback

# Import function from data_utils.py
from data_utils import get_param_value


# --- Parameter Loading Functions ---
def load_params(param_file):
    """Load parameter file, handling comments and empty lines"""
    params = []
    try:
        with open(param_file) as f:
            for line in f:
                # Use get_param_value from data_utils.py to remove comments
                line_content = get_param_value(line)
                # Add only non-empty lines to the list
                if line_content:
                    params.append(line_content)
                else:
                    # Add empty string for empty lines (to preserve line numbers)
                    params.append('')
        return params
    except FileNotFoundError:
        print(f"Configuration file not found: {param_file}")
        sys.exit(1)


def parse_c_y_values(c_y_param):
    """Parse c_y parameter and return a list of numeric values"""
    if not c_y_param:
        return [400]  # Default value
    
    # Extract numeric values from string
    c_y_values = []
    
    # Split by comma, space, tab, or newline
    import re
    parts = re.split(r'[,\s\t\n]+', str(c_y_param).strip())
    
    for part in parts:
        part = part.strip()
        if part:  # If not empty
            try:
                value = int(part)
                c_y_values.append(value)
            except ValueError:
                print(f"Warning: Cannot convert c_y value '{part}' to number. Skipping.")
    
    # Use default value if no valid values found
    if not c_y_values:
        print("Warning: No valid c_y values found. Using default value 400.")
        c_y_values = [400]
    
    print(f"Detected c_y values: {c_y_values}")
    return c_y_values


# --- Parameter File Copy Function ---
def copy_param_file(param_file_path, base_dir):
    """Copy parameter file to base_dir/param directory"""
    try:
        param_dir = base_dir / 'param'
        param_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = Path(param_file_path).name
        copied_name = f"{timestamp}_{original_name}"
        copied_path = param_dir / copied_name
        
        # Copy the file
        shutil.copy2(param_file_path, copied_path)
        print(f"Parameter file copied: {copied_path}")
        return copied_path
        
    except Exception as e:
        print(f"Failed to copy parameter file: {e}")
        return None


# --- Directory Preparation Function ---
def prepare_directories(base_dir):
    """Create directories recursively if they don't exist"""
    raw_temp = base_dir / 'raw_temp'
    
    # Create parent directory recursively if it doesn't exist
    if not base_dir.parent.exists():
        base_dir.parent.mkdir(parents=True, exist_ok=True)
        
    for d in [base_dir, raw_temp]:
        d.mkdir(parents=True, exist_ok=True)
    return raw_temp


# --- Camera Initialization Function ---
def initialize_camera(device_index=0):
    """Initialize the camera"""
    try:
        # Convert device_index to integer if it's a string
        if isinstance(device_index, str):
            if device_index.lower() == "default":
                device_index = 0
            else:
                try:
                    device_index = int(device_index)
                except ValueError:
                    print(f"Warning: Cannot convert device name '{device_index}' to number. Using default (0).")
                    device_index = 0
        
        camera = cv2.VideoCapture(device_index)
        
        if not camera.isOpened():
            print(f"Failed to initialize camera {device_index}")
            return None
        
        # Camera settings (720p quality as default)
        # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Other settings use default values

        print(f"Initialized camera {device_index}")
        return camera
        
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return None


# --- Image Capture Function ---
def capture_single_image(camera, save_dir=None, fixed_filename=None, return_array=False):
    """Capture a single image using OpenCV VideoCapture
    
    Args:
        camera: Camera object
        save_dir: Save directory (not required when return_array=True)
        fixed_filename: Fixed filename
        return_array: If True, returns image array; if False, saves to file
    
    Returns:
        return_array=True: Tuple of (success, frame, timestamp)
        return_array=False: File path as before (or None)
    """
    try:
        # Capture image from camera
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image from camera")
            if return_array:
                return False, None, None
            else:
                return None
        
        timestamp = datetime.datetime.now()
        
        if return_array:
            # Return image array directly
            return True, frame, timestamp
        else:
            # Save to file as before
            timestamp_str = timestamp.strftime("%Y%m%d%H%M%S")
            
            # Use fixed filename if specified
            if fixed_filename:
                output_path = save_dir / fixed_filename
            else:
                output_path = save_dir / f"snapshot-{timestamp_str}.jpg"
            
            # Save image
            success = cv2.imwrite(str(output_path), frame)
            if success:
                print(f"Image captured: {output_path}")
                return output_path
            else:
                print("Failed to save image")
                return None
            
    except Exception as e:
        print(f"Failed to capture image: {e}")
        if return_array:
            return False, None, None
        else:
            return None


# --- Image Center Position Calculation ---
def calc_center_position(image_path, c_y, n_conv, is_auto_mode=False):
    """Calculate center position from a single image"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot read image: {image_path}")
        return 0, 0, datetime.datetime.now()
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return calc_center_from_array(img_rgb, c_y, n_conv, image_path, False, is_auto_mode)


def calc_center_from_frame(frame_bgr, c_y, n_conv, timestamp=None, is_auto_mode=False):
    """Calculate center position from BGR frame (directly from OpenCV camera)"""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return calc_center_from_array(img_rgb, c_y, n_conv, None, False, is_auto_mode)


def calc_multiple_spots_from_frame(frame_bgr, c_y_list, n_conv, timestamp=None, auto_mode_flags=None):
    """Calculate center positions of multiple spots from BGR frame"""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if timestamp is None:
        timestamp = datetime.datetime.now()
    
    results = []
    for i, c_y in enumerate(c_y_list):
        # Determine auto mode flag (treat as False if None)
        is_auto_mode = auto_mode_flags[i] if auto_mode_flags and i < len(auto_mode_flags) else False
        cen_x, cen_y, _ = calc_center_from_array(img_rgb, c_y, n_conv, None, False, is_auto_mode)
        results.append({
            'spot_id': i + 1,
            'c_y': c_y,
            'cen_x': cen_x,
            'cen_y': cen_y,
            'timestamp': timestamp
        })
    
    return results


def calc_multiple_spots_from_path(image_path, c_y_list, n_conv, auto_mode_flags=None):
    """Calculate center positions of multiple spots from image file"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot read image: {image_path}")
        return []
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(image_path))
    
    results = []
    for i, c_y in enumerate(c_y_list):
        # Determine auto mode flag (treat as False if None)
        is_auto_mode = auto_mode_flags[i] if auto_mode_flags and i < len(auto_mode_flags) else False
        cen_x, cen_y, _ = calc_center_from_array(img_rgb, c_y, n_conv, image_path, False, is_auto_mode)
        results.append({
            'spot_id': i + 1,
            'c_y': c_y,
            'cen_x': cen_x,
            'cen_y': cen_y,
            'timestamp': timestamp
        })
    
    return results


def calc_center_from_array(img_rgb, c_y, n_conv, image_path=None, return_details=False, is_auto_mode=False):
    """Calculate center position from RGB image array
    
    Args:
        img_rgb: RGB image array
        c_y: Center y-coordinate for analysis
        n_conv: Convolution width (smoothing window size)
        image_path: Image file path (optional)
        return_details: Whether to return detailed data
        is_auto_mode: Whether in auto mode (if True, averaging width is set to ±10)
    """
    f_img_rgb = img_rgb.astype(np.float32)
    img_r = f_img_rgb[:, :, 0]
    
    # Determine y-range (±10 for auto mode, ±75 for manual mode)
    y_half_width = 10 if is_auto_mode else 75
    yrange = [c_y - y_half_width, c_y + y_half_width]
    if yrange[0] < 0:
        yrange[0] = 0
    if yrange[1] >= img_r.shape[0]:
        yrange[1] = img_r.shape[0] - 1
        
    redmap_cut = img_r[yrange[0]:yrange[1], :]
    x_dist = np.mean(redmap_cut, axis=0)
    b = np.ones(n_conv) / n_conv
    x_mean = np.convolve(x_dist, b, mode="same")
    cen_x = np.mean(np.where(x_mean == np.max(x_mean)))
    if np.isnan(cen_x):
        cen_x = 0
        
    # y-direction center (search within yrange to avoid picking up other spots)
    if cen_x > 0 and cen_x < img_r.shape[1]:
        # Get y-direction intensity distribution within yrange
        redmap_cut_y = img_r[yrange[0]:yrange[1], round(cen_x)]
        y_mean = np.convolve(redmap_cut_y, b, mode="same")
        # Get relative position within yrange
        cen_y_relative = np.mean(np.where(y_mean == np.max(y_mean)))
        if np.isnan(cen_y_relative):
            cen_y = c_y  # Use c_y as default
        else:
            # Convert to absolute position (add yrange[0])
            cen_y = yrange[0] + cen_y_relative
    else:
        cen_y = c_y  # Use c_y as default
        
    # Get timestamp
    if image_path:
        try:
            timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(image_path))
        except (OSError, TypeError):
            timestamp = datetime.datetime.now()
    else:
        timestamp = datetime.datetime.now()
    
    # Return detailed data if requested
    if return_details:
        x_pixels = np.arange(len(x_dist))
        return cen_x, cen_y, timestamp, {
            'yrange': yrange,
            'cross_section': x_dist,
            'smoothed': x_mean,
            'x_pixels': x_pixels
        }
    else:
        return cen_x, cen_y, timestamp


# --- Auto c_y Determination Function ---
def get_auto_c_y_from_buffer(y_buffer):
    """Automatically determine c_y from median of buffered y-coordinates"""
    if len(y_buffer) > 0:
        # Calculate median of y-coordinates
        c_y = int(np.median(y_buffer))
        print(f"Auto-determined c_y: {c_y} (median of past {len(y_buffer)} points)")
        return c_y
    else:
        print(f"Buffer is empty. Using default value 400.")
        return 400


# --- Y-coordinate Buffer Management Function ---
def update_y_buffer(y_buffer, new_y, max_size=10):
    """Update y-coordinate buffer (maintain maximum size)"""
    y_buffer.append(new_y)
    # Limit buffer size (remove oldest data)
    if len(y_buffer) > max_size:
        y_buffer.pop(0)
    return y_buffer


# --- Multi-Spot Data Management Functions ---
def setup_multi_spot_directories(base_dir, num_spots):
    """Set up data directories and files for multiple spots"""
    spot_info = []
    data_dir = base_dir / 'data'
    
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to create data directory: {data_dir}")
        print(f"  Error: {e}")
        print(f"  Processing will continue, but data saving may fail.")
    
    for i in range(num_spots):
        spot_id = i + 1
        spot_dir = data_dir / f'p{spot_id}'
        
        try:
            spot_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create directory for spot {spot_id}: {spot_dir}")
            print(f"  Error: {e}")
        
        output_file = spot_dir / 'pix_magdata.txt'
        file_exists = output_file.exists()
        
        spot_info.append({
            'spot_id': spot_id,
            'dir': spot_dir,
            'file': output_file,
            'exists': file_exists,
            'buffer': []  # Add data buffer
        })
        
        print(f"Spot {spot_id}: {spot_dir}/pix_magdata.txt")
    
    return spot_info


def buffer_multi_spot_data(spot_info, results):
    """Add multiple spot data to buffer (without writing to file)"""
    for result in results:
        spot_id = result['spot_id']
        timestamp = result['timestamp']
        cen_x = result['cen_x']
        cen_y = result['cen_y']
        
        # Find corresponding spot info
        spot_data = next((s for s in spot_info if s['spot_id'] == spot_id), None)
        if spot_data:
            time_str = timestamp.strftime('%Y/%m/%d %H:%M:%S')
            line = f"{time_str}, {int(cen_x)}, {int(cen_y)}\n"
            spot_data['buffer'].append(line)


def flush_multi_spot_data(spot_info):
    """Write buffered data to files"""
    for spot_data in spot_info:
        if spot_data['buffer']:
            try:
                # Create parent directory if it doesn't exist
                spot_data['file'].parent.mkdir(parents=True, exist_ok=True)
                
                with open(spot_data['file'], 'a') as f:
                    f.writelines(spot_data['buffer'])
                # Clear buffer
                spot_data['buffer'].clear()
            except FileNotFoundError as e:
                print(f"Warning: Failed to write file: {spot_data['file']}")
                print(f"  Error: {e}")
                print(f"  Attempting to create directory and retry...")
                try:
                    spot_data['file'].parent.mkdir(parents=True, exist_ok=True)
                    with open(spot_data['file'], 'a') as f:
                        f.writelines(spot_data['buffer'])
                    spot_data['buffer'].clear()
                    print(f"  Retry successful: {spot_data['file']}")
                except Exception as retry_error:
                    print(f"  Retry failed: {retry_error}")
                    print(f"  Data will be kept in buffer.")
            except Exception as e:
                print(f"Warning: Error while writing file: {spot_data['file']}")
                print(f"  Error: {e}")
                print(f"  Data will be kept in buffer.")


def write_multi_spot_data(spot_info, results):
    """Write multiple spot data to files immediately (kept for backward compatibility)"""
    for result in results:
        spot_id = result['spot_id']
        timestamp = result['timestamp']
        cen_x = result['cen_x']
        cen_y = result['cen_y']
        
        # Find corresponding spot info
        spot_data = next((s for s in spot_info if s['spot_id'] == spot_id), None)
        if spot_data:
            try:
                # Create parent directory if it doesn't exist
                spot_data['file'].parent.mkdir(parents=True, exist_ok=True)
                
                with open(spot_data['file'], 'a') as f:
                    time_str = timestamp.strftime('%Y/%m/%d %H:%M:%S')
                    f.write(f"{time_str}, {int(cen_x)}, {int(cen_y)}\n")
            except FileNotFoundError as e:
                print(f"Warning: Failed to write file: {spot_data['file']}")
                print(f"  Error: {e}")
                print(f"  This data will be skipped.")
            except Exception as e:
                print(f"Warning: Error while writing file: {spot_data['file']}")
                print(f"  Error: {e}")
                print(f"  This data will be skipped.")


# --- Archive Directory Creation Function ---
def create_archive_directory(archive_base_dir):
    """Create archive directory for image storage"""
    archive_dir = archive_base_dir / 'archived_images'
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


# --- Main Processing ---
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image Capture and Analysis Program')
    parser.add_argument('param_file', help='Path to parameter file')
    parser.add_argument('--realtime', '-r', action='store_true', 
                       help='Real-time display mode (for camera position adjustment)')
    
    args = parser.parse_args()
    
    # Normal capture and analysis mode (with real-time display option)
    if len(sys.argv) < 2:
        print("Usage: python rec_data.py <config_file> [--realtime]")
        sys.exit(1)
        
    # Save parameter file path
    param_file_path = args.param_file
    
    # Initial parameter loading
    params = load_params(param_file_path)
    if len(params) < 7:
        print("Insufficient parameters in configuration file")
        sys.exit(1)
    
    # Initial settings (parameters that don't change)
    base_dir = Path(params[6])
    device = params[8] if len(params) > 8 else "default"
    
    # Set temporary photo storage directory from parameter file
    if len(params) > 9 and params[9].strip():
        photo_temp_dir = Path(params[9].strip()).expanduser()
    else:
        photo_temp_dir = base_dir / 'temp'  # Default: use 'temp' under base_dir
    
    # Set archive directory from parameter file
    if len(params) > 10 and params[10].strip():
        archive_base_dir = Path(params[10].strip()).expanduser()
    else:
        archive_base_dir = base_dir / 'images'
    
    # Copy parameter file
    copy_param_file(param_file_path, base_dir)
    
    print(f"Initial setup complete:")
    print(f"  Base directory: {base_dir}")
    print(f"  Camera device: {device}")
    print(f"  Temporary storage: {photo_temp_dir}")
    print(f"  Archive storage: {archive_base_dir}")
    print(f"  Parameter file: {param_file_path}")
    print(f"  * c_y, n_conv, capture interval, and write frequency are dynamically reloaded")
    
    # Initialize camera
    camera = initialize_camera(device)
    if camera is None:
        print("Camera initialization failed. Exiting program.")
        sys.exit(1)
    
    # Prepare directories
    try:
        raw_temp = prepare_directories(photo_temp_dir)
        archive_dir = create_archive_directory(archive_base_dir)
        print(f"Temporary file storage: {raw_temp}")
        print(f"Archive storage: {archive_dir}")
    except Exception as e:
        print(f"Directory creation error: {e}")
        # Use current directory as fallback
        fallback_temp_dir = Path('./photo_temp')
        raw_temp = prepare_directories(fallback_temp_dir)
        archive_dir = create_archive_directory(fallback_temp_dir)
        print(f"Using fallback path: {fallback_temp_dir}")
    
    # Initial parameter loading (dynamic parameters)
    params = load_params(param_file_path)
    c_y_list = parse_c_y_values(params[0])
    num_spots = len(c_y_list)
    n_conv = int(params[3])
    
    # Capture interval setting (in seconds)
    if len(params) > 11 and params[11].strip():
        try:
            capture_interval = float(params[11].strip())
        except ValueError:
            print(f"Warning: Cannot convert capture interval '{params[11]}' to number. Using default (1.0 sec).")
            capture_interval = 1.0
    else:
        capture_interval = 1.0  # Default value
    
    # File write frequency setting
    if len(params) > 13 and params[13].strip():
        try:
            flush_interval = int(params[13].strip())
        except ValueError:
            print(f"Warning: Cannot convert write frequency '{params[13]}' to number. Using default (60 times).")
            flush_interval = 60
    else:
        flush_interval = 60  # Default value
    
    print(f"\nDynamic parameters (can be changed):")
    print(f"  Number of spots: {num_spots}")
    print(f"  c_y values: {c_y_list}")
    print(f"  Convolution width (n_conv): {n_conv}")
    print(f"  Capture interval: {capture_interval} seconds")
    print(f"  Write frequency: every {flush_interval} times (approximately every {flush_interval * capture_interval:.0f} seconds)")
    
    # Prepare output files - same processing for single and multiple spots
    spot_info = setup_multi_spot_directories(base_dir, num_spots)
    
    # Initialize y-coordinate buffers (10-point buffer for each spot)
    y_buffers = [[] for _ in range(num_spots)]
    
    # Check for auto c_y mode (-1 means auto mode)
    auto_mode_flags = [c_y == -1 for c_y in c_y_list]
    
    # Set initial c_y values (use default for auto mode)
    for i, c_y in enumerate(c_y_list):
        if c_y == -1:
            c_y_list[i] = 400  # Initial default value
            print(f"Spot {i+1}: Auto mode (initial value 400, auto-adjusts after measurement starts)")
    
    print(f"Analysis target: {num_spots} spot(s), c_y values: {c_y_list}")
    
    # Initialize real-time display (if --realtime option specified)
    if args.realtime:
        print("Starting real-time display mode...")
        print("Controls: Close window to exit")
        
        # matplotlib settings - adjust according to number of spots (same layout for single spot)
        plt.style.use('default')
        # Same layout for single and multiple spots
        fig, axes = plt.subplots(2, max(2, num_spots), figsize=(max(12, num_spots*4), 8))
        fig.suptitle(f'Real-time Multi-Spot Analysis ({num_spots} spot(s))', fontsize=14)
        if num_spots == 1:
            axes = axes.reshape(2, 1)
        ax1 = axes[0, 0]  # For image display
        cross_section_axes = [axes[1, i] for i in range(num_spots)]  # Cross-section for each spot
        ax3 = axes[0, -1]  # For info display
        
        # Subplot for info text
        ax3.axis('off')
        text_info = ax3.text(0.1, 0.9, '', fontsize=10, transform=ax3.transAxes, 
                            verticalalignment='top')
        text_controls = ax3.text(0.1, 0.3, 
                               f"Real-time Display\n\nSettings:\n• {num_spots} spot(s) analysis\n• c_y values: {c_y_list}", 
                               fontsize=9, transform=ax3.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.ion()  # Interactive mode ON
        plt.show()
        
        print(f"Starting capture and analysis (real-time display, {num_spots} spot(s)). Close window or Ctrl+C to exit.")
    else:
        print(f"Starting capture and analysis ({num_spots} spot(s)). Press Ctrl+C to exit.")
    
    # Initialize image counter
    image_counter = 0
    
    try:
        while True:
            try:
                # Reload parameter file (to support dynamic changes)
                params = load_params(param_file_path)
                c_y_list_new = parse_c_y_values(params[0])
                n_conv_new = int(params[3])
                
                # Capture interval
                if len(params) > 11 and params[11].strip():
                    try:
                        capture_interval_new = float(params[11].strip())
                    except ValueError:
                        capture_interval_new = capture_interval
                else:
                    capture_interval_new = 1.0
                
                # Write frequency
                if len(params) > 13 and params[13].strip():
                    try:
                        flush_interval_new = int(params[13].strip())
                    except ValueError:
                        flush_interval_new = flush_interval
                else:
                    flush_interval_new = 60
                
                # Notify if parameters changed
                param_changed = False
                if c_y_list_new != c_y_list:
                    print(f"\n[Parameter changed] c_y values: {c_y_list} → {c_y_list_new}")
                    
                    # Update auto mode flags (-1 means auto mode)
                    auto_mode_flags_new = [c_y == -1 for c_y in c_y_list_new]
                    
                    # Check spots that switched to auto mode
                    for i, (old_flag, new_flag) in enumerate(zip(auto_mode_flags, auto_mode_flags_new)):
                        if not old_flag and new_flag:
                            print(f"  Spot {i+1}: Manual mode → Auto mode")
                            # Keep current c_y value when switching to auto mode
                            c_y_list_new[i] = c_y_list[i]
                        elif old_flag and not new_flag:
                            print(f"  Spot {i+1}: Auto mode → Manual mode (c_y={c_y_list_new[i]})")
                        # If already in auto mode and -1 specified in parameter file
                        elif old_flag and new_flag:
                            # Keep current c_y value (use actual value, not -1)
                            c_y_list_new[i] = c_y_list[i]
                    
                    auto_mode_flags = auto_mode_flags_new
                    c_y_list = c_y_list_new
                    num_spots = len(c_y_list)
                    
                    # Reset directories and buffers if number of spots changed
                    if len(c_y_list) != len(y_buffers):
                        spot_info = setup_multi_spot_directories(base_dir, num_spots)
                        y_buffers = [[] for _ in range(num_spots)]
                        print(f"  Number of spots changed. Buffers initialized.")
                    
                    param_changed = True
                    
                if n_conv_new != n_conv:
                    print(f"[Parameter changed] n_conv: {n_conv} → {n_conv_new}")
                    n_conv = n_conv_new
                    param_changed = True
                    
                if capture_interval_new != capture_interval:
                    print(f"[Parameter changed] Capture interval: {capture_interval} sec → {capture_interval_new} sec")
                    capture_interval = capture_interval_new
                    param_changed = True
                    
                if flush_interval_new != flush_interval:
                    print(f"[Parameter changed] Write frequency: {flush_interval} times → {flush_interval_new} times")
                    flush_interval = flush_interval_new
                    param_changed = True
                
                if param_changed and args.realtime:
                    # Update real-time display info text
                    text_controls.set_text(
                        f"Real-time Display\n\nSettings:\n• {num_spots} spot(s) analysis\n• c_y values: {c_y_list}")
                
                # Exit if window closed in real-time mode
                if args.realtime and not plt.fignum_exists(fig.number):
                    break
                    
                image_counter += 1
                save_permanent = (image_counter == 1 or image_counter % 300 == 0)
                
                if save_permanent and not args.realtime:
                    # Save file only when archiving
                    archive_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_filename = f"snapshot-{archive_timestamp}.jpg"
                    image_path = capture_single_image(camera, archive_dir, archive_filename, return_array=False)
                    if not image_path:
                        print("Capture failed. Retrying in 5 seconds.")
                        time.sleep(5)
                        continue
                    
                    # Image analysis (from file) - always same processing for multiple spots
                    results = calc_multiple_spots_from_path(image_path, c_y_list, n_conv, auto_mode_flags)
                    
                    # Flush buffer when saving image (to maintain data consistency)
                    flush_multi_spot_data(spot_info)
                    
                    print(f"Image saved (capture #{image_counter}): {image_path}")
                else:
                    # Normally get image array directly for analysis
                    success, frame, timestamp = capture_single_image(camera, return_array=True)
                    if not success:
                        print("Capture failed. Retrying in 5 seconds.")
                        time.sleep(5)
                        continue
                    
                    # Image analysis (directly from array) - always same processing for multiple spots
                    results = calc_multiple_spots_from_frame(frame, c_y_list, n_conv, timestamp, auto_mode_flags)
                
                # Update real-time display
                if args.realtime:
                    # Use already captured frame for non-archive, capture new for archive
                    if save_permanent:
                        ret, frame = camera.read()
                        if not ret:
                            frame = None
                    
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Update image display
                        ax1.clear()
                        ax1.imshow(frame_rgb, aspect='auto')
                        ax1.set_title('Camera Image (Analysis Range)')
                        ax1.set_xlabel('X pixel')
                        ax1.set_ylabel('Y pixel')
                        
                        # Display analysis results for each spot
                        spot_info_text = f"Time: {results[0]['timestamp'].strftime('%H:%M:%S')}\n"
                        spot_info_text += f"Spots: {num_spots}\n\n"
                        
                        for i, (result, c_y) in enumerate(zip(results, c_y_list)):
                            cen_x_rt = result['cen_x']
                            cen_y_rt = result['cen_y']
                            
                            # Determine if auto mode
                            is_auto_mode = auto_mode_flags[i] if i < len(auto_mode_flags) else False
                            
                            # Display analysis range and peak position on image
                            y_half_width = 25 if is_auto_mode else 75
                            yrange = [c_y - y_half_width, c_y + y_half_width]
                            if yrange[0] < 0:
                                yrange[0] = 0
                            if yrange[1] >= frame_rgb.shape[0]:
                                yrange[1] = frame_rgb.shape[0] - 1
                            
                            # Display each spot in different color
                            colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
                            color = colors[i % len(colors)]
                            
                            ax1.axhline(y=yrange[0], color=color, linestyle='--', alpha=0.8, linewidth=2)
                            ax1.axhline(y=yrange[1], color=color, linestyle='--', alpha=0.8, linewidth=2)
                            ax1.axvline(x=cen_x_rt, color=color, linestyle='-', linewidth=1, alpha=0.9)
                            ax1.text(cen_x_rt + 10, c_y, f'P{i+1}', color=color, fontsize=12, fontweight='bold')
                            
                            # Detailed analysis (for cross-section display)
                            cen_x_detail, cen_y_detail, _, details = calc_center_from_array(
                                frame_rgb, c_y, n_conv, return_details=True, is_auto_mode=is_auto_mode)
                            
                            # Update cross-section plot (multi-spot support)
                            if i < len(cross_section_axes):
                                ax_cross = cross_section_axes[i]
                                ax_cross.clear()
                                
                                x_pixels = details['x_pixels']
                                cross_section = details['cross_section']
                                smoothed = details['smoothed']
                                
                                ax_cross.plot(x_pixels, cross_section, color=color, linewidth=2, label='Raw Data', alpha=0.7)
                                ax_cross.plot(x_pixels, smoothed, color=color, linewidth=2, label='Smoothed', linestyle='--')
                                ax_cross.axvline(x=cen_x_detail, color=color, linestyle='-', linewidth=1, alpha=0.9, 
                                               label=f'Peak: {int(cen_x_detail)}px')
                                ax_cross.set_title(f'Spot {i+1} (c_y={c_y})')
                                ax_cross.set_xlabel('X pixel')
                                ax_cross.set_ylabel('Intensity')
                                ax_cross.grid(True, alpha=0.3)
                                ax_cross.legend(fontsize=8)
                            
                            # Add to info text
                            spot_info_text += f"Spot {i+1} (c_y={c_y}):\n"
                            spot_info_text += f"  X = {int(cen_x_rt)} px\n"
                            spot_info_text += f"  Y = {int(cen_y_rt)} px\n\n"
                        
                        # Update info display
                        spot_info_text += f"Parameters:\n  Convolution Width: {n_conv} points\n"
                        spot_info_text += f"Camera: {device}\n"
                        spot_info_text += f"Image Count: {image_counter}"
                        text_info.set_text(spot_info_text)
                        
                        plt.draw()
                        plt.pause(0.05)  # Frame rate adjustment
                        fig.canvas.flush_events()
                
                # Record results (not recorded in real-time mode)
                if not args.realtime:
                    # Add data to buffer (don't write to file immediately)
                    buffer_multi_spot_data(spot_info, results)
                    
                    # Periodically flush buffer to write to file
                    if image_counter % flush_interval == 0:
                        flush_multi_spot_data(spot_info)
                        print(f"  → Data written to file (capture #{image_counter})")
                
                # Add y-coordinate to buffer and update c_y in auto mode
                for result in results:
                    spot_id = result['spot_id']
                    cen_y = result['cen_y']
                    spot_index = spot_id - 1
                    
                    # Add y-coordinate to buffer
                    y_buffers[spot_index] = update_y_buffer(y_buffers[spot_index], cen_y, max_size=10)
                    
                    # In auto mode, calculate median from buffer every 100 times to update c_y
                    if auto_mode_flags[spot_index] and image_counter % 100 == 0:
                        new_c_y = get_auto_c_y_from_buffer(y_buffers[spot_index])
                        if new_c_y != c_y_list[spot_index]:
                            old_c_y = c_y_list[spot_index]
                            c_y_list[spot_index] = new_c_y
                            print(f"  [Auto update] Spot {spot_id}: c_y {old_c_y} → {new_c_y} (update every 100 times)")
                    
                    # Set c_y value in result (for display)
                    if auto_mode_flags[spot_index]:
                        result['c_y'] = c_y_list[spot_index]
                    
                # Display results - always same format
                for result in results:
                    spot_id = result['spot_id']
                    timestamp = result['timestamp']
                    cen_x = result['cen_x']
                    cen_y = result['cen_y']
                    c_y = result['c_y']
                    auto_flag = " (auto)" if auto_mode_flags[spot_id - 1] else ""
                    print(f"[{timestamp.strftime('%Y/%m/%d %H:%M:%S')}] Spot {spot_id} (c_y={c_y}{auto_flag}): X={int(cen_x)}, Y={int(cen_y)}")
                
                # Delay (capture interval specified in parameters)
                time.sleep(capture_interval)
                
            except Exception as loop_error:
                # Catch loop errors and continue processing
                print(f"\nWarning: An error occurred during processing, but measurement will continue.")
                print(f"Error: {loop_error}")
                print("Stack trace:")
                traceback.print_exc()
                print()
                # Wait a bit before next processing on error
                time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nCapture and analysis ended.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Write remaining buffered data to file
        if 'spot_info' in locals() and spot_info and not args.realtime:
            flush_multi_spot_data(spot_info)
            print("Remaining data written to file.")
        
        # Release camera resources
        if 'camera' in locals() and camera is not None:
            camera.release()
            print("Camera released.")
        
        # Clean up real-time display
        if args.realtime:
            plt.ioff()
            plt.close('all')
            print("Real-time display ended.")
    

if __name__ == "__main__":
    main()
