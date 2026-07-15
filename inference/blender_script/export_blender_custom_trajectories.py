"""
Export camera and ellipsoid trajectories from Blender animation.
"""
import bpy
import numpy as np
import json
from pathlib import Path
import os

# ================== Configuration ==================
# Export options
EXPORT_CAMERA = True
EXPORT_ELLIPSOIDS = True

ROOT_DIR = "/absolute/path/to/demo_data/your_folder"
INPUT_JSON_PATH = f"{ROOT_DIR}/fitted_3D_gaussian/gaussian_params.json"
OUTPUT_NPZ_PATH = f"{ROOT_DIR}/camera_object_0/custom_camera_trajectory.npz" 
OUTPUT_JSON_PATH = f"{ROOT_DIR}/camera_object_0/custom_3D_gaussian_trajectory.json"

os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)

# Frame configuration
TOTAL_FRAMES = 81
FRAME_STEP = 1

# ================== Validate Export Options ==================
if not EXPORT_CAMERA and not EXPORT_ELLIPSOIDS:
    raise RuntimeError("At least one of EXPORT_CAMERA or EXPORT_ELLIPSOIDS must be True!")

print("[INFO] Export configuration:")
print(f"  - Export camera trajectory: {EXPORT_CAMERA}")
print(f"  - Export ellipsoid trajectories: {EXPORT_ELLIPSOIDS}")
print(f"  - Total frames: {TOTAL_FRAMES}")
print(f"  - Frame step: {FRAME_STEP}")

# ================== Export Camera Extrinsics ==================
if EXPORT_CAMERA:
    print("[INFO] Exporting camera extrinsics...")
    
    scene = bpy.context.scene
    
    # Find all camera objects
    camera_objects = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    
    if len(camera_objects) == 0:
        print(f"[WARN] No camera objects found! Skipping camera export.")
    elif len(camera_objects) > 1:
        camera_names = [obj.name for obj in camera_objects]
        raise RuntimeError(
            f"Multiple cameras found: {camera_names}. Please ensure only one camera exists in the scene!"
        )
    else:
        cam_obj = camera_objects[0]
        print(f"[INFO] Found camera: '{cam_obj.name}'")
        
        # Store extrinsics for all frames
        extrinsics_list = []

        for i in range(0, TOTAL_FRAMES, FRAME_STEP):
            frame_no = i + 1
            scene.frame_set(frame_no)

            # Get camera world matrix (Blender format: camera-to-world)
            cam_matrix = cam_obj.matrix_world
            cam_matrix_np = np.array(cam_matrix, dtype=np.float32)

            # Export camera-to-world matrix (Blender coordinate system)
            extrinsics_list.append(cam_matrix_np)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processed frame {frame_no}/{TOTAL_FRAMES}")

        # Convert to numpy array [T, 4, 4]
        extrinsics_array = np.stack(extrinsics_list, axis=0)
        print(f"[INFO] Camera extrinsics shape: {extrinsics_array.shape}")
        print(f"[INFO] Exported format: camera-to-world matrices (Blender coordinate system)")

        # Save as npz file
        np.savez(OUTPUT_NPZ_PATH, extrinsics=extrinsics_array)
        print(f"[INFO] Camera extrinsics saved to: {OUTPUT_NPZ_PATH}")
        print(f"[INFO] Camera name: '{cam_obj.name}'")
else:
    print("[INFO] Skipping camera export (EXPORT_CAMERA=False)")

# ================== Export Ellipsoid Trajectories ==================
if EXPORT_ELLIPSOIDS:
    print("[INFO] Exporting ellipsoid trajectories...")
    
    # Load input JSON for color mapping
    obj_id_to_color_idx = {}
    if Path(INPUT_JSON_PATH).exists():
        print(f"[INFO] Loading color mapping from: {INPUT_JSON_PATH}")
        try:
            with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            if "obj_id_to_color_idx" in input_data:
                for key, value in input_data["obj_id_to_color_idx"].items():
                    obj_id_to_color_idx[key] = value
                print(f"[INFO] Loaded color mapping for {len(obj_id_to_color_idx)} objects")
            else:
                print("[WARN] No 'obj_id_to_color_idx' found in input JSON")
        except Exception as e:
            print(f"[WARN] Failed to load input JSON: {e}")
    else:
        print(f"[WARN] Input JSON file not found: {INPUT_JSON_PATH}")
    
    if 'scene' not in locals():
        scene = bpy.context.scene
    
    # Find all ellipsoid objects
    ellipsoid_objects = [obj for obj in bpy.data.objects if obj.name.startswith("Ellipsoid_")]
    
    num_ellipsoids = len(ellipsoid_objects)
    print(f"[INFO] Found {num_ellipsoids} ellipsoid object(s)")
    
    if num_ellipsoids == 0:
        print("[WARN] No ellipsoid objects found! Skipping ellipsoid export.")
    else:
        # Prepare JSON data structure
        export_data = {
            "metadata": {
                "num_objects": len(ellipsoid_objects),
                "num_frames": TOTAL_FRAMES,
                "frame_step": FRAME_STEP,
                "description": "Exported ellipsoid Gaussian parameters from Blender animation",
                "obj_id_to_color_idx": obj_id_to_color_idx
            },
            "frames": []
        }
        
        # Organize data by frame
        for i in range(0, TOTAL_FRAMES, FRAME_STEP):
            frame_no = i + 1
            scene.frame_set(frame_no)
            
            frame_data = {
                "frame_index": i,
                "objects": []
            }
            
            # Iterate each ellipsoid
            for ellipsoid_obj in ellipsoid_objects:
                # Extract object_id from name
                obj_id_str = ellipsoid_obj.name.replace("Ellipsoid_", "")
                try:
                    obj_id = obj_id_str
                except ValueError:
                    obj_id = obj_id_str
                
                mesh = ellipsoid_obj.data
                
                if len(mesh.vertices) > 0:
                    # Get ellipsoid center from world matrix
                    world_matrix = ellipsoid_obj.matrix_world
                    center_blender = np.array([
                        world_matrix[0][3],
                        world_matrix[1][3],
                        world_matrix[2][3]
                    ], dtype=np.float32)
                    mean = center_blender.tolist()

                    # Always compute covariance from current mesh vertices in world space
                    # This ensures all user edits are captured (Edit Mode changes, Object Mode transforms, constraints)
                    vertices_world = []
                    for v in mesh.vertices:
                        v_world = world_matrix @ v.co
                        vertices_world.append([v_world.x, v_world.y, v_world.z])
                    vertices_world = np.array(vertices_world, dtype=np.float32)
                    
                    # Compute covariance from vertex distribution
                    # For an ellipsoid created from unit sphere with scale_factor, we need to adjust
                    vertices_centered = vertices_world - center_blender
                    sample_cov = np.cov(vertices_centered.T)
                    
                    # The ellipsoid mesh is created by scaling a unit icosphere
                    # For a unit sphere, variance along each axis ≈ 1/3
                    # So the sample covariance needs to be adjusted to get the Gaussian covariance
                    # covariance = sample_cov / (scale_factor^2) * 3, where scale_factor=2.0 in make_ellipsoid_mesh
                    ellipsoid_scale_factor = 2.0
                    covariance_blender = sample_cov / (ellipsoid_scale_factor ** 2) * 3
                    cov = covariance_blender.tolist()
                    
                    # Keep parameters in Blender coordinate system
                    # (rendering_4D_control_maps.py expects Blender coordinates since point cloud is also in Blender coords)
                    
                    # Get color index from mapping
                    color_idx = 0
                    if isinstance(obj_id, int) and obj_id in obj_id_to_color_idx:
                        color_idx = obj_id_to_color_idx[obj_id]
                        print(f"  Object {obj_id}: using color_idx={color_idx} from mapping")
                    elif isinstance(obj_id, int):
                        color_idx = obj_id
                        print(f"  Object {obj_id}: no mapping found, using obj_id as color_idx")
                    
                    object_data = {
                        "object_id": obj_id,
                        "color_index": color_idx,
                        "gaussian_3d": {
                            "mean": mean,
                            "covariance": cov
                        }
                    }
                    
                    frame_data["objects"].append(object_data)
            
            export_data["frames"].append(frame_data)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processed frame {frame_no}/{TOTAL_FRAMES}")
        
        # Save as JSON file
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Ellipsoid trajectories saved to: {OUTPUT_JSON_PATH}")
        print(f"[INFO] Exported {len(ellipsoid_objects)} ellipsoids across {len(export_data['frames'])} frames")
        print(f"[INFO] Exported format: Blender coordinate system (X=right, Y=forward, Z=up)")
        if obj_id_to_color_idx:
            print(f"[INFO] Color mapping included: {len(obj_id_to_color_idx)} objects")
        else:
            print(f"[WARN] No color mapping found in metadata")
else:
    print("[INFO] Skipping ellipsoid export (EXPORT_ELLIPSOIDS=False)")

print("[INFO] Export completed!")
