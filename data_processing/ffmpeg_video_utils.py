import cv2
import os
import subprocess
from tqdm import tqdm


def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    # print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")

    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # source release
    video_writer.release()
    # print(f"Video saved at {output_video_path}")


def create_video_from_images_with_ffmpeg(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    # print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")

    # Ensure images are named sequentially for FFmpeg
    for idx, image_file in enumerate(image_files):
        src = os.path.join(image_folder, image_file)
        dst = os.path.join(image_folder, f"{idx:04d}.png")
        os.rename(src, dst)

    # Build FFmpeg command
    input_pattern = os.path.join(image_folder, "%04d.png")
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(frame_rate), "-c:v", "mjpeg", "-f", "image2", "-i", input_pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", output_video_path
    ]

    print(' '.join(ffmpeg_cmd))
    # Run FFmpeg
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Video saved at {output_video_path}")

