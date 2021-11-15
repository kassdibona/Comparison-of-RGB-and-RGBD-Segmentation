# Based upon https://github.com/stereolabs/zed-examples/blob/master/svo%20recording/playback/python/svo_playback.py
"""
This script helps read a .svo formated recording and saves the specified frames into the same directory.
The following images from the specified frames are saved:
- Full stereo RGB image (has ending `-full.png`)
- Left stereo RGB image (has ending `-left.png`)
- Depth image (has ending `-depth.png`)
- Depth layer (has ending `-measuredDepth.png`)

To run in a terminal:
  python3 svo_save_frames.py svo_file_name.svo <num_frames> <n_frames> <frame_start>

Note:
- num_frames is the total number of frames to be saved,
- n_frames-1 is the number of frames to be skipped before another frame is saved, i.e. every 15 frames are saved,
- frame_start is the frame number from which the saving will commence.
"""

import sys
import pyzed.sl as sl
import cv2

KEY_f = 102
KEY_b = 98
KEY_s = 115
KEY_q = 113

# Figure out len_int
len_int = 8

# default to saving all of the frames available
save_none = False
save_all = False
frame_start = 0 
num_frames = 150
n_frames = 15 

# If modifying the calculated depth 
# to have a specified min and max
setDist = False

min_dist = 1.5 # metres
max_dist = 250   # metres

################################################################################
# Functions

def condition_save_frame(frame_ID, num_frames, n_frames, frame_start):
    if save_none:
        return False

    if save_all:
        return True

    if frame_ID < frame_start:
        return False

    if frame_ID > frame_start+((num_frames-1)*n_frames):
        return False

    if (frame_ID-frame_start)%n_frames != 0:
        return False

    return True

def print_camera_information(cam):
    while True:
        res = input("Do you want to display camera information? [y/n]: ")
        if "y" in res:
            print()
            print("Distorsion factor of the right cam before calibration: {0}.".format(
                cam.get_camera_information().calibration_parameters_raw.right_cam.disto))
            print("Distorsion factor of the right cam after calibration: {0}.\n".format(
                cam.get_camera_information().calibration_parameters.right_cam.disto))

            print("Confidence threshold: {0}".format(cam.get_runtime_parameters().confidence_threshold))
            print("Depth min and max range values: {0}, {1}".format(cam.get_init_parameters().depth_minimum_distance,
                                                                    cam.get_init_parameters().depth_maximum_distance)
)
            print("Resolution: {0}, {1}.".format(round(cam.get_camera_information().camera_resolution.width, 2), cam.get_camera_information().camera_resolution.height))
            print("Camera FPS: {0}".format(cam.get_camera_information().camera_fps))
            print("Frame count: {0}.\n".format(cam.get_svo_number_of_frames()))
            break
        elif "n" in res:
            print("Camera information not displayed.\n")
            break
        else:
            print("Error, please enter [y/n].\n")


def save_image(frame_name, mat):
    img = sl.ERROR_CODE.FAILURE
    while img != sl.ERROR_CODE.SUCCESS:
        #print(f"Saving image: {frame_name}")
        img = mat.write(frame_name)
        print(f"Saved image: {frame_name}")
        if img == sl.ERROR_CODE.SUCCESS:
            break
        else:
            print("Help: something went wrong when trying to save the images.")

################################################################################
# MAIN
def main():
    if len(sys.argv) < 2:
        print("Please specify path to .svo file.")
        print("Example: python3 svo_save_playback-modified_10.py [num_frames] [n_frames] [frame_start]")
        exit()

    filepath = sys.argv[1]
    print("Reading SVO file: {0}".format(filepath))

    filename = filepath[:-4] # remove the ".svo"

    try:
        num_frames = int(sys.argv[2])
        n_frames = int(sys.argv[3])
        frame_start = int(sys.argv[4])
    except:
        pass

    print(f"num_frames = {num_frames}")
    print(f"n_frames = {n_frames}")
    print(f"frame_start = {frame_start}")

    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)

    # https://www.stereolabs.com/docs/depth-sensing/depth-settings/
    runtime_parameters =sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.METER
    if setDist:
        init.depth_minimum_distance = min_dist
        init.depth_maximum_distance = max_dist
    ##print(dir(init))

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    full = sl.Mat()
    left = sl.Mat()
    depth = sl.Mat()
    depth_val = sl.Mat()

    key = ''
    print("  Quit the video reading:     q\n") # 113
    frame_ID = 0
    frames_saved = 0
    while key != KEY_q:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            frame_ID += 1
            if frame_ID%50 == 0:
                print(f"frame_ID = {frame_ID}")
            # Retrieve image
            cam.retrieve_image(full, sl.VIEW.SIDE_BY_SIDE)
            # Retrieve left image
            cam.retrieve_image(left, sl.VIEW.LEFT)
            # Retrieve the normalized depth image
            cam.retrieve_image(depth, sl.VIEW.DEPTH)
            # Retrieve depth map. Depth is aligned on the left image
            cam.retrieve_measure(depth_val, sl.MEASURE.DEPTH)
            #cv2.imshow("ZED image", left.get_data())
            #cv2.imshow("ZED depth", depth.get_data())
            # Save the images available
            if condition_save_frame(frame_ID, num_frames, n_frames, frame_start):
                img_name = f"{filename}_frame-{frame_ID:0{len_int}}"
                save_image(f"{img_name}-full.png", full)
                save_image(f"{img_name}-left.png", left)
                save_image(f"{img_name}-depth.png", depth)
                save_image(f"{img_name}-measuredDepth.png", depth_val)
                frames_saved += 1
        key = cv2.waitKey(1)
    #cv2.destroyAllWindows()

    print_camera_information(cam)
    cam.close()
    print("\nFINISH")


if __name__ == "__main__":
    main()
