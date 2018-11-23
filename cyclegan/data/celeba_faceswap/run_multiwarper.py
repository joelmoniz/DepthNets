#!python3

import subprocess
import time
import os, sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', action='store_true',
                        help="If true, simply cuts out the face")
    parser.add_argument('--results_dir', default='extracted_images',
                        help="Output folder for images")
    args = parser.parse_args()
    return args

options = parse_args()

DATASET_DIR = os.path.abspath("./")

#####################################################
# CHANGE THESE LINES TO POINT TO FACE WARPER SERVER #
#####################################################
WARPER_WORKING_DIR = "/home/cjb60/github/DepthNets/FaceWarper/FaceWarperServer/build/"
WARPER_PATH = os.path.join(WARPER_WORKING_DIR, "FaceWarperServer")

AFFINE_IDENTITY_PATH = os.path.abspath(os.path.join(DATASET_DIR, "util/aff_identity.txt"))
DEPTH_IDENTITY_PATH = os.path.abspath(os.path.join(DATASET_DIR, "util/depth_identity.txt"))

READY = "ready"
#RESULTS_PATH = os.path.join("swap_samples", "multiwarped_images_crop")
# RESULTS_PATH = "extracted_images/"
# RESULTS_PATH = "frontalized_images/"
RESULTS_PATH = options.results_dir

def build_command(image, keypoints, depth, affine, result):
    return " ".join((image, keypoints, depth, affine, result))

def send_command(proc, command):
    proc.stdin.write(command + "\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if READY in line:
            break
        time.sleep(0.001)

def image_filepath(fileID):
    return os.path.join(DATASET_DIR, "images", fileID + "_crop.png")

def keypoints_filepath(fileID):
    return os.path.join(DATASET_DIR, "keypoints", fileID + "_crop.txt")

def depth_filepath(person):
    return os.path.join(DATASET_DIR, "swap_samples", "depth", person)

def affine_filepath(person):
    return os.path.join(DATASET_DIR, "swap_samples", "affine", person)  

def result_filepath(fileID, results_dir):
    return os.path.join(results_dir, fileID + ".png")

def extract_identity(person):
    # print img_path
    fileName = person.split('.')[0]
    fileID = fileName.split('_')[0]
    return fileName, fileID


def main():

    start_index = 0
    #start_index = 261000

    results_dir = os.path.join(DATASET_DIR, RESULTS_PATH)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print(WARPER_PATH)
    proc = subprocess.Popen([WARPER_PATH], cwd=WARPER_WORKING_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, shell=0)
    
    while True:
        line = proc.stdout.readline()
        if READY in line:
            break
        time.sleep(0.001)

    current_person = ""
    processed_count = 0
    start_time = time.clock()
    for person in os.listdir(os.path.join(DATASET_DIR, "swap_samples", "depth")):
        fileName, fileID = extract_identity(person)

        image = image_filepath(fileID)
        
        processed_count += 1

        keypoints = keypoints_filepath(fileID)
        if options.identity:
            affine = AFFINE_IDENTITY_PATH
            depth = DEPTH_IDENTITY_PATH
        else:
            depth = depth_filepath(person)
            affine = affine_filepath(person)
       
        result = result_filepath(fileName, results_dir)

        if not os.path.exists(image):
            print("PB image path")
        if not os.path.exists(keypoints):
            print("PB kp path")
        if not os.path.exists(depth):
            print("PB depth path")
        if not os.path.exists(affine):
            print("PB affine path")

        
        print ("Image:")
        print (image)
        print ("KPs:")
        print (keypoints)
        print ("Depth:")
        print (depth)
        print ("Affine:")
        print (affine)

        if current_person != person:
            delta = time.clock() - start_time
            # print(file_index, person, "(avg : {:.1f} faces/s)".format(processed_count / delta))
            current_person = person
        
        send_command(proc, build_command(image, keypoints, depth, affine, result))

    proc.terminate()

if __name__ == '__main__':
    main()
