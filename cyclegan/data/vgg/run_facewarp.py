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

AFFINE_IDENTITY_PATH = os.path.abspath(os.path.join(DATASET_DIR, "../util/aff_identity.txt"))
DEPTH_IDENTITY_PATH = os.path.abspath(os.path.join(DATASET_DIR, "../util/depth_identity.txt"))

READY = "ready"
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

def image_filepath(identity, fileID):
    return os.path.join(DATASET_DIR, "images", identity, fileID + ".png")

def keypoints_filepath(identity, fileID):
    return os.path.join(DATASET_DIR, "keypoints", identity, fileID + ".txt")

def depth_filepath(identity, fileID):
    return os.path.join(DATASET_DIR, "depth_anaff_frontal", "depth", identity, fileID + ".txt")

def affine_filepath(identity, fileID):
    return os.path.join(DATASET_DIR, "depth_anaff_frontal", "affine", identity, fileID + ".txt")  

def result_filepath(identity, fileID, results_dir):
    return os.path.join(results_dir, identity, fileID + ".png")

def extract_identity(img_path):
    # print img_path
    person, fileID = img_path.split("/")[-2:]
    fileID = os.path.splitext(fileID)[0]
    return (person, fileID)

def main():

    start_index = 0
    #start_index = 261000

    results_dir = os.path.abspath(os.path.join(DATASET_DIR, RESULTS_PATH))
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
    
    for identity in os.listdir(os.path.join(DATASET_DIR,"images")):
        #picture_path = DATASET_DIR + "images/" + person
        identity_path = os.path.join(DATASET_DIR, "images", identity)
        for person in os.listdir(os.path.join(identity_path)):
            picture_path = os.path.join(identity_path, person)

            img_src = picture_path
            _, fileID = extract_identity(img_src)

            image = image_filepath(identity, fileID)

            processed_count += 1

            keypoints = keypoints_filepath(identity, fileID)
            
            if options.identity:
                affine = AFFINE_IDENTITY_PATH
                depth = DEPTH_IDENTITY_PATH
            else:
                depth = depth_filepath(identity, fileID)
                affine = affine_filepath(identity, fileID)

            result = result_filepath(identity, fileID, results_dir)

            if not os.path.exists(os.path.dirname(result)):
                os.makedirs(os.path.dirname(result))

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
