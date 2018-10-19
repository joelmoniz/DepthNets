#!python3

import time
import os, sys
import argparse
import tempfile
import datetime

import FaceWarperClient as fwc

class Timer:
    def __init__(self):
        self._start_time = None
        self.restart()

    def restart(self):
        self._start_time = time.perf_counter()

    def time(self):
        return time.perf_counter() - self._start_time

class Dataset:
    def __init__(self, dataset_dir, affine_identity_path):
        self._dataset_dir = dataset_dir
        self._affine_identity_path = affine_identity_path

    def image_filepath(self, person, fileID):
        return os.path.join(self._dataset_dir, "images", person, fileID + ".png")

    def keypoints_filepath(self, person, fileID):
        return os.path.join(self._dataset_dir, "keypoints", person, fileID + ".txt")

    def depth_filepath(self, person, fileID):
        return os.path.join(self._dataset_dir, "depth", person, fileID + ".txt")

    def affine_filepath(self, person, fileID):
        return os.path.join(self._dataset_dir, "affine", person, fileID + ".txt")

    def affine_identity_filepath(self):
        return self._affine_identity_path

    def pairs_filepath(self):
        return os.path.join(self._dataset_dir, "pairs.txt")

class ResultsDestination:
    def __init__(self, path):
        self._path = path
        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def result_filepath(self, person, fileID):
        return os.path.join(self._path, person, fileID + ".png")

def write_affine_identity(filepath):
    affine_identity_matrix_string = "1.0 0.0 0.0 0.0\n0.0 1.0 0.0 0.0\n"
    with open(filepath, 'w') as f:
        f.write(affine_identity_matrix_string)

def test_create_affine_identity_file():
    filename = "facewarper_affine_identity.txt"
    directory = os.path.join(tempfile.gettempdir(), "FaceWarper")
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    write_affine_identity(filepath)
    return filepath

def extract_identity(img_path):
    person, fileID = img_path.split("/")
    fileID = os.path.splitext(fileID)[0]
    return (person, fileID)

def parse_pair_line(line):
    img_dst, img_src = line.split()
    return (img_dst, img_src)

def warp_all(server, dataset, results_destination, options):
    current_person = ""
    processed_count = 0
    total_timer = Timer()
    with open(dataset.pairs_filepath()) as f:
        for i, line in enumerate(f):
            if i < options.start_index:
                continue

            file_index = i + 1
            img_dst, img_src = parse_pair_line(line)
            person, fileID = extract_identity(img_src)

            image = dataset.image_filepath(person, fileID)
            
            if not os.path.exists(image):
                print("Skipping : %s %s %s" % (file_index, person, fileID))
                continue

            processed_count += 1

            keypoints = dataset.keypoints_filepath(person, fileID)
            depth = dataset.depth_filepath(person, fileID)
            if options.identity :
                affine = dataset.affine_identity_filepath()
            else:
                affine = dataset.affine_filepath(person, fileID)
            result = results_destination.result_filepath(person, fileID)

            if not os.path.exists(os.path.dirname(result)):
                os.makedirs(os.path.dirname(result))

            if current_person != person:
                delta = total_timer.time()
                print(file_index, person, "(avg : {:.1f} faces/s)".format(processed_count / delta))
                current_person = person
                
            server.send_command(fwc.build_command(image, keypoints, depth, affine, result))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to the directory containing the dataset.')
    parser.add_argument('--results', required=True, help='Path to the results directory.')
    parser.add_argument('--server_exec', default=None, help='Path to the server executable. If not set, the program assumes the server executable is named FaceWarperServer and is locating in system PATH.')
    parser.add_argument('--identity', action='store_true', help='Use identity affine transform.')
    parser.add_argument('--start_index', type=int, default=0, help='Index of the first image to warp in the dataset.')
    args = parser.parse_args()
    return args

def main():
    options = parse_args()
    affine_identity_filepath = test_create_affine_identity_file()
    dataset = Dataset(options.dataset_path, affine_identity_filepath)
    results_destination = ResultsDestination(options.results)
    with fwc.Server(options.server_exec) as server:
        warp_all(server, dataset, results_destination, options)

if __name__ == '__main__':
    main()
