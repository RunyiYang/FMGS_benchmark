#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def searchForMaxCKPTfile(folder):
    pth_files = [file for file in os.listdir(folder) if file.endswith(".pth")]

    if pth_files:
        # Extract numbers from filenames
        numbers = [int(file.split("chkpnt")[1].split(".")[0]) for file in pth_files]

        # Find the file with the maximum number
        max_number_index = numbers.index(max(numbers))
        max_number_file = pth_files[max_number_index]

        print(f"The .pth file in floder {folder} with the maximum number is: {max_number_file}")
        return max_number_file
    else:
        print(f"No .pth files found in the floder {folder}.")
        return None
