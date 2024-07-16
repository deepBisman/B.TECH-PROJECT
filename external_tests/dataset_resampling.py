import os
import sys

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Dynamic imports
from external_tests.utils.dataset_utils import resample_directory

if __name__ == '__main__':
    # Flag and dir Mapping
    flag_map = {
        'rgb': True,
        'sar': True,
        'dsm': True
    }

    # Setting up arguments to function
    scale = 0.8 / 5
    base_dir = '../data/ieeegrss_2023dfc_track2'
    resample_dir_name = f"ieeegrss_2023dfc_track2_{int(0.8 / scale)}M_{'bilinear' if flag_map['dsm'] else 'maxpool'}"
    resampled_base_dir = os.path.join("../data", resample_dir_name)

    # Resmapling directory wise
    for category in ['rgb', 'sar', 'dsm']:
        input_dir = os.path.join(base_dir, category)
        output_dir = os.path.join(resampled_base_dir, category)
        resample_directory(input_dir, output_dir, scale, flag = flag_map[category])




    