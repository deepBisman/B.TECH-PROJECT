import os
import sys
import torch

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Dynamic imports
from external_tests.utils.dataset_utils import analyze_raster_image
from external_tests.utils.pipelines import preprocessing_pipeline_3M, preprocessing_pipeline_5M
from src.models.unet_models import ResAttentionUNet
from src.models.composite_models import TwoStageUNetFusion

if __name__ == '__main__':
    # Set the Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Path to Images
    rgb_path = '../external_tests/images/rgb_test_images/20230515_042527_59_2415_3B_Visual.tif'
    target_path = '../external_tests/images/target_test_images/240211_BLR_BuildingHeight_WaterVegNullRemoved_ra_5m_utm43n_PT_V5.tif'
    
    # Image Analysis Results
    analyze_raster_image(target_path, True)

    # Set up regression and segmentation nets of the composite model
    segmentation_net = ResAttentionUNet(in_channels = 3, out_channels = 1, bias = False, pooling = 'spectralpool').to(device)
    regression_net = ResAttentionUNet(in_channels = 3, out_channels = 1, bias = False, pooling = 'spectralpool').to(device)
    model = TwoStageUNetFusion(segmentation_net, regression_net).to(device)
    model.load_state_dict(torch.load('../models_archive/three_channel/composite_models/composite_type2_resattentionunet.pth', map_location=device))

    # RGB Mean and Standard Deviation for normalization
    rgb_mean = [0.31832295656204224, 0.3443617522716522, 0.2820265591144562]
    rgb_stddev = [0.17934595048427582, 0.1642322838306427, 0.16852612793445587]
    
    # # Evaluation result using 3M Preprocessing Pipeline
    # print('----------------3M Pipeline-----------------------')
    # preprocessing_pipeline_3M(rgb_path, target_path, rgb_mean, rgb_stddev, model, device, 512, 'maxpool')

    # Evaluation result using 5M Preprocessing Pipeline
    print('----------------5M Pipeline-----------------------')
    preprocessing_pipeline_5M(rgb_path, target_path, rgb_mean, rgb_stddev, model, device, 512, 'bilinear')

