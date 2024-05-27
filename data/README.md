## Dataset

The dataset used in this project is from the IEEE GRSS 2023 DFC Track 2. Due to its size, it is not stored directly in this repository. Instead, you can download it from Kaggle using the provided instructions.

### Setting Up the Kaggle API

To download the dataset, you'll need to set up your Kaggle API credentials. Follow these steps:

1. **Create a `.kaggle` Folder**: 
   Create a folder named `.kaggle` in your home directory (or project directory if you prefer):
   ```bash
   mkdir .kaggle
2. **Move the Kaggle JSON File**:
   Move your `kaggle.json` file into the `.kaggle` folder:
   ```bash
   mv /path/to/your/kaggle.json .kaggle/
3. **Set File Permissions (Optional but Recommended)**:
   Set appropriate permissions for the `kaggle.json` file to ensure it is readable and writable only by the owner:
   ```bash
   chmod 600 .kaggle/kaggle.json
   ```
### Downloading the Dataset
Once you have set up your Kaggle API credentials, you can download the dataset by running the provided script. Make sure you have the Kaggle CLI installed:
   ```bash
   pip install kaggle
   bash data/download_data.sh
   ```
This will download the dataset into the `data/ieeegrss_2023dfc_track2` directory and also create other subdatasets required for experimentation later. The number of files in each dataset are :
| Dataset                          | Number of Images |
|-------------------------------------|-----------------|
| IEEEGRSS_2023DFC_TRACK2            | 1773            |
| NEWDELHI_SAOLUIS_BRASILIA_RIO      | 500             |
| NEWYORK_SANDIEGO_SYDNEY            | 250             |
| BARCELONA_COPENHAGEN_PORTSMOUTH      | 457             |
| BERLIN                              | 566             |

