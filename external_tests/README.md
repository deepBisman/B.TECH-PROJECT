## Dataset

This dataset is used for the purpose of testing the trained model on external images obtained from ```Planet Labs```. The datasets are stored on kaggle and are in ```Private``` mode and the user needs to ask to be added as a collaborator to access the datasets.

### Downloading the Dataset
Once you have set up your Kaggle API credentials, you can download the dataset by running the provided script. Make sure you have the Kaggle CLI installed:
   ```bash
   cd  external_tests
   pip install kaggle
   bash download_data.sh
   ```