# Download CICIDS2017 Dataset

The CICIDS2017 dataset is required for training the models. Follow these steps:

## Option 1: Download from Official Source

1. Go to: https://www.unb.ca/cic/datasets/ids-2017.html
2. Click on "Download" 
3. Download the CSV files (MachineLearningCVE folder)
4. Extract and place all CSV files in `data/raw/` folder

## Option 2: Direct Download Links

The dataset contains multiple CSV files for different days:
- Monday-WorkingHours.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Wednesday-workingHours.pcap_ISCX.csv
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Friday-WorkingHours-Morning.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

## Dataset Size

Total size: ~1GB compressed, ~5GB uncompressed

## Quick Start (Sample Data)

For testing purposes, you can start with just one file (e.g., Friday-DDos) which contains DDoS attacks.

## After Download

Place the CSV files in `data/raw/` folder and they will be automatically detected by the data loader.
