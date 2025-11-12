apt-get update
apt-get install wget unzip python3-pip -y
pip3 install gdown

if [ -d "Car_data-256" ] || [ -f "Car_data-256.zip" ]; then
    echo "Car_data-256 directory or zip file already exists"
elif [ -d "Car_data-256-processed" ]; then
    echo "Processed dataset already exists, skipping raw dataset download and extraction"
else
    gdown https://drive.google.com/uc?id=1qL-1PV1jvNDRF_yToKUGptVpvmZUfor1 -O Car_data-256.zip
    unzip Car_data-256.zip
    rm Car_data-256.zip
fi
