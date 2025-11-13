apt-get update
apt-get install wget unzip python3-pip -y
pip3 install gdown

if [ -d "car_train_split" ] || [ -f "Car_data-256.zip" ]; then
    echo "Car_data-256 directory or zip file already exists"
else
    gdown https://drive.google.com/uc?id=1qL-1PV1jvNDRF_yToKUGptVpvmZUfor1 -O Car_data-256.zip
    unzip Car_data-256.zip
    rm Car_data-256.zip
fi

if [ -d "car_test" ] || [ -f "Car_data_test-256.zip" ]; then
    echo "Car_data_test-256 directory or zip file already exists"
else
    gdown https://drive.google.com/uc?id=1E5mqA18Dto2l0MjqERjQm3BSAPbcwoeF -O Car_data_test-256.zip
    unzip Car_data_test-256.zip
    rm Car_data_test-256.zip
fi