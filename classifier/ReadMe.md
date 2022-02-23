# Classifier

You can follow the step below for training
```bash
python train.py
```
Script returns precision recall etc. params according to the results produced from test data 
```bash
python test_model.py
```
for single image inferance 
```bash
python single_image_inferance.py --img [IMG_PATH]
```
## Training

The dataset is divided into three folders, train val test. Dispersion ratio is Train(0.8) Val(0.1) Test(0.1). You can check split_train_val_test.py file. 'dataset_loader.py' for custom dataset class for torch environment.

## Models
Three models were trained and the results were compared. These are EfficentNet, VGG-16 and ShuffleNetv2. Models were chosen experimentally <br />
Referenced for model architecture : https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/

### EfficentNet_b0
![efficentb0_weights_result](https://user-images.githubusercontent.com/37477289/155380631-61a66c7c-d0b6-4c48-995f-e6d6f8e59da1.png)

### VGG-16
![vgg16_result](https://user-images.githubusercontent.com/37477289/155380650-bd1eefce-c0d9-415e-9b8b-1bdc56e33374.png)

### ShuffleNetv2
![shufflenet_v2_result](https://user-images.githubusercontent.com/37477289/155380637-fef51d33-fb60-4075-83b0-2e07710d0471.png)

## Benchmarks
