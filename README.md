# Airbus-ship-segmentation
For this challenge was used Unet model trained on patches of size (128,128). After first try of resizing I realize that small ships disappear. So cropping or patching seems more reasonable. Also I take patches with ships only, thinking that we already have strongly unbalanced data and background already present enough. Later it will be clear that this approach shifts model to be biased in a way that there is always must be a ship. In other words, there are too many false positives ahead.\
So, what's done?
#### Prepare_data.py:
Process starts through **def prepare_datafiles()**:
1. From **train_ship_segmentations_v2.csv** filter out blank annotations.
2. Decode pixels from strings to pixel coordinates in **def get_points()** (later I found approved funcs for decode/encode rle)
3. Create masks files from these decoded pixels in **def create_masks()**. Here is also masks are combined in one image.
4. Make patches of size (128,128) from images and masks. Save only patches with ships.

Then **def get_datasets(full_image=True)** takes logic flow:
1. Lists **x,y** are formed by walking through folders with images/masks files.
   - parameter **full_image** defines full-sized images (768,768) or patches (128,128) will go down the line.
2. Then **train_test_split** splits data into train and valid datasets. Valid takes only 5% of data, because it will have 5932 images, that seems enough for validation task.
3. **def make_dataset()** will return processed **tf.data.Dataset** for train and valid dataset respectively.
4. About processing: read file, decode image and resize full-sized (768,768) into expected size (128,128). Later I understand that it contradicts to my decision of not resizing, but cropping/patching images.
#### Train_model.py:
**def train():**
Here model architecture and dice loss are defined. After Input layer inputs are normalized by **/255**.\
Next Callbacks were added: **EarlyStopping**, **ModelCheckpoint**, **ReduceLROnPlateau**.\
Hyperparameters for model:\
  BATCH_SIZE = 128\
  BUFFER_SIZE = 4*BATCH_SIZE\
  LR = 1e-4\
  EPOCHS = 15\
After training hitsory and model weights are saved **new_patches_history_e4.pkl**, **new_patches_e-4.keras**\
Alternatively one could simply load already trained model and history in **def get_trained()**
#### Predict.py:
1. **def predict_full_images()** patches image, predict on these patches, binarize result and reconstruct full mask in return
2. **def get_true_accuracy()** calculates dice loss coeficient for a given dataset. That's what i mean by true accuracy. In training time right metrics were not chosen.
---

In conclusion, I want to try resnet34 as backbone for my model and give some no-ships images. Also it seems like a nice idea to train not only on patches, but on full images as well to model learn general patterns from big picture.

