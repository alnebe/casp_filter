"""training model"""
from models import ANNs, generator
from tensorflow.keras.callbacks import (ReduceLROnPlateau, ModelCheckpoint,
                                        TensorBoard, EarlyStopping)
from sklearn.model_selection import train_test_split

import pickle
import os


# load descriptors
with open("/tmp/LFP_4_4096.pickle", "rb") as pkl:
    desc_dict = pickle.load(pkl)

# split into input and output elements
indices = list(desc_dict.keys())

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
model_name = "my_model"
output_filename = f"{checkpoint_dir}/{model_name}"

tensorboard_callback = TensorBoard(log_dir="tb_logs",
                                   histogram_freq=1,
                                   write_images=True)

estp = EarlyStopping(monitor="val_loss",
                     patience=1,
                     mode="min",
                     min_delta=0.0001,
                     verbose=1)

mcp = ModelCheckpoint(filepath=output_filename,
                      save_best_only=True,
                      monitor="val_loss",
                      mode="min")

rlr = ReduceLROnPlateau(monitor="val_bac",
                        factor=0.5,
                        patience=2,
                        min_lr=1e-6,
                        mode="max",
                        min_delta=1e-4,
                        verbose=1,
                        cooldown=1)

pretrain, test = train_test_split(indices,
                                  test_size=0.1,
                                  random_state=42,
                                  shuffle=True)

train, validation_set = train_test_split(pretrain,
                                         test_size=0.2,
                                         random_state=41)

input_shape = desc_dict["1"][0][0].shape
model = ANNs.create_baseline(input_shape)

history = model.fit(generator(train, 200, desc_dict),
                    steps_per_epoch=len(train)//200,
                    validation_data=generator(validation_set, 200, desc_dict),
                    validation_steps=len(validation_set)//200,
                    callbacks=[rlr, estp, mcp, tensorboard_callback],
                    verbose=1,
                    epochs=100,
                    use_multiprocessing=False,
                    workers=1)

result = model.evaluate(generator(test, 200, desc_dict),
                        batch_size=200,
                        steps=len(test)//200,
                        verbose=1,
                        use_multiprocessing=False,
                        workers=1)

dict(zip(model.metrics_names, result))