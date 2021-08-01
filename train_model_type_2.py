#!/usr/bin/env python3

import argparse
import random
import os
import pickle
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from generator import DataGenerator
from build_model import build_model

def load_data(path_image):
    print("LOADING IMAGES...")

    #Load image from path_image
    image_path = os.listdir(path_image)
    image_path = list(filter(lambda x: x[-3:].lower() == 'jpg' or x[-3:].lower() == 'png', image_path))
    #image_path = np.repeat(image_path, 10)

    #Result variable
    list_image = []
    list_label = []

    #Mapping image with label
    for (j, imagePath) in enumerate(image_path):
        listPath = imagePath.split('-')
        list_image.append(imagePath)
        list_label.append(listPath[0])
        
    num_img = len(list_image)
    print("Total images: %d" % num_img)
    return list_image, list_label

def encode_label(list_label, save_file):
    print("ENCODING LABELS...")
    dir = './'
    if os.path.exists(os.path.join(dir, save_file)):
        print("LOADING LABEL MAP")
        label_map = pickle.load(open(os.path.join(dir, save_file), 'rb'))
    else:
        print("SAVE LABEL MAP")
        set_list_label = set(list_label)
        set_list_label = sorted(set_list_label)
        label_map = dict((c, i) for i, c in enumerate(set_list_label))
        pickle.dump(label_map, open(os.path.join(dir, save_file), 'wb'))

    print("LABEL MAP", label_map)   
    encoded = [label_map[x] for x in list_label]
    encoded = to_categorical(encoded)
    print("Load or Save file %s success" % save_file)
    return encoded


def train_model(model, baseModel, X_train, y_train, X_test=None, y_test=None, args=None, n_classes=0, batch_size=32, ckpt_path='./ckpt', model_ckpt='model_best_ckpt.h5'):
    """
    TRAIN MODEL HAZARD DETECT
    """
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    aug_train = DataGenerator(X_train, y_train, args.img_path, 3, batch_size=batch_size, n_classes=n_classes)
    aug_valid = DataGenerator(X_valid, y_valid, args.img_path, 3, batch_size=batch_size, n_classes=n_classes)
    aug_test = DataGenerator(X_test, y_test, args.img_path, 3, batch_size=batch_size, n_classes=n_classes)
    checkpoint = ModelCheckpoint(os.path.join(ckpt_path, model_ckpt), monitor="val_loss",
                                 save_best_only=True, mode='min', save_weights_only=True, save_freq='epoch')
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    # Load checkpoint
    if os.path.exists(os.path.join(ckpt_path, model_ckpt)):
        print("LOADING MODEL WEIGHT...")
        model.load_weights(os.path.join(ckpt_path, model_ckpt))
    else:
        print("CREATE MODEL WEIGHT FILE...")
    
    if(args.step <= 1):
        print("TRAINING MODEL STEP 1...")
        # freeze EfficientNetB2 model
        for layer in baseModel.layers:
            layer.trainable = False
        opt = RMSprop(0.001)
        model.compile(opt, 'categorical_crossentropy', ['accuracy'])
        if (args.validation):
            H = model.fit(aug_train, validation_data=aug_valid, epochs=args.epoch_step_1, callbacks=[checkpoint, early_stop])
        else:
            H = model.fit(aug_train, epochs=args.epoch_step_1, callbacks=[checkpoint, early_stop])

    if(args.step <= 2):
        print("TRAINING MODEL STEP 2...")
        # unfreeze all CNN layer in EfficientNetB2:
        for layer in baseModel.layers[182:]:
            layer.trainable = True

        opt = Adam(learning_rate=0.001, decay=5e-5)
        model.compile(opt, 'categorical_crossentropy', ['accuracy'])
        if (args.validation):
            H = model.fit(aug_train, validation_data=aug_valid, epochs=args.epoch_step_2, callbacks=[checkpoint,  early_stop])
        else:
            H = model.fit(aug_train, epochs=args.epoch_step_2, callbacks=[checkpoint,  early_stop])
    if(args.step <= 3):
        print("EVALUTE MODEL STEP 3...")
        opt = Adam(learning_rate=0.001, decay=5e-5)
        model.compile(opt, 'categorical_crossentropy', ['accuracy'])
        score = model.evaluate(aug_test, verbose=1, batch_size=batch_size)
        print("TEST LOST, TEST ACCURACY:",score)
    print("FINISH TRAINING MODEL...")

def main(args):

    print("START MAIN CLASS TRAINING MODEL")
    list_image, list_label = load_data(args.img_path)
    print("LIST CLASSES BEFORE SHUFFLE", set(list_label))
    labels = encode_label(list_label, args.mapping_file)
    n_classes = len(set(list_label))
    print("NUM CLASSES", n_classes)
    print("LIST CLASSES AFTER SHUFFLE", set(list_label))
    baseModel, mainModel = build_model(n_classes, args.appEfficientNet)

    
    batch_size = args.batch_size
    model_ckpt = args.model_ckpt
    if (args.validation):
        X_train, X_test, y_train, y_test = train_test_split(list_image, labels, test_size=0.2, random_state=42)
        train_model(model=mainModel, baseModel=baseModel, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, args=args, n_classes=n_classes, batch_size=batch_size, model_ckpt=model_ckpt)
    else:
        train_model(model=mainModel, baseModel=baseModel, X_train=list_image, y_train=labels, args=args, n_classes=n_classes, batch_size=batch_size, model_ckpt=model_ckpt)
    print("FINISH MAIN CLASS TRAINING MODEL")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='Path to folder which contains images.', type=str, default='./images-cropped')
    parser.add_argument('--mapping_file', help='Path to save label map file.', type=str, default='label_map.pkl')
    parser.add_argument('--epoch_step_1', help='Number of epochs for training step 1.', type=int, default=30)
    parser.add_argument('--epoch_step_2', help='Number of epochs for training step 2.', type=int, default=100)
    parser.add_argument('--validation', help='Wheather to split data for validation.', type=bool, default=True)
    parser.add_argument('--step', help='Training model step (1, 2, 3)', type=int, default=0)
    parser.add_argument('--appEfficientNet', help='EfficientNetB0, 1, 2, 3, 4, 5, 6, 7', type=int, default=0)
    parser.add_argument('--batch_size', help='Number of batch size for training', type=int, default=32)
    parser.add_argument('--model_ckpt', help='File name model ckpt h5', type=str, default='model_best_ckpt.h5')

    args = parser.parse_args()
    main(args)
