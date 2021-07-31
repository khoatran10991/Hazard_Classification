#!/usr/bin/env python3

import numpy as np
import os
import cv2
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.optimizers import Adam, RMSprop
import pickle
from build_model import build_model

class HazardClassification():
    def __init__(self, appEfficientNet=0, dir='./', label='label_map.pkl',ckpt='model_best_ckpt.h5'):
        self.label_map = pickle.load(open(os.path.join(dir,'label', label), 'rb'))
        self.num_class = len(self.label_map)
        self.label_key = list(self.label_map.keys())
        self.label_value = list(self.label_map.values())

        _, self.model = build_model(self.num_class, appEfficientNet)
        self.model.load_weights(os.path.join(dir, 'ckpt', ckpt))
        self.model.compile(Adam(learning_rate=0.001, decay=5e-5), 'categorical_crossentropy', ['accuracy'])
        print("LABEL", self.label_map)

    def run(self, image, return_label=True):
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.expand_dims(image, 0)
        image = imagenet_utils.preprocess_input(image)

        result = self.model.predict(image)
        print("result {}".format(result))
        result = np.argmax(result, axis=1)
        print("result argmax {}".format(result))

        if return_label:
            return self.decode_label(result)
        else:
            return result

    def decode_label(self, result):
        return self.label_key[self.label_value.index(result)]
