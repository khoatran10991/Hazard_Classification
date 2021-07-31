#!/usr/bin/env python3

from tensorflow.keras import utils
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.models import Model

def build_model(num_class, appEfficientNet):
    print("BUILDING MODEL...")
    print("CREATE EFFICIENTNET B",appEfficientNet)
    # Load model EfficientNetB2 16 của ImageNet dataset, include_top=False để bỏ phần Fully connected layer ở cuối.
    if(appEfficientNet == 0):
        baseModel = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif(appEfficientNet == 1):
        baseModel = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif(appEfficientNet == 2):
        baseModel = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif(appEfficientNet == 3):
        baseModel = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif(appEfficientNet == 4):
        baseModel = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif(appEfficientNet == 5):
        baseModel = EfficientNetB5(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif(appEfficientNet == 6):
        baseModel = EfficientNetB6(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif(appEfficientNet == 7):
        baseModel = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Xây thêm các layer
    # Lấy output của ConvNet trong EfficientNetB2
    fcHead = baseModel.output

    # Flatten trước khi dùng FCs
    fcHead = Flatten()(fcHead)

    # Thêm FC
    fcHead = Dense(2048, activation='relu')(fcHead)

    fcHead = Dense(512, activation='relu')(fcHead)

    fcHead = Dense(256, activation='relu')(fcHead)
    
    # Output layer với softmax activation
    fcHead = Dense(num_class, activation='softmax')(fcHead)

    # Xây dựng model bằng việc nối ConvNet của EfficientNetB2 và fcHead
    model = Model(inputs=baseModel.input, outputs=fcHead)
    return baseModel, model
