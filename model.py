from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import InceptionResNetV2

def get_model():
    # base_model = VGG16(weights='imagenet', include_top=True)
    # model = Model(inputs=base_model.input,
    #             outputs=base_model.get_layer(layer).output)
    model = InceptionResNetV2(weights='imagenet', include_top=True)
    
    return model

# from torchvision import models
# import torch 
# def get_model(weight):
#     model = models.resnet50()
#     # model.eval()