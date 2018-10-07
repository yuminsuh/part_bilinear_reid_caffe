#!/usr/bin/env bash

# Download pretrained network weights
if ! [ -f "test.prototxt" ]; then
    wget "https://www.dropbox.com/s/exwqft6f5rcyzz7/test.prototxt?dl=0" -O "test.prototxt"
fi
if ! [ -f "model_iter_75000.caffemodel" ]; then
    wget "https://www.dropbox.com/s/2saef01f8j2rf70/model_iter_75000.caffemodel?dl=0" -O "model_iter_75000.caffemodel"
    
fi
