# How to extract features
1. Run ```downloads.sh``` to download [caffe model](https://www.dropbox.com/s/exwqft6f5rcyzz7/test.prototxt?dl=0) and [weights](https://www.dropbox.com/s/2saef01f8j2rf70/model_iter_75000.caffemodel?dl=0) to test.prototxt and  model_iter_75000.caffemodel, respectively
2. Run ```mars_eval.py``` to extract features. It will produce mars_feat.mat in the directory. To run the code, you need to prepare the followings:

    - Download and build custom [caffe](https://github.com/yuminsuh/caffe_retrieval)
    - Download the [dataset](http://www.liangzheng.com.cn/Project/project_mars.html) and extract files to MARS_DATASET_ROOT. The data structure should look like ```MARS_DATASET_ROOT/bbox_test/0000/*.jpg```.
    - Set paths, CAFFE_ROOT and MARS_DATASET_ROOT, in mars_eval.py

# How to evaluate
1. Extract feutures or download [features](https://www.dropbox.com/s/i38ofh0vhm8zalc/mars_feat.mat?dl=0) to mars_feat.mat
2. Run ```mars_eval.m``` to get the accuracy
