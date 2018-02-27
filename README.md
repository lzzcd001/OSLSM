
# One-shot Learning for Semantic Segmentation (OSLSM)

By Amirreza Shaban, Shray Bansal, Zhen Liu, Irfan Essa and Byron Boots

### Paper

You can find our paper at https://arxiv.org/abs/1709.03410


### Citation

If you find OSLSM useful in your research, please consider to cite:

 @inproceedings{shaban2017one,
	  title={One-Shot Learning for Semantic Segmentation},
	  author={Shaban, Amirreza and Bansal, Shray and Liu, Zhen and Essa, Irfan and Boots, Byron},
	  journal={arXiv preprint arXiv:1709.03410},
	  year={2017}
 }


### Instructions for Testing (tested on Ubuntu 16.04)
We assume you have downloaded the repository into ${OSLSM_HOME} path.

1. Install Caffe prerequisites and build the Caffe code (with PyCaffe). See http://caffe.berkeleyvision.org/installation.html for more details

```shell 
cd ${OSLSM_HOME}
mkdir build
cd build
cmake ..
make all -j8
```

If you prefer Make, set BLAS to your desired one in Makefile.config. Then run:

```shell
cd ${OSLSM_HOME}
make all -j8
make pycaffe
```

2. Update the `$PYTHONPATH`: 

```shell
export PYTHONPATH=${OSLSM_HOME}/OSLSM/code:${OSLSM_HOME}/python:$PYTHONPATH
```

3. Download PASCAL VOC dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

4. Download trained models from: https://gtvault-my.sharepoint.com/:u:/g/personal/ashaban6_gatech_edu/EXS5Cj8nrL9CnIJjv5YkhEgBQt9WAcIabDQv22AERZEeUQ

5. Set `CAFFE_PATH=${OSLSM_HOME}` and `PASCAL_PATH` in `${OSLSM_HOME}/OSLSM/code/db_path.py` file

6. Run the following to test the models in one-shot setting:

```shell
cd ${OSLSM_HOME}/OSLSM/os_semantic_segmentation
python test.py deploy_1shot.prototxt ${TRAINED_MODEL} ${RESULTS_PATH} 1000 fold${FOLD_ID}_1shot_test
```

Where ${FOLD_ID} can be 0,1,2, or 3 and ${TRAIN_MODEL} is the path to the trained caffe model. Please note that we have included different caffe models for each ${FOLD_ID}.

Simillarly, run the following to test the models in 5-shot setting:

```shell
cd ${OSLSM_HOME}/OSLSM/os_semantic_segmentation
python test.py deploy_5shot.prototxt ${TRAINED_MODEL} ${RESULTS_PATH} 1000 fold${FOLD_ID}_5shot_test
```

7. For training your own models, we have included all prototxts in `${OSLSM_HOME}/OSLSM/os_semantic_segmentation/training` directory and the vgg pre-trained model can be found in `snapshots/os_pretrained.caffemodel`.

You will also need to

1) Download/Prepare SBD dataset (http://home.bharathh.info/pubs/codes/SBD/download.html).

2) Set `SBD_PATH` in `${OSLSM_HOME}/OSLSM/code/db_path.py`

3) Set the profile to `fold${FOLD_ID}\_train` for our data layer (check the prototxt files and `${OSLSM_HOME}/OSLSM/code/ss_datalayer.py`) to work.

### License

The code and models here are available under the same license as Caffe (BSD-2) and the Caffe-bundled models (that is, unrestricted use; see the BVLC model license).


### Contact

For further questions, you can leave them as issues in the repository, or contact the authors directly:

Amirreza Shaban amirreza@gatech.edu

Shray Bansal sbansal34@gatech.edu

Zhen Liu liuzhen1994@gatech.edu



# About Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
