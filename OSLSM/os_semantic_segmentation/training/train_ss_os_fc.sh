CUDA_VISIBLE_DEVICES=0 /storage/dev/video_seg/caffe/build/tools/caffe train \
    -solver=solver_ss_os_fc.prototxt \
    -weights=ss_os_fc.caffemodel \
    -gpu 0 \
    2>&1 | tee ./log/train_ss_os_fc"log$(date +'%m_%d_%y')".log
