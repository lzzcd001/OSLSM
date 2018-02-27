"""
test.py - Test network using low-shot pairs
"""

# TODO: Put all the copied files in a temp folder and operate out of there. Or no copy, just shift to the fcn dir
# current system is bad
import caffe
import os
import sys
import numpy as np
import ss_datalayer
import csv
from skimage.io import imsave
from matplotlib.pyplot import imshow,show, figure
import time

# Get image pairs
class LoaderOfPairs(object):
    def __init__(self, profile):
        profile_copy = profile.copy()
        profile_copy['first_label_params'].append(('original_first_label', 1.0, 0.0))
	profile_copy['deploy_mode'] = True
        dbi = ss_datalayer.DBInterface(profile)
        self.PLP = ss_datalayer.PairLoaderProcess(None, None, dbi, profile_copy)
    def get_items(self):
        self.out = self.PLP.load_next_frame(try_mode=False)
        return (np.asarray(self.out['first_img']), #[np.newaxis,:,:,:], 
                np.asarray(self.out['first_label']), #[np.newaxis,:,:,:], 
                np.asarray(self.out['second_img']), #[np.newaxis,:,:,:], 
                np.asarray(self.out['second_label']), #[np.newaxis,:,:,:], 
                self.out['deploy_info'])
    def get_items_no_return(self):
        self.out = self.PLP.load_next_frame(try_mode=False)
    # corrects the image to be displayable
    def correct_im(self, im):
        im = (np.transpose(im, (0,2,3,1)))/255.
        im += np.array([0.40787055,  0.45752459,  0.4810938])
        return im[:,:,:,::-1]
    # returns the outputs as images and also the first label in original img size
    def get_items_im(self):
        self.out = self.PLP.load_next_frame(try_mode=False)
        return (self.correct_im(self.out['first_img']), 
                self.out['original_first_label'], 
                self.correct_im(self.out['second_img']), 
                self.out['second_label'][0], 
                self.out['deploy_info'])

def __stack(obj_list):
    isinstance(obj_list, list)
    if isinstance(obj_list[0], np.ndarray):
        #arrays = np.stack(obj_list, axis = 0)
        arrays = np.concatenate(obj_list, axis = 0).reshape((len(obj_list),) + obj_list[0].shape)
    else:
        #It should be a list of numbers
        assert(not hasattr(obj_list[0], '__len__'))
        arrays = np.array(obj_list)
    return arrays

def compute_net_inputs(pair_item, input_names):
    inputs = dict()
    for input_name in input_names:
        inputs[input_name] = __stack(pair_item[input_name])
    return inputs   
# the z(h)en function for measuring IOU  
def measure(y_in, pred_in):
    thresh = .5
    y = y_in>thresh
    pred = pred_in>thresh
    tp = np.logical_and(y,pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn

def test_net(loader, model_path, weights_path, output_dir, test_iters, gpu, test_cats_indicator=range(20), save_images=True):
    '''
        Test our one-shot net on test categories

        Inputs:

            loader: data loader
            model_path: specifies the model path, i.e. deploy.prototxt
            weights_path: path to weights file, i.e. caffemodel
            output_dir: where to store your test results
            gpu: specifies which gpu to run on
            test_cats_indicator: a list of indices (from 0 to 19, since we have 20 image classes) that specifies which categories are used for testing
            save_images: whether to visualize the segmented images and save them into output_dir

    '''

    caffe.set_mode_gpu()
    caffe.set_device(gpu)


    # load the parent net
    print model_path
    net = caffe.Net(model_path, weights_path, caffe.TEST)
    pred_dir = os.path.join(output_dir, 'pred')
    soft_mask_dir = os.path.join(pred_dir, 'soft')
    final_mask_dir = os.path.join(pred_dir, 'hard')
    first_label_dir = os.path.join(output_dir, 'first_label')
    os.makedirs(pred_dir)
    os.makedirs(first_label_dir)
    os.makedirs(soft_mask_dir)
    os.makedirs(final_mask_dir)

    # record all the stats
    recorder=[]


    # list of True Pos, True neg, False Pos, False Neg
    # Note: IOUs for all training categories will be zero; they are simply placeholders
    num_classes = 20
    tp_list = [0]*num_classes 
    fp_list = [0]*num_classes 
    fn_list = [0]*num_classes
    iou_list = [0]*num_classes

    total_time = 0

    total_time_iter = 0

    for ti in range(test_iters):
        iter_start_time = time.time()
        first_img, first_label, second_img, second_label, deploy_info = loader.get_items()
        print '>'*10, 'Test Iteration: ', ti
        inputs = compute_net_inputs(loader.out, net.inputs)
        #TODO: REMOVE THIS OR THE NET DOESNT DO GRAD DESCENT
        # make learning rate zero
        #net.params['w1s'][2].data[...] = 0.0

        #old_w1 = net.params['w1s'][0].data.copy()  

        start_time = time.time()
        net.forward(**inputs)
        total_time += time.time()-start_time
        #assert((np.absolute(old_w1 -net.blobs['w1s'].data)).sum()==0.0)
        
        
        # get the input, output mask
        pred = net.blobs['score'].data[0,0]
        tp, tn, fp, fn = measure(second_label[0,0], net.blobs['score'].data[0,0])
        iou_img = tp/float(max(tp+fp+fn,1))
        class_ind = int(deploy_info['first_semantic_labels'][0][0])-1 # because class indices from 1 in data layer
        tp_list[class_ind] += tp
        fp_list[class_ind] += fp
        fn_list[class_ind] += fn
        # max in case both pred and label are zero 
        iou_list = [tp_list[ic] / 
                       float(max(tp_list[ic] + fp_list[ic] + fn_list[ic],1)) 
                       for ic in range(num_classes)]
        # record
        recorder.append([ti, 0, class_ind, tp, tn, fp, fn, iou_img])
        #debug
        #print 'Min, Max of mask = ', pred.min(), pred.max()
        
        if save_images:
            # save image
            imsave(os.path.join(soft_mask_dir, '%05d.png' % (ti)), pred)

        Threshold = 0.5
        pred[pred < Threshold] = 0
        pred[pred > 0] = 1

        if save_images:
            imsave(os.path.join(final_mask_dir, '%05d.png' % (ti)), pred)    
        # # Debug: Show Stuff
        # imshow(net.blobs['score'].data[0,0])
        # figure()
        # imshow(net.blobs['second_img'].data[0,0])
        # figure()
        # # run net for first img
        # imshow(net.blobs['first_img'].data[0,0])
        # figure()
        # imshow(net.blobs['first_label'].data[0,0])
        # #print iou
        # show()

        # record everything
        if ti%10==0:
            with open(os.path.join(output_dir,'iou_results.txt'), 'w') as iou_file:
                iou_file.write('tp: ' +  ' '.join([str(x) for x in tp_list]) + '\n')
                iou_file.write('fp: ' +  ' '.join([str(x) for x in fp_list]) + '\n')
                iou_file.write('fn: ' +  ' '.join([str(x) for x in fn_list]) + '\n')
                iou_file.write('iou (all categories): ' +  ' '.join([str(x) for x in iou_list]) + '\n')
                iou_file.write('mean_iou (only test categories): ' + str(np.mean(np.take(iou_list, test_cats_indicator))) + '\n')
            with open(os.path.join(output_dir, 'all_results.csv'), 'wb') as allresults:
                    writer = csv.writer(allresults)
                    writer.writerows(recorder)
        total_time_iter += time.time() - iter_start_time

    # record the time per iter
    print '\n--Time per the iteration forward is {}'.format(total_time/float(test_iters))
    print '\n--Time per the whole iteration is {}'.format(total_time_iter/float(test_iters))

    # Print meanIOU
    print '\n--The mean IOU for this fold is {}'.format(np.mean(np.take(iou_list, test_cats_indicator)))

"""
    Test FCN on 15 training categories
"""
def test_net_fcn(loader, model_path, weights_path, output_dir, test_iters, gpu):
    caffe.set_mode_gpu()
    caffe.set_device(gpu)
    # load the parent net
    print model_path
    net = caffe.Net(model_path, weights_path, caffe.TEST)
    # record all the stats
    recorder=[]
    # list of True Pos, True neg, False Pos, False Neg
    num_classes=15
    tp_list = [0]*num_classes 
    fp_list = [0]*num_classes 
    fn_list = [0]*num_classes
    iou_list = [0]*num_classes

    for ti in range(test_iters):
        loader.get_items_no_return()
        print '>'*10, 'Test Iteration: ', ti
        inputs = compute_net_inputs(loader.out, net.inputs)
        first_img = inputs['first_img']
        first_label = np.asarray(loader.out['first_label'])
        print 'First Img = ', first_img.shape
        print 'First Label = ', first_label.shape
        print 'Label Max = ', first_label.max()
        net.forward(**inputs)
        # get the input, output mask
        pred = net.blobs['score'].data[0,0]
        
        for c in range(num_classes):
            cind =c+1 # ignore background class
            tp, tn, fp, fn = measure(first_label==cind, net.blobs['score'].data[0,cind])
            tp_list[c] += tp
            fp_list[c] += fp
            fn_list[c] += fn
            # if (first_label==cind).sum() > 5:
            #     # Debug: Show Stuff
            #     imshow(net.blobs['score'].data[0,cind])
            #     figure()
            #     imshow(net.blobs['first_img'].data[0,0])
            #     figure()
            #     imshow(first_label[0,0]==cind)
            #     print (first_label==cind).sum()
            #     print tp, fp, fn
            #     show()

        # record
        recorder.append([ti, 0, 0, tp, tn, fp, fn, 0.0])
        Threshold = 0.5
        pred[pred < Threshold] = 0
        pred[pred > 0] = 1
        #imsave(os.path.join(final_mask_dir, '%05d.png' % (ti)), pred)    

    # record everything
    # max in case both pred and label are zero 
    iou_list = [tp_list[ic] / 
                float(max(tp_list[ic] + fp_list[ic] + fn_list[ic],1)) 
                for ic in range(num_classes)]
    with open(os.path.join(output_dir,'iou_results.txt'), 'w') as iou_file:
        iou_file.write('tp: ' +  ' '.join([str(x) for x in tp_list]) + '\n')
        iou_file.write('fp: ' +  ' '.join([str(x) for x in fp_list]) + '\n')
        iou_file.write('fn: ' +  ' '.join([str(x) for x in fn_list]) + '\n')
        iou_file.write('iou: ' +  ' '.join([str(x) for x in iou_list]) + '\n')
        iou_file.write('mean_iou: ' + str(np.mean(iou_list)) + '\n')
    with open(os.path.join(output_dir, 'all_results.csv'), 'wb') as allresults:
        writer = csv.writer(allresults)
        writer.writerows(recorder)
                
if __name__=='__main__':
    # check commandline args
    assert (len(sys.argv) >= 6), 'Usage: test.py [model_path] [weights_path] [output_dir] [test_iters] [ss_profile] [gpu_number=0] [save_images=0/1]'
    
    model_path = os.path.abspath(sys.argv[1])
    weights_path = os.path.abspath(sys.argv[2])
    output_dir = os.path.abspath(sys.argv[3])
    test_iters = int(sys.argv[4])
    ss_profile = sys.argv[5]
    gpu = 0
    if (len(sys.argv)>6):
        gpu = int(sys.argv[6])
    save_images = True
    if (len(sys.argv)>7):
        save_images = bool(int(sys.argv[7]))

    settings = __import__('ss_settings')
    profile = getattr(settings, ss_profile)

    # Specify what the test categories are (for calculating final meanIOU)
    test_cats_indicator = [ind for ind in range(20) if (profile.default_pascal_cats[ind] in profile.pascal_cats)]


    os.makedirs(output_dir)

    # Pair Loader Process
    loader = LoaderOfPairs(profile)

    # test
    test_net(loader, model_path, weights_path, output_dir, test_iters, gpu, test_cats_indicator, save_images)
