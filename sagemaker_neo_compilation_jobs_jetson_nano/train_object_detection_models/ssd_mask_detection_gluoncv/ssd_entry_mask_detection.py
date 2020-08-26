import os
import json
import time
import random
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon

import subprocess
subprocess.run(["pip",  "install", "matplotlib==3.2.0"])
subprocess.run(["pip",  "install", "gluoncv==0.4.0"])

import gluoncv as gcv
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv import model_zoo, data, utils
import argparse
import matplotlib.pyplot as plt
import logging

    
# set logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Finetune SSD networks.')
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training mini-batch size')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Training epochs.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-interval', type=int, default=60,
                        help='epochs at which learning rate decays.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--task', type=str, default='mask-det',
                        help='Labeling name used in manifest file.')
    parser.add_argument('--base-dataset', type=str, default='voc',
                        help='dataset used to train base network, default is voc dataset')
    parser.add_argument('--label-file', type=str, default='output.manifest',
                        help='manifest file containing labels, default name is output.manifest')
    parser.add_argument('--classes-file', type=str, default='classes.lst',
                        help='text file containing class names')
    return parser.parse_args()



class GetS3Dataset(gluon.data.Dataset):
    """Class to generate the dataset with labels.
    """
    def __init__(self, label_file = 'output.manifest',s3_label_path='label', s3_data_path='data', split='train', task='mask-det'):

        self.s3_data_path = s3_data_path
        self.image_info = []
        self.task = task
        with open(os.path.join(s3_label_path,label_file)) as f:
            lines = f.readlines()
            for line in lines:
                info = json.loads(line[:-1])
                if len(info[self.task]['annotations']):
                    self.image_info.append(info)
                    
        random.seed(1950)
        random.shuffle(self.image_info)
        l = len(self.image_info)
        
        if split == 'train':
            self.image_info = self.image_info[:int(0.85*l)]
        if split == 'val':
            self.image_info = self.image_info[int(0.85*l):int(l)]

        
        
    def __getitem__(self, idx):
        info = self.image_info[idx]
        # read image 
        image = mx.image.imread(os.path.join(self.s3_data_path,info['source-ref'].split('/')[-1]))
        # get bounding boxes and labels
        boxes = info[self.task]['annotations']
        label = []
        for box in boxes:
            label.append([box['left'], box['top'], box['left']+box['width'], box['top']+box['height'], box['class_id']])
     
        return image, np.array(label)
        
    def __len__(self):
        return len(self.image_info)

    
def get_dataloader(model, train_dataset, validation_dataset, height, width, batch_size, num_workers):
    """Data pre-processing. Returns mini batches of dataset with transformations

    Args:
        model (SSD model): Object detection model
        train_dataset (Dataset): Training images and labels
        validation_dataset (Dataset): Validation images and labels
        height (int): Height of the training image
        width (int): Width of training image
        batch_size (int): Number of images in a mini batch
        num_workers (int): Number of multiprocessing workers

    Returns:
        Dataloader : Mini batches of data
    """
    
    with autograd.train_mode():
        _, _, anchors = model(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  
    
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(height, width, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    
    return train_loader

    
def train(model, train_loader,ctx,args): 
    """Train a pretrained model

    Args:
        model (HybridBlock): SSD object detection model. Network with initial weights
        train_loader (Dataloader): Preprocessed data for training
        ctx (Context): A list of Contexts
        args (argument list): Training arguments such as hyperparameters for fine-tuning

    Returns:
        Model: A trained object detection model
    """
    
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': args.lr, 'wd': args.wd,'momentum':args.momentum})

    # define losses - Confidence Loss (Cross entropy) and  Location Loss (L2 loss)
    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')
    
    epochs = args.epochs
    lr_step = args.lr_decay_interval
    lr_steps = []
    if lr_step != 0:
        lr_steps = [(i-1) for i in range(1,epochs) if i%lr_step==0]
        lr_decay = args.lr_decay
    num_batch = len(train_loader)
    

    logger.info('Training started')
    # start learning
    for epoch in range(0, epochs):
        
        #Update learning based on the lr_decay
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)

            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        
        model.hybridize(static_alloc=True, static_shape=True)
        
        #iterate over training images
        for i, batch in enumerate(train_loader):
            
            batch_size = batch[0].shape[0]
            
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = model(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
                
            trainer.step(1)
            
            #update and print losses
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()

    
    #export and save model
    model.set_nms(nms_thresh=0.45, nms_topk=400, post_nms=100)
    model(mx.nd.ones((1,3,args.data_shape,args.data_shape), ctx=ctx[0]))
    model.export('%s/model' % os.environ['SM_MODEL_DIR'])
    return model


def read_classes(file):
    classes = []
    with open(file,'r') as f:
        lines = f.readlines()
        for l in lines:
            classes.append(l.split('\n')[0])
    f.close()
    return classes


if __name__ == '__main__':
    args = parse_args()
    print('Arguments initialized. The training arguemnts are\n')
    print(args)

    logger.info(args)
    
    # get the pretrained model and set classes
    class_list = os.path.join(os.environ["SM_CHANNEL_LABELS"],args.classes_file) 
    classes = read_classes(class_list)
    logger.info('Classes loaded')
    logger.info('Classes = {}'.format(classes))
    
    model = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False, transfer=args.base_dataset)
    

    #images and labels from Groundtruth are downloaded by Sagemaker into training instance
    train_dataset = GetS3Dataset(split='train', label_file = args.label_file, s3_label_path=os.environ["SM_CHANNEL_LABELS"], s3_data_path=os.environ["SM_CHANNEL_TRAIN"], task=args.task)
    val_dataset = GetS3Dataset(split='val', label_file= args.label_file, s3_label_path=os.environ["SM_CHANNEL_LABELS"], s3_data_path=os.environ["SM_CHANNEL_TRAIN"], task=args.task)
    
    logger.info('Training data loaded')

    train_loader= get_dataloader(model, train_dataset, val_dataset, args.data_shape,args.data_shape, args.batch_size, 1)
    
    #check if GPUs are available
    ctx = [mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()]
    
    #reassign parameters to context ctx
    model.collect_params().reset_ctx(ctx)
    
    train(model,train_loader,ctx,args)
