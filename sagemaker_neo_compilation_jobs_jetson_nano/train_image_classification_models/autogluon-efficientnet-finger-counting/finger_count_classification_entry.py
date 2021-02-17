import subprocess
import os

subprocess.run(["pip",  "install", "autogluon==0.0.15"])
subprocess.run(["pip",  "install", "autogluon.vision==0.0.15b20201108"])
subprocess.run(["pip",  "install", "autogluon.extra==0.0.15b20201108"])
subprocess.run(["pip",  "install", "gluoncv"])  
subprocess.run(["pip",  "install", "mxboard"]) 

from mxnet.gluon import Block
import autogluon.core as ag
from autogluon.vision import ImageClassification as task
from autogluon import model_zoo
import mxnet as mx
from mxnet import nd, image
from mxnet import optimizer as optim
import logging 
import gluoncv
import autogluon.mxnet.task.dataset as ds
from mxnet.gluon.data.vision import transforms
import math
import argparse
import ast
import random
import gluoncv.data.transforms as gcvtransforms


class RandomColorDistort(Block):
    def __init__(self):
        super(RandomColorDistort, self).__init__()

    def forward(self, x):
        if random.random() > 0.4:
            return x
        else:
            return gcvtransforms.experimental.image.random_color_distort(x)


class RandomExpansion(Block):
    def __init__(self):
        super(RandomExpansion, self).__init__()

    def forward(self, x):
        if random.random() > 0.5:
            return x
        else:
            return gcvtransforms.image.random_expand(x, max_ratio=2)[0]
        
def parse_args():
    parser = argparse.ArgumentParser(
        description='Finetune image classification networks.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training mini-batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs for best model search')
    parser.add_argument('--final-fit-epochs', type=int, default=150,
                        help='Training epochs for the final model')
    parser.add_argument('--ngpus-per-trial', type=int, default=1,
                        help='Number of gpus used per trial')
    parser.add_argument('--networks', type=ast.literal_eval,
                        default=['efficientnet_b0'], help='Networks for training or network search')
    parser.add_argument('--num-trials', type=int, default=5,
                        help='Number of trails for parameter search')
    return parser.parse_args()



def generate_transform(train, input_size, jitter_param=0.4, crop_ratio=1):
    resize = int(math.ceil(input_size / crop_ratio))
    if train:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.6, 1)),
                transforms.RandomFlipTopBottom(),
                transforms.RandomFlipLeftRight(),
                RandomColorDistort(),
                RandomExpansion(),
                transforms.RandomLighting(0.2),
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]
        )
    else:
        transform=transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]
        )
    return transform

def create_dataset(path,train=True,input_size=224,jitter_param=0.4,crop_ratio=1):
    transform = generate_transform(train=True, input_size=input_size,jitter_param=jitter_param,crop_ratio=crop_ratio)
    dataset = ds.NativeImageFolderDataset(path,transform=ds._TransformFirstClosure(transform))
    return dataset

streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(streamhandler)

if __name__ == '__main__':
    args = parse_args()
    logging.info('Arguments initialized. The training arguemnts are\n')

    f = open(os.environ["SM_OUTPUT_DATA_DIR"] + "model_summary.txt","w")
    f.write("test file")
    f.close()
    dataset = create_dataset(path=os.environ["SM_CHANNEL_TRAIN"])
    
    @ag.obj(
        learning_rate=ag.space.Real(1e-3, 1e-1, log=True),
        momentum = ag.space.Real(0.89, 0.99, log=True),
        wd=ag.space.Real(1e-4, 1e-3, log=True)
    )
    
    class NAG(optim.NAG):
        pass
    
    optimizer = NAG()

    lr_config = ag.Dict(lr_mode='cosine', lr_decay=0.1, lr_decay_period=0, lr_decay_epoch='50,100', warmup_lr=0.0, warmup_epochs=5)

    network_list = args.networks
    if len(network_list) > 1:
        nets = ag.Categorical(*network_list)
    else:
        nets = network_list[0]

    classifier = task.fit(dataset,
                          net=nets,
                          optimizer=optimizer,
                          search_strategy='skopt',
                          search_options={'base_estimator': 'GP', 'acq_func': 'EI'},
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          metric='accuracy',
                          lr_config=lr_config,
                          verborse=True,
                          plot_results=True,
                          visualizer = 'tensorboard',
                          ngpus_per_trial=args.ngpus_per_trial,
                          num_trials=args.num_trials,
                          final_fit_epochs=args.final_fit_epochs,
                          use_pretrained=True,
                          batch_norm=False,
                          output_directory=os.environ["SM_OUTPUT_DATA_DIR"]
                              )

    test_dataset = create_dataset(os.environ["SM_CHANNEL_TEST"],train=False)  
    test_acc = classifier.evaluate(test_dataset)
    logging.info('Test accuracy: %f' % test_acc)

    input_size = classifier.model.input_size
    num_classes = dataset.rand.num_classes
    logging.info("Input data size for the final model = {}\n Number of output classes = {}".format(input_size,num_classes))
    
    classifier.model.hybridize()
    classifier.model(nd.ones((1,3,input_size,input_size)))
    classifier.model.export('{}/model'.format(os.environ['SM_MODEL_DIR']))
    
    summary = classifier.fit_summary(output_directory='./', verbosity=2)
    logging.info('Top-1 val acc: %3f' %classifier.results['best_reward'])
    logging.info(summary)


