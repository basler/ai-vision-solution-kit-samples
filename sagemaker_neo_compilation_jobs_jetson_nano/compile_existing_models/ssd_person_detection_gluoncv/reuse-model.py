import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import nd
import argparse
import logging

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune resuse weights')
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape, use 512.")
    parser.add_argument('--base-model', type=str, default='ssd_512_mobilenet1.0_voc',
                        help='Pretrained model')
    parser.add_argument('--classes', nargs="+", default=['Person'],
                        help='class names default value contains person detection class')
    parser.add_argument('--reuse-classes', nargs="+", default=['person'],
                        help='corresponding class names in the old model')
    parser.add_argument('--save-as', type=str, default='person-detection',
                        help='Name of exported model')
    return parser.parse_args()

if __name__ == '__main__':
    
    logging.info('Parsing arguments')
    args = parse_args()
                                  
    logging.info('Argument values are Data shape = {}, Base model = {}, Classes = {}, Reuse classes ={}, Save as = {}'.format(args.data_shape,args.base_model, args.classes, args.reuse_classes, args.save_as))
    
    class_dict = {}
    for c in range(0,len(args.classes)):
        class_dict[args.classes[c]] = args.reuse_classes[c] 
    
    logging.info('Mapping to new classes {}'.format(class_dict))
    
    net = gcv.model_zoo.get_model(args.base_model, pretrained=True)
    net.reset_class(classes=args.classes, reuse_weights=class_dict)
    
    net.hybridize(static_alloc=True, static_shape=True)
    ctx = [mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()]
    
    net.set_nms(nms_thresh=0.45, nms_topk=400, post_nms=100)
    net(mx.nd.ones((1,3,args.data_shape,args.data_shape), ctx=ctx[0]))
    logging.info('Exporting model')
    net.export(args.save_as)
