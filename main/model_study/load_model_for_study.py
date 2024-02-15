import os
import sys
import pickle
import logging
import argparse
import pandas as pd
import torch
import numpy as np

from helpers import *
from models.general import *
from models.sequential import *
from models.developing import *
from models.context import *
from models.context_seq import *
from utils import utils

def parse_global_args(parser):
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed of numpy and pytorch')
    parser.add_argument('--gpu', type=str, default='3',
                        help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--load', type=int, default=1,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=0,
                        help='To train the model or not.')
    parser.add_argument('--test', type=int, default=1,
                        help='To test the model or not.')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files')
    # parser.add_argument('-model_path', type=str, default='',
    #                     help='Loading model path')
    parser.add_argument('--log_immers_dir', type=str, default='../log_immers/')
    parser.add_argument('--save_source_immers', type=int, default=0,
                        help="Wheter to save source immersion")
    parser.add_argument('--study_all',type=int, default=0,
                        help="Wheter to tested on all")
    return parser


def main():
    exclude = [ 'model_path', 'path', 'pin_memory', 'load',
               'regenerate', 'sep', 'train', 'metric', 'buffer']
    # Random seed
    utils.init_seed(args.random_seed)

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cpu')
    if args.gpu != '' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    logging.info('Device: {}'.format(args.device))

    print('read data')
    # Read data
    corpus_path = os.path.join(args.path, args.dataset, model_name.reader+args.data_appendix+ '.pkl')
    if not args.regenerate and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        print ('generate data')
        corpus = reader_name(args)
        logging.info('Save corpus to {}'.format(corpus_path))
        pickle.dump(corpus, open(corpus_path, 'wb'))

    # Define model
    model = model_name(args, corpus).to(args.device)
    if args.load > 0:
        model.load_model(args.model_path)
    for name, param in model.named_parameters():
        # if "im_params" in name:
        #     print(f"{name}: {param}")
        if "im" in name:
            print(f"{name}: {param}")

    # Predict model
    if args.test>0:
        if not os.path.exists(args.log_immers_dir):
            os.makedirs(args.log_immers_dir)
        if not os.path.exists(os.path.join(args.log_immers_dir,init_args.model_name)):
            os.makedirs(os.path.join(args.log_immers_dir,init_args.model_name))
        runner = runner_name(args)
        data_dict = dict()
        if args.study_all:
            print('start preparing data')
            data_dict['train']= model_name.Dataset(model, corpus, 'train')
            save_immersion_all(data_dict['train'],runner, 1)
        else:
            result_path = os.path.join(args.log_immers_dir, args.model_path.split('model/')[1].split('.pt')[0]+'.csv')
            if os.path.exists(result_path):
                print('result already exists! Pay attention to mode a for adding!')
            for phase in ['train', 'dev', 'test']:
                data_dict[phase] = model_name.Dataset(model, corpus, phase)
                data_dict[phase].prepare()
                if phase=='train':
                    continue
                    save_rec_results(data_dict[phase], runner, 1, result_path)
                else:
                    save_rec_results(data_dict[phase], runner, 50, result_path)
                
def save_immersion_all(dataset,runner,topk):
    result_path_foot = os.path.join(args.log_immers_dir, args.model_path.split('model/')[1].split('.pt')[0]+'_all_')
    print('Saving top-{} recommendation results to: {}'.format(topk, result_path_foot))
    predictions,immersions = runner.predict(dataset)
    np.save(result_path_foot+'predictions.npy', predictions)
    np.save(result_path_foot+'immersions.npy', immersions)


def save_rec_results(dataset, runner, topk,result_path):
    print('Saving top-{} recommendation results to: {}'.format(topk, result_path))
    predictions,immersions = runner.predict(dataset)  # n_users, n_candidates
    users, rec_items, item_ids, scores, immers = list(), list(), list(), list(), list()
    print('finish prediction, start alignment')
    for i in range(len(dataset)):
        info = dataset[i]
        users.append(info['user_id'])
        item_scores = zip(info['item_id'], predictions[i])
        sorted_lst = sorted(item_scores, key=lambda x: x[1], reverse=True)[:topk]
        rec_items.append([x[0] for x in sorted_lst])
        item_ids.append(list(info['item_id']))
        scores.append(list(predictions[i]))
        immers.append(list(immersions[i]))
    print('finish alignment, start saving')
    rec_df = pd.DataFrame(columns=['user_id',  'rec_items', 'item_ids', 'predictions','immersions'])
    rec_df['user_id'] = users
    rec_df['rec_items'] = rec_items
    rec_df['item_ids'] = item_ids
    rec_df['predictions'] = scores
    rec_df['immersions'] = immers
    rec_df.to_csv(result_path, sep=args.sep, mode='a', index=False, header=not os.path.exists(result_path))


if __name__ == '__main__':
    print('start')
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='WideDeep', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    reader_name = eval('{0}.{0}'.format(model_name.reader))  # model chooses the reader
    runner_name = eval('{0}.{0}'.format(model_name.runner))  # model chooses the runner
    
    

    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, extras = parser.parse_known_args()
    
    args.data_appendix = ''
    if model_name.reader in  ['ContextReader','ContextSeqReader']:
        args.data_appendix = '_context%d%d%d'%(args.include_item_features,args.include_user_features,
                                        args.include_context_features)

    # Logging configuration
    log_args = [init_args.model_name, args.dataset+args.data_appendix, str(args.random_seed)]
    for arg in ['lr', 'l2', 'tradeoff_DA', 'tradeoff_Clf'] + model_name.extra_log_args:
        log_args.append(arg + '=' + str(eval('args.' + arg)))
    log_file_name = '__'.join(log_args).replace(' ', '__')
    if args.log_file == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    # if args.model_path == '':
    #     args.model_path = '../model/{}/{}.pt'.format(init_args.model_name, log_file_name)

    print('main function')
    main()

