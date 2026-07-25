from __future__ import division
from __future__ import print_function
import datetime
import json
import logging
import os
import pickle
import time
import numpy as np
import torch
from geoopt import ManifoldParameter
from config import parser
from models.base_models import HTSGNodeClassifier
from optim import RiemannianAdam, RiemannianSGD
from utils.data_utils import load_data
from utils.train_utils import format_metrics, get_dir_name

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], 'train', date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(save_dir, 'log.txt')), logging.StreamHandler()])
    logging.info(f'Using: {args.device}')
    logging.info(f'Using seed {args.seed}.')
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, _ = data['text_features'].shape
    args.feat_dim = args.dim
    args.n_classes = int(data['labels'].max() + 1)
    args.data = data
    logging.info(f'Num classes: {args.n_classes}')
    model = HTSGNodeClassifier(args)
    if hasattr(model, 'set_class_weights'):
        train_labels = data['labels'][data['idx_train']]
        model.set_class_weights(train_labels, args.device)
    no_decay = ['bias', 'scale']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay) and not isinstance(p, ManifoldParameter)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if (p.requires_grad and any(nd in n for nd in no_decay)) or isinstance(p, ManifoldParameter)],
            'weight_decay': 0.0,
        },
    ]
    if args.optimizer == 'radam':
        optimizer = RiemannianAdam(params=optimizer_grouped_parameters, lr=args.lr, stabilize=10)
    elif args.optimizer == 'rsgd':
        optimizer = RiemannianSGD(params=optimizer_grouped_parameters, lr=args.lr, stabilize=10)
    else:
        raise ValueError(f'Unsupported optimizer: {args.optimizer}')
    tot_params = sum(np.prod(p.size()) for p in model.parameters())
    model = model.to(args.device)
    for key, val in data.items():
        if torch.is_tensor(val):
            data[key] = val.to(args.device)
    logging.info(f'Total number of parameters: {tot_params}')
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_emb = None
    best_state = None
    adj_indices = data['adj_train_norm'].coalesce().indices()
    adj_values = data['adj_train_norm'].coalesce().values()
    adj_shape = data['adj_train_norm'].shape
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        txt_feat = data['text_features']
        img_feat = data['image_features']
        adj_train = data['adj_train_norm']
        embeddings = model.encode(txt_feat, img_feat, adj_train)
        train_metrics, _ = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(' '.join([
                f'Epoch: {epoch + 1:04d}',
                f"lr: {optimizer.param_groups[0]['lr']}",
                format_metrics(train_metrics, 'train'),
                f'time: {time.time() - t:.4f}s'
            ]))
        with torch.no_grad():
            if (epoch + 1) % args.eval_freq == 0:
                model.eval()
                embeddings = model.encode(data['text_features'], data['image_features'], data['adj_train_norm'])
                val_metrics, _ = model.compute_metrics(embeddings, data, 'val')
                if (epoch + 1) % args.log_freq == 0:
                    logging.info(' '.join([f'Epoch: {epoch + 1:04d}', format_metrics(val_metrics, 'val')]))
                if model.has_improved(best_val_metrics, val_metrics):
                    best_val_metrics = val_metrics
                    best_emb = embeddings.detach().cpu()
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    counter = 0
                    if args.save:
                        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.numpy())
                else:
                    counter += 1
                    if counter == args.patience and epoch > args.min_epochs:
                        logging.info('Early stopping')
                        break
    logging.info('Optimization Finished!')
    logging.info(f'Total time elapsed: {time.time() - t_total:.4f}s')
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        model.eval()
        with torch.no_grad():
            best_emb = model.encode(data['text_features'], data['image_features'], data['adj_train_norm']).detach().cpu()

    model.eval()
    with torch.no_grad():
        final_emb = model.encode(data['text_features'], data['image_features'], data['adj_train_norm'])
        best_test_metrics, classification_report = model.compute_metrics(final_emb, data, 'test')
        best_emb = final_emb.detach().cpu()

    logging.info(' '.join(['Val set results:', format_metrics(best_val_metrics, 'val')]))
    logging.info(' '.join(['Test set results:', format_metrics(best_test_metrics, 'test')]))
    logging.info(' '.join(['Test set report:', classification_report]))

    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)
        args_to_save = {}
        for key, value in vars(args).items():
            if key == 'data':
                continue
            try:
                json.dumps(value)
                args_to_save[key] = value
            except TypeError:
                args_to_save[key] = str(value)
        json.dump(args_to_save, open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f'Saved model in {save_dir}')


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
