import os
import glob
import torch
from torch.utils.tensorboard import SummaryWriter
from absl import app, flags, logging
from math import inf
from time import time
import numpy as np
import random
from pickle import Pickler, Unpickler
from shutil import copyfile

from models.config import args
from models.utils import AverageMeter
from graph_data import GraphDataBatch

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'GIN', 'configuration parameter set')
flags.DEFINE_string('note', '', 'comment to add to run logs')
flags.DEFINE_string('root_log', './logs/hnet', 'root directory of tensorboard logging')
flags.DEFINE_string('job_id', '', 'process id for batch job')
flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_integer('lr_patience', 100, 'Number of epochs with no improvement after which learning rate will be reduced.')
flags.DEFINE_float('wd', 0.01, 'weight decay')
flags.DEFINE_float('train_split', 0.9, 'train split fraction .9 => 90% train 10% validation')
flags.DEFINE_integer('epochs', 500, 'number of full passes over training data')
flags.DEFINE_integer('batch_size', 128, 'number of graphs per batch')
flags.DEFINE_string('train_data_search', './data/train_*.pkl', 'search path for training data')
flags.DEFINE_string('from_checkpoint', None, 'start training from the pretrained parameters in this file')
flags.DEFINE_list('exclude_augmentation', [], 'augmentations to exclude from training (symetry, path_adjacent)')

def train_arg_text():
    return "lr: {}\n weight_decay {}\n batch_size: {}\n config: {}\n notes: {}".format(
        FLAGS.lr,
        FLAGS.wd,
        FLAGS.batch_size,
        FLAGS.config,
        FLAGS.note
    )

class Data():
    def __init__(self, examples, batch_size, device, train_split, duplicates):
        random.shuffle(examples)
        val_ndx = int(len(examples) * train_split)
        self.train_examples = examples[:val_ndx]
        self.val_examples = examples[val_ndx:]
        self.batch_size = batch_size
        self.device = device
        self.duplicates = duplicates

    def get_batch(self, train):
        examples = self.train_examples if train else self.val_examples
        i = 0
        ex_count = len(examples)
        while i < ex_count:
            ex_batch = examples[i:i+self.batch_size]
            i += self.batch_size
            graphs, h_target, h_dist, _ = zip(*ex_batch)           
            g_batch = GraphDataBatch(graphs=graphs, device=self.device)
            h_target = torch.tensor(h_target, dtype=torch.float, device=self.device).unsqueeze(dim=1)
            h_dist = torch.tensor(h_dist, dtype=torch.float, device=self.device).unsqueeze(dim=1)

            yield g_batch, h_target, h_dist

    @classmethod
    def load(cls, train_data_search, batch_size, device, train_split=0.9, exclude_augmentation=[]):
        examples = []
        g_set = set()
        duplicates = 0

        for file_path in glob.glob(train_data_search):          
            with open(file_path, "rb") as f:
                exl = Unpickler(f).load()
                # de-duplicate graphs
                for ex in exl:
                    cannon_str = ex[0].get_cannon_str()
                    ex[0].zero_edge_attr()
                    if cannon_str not in g_set:
                        examples.append(ex)
                        g_set.add(cannon_str)
                    else:
                        duplicates += 1

        # filter out any excluded augmenations
        logging.info("Total examples: {}".format(len(examples)))
        logging.info("ASH examples: {}".format(len([ex for ex in examples if 'path_adjacent' in ex[3]])))
        examples = [ex for ex in examples if not set(ex[3]) & set(exclude_augmentation)]

        return cls(examples, batch_size, device, train_split, duplicates)


def main(_argv):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = Data.load(
        train_data_search=FLAGS.train_data_search, 
        batch_size=FLAGS.batch_size, 
        device=device,
        train_split=FLAGS.train_split,
        exclude_augmentation=FLAGS.exclude_augmentation
    )

    model_args = args[FLAGS.config]
    model = model_args['nnet'](model_args).to(device)    
    if FLAGS.from_checkpoint:
        checkpoint = torch.load(FLAGS.from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=FLAGS.lr_patience, verbose=True)
    loss_f = torch.nn.MSELoss().to(device)

    job_id = FLAGS.job_id if FLAGS.job_id != '' else "{}_{}".format(FLAGS.config, int(time()))
    log_dir = os.path.join(FLAGS.root_log, job_id)
    checkpoint_fp = os.path.join(log_dir, "checkpoint.pt")
    best_metric = inf
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text(tag='model args', text_string=str(model_args))
    writer.add_text(tag='training', text_string=train_arg_text())
    writer.add_text(tag='training', text_string="{} duplicate examples in training data".format(train_data.duplicates))
    with open(os.path.join(log_dir, "model_args.pkl"), "wb") as f:
        Pickler(f).dump(model_args)

    for epoch in range(FLAGS.epochs):
        print('epoch: ', epoch)
        train_loss = AverageMeter()
        val_loss = AverageMeter()
        val_mean_error = AverageMeter()
        
        # Training
        model.train(True)
        for batch, h_target, h_dist in train_data.get_batch(train=True):
            h_pred = model(batch)
            loss = loss_f(h_pred, h_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())

        writer.add_scalar('Loss/train', train_loss.avg, epoch)

        # Validation
        model.train(False)
        with torch.no_grad():
            for batch, h_target, h_dist in train_data.get_batch(train=False):
                h_pred = model(batch)
                loss = loss_f(h_pred, h_target)
                val_loss.update(loss.item())
                val_mean_error.update(torch.nn.functional.l1_loss(h_pred, h_target))

        writer.add_scalar('Loss/validation', val_loss.avg, epoch)
        writer.add_scalar('MAE/validation', val_mean_error.avg, epoch)
        scheduler.step(val_loss.avg)

        if val_loss.avg < best_metric:
            best_metric = val_loss.avg
            torch.save({
                'state_dict': model.state_dict(),
            }, checkpoint_fp)
            txt = "Saving checkpoint, loss:{}, mean error:{}".format(best_metric, val_mean_error.avg)
            writer.add_text(tag='training', text_string=txt, global_step=epoch)

        writer.flush()
    writer.close()
    copyfile(checkpoint_fp, "logs/{}.pt".format(FLAGS.config))
    

if __name__ == '__main__':
    app.run(main)