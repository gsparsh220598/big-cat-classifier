import argparse, os
from fastcore.all import *
from fastai.vision.widgets import *
from fastdownload import download_url
from fastai.vision.all import *
from time import sleep
import pandas as pd
import wandb
import params
import utils
import torchvision.models as tvmodels
from fastai.callback.wandb import WandbCallback

# defaults
default_config = SimpleNamespace(
    framework="fastai",
    img_size=180, #(180, 320) in 16:9 proportions,
    batch_size=8, #8 keep small in Colab to be manageable
    augment=True, # use data augmentation
    epochs=10, # for brevity, increase for better results :)
    lr=2e-3,
    pretrained=True,  # whether to use pretrained encoder,
    mixed_precision=True, # use automatic mixed precision
    arch="resnet18",
    seed=42,
    log_preds=False,
)

def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--img_size', type=int, default=default_config.img_size, help='image size')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--arch', type=str, default=default_config.arch, help='timm backbone architecture')
    argparser.add_argument('--augment', type=bool, default=default_config.augment, help='Use image augmentation')
    argparser.add_argument('--seed', type=int, default=default_config.seed, help='random seed')
    argparser.add_argument('--log_preds', type=bool, default=default_config.log_preds, help='log model predictions')
    argparser.add_argument('--pretrained', type=bool, default=default_config.pretrained, help='Use pretrained model')
    argparser.add_argument('--mixed_precision', type=bool, default=default_config.mixed_precision, help='use fp16')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


def download_data():
    processed_data_at = wandb.use_artifact(f'{params.PROCESSED_DATA_AT}:latest')
    processed_data_dir = Path(processed_data_at.download())
    return processed_data_dir

def get_df(processed_dataset_dir, is_test=False):
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    path = processed_dataset_dir/'bcc_images'

    #assign paths
    df['image_fname'] = [f for f in get_image_files(path)]
    df['label'] = [utils.label_func(f) for f in df.image_fname.values]
    
    #val/test mode
    if not is_test:
        df = df[df.stage != 'test'].reset_index(drop=True)
        df['is_valid'] = df.stage == 'valid'
    else:
        df = df[df.stage == 'test'].reset_index(drop=True)

    return df

def get_data(df, bs=4, img_size=(180, 320), augment=True):
    block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                  get_x=ColReader("image_fname"),
                  get_y=ColReader("label"),
                  splitter=ColSplitter(),
                  item_tfms=Resize(img_size),
                  batch_tfms=aug_transforms() if augment else None,
                 )
    return block.dataloaders(df, bs=bs, shuffle=True)

def log_predicitons(learner):
    "Log Predictions with class probabilities"
    samples, outputs, predictions = utils.get_predictions(learn)
    table = utils.create_prob_table(samples, outputs, predictions, params.BIG_CAT_CLASSES)
    wandb.log({"pred_table":table})
    
def log_metrics(learner):
    scores = learner.validate()
    metric_names = ['final_loss'] + [f'final_{x}' for x in ['accuracy', 'error_rate', 'f1score_weighted', 'hamming_loss']]
    final_results = {metric_names[i] : scores[i] for i in range(len(scores))}
    for k,v in final_results.items():
        wandb.summary[k] = v
        
def train(config):
    set_seed(config.seed, reproducible=True)
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="training", config=config)
    
    config = wandb.config
    processed_dataset_dir = download_data()
    df = get_df(processed_dataset_dir)
    dls = get_data(df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)

    metrics = [accuracy, error_rate, F1Score(average='weighted'), HammingLoss()]
    learn = vision_learner(dls, arch=getattr(tvmodels, config.arch), pretrained=config.pretrained, metrics=metrics)

    cbs = [
        SaveModelCallback(monitor='accuracy'),
        WandbCallback(log_preds=False, log_model=True)
    ]
    cbs += ([MixedPrecision()] if config.mixed_precision else [])

    learn.fit_one_cycle(config.epochs, config.lr, cbs=cbs)
    
    if config.log_preds:
        log_predictions(learn)
    log_metrics(learn)
    
    wandb.finish()
    
if __name__ == '__main__':
    parse_args()
    train(default_config)