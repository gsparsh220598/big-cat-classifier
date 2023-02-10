from duckduckgo_search import ddg_images
from fastcore.all import *
from fastai.vision.widgets import *
from fastdownload import download_url
from fastai.vision.all import *
from time import sleep
import wandb

def label_func(fname):
    return fname.parent.name

def search_images(term, max_images=20):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

def _download_images(download_path, searches):
    for o in searches:
        dest = (download_path/o)
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{o} day photo'))
        sleep(10)  # Pause between searches to avoid over-loading server
        download_images(dest, urls=search_images(f'{o} female photo'))
        sleep(10)
        download_images(dest, urls=search_images(f'{o} at night photo'))
        sleep(10)
        resize_images(path/o, max_size=400, dest=path/o)
        
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(f"Failed images: {len(failed)}")
        
def _prepare_data(path, class_labels):
    df = pd.DataFrame(columns = ['Image', 'Label'])
    for cl in class_labels:
        temp = []
        temp_df = pd.DataFrame(columns = ['Image', 'Label'])
        for i in range(len(list(get_image_files(path/cl)))):
            temp.append(str(list(get_image_files(path/cl))[i]).split('/')[-1])
        temp_df['Image'] = temp
        temp_df['Label'] = cl
        df = pd.concat([df, temp_df])
    return df

def _create_table(image_files, image_labels, dataset_name):
    "Create a table with the dataset"
    table = wandb.Table(columns=["File_Name", "Images", "Dataset", "Labels"])
    
    for i, image_file in progress_bar(enumerate(image_files), total=len(image_files)):
        image = Image.open(image_file)
        class_in_image = image_labels[image_labels['Image'] == str(image_file).split('/')[-1]]['Label'].values[0]
        table.add_data(
            image_file.stem,
            wandb.Image(image),
            dataset_name,
            class_in_image
        )
        
    return table

def get_predictions(learner, mode='train', max_n=None):
    """Return the samples = (x,y) and outputs (model predictions decoded), and predictions (raw preds)"""
    if mode.lower() == 'train':
        idx = 0
    elif mode.lower() == 'eval':
        idx = 1
    inputs, predictions, targets, outputs = learner.get_preds(
        ds_idx=idx, with_input=True, with_decoded=True
    )
    x, y, samples, outputs = learner.dls.valid.show_results(
        tuplify(inputs) + tuplify(targets), outputs, show=False, max_n=max_n
    )
    return samples, outputs, predictions

def metric_per_class(inp, targ):
    "Compute metric per class"
    metric_scores = []
    for c in range(inp.shape[0]):
        dec_preds = inp.argmax(dim=0)
        p = torch.where(dec_preds == c, 1, 0)
        t = torch.where(targ == c, 1, 0)
        c_inter = (p * t).float().sum().item()
        c_union = (p + t).float().sum().item()
        iou_scores.append(c_inter / (c_union - c_inter) if c_union > 0 else np.nan)
    return metric_scores

def create_row(sample, pred_label, prediction, class_labels):
    """"A simple function to create a row of (img, target, prediction, and scores...)"""
    (image, label) = sample
    # compute metrics
    # iou_scores = iou_per_class(prediction, label)
    image = image.permute(1, 2, 0)
    row =[wandb.Image(image), label, str(pred_label[0])]+[prob for prob in prediction.numpy()]
    return row

def create_prob_table(samples, outputs, predictions, class_labels):
    "Creates a wandb table with predictions and targets side by side"

    def _to_str(l):
        return [f'{str(x)} Probability' for x in l]
    
    items = list(zip(samples, outputs, predictions))
    
    table = wandb.Table(columns=["Image", "Label", "Prediction"]+ _to_str(class_labels.values()))
    # we create one row per sample
    for item in progress_bar(items):
        table.add_data(*create_row(*item, class_labels=class_labels))
    
    return table