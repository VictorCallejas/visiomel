from multiprocessing import Pool
from PIL import Image, ImageSequence
Image.MAX_IMAGE_PIXELS = None
import os

import boto3
import io
from botocore import UNSIGNED
from botocore.config import Config

import pandas as pd

from tqdm import tqdm

def download_image_resize(filename):

    object = bucket.Object('images/'+filename)
    img_stream = io.BytesIO()
    object.download_fileobj(img_stream)

    image = Image.open(img_stream)

    #n_frames = image.n_frames - 1

    it = ImageSequence.Iterator(image)
    page = it[3]
    page = page.resize((4096,4096))
    #while True:
        
        #page = it[n_frames]

        #if ((page.size[0] < 4096) or (page.size[1] < 4096)) and (n_frames > 0):
            #n_frames = n_frames - 1
            #continue

        #page = page.resize((4096,4096))
        #break
    
    page.save('data/images/' + filename)

        
train_labels = pd.read_csv('data/train_labels.csv')
train_metadata = pd.read_csv('data/train_metadata.csv')

train = train_metadata.merge(train_labels, on='filename', how='inner')

s3 = boto3.resource('s3',config=Config(signature_version=UNSIGNED))
bucket = s3.Bucket('drivendata-competition-visiomel-public-eu')

curr_files = set(os.listdir('data/images/'))

if __name__ == '__main__':

    #with Pool(processes=20) as pool:
    #with tqdm(total=train.shape[0]) as pbar:
            #for _ in pool.imap_unordered(download_image_resize, train.filename.tolist()):
                #pbar.update()
                
    for filename in tqdm(train.filename, total=train.shape[0]):
        if filename not in curr_files:
            download_image_resize(filename)
    #train.filename[f:], total=train.shape[0] - 370 - 230):