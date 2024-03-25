import numpy as np
from PIL import Image
from vgg.vgg_feature_extractor import VGGFeatureExtractor
from flask import Flask, request
from pathlib import Path
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


app = Flask(__name__)

fe = VGGFeatureExtractor()


@app.route('/<string:category>', methods=['POST'])
def search(category):
    feature_dir = f"./vgg/static/feature/{category}"
    img_dir = f"./vgg/static/img/{category}"

    features = []
    img_paths = []
    for feature_path in Path(feature_dir).glob("*.npy"):
        features.append(np.load(feature_path))
        img_paths.append(Path(img_dir) / (feature_path.stem + ".png"))
    features = np.array(features)

    if request.method == "POST":
        file = request.files['query_img']
        img = Image.open(file.stream)

        # run search
        query = fe.extract_features(img)
        dists = np.linalg.norm(features - query, axis=1)
        ids = np.argsort(dists)[:10]
        scores = [(float(dists[id]), str(img_paths[id])) for id in ids if dists[id] <= 1]
        data = {"results": [{"score": float(d), "path": os.path.basename(str(p))} for d, p in scores]}
        return data


if __name__ == '__main__':
    app.run('0.0.0.0')
