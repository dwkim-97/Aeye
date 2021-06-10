from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import argparse
import utils
import model
import os

if __name__ == "__main__":
    """
    1.get video frame
    2.YOLO detection
    3.Crop person image
    4.Clustering 
    """
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    path = '/path/'

    ## Get Frame from Video
    utils.get_frame_from_video(path, '1554.mp4')
    utils.write_train_list(path)

    ## YOLO & get Coordinate
    utils.excute_YOLO()
    coord = utils.get_person_coord(path)
    utils.crop_person_image(coord, path)

    ## Feature Extraction
    filenames = os.listdir(os.path.join(path, 'persons'))
    imgs_dict = utils.get_images(path, filenames, target_size=(299, 299))

    model = model.get_model()
    img_feature_vector = utils.feature_vectors(imgs_dict, model)

    ## croped person image Train
    # model.fit()
    images = list(img_feature_vector.values())
    
    ## Clustering
    # kmeans = KMeans(n_clusters=5, init='k-means++')
    # kmeans.fit(images)
    # y_kmeans = kmeans.predict(images)
    dbscan = DBSCAN()
    y_dbscan = dbscan.fit(images)

    file_names = list(imgs_dict.keys())
    groups = utils.make_group(file_names, dbscan)
    # for i in groups:
    #     print(i)
    print(groups)

