import cv2
from pprint import pprint
import copy
import os
import numpy as np
from keras.models import load_model, Model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib.image as mpimg
import pickle
from tqdm import tqdm
import json
from pprint import pprint
from sklearn.decomposition import PCA
from absl import app
from absl import flags
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from scipy import spatial





FLAGS = flags.FLAGS
flags.DEFINE_string("target_image", None, "target image.")
# Required flag.
# 1.target vidoe to execute yolo
# 2.number of frame to cut default 10
flags.mark_flag_as_required("target_image")




#Set path as your own directory
path = "/Users/suminbae/PycharmProjects/tf_cv2/aeye"
#p_path = "/Users/suminbae/PycharmProjects/tf_cv2/aeye/persons"
p_path="/Users/suminbae/PycharmProjects/tf_cv2/aeye/test_data/persons"
d_path = "/Users/suminbae/PycharmProjects/tf_cv2/aeye/debug_persons"
save_path="/Users/suminbae/PycharmProjects/tf_cv2/aeye/test_data/matching_result"

# change the working directory to the path where the images are located
#path 설정을 잘해야 에러 안남
os.chdir(p_path)


# this list holds all the image filename
persons = []



# 저장해준다 filename들
def save_file_name(p_path):
    # creates a ScandirIterator aliased as files
    with os.scandir(p_path) as files:
        # loops through each file in the directory
        for file in files:
            if file.name.endswith('.jpg'):
                # adds only the image files to the flowers list
                persons.append(file.name)

def get_model():
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


#input: person file name directory
def extract_cropped_features(persons,model):
    data = {}
    p = r"CHANGE TO A LOCATION TO SAVE FEATURE VECTORS"

    # lop through each image in the dataset
    for person in tqdm(persons):
        # try to extract the features and update the dictionary
        try:
            feat = extract_features(person,model)
            data[person] = feat
        # if something fails, save the extracted features as a pickle file (optional)
        except:
            with open(p, 'wb') as file:
                pickle.dump(data, file)
    return data



def reshape_feature_vector(img_feature_vector):
    keys_values = img_feature_vector.items()
    new_d = {key: value.reshape(4096) for key, value in keys_values}
    pure_values =  list(new_d.values())
    return new_d, pure_values


def plot_k_value(img_feature_vector,images):
    plt.figure(figsize=(10, 5))
    fns = list(img_feature_vector.keys())
    sum_of_squared_distances = []
    K = range(1, 10)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(images)
        sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def cluster_data(persons,n_clusters,images):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    kmeans.fit(images)
    y_kmeans = kmeans.predict(images)
    kmeans_labels_int = list(kmeans.labels_)
    kmeans_labels_int = list(map(int, kmeans_labels_int))

    groups = {}
    for file, cluster in zip(persons, kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    keys_values = groups.items()
    groups = {int(key): value for key, value in keys_values}

    return groups

def save_clustering_result(groups,path,result_name):
    os.chdir(path)

    with open(result_name, 'w') as f:
        json.dump(groups, f)

    print("Result Saved as json file")


# 코사인 유사도 함수
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def cos_sim_sci(A, B):
    return 1 - spatial.distance.cosine(A, B)

def cluster_average(groups,data):
    cluster_mean = {}
    for i in groups.keys():
        feature_lst = []
        for val in groups[i]:
            feature_lst.append(data[val])

        feature_lst_mean = np.array(feature_lst)
        feature_lst_mean = feature_lst_mean.mean(axis=0)
        #print(feature_lst_mean.shape)
        cluster_mean[i] = feature_lst_mean
    return cluster_mean


def compare_sim(cluster_mean,target_feature):
    cluster_compare_result = {}
    for key, val in cluster_mean.items():
        sim = cos_sim_sci(val, target_feature)
        print(f"Compare to this cluster{key} sim with target: ", sim)
        cluster_compare_result[key] = sim

    return cluster_compare_result


def recommend_matching_result(cluster_compare_result,groups):
    max_result = max(cluster_compare_result, key=cluster_compare_result.get)
    matching_person = p_path + "/" + groups[max_result][0]
    #matching_person = mpimg.imread(matching_person)
    matching_person = cv2.imread(matching_person)
    cv2.imshow("Is this the person looking for?",matching_person)
    os.chdir(save_path)
    cv2.imwrite("matching_person.jpg",matching_person)





def main(argv):
    del argv
    # 1.success
    # for real process change d_path -> p_path
    save_file_name(p_path)
    # 2. call model(success)
    model = get_model()
    # 3. persons feature
    img_feature_vector = extract_cropped_features(persons,model)
    data = copy.deepcopy(img_feature_vector)

    # 4. reshaping image (1,4096) ->(4096,)
    img_feature_vector, images = reshape_feature_vector(img_feature_vector)

    # plot kvalue sucess doens't work on shell process
    #plot_k_value(img_feature_vector, images)

    # set k based on plotting data
    k = 5

    groups = cluster_data(persons, k, images)

    cluster_avg = cluster_average(groups,data)

    #target_img = cv2.imread('test.jpg')
    #target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    #os.chdir(path)

    target_feature = extract_features(FLAGS.target_image, model)
    compare_result = compare_sim(cluster_avg,target_feature)
    print(compare_result)


    #pops up best recommendation
    recommend_matching_result(compare_result,groups)



    #yolo_on_video(FLAGS.video,FLAGS.frame_rate)




if __name__=="__main__":
    #10/27
    app.run(main)








