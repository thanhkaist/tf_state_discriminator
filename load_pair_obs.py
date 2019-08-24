import numpy as np
from PIL import Image
import time

def visualize_obs(obs):
    obs = unnorm_obs(obs)
    obs = Image.fromarray(obs, 'RGB')
    obs.show()


def unnorm_obs(obs):
    obs = obs * (255.0 / 2)
    obs = obs + (255.0 / 2)
    obs = np.uint8(obs)
    return obs


def load_data_from_npz(path='pair_obs.npz'):
    data = np.load(path)
    pair_obs_arr = data['pair_obs_arr']
    img1s = pair_obs_arr[:,0]
    img2s = pair_obs_arr[:,1]
    labels = data['labels']
    return img1s,img2s,labels


if __name__ == '__main__':
    np.random.seed(int(time.time()))
    print('Loading pair of observations')

    # Load img
    img1s, img2s, labels = load_data_from_npz()
    print(img1s.shape)
    print(img2s.shape)
    print(labels.shape)
    print(labels[0:100])
    exit(0)
	
    data = np.load('pair_obs.npz')
    pair_obs_arr = data['pair_obs_arr']
    labels = data['labels']
    print('Finish!!')
    len = pair_obs_arr.shape[0]
    print('No pair of observations: ', len)
    for i in range(1):
        idx = np.random.randint(0, len)
        visualize_obs(pair_obs_arr[idx][0])
        visualize_obs(pair_obs_arr[idx][1])
        print(labels[idx])
