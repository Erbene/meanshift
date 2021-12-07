from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import image, pyplot
import os, time, imageio
from sklearn.decomposition import NMF


class SimpleMeanShift:
    def __init__(self, window_size, cluster_tolerance):
        self.window_size = window_size
        self.tolerance = cluster_tolerance

    def shift(self, points):
        original_data = points
        
        result = np.empty(original_data.shape)

        for idx, point in enumerate(points):
            mean_shift = self.tolerance + 1
            while mean_shift > self.tolerance:
                new_mean = self.get_near_points_mean(original_data, point)
                mean_shift = np.linalg.norm(point - new_mean)
                point = new_mean
            
            result[idx] = point
        
        return result
    
    def get_near_points_mean(self, data, center):
        upper_boundary = center + self.window_size / 2
        lower_boundary = center - self.window_size / 2
        
        condition = np.all(np.logical_and(np.greater_equal(data, lower_boundary), np.less_equal(data, upper_boundary)), axis=1)
        near = data[condition]
        
        return near.mean(axis=0)
    
    def cluster(self, points):
        shifted = self.shift(points)
        rounded = np.around(shifted, 2)
        unique_vectors = np.unique(rounded, axis=0)
        idx = 1
        idx_map = {}
        for v in unique_vectors:
            k = repr(v)
            idx_map[k] = idx
            idx += 1
        
        r = []

        for i, v in enumerate(rounded):
            r.append(idx_map[repr(v)])
        
        return np.array(r)



# Calculate local histograms using numpy integral histograms
# For a single band image
# Algorithm Modified from: https://medium.com/@jiangye07/fast-local-histogram-computation-using-numpy-array-operations-d96eda02d3c
def calculate_local_histogram(mtx, ws, BinN=11):
    h, w = mtx.shape
    
    # quantize values at each pixel into bin ID
    b_max = np.max(mtx[:, :])
    b_min = np.min(mtx[:, :])
    assert b_max != b_min, "Image has only one value!"

    b_interval = (b_max - b_min) * 1. / BinN
    mtx[:, :] = np.floor((mtx[:, :] - b_min) / b_interval)

    mtx[mtx >= BinN] = BinN - 1
    mtx = np.int32(mtx)

    # convert matrix to one hot encoding (https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f)
    # one hot encoding converts h x w matrix into a h*w x BinN matrix 
    one_hot_pix_b = np.zeros((h*w, BinN), dtype=np.int32)
    one_hot_pix_b[np.arange(h*w), mtx[:, :].flatten()] = 1
    one_hot_pix = one_hot_pix_b.reshape((h, w, BinN))
    
    integral_hist = one_hot_pix
    np.cumsum(one_hot_pix, axis=1, out=integral_hist)
    np.cumsum(integral_hist, axis=0, out=integral_hist)

    padding_l = np.zeros((h, ws + 1, BinN), dtype=np.int32)
    padding_r = np.tile(integral_hist[:, -1:, :], (1, ws, 1))

    integral_hist_pad_tmp = np.concatenate([padding_l, integral_hist, padding_r], axis=1)

    padding_t = np.zeros((ws + 1, integral_hist_pad_tmp.shape[1], BinN), dtype=np.int32)
    padding_b = np.tile(integral_hist_pad_tmp[-1:, :, :], (ws, 1, 1))

    integral_hist_pad = np.concatenate([padding_t, integral_hist_pad_tmp, padding_b], axis=0)

    integral_hist_1 = integral_hist_pad[ws + 1 + ws:, ws + 1 + ws:, :]
    integral_hist_2 = integral_hist_pad[:-ws - ws - 1, :-ws - ws - 1, :]
    integral_hist_3 = integral_hist_pad[ws + 1 + ws:, :-ws - ws -1, :]
    integral_hist_4 = integral_hist_pad[:-ws - ws - 1, ws + 1 + ws:, :]

    sh_mtx = integral_hist_1 + integral_hist_2 - integral_hist_3 - integral_hist_4

    histsum = np.sum(sh_mtx, axis=-1, keepdims=True) * 1. 

    sh_mtx = np.float32(sh_mtx) / np.float32(histsum)

    return sh_mtx

def find_predominant_feature(w):
    def predominant_feature(arr):
        return np.argmax(arr, axis=0)
    return np.apply_along_axis(predominant_feature, axis=1, arr=w)

def create_overlay(path, texture, source, txt):
    foreground = Image.open(texture).convert('RGBA')
    background = Image.open(source).convert('RGBA')
    foreground.putalpha(128)

    result = Image.blend(foreground, background, 0.7)
    if txt:
        draw = ImageDraw.Draw(result)
        font = ImageFont.truetype("arial.ttf", 16)
        draw.text((0, 0), txt, font=font)

    result.save(path)

def create_side_by_side_images(path, arr, txt):
    images = [Image.open(x).convert('RGB') for x in arr]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    if txt:
        draw = ImageDraw.Draw(new_im)
        font = ImageFont.truetype("arial.ttf", 16)
        draw.text((0, 0), txt, font=font)

    new_im.save(path)
    return new_im


def NMF_model(samples, rfeatures):
    model = NMF(n_components=rfeatures, init='random', random_state=0, max_iter=1600)
    
    W = model.fit_transform(samples)
    H = model.components_
    
    return (W, H)

def get_most_significant_features(model):
    (W, H) = model

    feature_classification = find_predominant_feature(W)[:]

    return feature_classification


def main():
    
    image_path = 'brain.tiff'
    number_of_hist_bins = 5
    image_name = image_path.split('.')[0]
    meanshift_bandwidth = 0.1
    cluster_tolerance = 0.01

    texture_path = f'{image_name}_meanshift_texture.png'
    texture_overlay_path = f'{image_name}_meanshift_overlay.png'
    
    img = Image.open(image_path).convert('L')
    
    # convert image to numpy array
    image_mtx = np.asarray(img)
    
    # Start timer 
    start_time = time.time()
    
    # Create Local Histogram Matrix
    localhist_mtx = calculate_local_histogram(image_mtx, 10, number_of_hist_bins)
    
    h, w, sh_dim = localhist_mtx.shape
    
    pix_samples = (localhist_mtx.reshape((h * w, sh_dim)))
    
    print("Calculated local histograms matrix: %s seconds" % (time.time() - start_time))
    
    start_time = time.time()
    
    # Calculate mean shift labels for each pixel
    model = SimpleMeanShift(meanshift_bandwidth, cluster_tolerance).cluster(pix_samples)
    
    print("Ran MeanShift: %s seconds" % (time.time() - start_time))
    
    # Create texture image and texture overlay on original image
    pyplot.imsave(texture_path, model.reshape(h,w), cmap='ocean')
    create_overlay(texture_overlay_path, texture_path, image_path, "Mean-shift")
    
    # Generate NMF model with 4 representative features
    start_time = time.time()
    nmf_model = NMF_model(pix_samples, 4)
    nmf_texture = get_most_significant_features(nmf_model)
    print("Generated NMF overlay: %s seconds" % (time.time() - start_time))

    nmf_texture_path = f'{image_name}_nmf_texture.png'
    nmf_overlay_path = f'{image_name}_nmf_overlay.png'
    
    # Create texture image and texture overlay on original image
    pyplot.imsave(nmf_texture_path, nmf_texture.reshape(h,w), cmap='ocean')
    create_overlay(nmf_overlay_path, nmf_texture_path, image_path, f"NMF:")
    
    # Generate side by side image
    side_by_side = f'side_by_side_{image_name}.png'
    create_side_by_side_images(side_by_side, [image_path, texture_overlay_path, nmf_overlay_path] ,f"Original Image")

if __name__ == "__main__":
    main()
