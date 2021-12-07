# Mean-shift Algorithm for image segmentation

## Objective

The objective of this work is perform image segmentation using [mean-shift algorithm](https://en.wikipedia.org/wiki/Mean_shift), and plot images showing pixel cluster regions on brain MRI. It also uses [NMF](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) clustering properties to generate separate image segmentations using the same dataset. The mean-shift algorithm implementation is very simplified. The considerations used are discussed in a section ahead. 

## Dataset
The brain MRI used for testing are pulled from [Kaggle's brain tumor detection challenge dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).

<p style="text-align: center;">
  <img src="https://user-images.githubusercontent.com/8680272/145035738-3ce943d7-47a0-4c2f-be78-bc2edc7d0057.png" />
  <br>
  <em>Fig.1 - Sample Image 1</em>
</p>
<p style="text-align: center;">
  <img src="https://user-images.githubusercontent.com/8680272/145035762-57611cf8-9a72-468d-8fb1-b5eaa1781256.png" />
  <br>
  <em>Fig.2 - Sample Image 2</em>
</p>


## Brain MRI image features
To generate sample for each image, we consider each pixel local histograms as a sample. To construct the sample input matrix, the [calculate local histograms using integral histograms technique](https://medium.com/@jiangye07/fast-local-histogram-computation-using-numpy-array-operations-d96eda02d3c) is used. 
The algorithm was adapted from the aforementioned blog to use a single-band image.

## Mean-shift algorithm implementation considerations
The mean-shift algorithm implementation considers the following to simplify the approach
- Considers a rectangular (square shaped) window for calculating density
- Considers a flat kernel, e.g. each point is shifted using the unweighted mean of its neighbors within the window

The following parameters should be configured, in the `main()` function when running the script:
```
image_path = 'Y2.jpg' # the path of the image to be segmented
number_of_hist_bins = 5 # the number of bins to be used when calculating local histograms
meanshift_bandwidth = 0.1 # the window square side size 
cluster_tolerance = 0.01 # cluster tolerance refers is used as a stop measure when calculating mean shifts. If the distance shifter is less than tolerance, the algorithm stops.
```

## Mean-shift image segmentation results

When running the algorithm for sample image 1, the following results are obtained:

**Parameters used:**
```
image_path = 'Y2.jpg'
number_of_hist_bins = 5
meanshift_bandwidth = 0.1 
cluster_tolerance = 0.01
```
**Image segmentation results:**
<div >
  <div>
    <img src="https://user-images.githubusercontent.com/8680272/145040557-cf688463-fcb8-4219-a63d-36eb546cefc5.png" />
    <br>
    <em>Fig.3 - Sample Image 1 segmentation result</em>
  <div/>
  <div>
    <img src="https://user-images.githubusercontent.com/8680272/145041878-f312458e-e8ea-4f3d-9816-2f15468e2982.png" />
    <br>
    <em>Fig.4 - Sample Image 1 segmentation result as an overlay in original image</em>
  <div/>
</div>

For sample image 2, the following results are obtained:
**Parameters used:**
```
image_path = 'Y15.jpg'
number_of_hist_bins = 5
meanshift_bandwidth = 0.1 
cluster_tolerance = 0.01
```
**Image segmentation results:**
<div >
  <div>
    <img src="https://user-images.githubusercontent.com/8680272/145045236-73cdb6ae-32ec-41c8-9a20-914717daf328.png" />
    <br>
    <em>Fig.5 - Sample Image 2 segmentation result</em>
  <div/>
  <div>
    <img src="https://user-images.githubusercontent.com/8680272/145045378-88ac0535-0f54-419b-8258-cb3b27681049.png" />
    <br>
    <em>Fig.6 - Sample Image 2 segmentation result as an overlay in original image</em>
  <div/>
</div>




