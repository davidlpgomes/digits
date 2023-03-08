# digits
Digits image feature extraction and classification.

The feature extraction method used for the digit images is based on the paper "Diagonal based feature extraction for handwritten character recognition system using neural network" ([Pradeep et al., 2011](https://ieeexplore.ieee.org/abstract/document/5941921)). The authors propose the use of counting black pixels on diagonal of  normalized images. The normalization consists of transforming the image into grayscale and resizing it to 90x60 resolution

The classification method used is the sklearn's KNN, using 3 neighbors and the euclidean distance. The achieved accuracy was 94.72%.
