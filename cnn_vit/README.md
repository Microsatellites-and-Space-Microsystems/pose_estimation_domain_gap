# Hybrid CNN-ViT model

This folder contains the Colab Notebooks to train and test the CNN-ViT model.

In this context, the network is trained to directly regress a set of pre-defined landmarks on the images which are then fed into a PnP solver.

To use the code as it is, the dataset shall be pre-processed through the functions provided in [preprocess_dataset](https://github.com/Microsatellites-and-Space-Microsystems/pose_estimation_domain_gap/tree/main/preprocess_dataset). A 3D model of the Tango spacecraft is also required (check [multiview_triangulation](https://github.com/Microsatellites-and-Space-Microsystems/pose_estimation_domain_gap/tree/main/multiview_triangulation)).
