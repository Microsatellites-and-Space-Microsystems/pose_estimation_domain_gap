# Adversarial Domain Adaptation with Swin Transformers

This folder contains the Colab Notebooks implementing our three-stage domain adversarial approach.

The three stages consists in: satellite detection, landmarks regression, and PnP solver.

<p align="center">
  <img src="../images/3_stage_pipeline.PNG" width="700" title="hover text">
</p>

Both detection and regression networks are trained through adversial domain adaptation adopting the scheme illustrated below.

<p align="center">
  <img src="../images/adversarial_training.png" width="700" title="hover text">
</p>

To use the code as it is, the dataset shall be pre-processed through the functions provided in [preprocess_dataset](https://github.com/Microsatellites-and-Space-Microsystems/pose_estimation_domain_gap/tree/main/preprocess_dataset). A 3D model of the Tango spacecraft is also required (check [multiview_triangulation](https://github.com/Microsatellites-and-Space-Microsystems/pose_estimation_domain_gap/tree/main/multiview_triangulation)).

