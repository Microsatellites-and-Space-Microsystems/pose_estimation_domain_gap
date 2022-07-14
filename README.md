# Investigating Vision Transformers for Bridging Domain Gap in Satellite Pose Estimation

Autonomous onboard estimation of the pose from a single monocular image is a key technology for all space missions requiring an active chaser to service an uncooperative target. In recent years, many algorithms have been proposed who leverages neural networks to infer the three dimensional pose of a known satellite from a two dimensional image. However, their adoption is still facing two main limitations: 

1.	Poor robustness against domain shifts: as it is not feasible to collect large collections of satellite pictures in orbit to train the networks, they are usually trained on synthetic images only and their accuracy does not transfers to real pictures.

2.	Poor accuracy-latency trade off: current methods are often too computational expensive for real time onboard execution or not enough accurate especially under domain shifts.

A recent effort to push the research forward is represented by the 2021 edition of the [Satellite Pose Estimation Competition (SPEC)](https://kelvins.esa.int/pose-estimation-2021/challenge/) hosted by the European Space Agency. The contest was based on SPEED+ [1], the first dataset focusing on domain gap for space applications, including synthetic images for training and two sets of hardware-in-the-loop pictures of a spacecraft mockup for testing. 

In this repository we provide the codes related to the methods we developed as part of our participation to SPEC 2021 that we also discussed in the paper "Investigating Vision Transformers for Bridging Domain Gap in Satellite Pose Estimation". The work will be presented at the 1st International Workshop on "The use of Artificial Intelligence for Space Applications” co-located with the 2nd International Conference of Applied Intelligence and Informatics, 1-3 September 2022, Reggio Calabria, Italy.

More in details, we propose two alternative solutions to approach the problem.

The first one [(adversarial_domain_adaptation)](https://github.com/Microsatellites-and-Space-Microsystems/pose_estimation_domain_gap/tree/main/adversarial_domain_adaptation) consists of a classical three-stage pose estimation pipeline (detection, landmarks regression, PnP solver) which however leverages Swin Transformers in place of standard Convolutional Neural Networks (CNN) and adversarial domain adaptation to promote the emergence of domain invariant features. Our algorithm reached the fourth and fifth places on the sunlamp and lightbox leaderboards of SPEC 2021 respectively. 

The second method [(cnn_vit)](https://github.com/Microsatellites-and-Space-Microsystems/pose_estimation_domain_gap/tree/main/cnn_vit) consists of a lighter dual stage pose estimation pipeline (a single neural network and the PnP solver), built on top of an hybrid CNN+Vision Transformer (ViT) model. In this case access to test images is not required: we exploit data augmentations to promote domain generalization.

# References

[1] Park, T. H., Märtens, M., Lecuyer, G., Izzo, D., D'Amico, S. (2021). Next Generation Spacecraft Pose Estimation Dataset (SPEED+). Stanford Digital Repository. Available at https://purl.stanford.edu/wv398fc4383. https://doi.org/10.25740/wv398fc4383
