# Dataset Pre-Processing

This folder contains all the MATLAB and Python codes we adopted to pre-process the SPEED+ [1] dataset.

<b>Basic Processing</b>
1.	json_enrich.m: retrieve train and validation labels. Add keypoints coordinates and bounding box vertices to the json files containing pose information coming with SPEED+.
2.	train_data_save_to_csv.m and validation_data_save_to_csv.m: convert the enriched json files to csv for TFRecords conversion.
3.	TFRecords_from_csv.ipynb: convert csv data and images to TFRecords format.

More information and instructions are provided in the comments.

To train the NNs on Colab TPUs, the resulting TFRecords file shall be uploaded on a [Google Cloud Bucket](https://cloud.google.com/storage/docs/creating-buckets). Please note that this is a paid service. The authors do not have any relationship with the service provider.

<b>Adversarial Approach: Train a LRN</b>

In the framework of adversarial domain adaptation, in order to achieve the same zoom level on real pictures while training a LRN, one should:
-	Perform inference on sunlamp and lightbox images with a trained SDN
-	Add the coordinates of the bounding boxes for sunlamp and lightbox images to the enriched json files (we provide update_bbox_values_for_lrn.m for that).
-	Re-process the training data through steps 2-3 (check line 61 in train_data_save_to_csv.m  for additional instructions).


<b>Further Image Processing </b>
-	offline_sunflare_addition.ipynb and offline_sunflare_and_style_randomization.ipynb: process images offline by adding sunflare [2] only (first notebook) and sunflare or style randomization [3] (second notebook).

# References

[1] Park, T. H., Märtens, M., Lecuyer, G., Izzo, D., D'Amico, S. (2021). Next Generation Spacecraft Pose Estimation Dataset (SPEED+). Stanford Digital Repository. Available at https://purl.stanford.edu/wv398fc4383. https://doi.org/10.25740/wv398fc4383

[2] Albumentations, RandomSunflare, https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSunFlare

[3] Jackson, P.T., Atapour-Abarghouei, A., Bonner, S., Breckon, T.P., Obara, B.: Style Augmentation: Data Augmentation via Style Randomization. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2019. 

