function keypoints_selection(selected_images,filename)
%KEYPOINTS_SELECTION Manually select pre-defined keypoints on a set of images
%   [image_points, poses]=keypoints_selection(selected_images) allows to
%   retrieve the coordinates of some pre-defined keypoints by manually
%   selecting them on the images. Images must be stored in the
%   "selected_images" folder. The function returns a ".mat" file containing
%   2 structures (each line of the structures corresponds to an image):
%
%       "image_points":
%           - feature:  row vector containing the numeric identifiers of
%                       the keypoints selected on the corresponding image
%           - x:        row vector containing the x coordinates of the
%                       keypoints (image frame), in the same order they
%                       appear in "feature"
%           - y:        row vector containing the y coordinates of the
%                       keypoints (image frame), in the same order they
%                       appear in "feature"
%
%       "poses":
%           - rotm:     3x3 matrix associated to the satellite attitude
%           - pos:      position vector of the satellite in the camera
%                       frame [m]
%
%   Before starting, decide which keypoints to use and assign them a unique
%   numeric identifier (starting from 1 with step 1).
%
%       "selected_images":  vector containing image numbers
%       "filename":         name of the output ".m" file

%Load all training poses
text = fileread('train.json');
train_data=jsondecode(text);

%Store all the poses in a temporary structure
temp_poses=struct('name','','rotm','','rel_pos','');
for i=1:length(train_data)
    temp_poses(i).name = train_data(i).filename;
    temp_poses(i).rotm = quat2rotm(train_data(i).q_vbs2tango');
    temp_poses(i).rel_pos=train_data(i).r_Vo2To_vbs_true;
end

%Since some images have been removed from the syntetic dataset to create 
%the test set, we need to extract the poses corresponding to the
%selected images from train.json
poses=struct('rotm','','pos','');
for z=1:length(selected_images)
    image_nr=num2str(selected_images(z));
    digits=numel(image_nr);
    for w=1:length(temp_poses)
        if strcmp(temp_poses(w).name(10-digits:9),image_nr)
            poses(z).rotm=temp_poses(w).rotm;
            poses(z).pos=temp_poses(w).rel_pos;
            break
        end
    end
end

%Directory to images selected for 3D model reconstruction
imageDir = fullfile('selected_images');

%Manage a collection of images
images = imageDatastore(imageDir);
 
image_points=struct('feature',[],'x',[],'y',[]);

for i=1:length(selected_images)
    k=0; %feature vector index
    img = readimage(images,i);
    first_time=1;
    y='y';
    while y=='y'
        if first_time==1
            fprintf('Let''s move to the next image \n\n')
            first_time=0;
        else
            prompt='Is there any other feature to detect in the image? y/n: ';
            y=input(prompt,'s');
            if y=='n'
                break
            end
        end

        k=k+1;
        imshow(img);
        prompt='Please enter the feature Id: ';
        get_input=input(prompt);
        image_points(i).feature(k)=get_input;
        [image_points(i).x(k),image_points(i).y(k)] = getpts;
        close

    end
end
save(filename,'image_points','poses')
end