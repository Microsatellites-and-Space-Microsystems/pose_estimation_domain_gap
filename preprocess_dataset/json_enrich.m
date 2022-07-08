% This script allows to create augmented json files with class labels,
% poses, keypoints coordinates, bounding boxes 
% If you are preparing the dataset for domain adversarial training, process
% also sunlamp and lighbox json files through this script.
clear
clc

%% EDIT
output_filename='synthetic_train_enriched.json';

% Original json to augment
original_json = jsondecode(fileread('speedplus\speedplus\synthetic\train.json'));

% Assign a class: synthetic = 1; lightbox = 2; sunlamp = 3;
dataset_class = 1;

% Camera parameters
camera = jsondecode(fileread('speedplus\speedplus\camera.json'));

% Satellite 3D model
load('Tango.mat')

%% We have pose information only for synthetic data

is_synthetic = false;
if dataset_class == 1
    is_synthetic = true;
end

%% Camera parameters

K=camera.cameraMatrix;

k1 = camera.distCoeffs(1);
k2 = camera.distCoeffs(2);
k3 = camera.distCoeffs(5);

p1 = camera.distCoeffs(3);
p2 = camera.distCoeffs(4);

%% Satellite 3D model

number_of_model_keypoints=length(xyzPoints);

Body_points=struct('Body_coord',[]);
for i=1:number_of_model_keypoints
    Body_points(i).Body_coord=xyzPoints(i,:);
end

image_data=struct('filename','','quaternions',[],'position',[],'bbox_coords',[],'kpts_coords',[],'class',[]);


%% Keypoints projection

for i=1:length(original_json)
    pose_data=original_json(i);
    image_data(i).class = dataset_class;
    image_data(i).filename=pose_data.filename;
    
    if is_synthetic
        
        image_data(i).quaternions=pose_data.q_vbs2tango_true;
        image_data(i).position=pose_data.r_Vo2To_vbs_true;

        for k=1:length(Body_points)
            % projection with dist coeffs:
            % https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
            projection=quat2rotm(pose_data.q_vbs2tango_true')*Body_points(k).Body_coord'+pose_data.r_Vo2To_vbs_true;
            xprime = projection(1)/projection(3);
            yprime = projection(2)/projection(3);

            r2 = xprime^2+yprime^2;
            xsecond = xprime*(1+k1*r2+k2*r2^2+k3*r2^3)+2*p1*xprime*yprime+p2*(r2+2*xprime^2);
            ysecond = yprime*(1+k1*r2+k2*r2^2+k3*r2^3)+2*p2*xprime*yprime+p1*(r2+2*yprime^2);

            ui = camera.cameraMatrix(1,1)*xsecond+camera.ccx;
            vi = camera.cameraMatrix(2,2)*ysecond+camera.ccy;
            image_data(i).kpts_coords=[image_data(i).kpts_coords; ui vi];
        end
        %bbox coordinates truncated at 0 and image dimensions
        xmin=max(min(image_data(i).kpts_coords(:,1)),0);
        ymin=max(min(image_data(i).kpts_coords(:,2)),0);
        xmax=min(max(image_data(i).kpts_coords(:,1)),camera.Nu);
        ymax=min(max(image_data(i).kpts_coords(:,2)),camera.Nv);
        image_data(i).bbox_coords=[xmin,ymin,xmax,ymax];

    end
    
    % Plot images for double-checking
    
%     pause
%     imshow(imread(strcat('speedplus\speedplus\synthetic\images\',pose_data.filename)));
%     hold on
%     plot(image_data(i).kpts_coords(:,1),image_data(i).kpts_coords(:,2),'.r')
%     rectangle('Position',[xmin, ymin, xmax-xmin, ymax-ymin],'EdgeColor','yellow')
%     plot(image_data(i).kpts_coords(:,1),image_data(i).kpts_coords(:,2),'.y')
%     hold off
end

%save data
image_data_export =jsonencode(image_data,'PrettyPrint',true);
fileID = fopen(output_filename,'w');
fprintf(fileID,image_data_export); 
fclose(fileID);