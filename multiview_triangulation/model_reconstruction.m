%This script allows to reconstruct the satellite 3D model through multiview
%triangulation, exploiting the information stored in the ".mat" file
%returned by keypoints_selection.m 

%The output is a matrix xyzPoints containing x, y, and z coordinates [m] of
%the choosen keypoints in the satellite body frame. For more info, refer to
%https://it.mathworks.com/help/vision/ref/triangulatemultiview.html

%Load data
log_file=dir(fullfile('*.mat'));
load(log_file.name)

%Define camera parameters
pixel_length=5.86e-6; %m
cx=1920/2; %px
cy=1200/2; %px
fx=0.0176/pixel_length; %px
fy=0.0176/pixel_length; %px
focalLength=[fx,fy];
principalPoint=[cx,cy];
imageSize=[1200, 1920];
intrinsics=cameraIntrinsics(focalLength,principalPoint,imageSize);

%Number of keypoints to track across images:
num_keypoints_to_track = max([image_points.feature]);

ViewId=uint32((1:length(poses))');

cameraPoses=table(ViewId);

for j=1:length(poses)
    cameraPoses.AbsolutePose(j)=rigid3d(poses(j).rotm,(poses(j).rotm'*(-poses(j).pos))');
end

tracks(1:num_keypoints_to_track)=pointTrack;
for i=1:length(tracks)
    for k=1:length(image_points)
        w=find(image_points(k).feature==i);
        
        if ~isempty(w)
            tracks(i).ViewIds=[tracks(i).ViewIds uint32(k)];
            tracks(i).Points=[tracks(i).Points; image_points(k).x(w) image_points(k).y(w)];
        end
    end
end
[xyzPoints,errors] = triangulateMultiview(tracks,cameraPoses,intrinsics);    