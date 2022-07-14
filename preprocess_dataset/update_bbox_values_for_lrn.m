% Read json enriched files and update 
% bounding boxes for sunlamp and lightbox
% images with SDN predictions
clear
%% Edit
old_json_file = 'sunlamp_enriched.json';
SDN_json_output = 'SDN_inference.json';

%Output filename
new_json_file = 'sunlamp_enriched_lrn.json';

%%

old_data = jsondecode(fileread(old_json_file));
SDN_inference_data = jsondecode(fileread(SDN_json_output));

for i=1:length(SDN_inference_data)
    y_min_nn=SDN_inference_data(i).bbox_coords(1);
    x_min_nn=SDN_inference_data(i).bbox_coords(2);
    y_max_nn=SDN_inference_data(i).bbox_coords(3);
    x_max_nn=SDN_inference_data(i).bbox_coords(4);
    
    bbox_coords = [x_min_nn, y_min_nn, x_max_nn, y_max_nn];
    
    for j=1:length(old_data)
        if strcmp(old_data(j).filename,SDN_inference_data(i).image)
            old_data(j).bbox_coords=bbox_coords;
            break
        end
    end
    
end

%save data
image_data_export =jsonencode(old_data,'PrettyPrint',true);
fileID = fopen(new_json_file,'w');
fprintf(fileID,image_data_export); 
fclose(fileID);