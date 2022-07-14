%Save train dataset information to csv file
%CSV file Structure:
%Image xmin ymin xmax ymax x_A y_A x_B y_B etc
clear
clc

%% 1. EDIT
output_filename = 'train.csv';

% Edit section 3 (see instructions below)
% Edit section 4 based on the number of satellite points you are working with  

%% 2. Load train dataset

text_read = fileread('synthetic_train_enriched.json');
image_data=jsondecode(text_read);


%% 3. Domain Agnostic / Domain Adversarial Training
% Domain agnostic: comment this section as you are not going to need test
% images for training.

% Domain adversarial: you shall also augment sunlamp and lightbox json files
% (json_enrich.m) and provide the filenames here.

text_read = fileread('sunlamp_enriched.json');
image_data=[image_data; jsondecode(text_read)];

text_read = fileread('lightbox_enriched.json');
image_data=[image_data; jsondecode(text_read)];

%% 4. Random Shuffling
permutation_indices = randperm(length(image_data));
image_data=image_data(permutation_indices);

%% 5. Convert to CSV

all_data_table=table();
for i=1:length(image_data)
    NAME_ID=convertCharsToStrings(image_data(i).filename);
    class=image_data(i).class;
    
    % if the class is not 1 (synthetic) we export zero values
    q_1=0; q_2=0; q_3=0; q_4=0; Xc=0; Yc=0; Zc=0;
    xmin=0; xmax=0; ymin=0; ymax=0;
    
    X_A=0; X_B=0; X_C=0; X_D=0; X_E=0; X_F=0; X_G=0; X_H=0; X_I=0; X_L=0; X_M=0;
    Y_A=0; Y_B=0; Y_C=0; Y_D=0; Y_E=0; Y_F=0; Y_G=0; Y_H=0; Y_I=0; Y_L=0; Y_M=0;
    
    if class == 1
        
        % Note that pose information are not required for NNs in our method
        % but might be useful for subsequent development
        q_1=image_data(i).quaternions(1);
        q_2=image_data(i).quaternions(2);
        q_3=image_data(i).quaternions(3);
        q_4=image_data(i).quaternions(4);

        Xc=image_data(i).position(1);
        Yc=image_data(i).position(2);
        Zc=image_data(i).position(3);
                
        %Note: if you are enriching lightbox and sunlamp json files with the 
        %bounding boxes regressed by the SDN, move the next 4 lines
        %out of the if statement 
        xmin=image_data(i).bbox_coords(1);
        ymin=image_data(i).bbox_coords(2);
        xmax=image_data(i).bbox_coords(3);
        ymax=image_data(i).bbox_coords(4);

        X_A=image_data(i).kpts_coords(1,1);
        Y_A=image_data(i).kpts_coords(1,2);
        X_B=image_data(i).kpts_coords(2,1);
        Y_B=image_data(i).kpts_coords(2,2);
        X_C=image_data(i).kpts_coords(3,1);
        Y_C=image_data(i).kpts_coords(3,2);
        X_D=image_data(i).kpts_coords(4,1);
        Y_D=image_data(i).kpts_coords(4,2);
        X_E=image_data(i).kpts_coords(5,1);
        Y_E=image_data(i).kpts_coords(5,2);
        X_F=image_data(i).kpts_coords(6,1);
        Y_F=image_data(i).kpts_coords(6,2);
        X_G=image_data(i).kpts_coords(7,1);
        Y_G=image_data(i).kpts_coords(7,2);
        X_H=image_data(i).kpts_coords(8,1);
        Y_H=image_data(i).kpts_coords(8,2);
        X_I=image_data(i).kpts_coords(9,1);
        Y_I=image_data(i).kpts_coords(9,2);
        X_L=image_data(i).kpts_coords(10,1);
        Y_L=image_data(i).kpts_coords(10,2);
        X_M=image_data(i).kpts_coords(11,1);
        Y_M=image_data(i).kpts_coords(11,2);

    end
    all_data_table=[all_data_table; table(NAME_ID, q_1, q_2, q_3, q_4, Xc, Yc, Zc, xmin, ymin, xmax, ymax, X_A, Y_A, X_B, Y_B, X_C, Y_C, X_D, Y_D, X_E, Y_E, X_F, Y_F, X_G, Y_G, X_H, Y_H, X_I, Y_I, X_L, Y_L, X_M, Y_M, class)];
end
writetable(all_data_table,output_filename)