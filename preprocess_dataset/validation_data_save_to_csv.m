%Save validation to csv file
%CSV file Structure:
%Image xmin ymin xmax ymax x_A y_A x_B y_B etc

clear
clc

%% EDIT
output_filename = 'validation.csv';

% Edit the code below based on the number of points you are working with  

%% ONLY SYNTHETIC VALIDATION DATASET

text_read = fileread('synthetic_validation_enriched.json');
image_data=jsondecode(text_read);

all_data_table=table();
for i=1:length(image_data)
    NAME_ID=convertCharsToStrings(image_data(i).filename);
    class=image_data(i).class;
    
    q_1=image_data(i).quaternions(1);
    q_2=image_data(i).quaternions(2);
    q_3=image_data(i).quaternions(3);
    q_4=image_data(i).quaternions(4);

    Xc=image_data(i).position(1);
    Yc=image_data(i).position(2);
    Zc=image_data(i).position(3);

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

    all_data_table=[all_data_table; table(NAME_ID, q_1, q_2, q_3, q_4, Xc, Yc, Zc, xmin, ymin, xmax, ymax, X_A, Y_A, X_B, Y_B, X_C, Y_C, X_D, Y_D, X_E, Y_E, X_F, Y_F, X_G, Y_G, X_H, Y_H, X_I, Y_I, X_L, Y_L, X_M, Y_M, class)];
end
writetable(all_data_table,output_filename)