%cleaning stuff up

clear all;
close all;
clc

%Pre processing images
imagesDir = "/Users/drechi/Documents/FIAP/IA/mlp/images/";
images = readdir(imagesDir);
imageCount = length(images);

disp(strcat("Processing",num2str(imageCount)," images"))

for i = 1:imageCount
  image = images{i};
  if(strcmp(image,".") > 0 || strcmp(image,"..") > 0)
    continue;
  endif;
   
  imageLocation = strcat(imagesDir,image);
  I = imread(imageLocation);
  
endfor;

disp(strcat("Processed",num2str(imageCount)," images"))

