%cleaning stuff up

clear all;
close all;
clc

%Class indexes
CLASS_SUNFLOWER = 1;
CLASS_IRIS = 2;
CLASS_ROSE = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% IMAGE GATHERING AND CLASSIFICATION
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentDir = pwd()
imagesDir = strcat(currentDir,"/images/");

images = readdir(imagesDir);
imageCount = length(images);

disp(strcat("Processing",num2str(imageCount)," images"))

classification = [];
for i = 1:imageCount
  image = images{i};
  if(strcmp(image,".") > 0 || strcmp(image,"..") > 0)
    continue;
  endif;
  
  %Image class must be the first name before the _ char
  imageNameSplitted = strsplit(image,"_");
  imageClassName = imageNameSplitted([:,1]);
  
  %Defining the current image class
  imgClassIndex = -1;
  if(strcmp(imageClassName,"sunflower") > 0)
    imgClassIndex = CLASS_SUNFLOWER;
  endif;
  if(strcmp(imageClassName,"rose") > 0)
    imgClassIndex = CLASS_ROSE;
  endif;
  if(strcmp(imageClassName,"iris") > 0)
    imgClassIndex = CLASS_IRIS;
  endif;
  
  %Ignore images which does not belong to any class
  if(imgClassIndex < 0)
    continue;
  endif;
  
  imageLocation = strcat(imagesDir,image);
  I = imread(imageLocation);
  
  %Get image's RGB colors average
  colorAvg = mean(reshape(I, size(I,1) * size(I,2), size(I,3)));
  
  result = [colorAvg, imgClassIndex];
  
  if(size(classification) == 0)
    classification = [result];
  else
    classification = [classification; result];
  endif;
  
endfor;

disp(strcat("Processed",num2str(imageCount)," images"))



