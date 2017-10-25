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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Training
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Splitting classificatory values
R = classification(:,1);
G = classification(:,2);
B = classification(:,3);
CLASS = classification(:,4);

%Normalizing classificatory values
R_N = (R - mean(R)/std(R));
G_N = (G - mean(G)/std(G));
B_N = (R - mean(B)/std(B));

data = [R_N G_N B_N];

target = zeros(length(R),3);
target(CLASS_SUNFLOWER, :) = repmat([1 0 0], sum(CLASS_SUNFLOWER),1);
target(CLASS_ROSE, :) = repmat([0 1 0], sum(CLASS_ROSE),1);
target(CLASS_IRIS, :) = repmat([0 0 1], sum(CLASS_IRIS),1);

eta = 1;
nHidden = 4;
type= "sigmoid";
epochCout = 1000;
minErr = 0.001;
[NET, errors] = backpropagation(data,target,nHidden,type,eta,epochCount,minErr);
plot(errVec');

% Display the graph
figure
hold_on
scatter(plen(CLASS_SUNFLOWER),pwid(CLASS_SUNFLOWER),8,'k',"filled")
scatter(plen(CLASS_ROSE),pwid(CLASS_ROSE),8,'r',"filled")
scatter(plen(CLASS_IRIS),pwid(CLASS_IRIS),8,'g',"filled")
title("RGB Variation")


