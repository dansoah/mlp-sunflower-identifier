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

idx_sf = CLASS == CLASS_SUNFLOWER;
idx_rs = CLASS == CLASS_ROSE;
idx_ir = CLASS == CLASS_IRIS;

%Normalizing classificatory values
R_N = (R - mean(R)/std(R));
G_N = (G - mean(G)/std(G));
B_N = (R - mean(B)/std(B));

data = [R_N G_N];
sum(CLASS_SUNFLOWER)
target = zeros(length(R),3);
target(idx_sf, :) = repmat([1 0 0], sum(idx_sf),1);
target(idx_rs, :) = repmat([0 1 0], sum(idx_rs),1);
target(idx_ir, :) = repmat([0 0 1], sum(idx_ir),1);

eta = 1;
nHidden = 4;
epochCount = 1000;
minErr = 0.001;
[R_OUT, G_OUT, errVec] = backprop_sigmoid(data,target,nHidden,eta,epochCount,minErr);
plot(errVec);

% Display the graph
figure
hold on
scatter(R(idx_sf),G(idx_sf),4,'k',"filled")
scatter(R(idx_rs),G(idx_rs),4,'r',"filled")
scatter(R(idx_ir),G(idx_ir),4,'g',"filled")
title("RGB Variation")

% Testing an image

testImagePath = strcat(imagesDir,"sunflower_1.jpeg");
testImage = imread(testImagePath);

testAvgRgb = mean(reshape(testImage, size(testImage,1) * size(testImage,2), size(testImage,3)));
testData = [ testAvgRgb(1) testAvgRgb(2) ];

result = mlp_sigmoid(R_OUT, G_OUT, testData); %sunflower
result

