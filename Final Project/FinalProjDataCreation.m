close all
clear all
clc
%%

% pathname to STL10 downloaded data from Kaggle
% we will use 20,000 of the 100,000 unlabeled images
folderPath = '/Users/henrymorris/Downloads/archive/unlabeled_images';

% Get a list of all PNG files in the folder
pngFiles = dir(fullfile(folderPath, '*.png'));

% Initialize a 3D matrix to store the grayscale images
imageData = zeros(20000, 96, 96);

% Loop through each image
for i = 1:20000
    currentImagePath = fullfile(folderPath, pngFiles(i).name);
    colorImage = imread(currentImagePath);
    
    % Convert to grayscale
    grayImage = rgb2gray(colorImage);
    imageData(i, :, :) = grayImage;
end

%%
% save the matrix
save('STL10Images.mat', 'imageData');
%%
% check that we loaded in images correctly by randomly showing one of the 
% images

% Choose an index (270 chosen for no particular reason)
randomIndex = 270;

% Retrieve the randomly chosen image
randomImage = squeeze(imageData(randomIndex, :, :));

% Display the random image
figure;
imshow(randomImage, []);
title(['Random Image - Index ' num2str(randomIndex)]);

%%
% Downsample each image to create a low-resolution version (32 by 32 pixels)
% this is scaling by 3 down
lowResImageData = zeros(20000, 32, 32);
for i = 1:20000
    % Downsample each image
    lowResImageData(i, :, :) = imresize(squeeze(imageData(i, :, :)), [32, 32]);
end

% Save the low-resolution imageData matrix to a MAT file named LowResSTL10Images
save('LowResSTL10Images.mat', 'lowResImageData');

% check this was done correctly by showing one of the low res images

% Retrieve the randomly chosen image
randomImage = squeeze(lowResImageData(randomIndex, :, :));

% Display the random image
figure;
imshow(randomImage, []);
title(['Random Low Res Image - Index ' num2str(randomIndex)]);

%%
% Downsample each image to create a low-resolution version (48 by 48 pixels)
% this is scaling down by 3
lowResImageData2 = zeros(20000, 48, 48);
for i = 1:20000
    % Downsample each image
    lowResImageData2(i, :, :) = imresize(squeeze(imageData(i, :, :)), [48, 48]);
end

% Save this second low-resolution imageData matrix to a MAT file named LowRes2STL10Images
save('LowRes2STL10Images.mat', 'lowResImageData2');

% check this was done correctly by showing one of the low res images

% Retrieve the randomly chosen image
randomImage = squeeze(lowResImageData2(randomIndex, :, :));

% Display the random image
figure;
imshow(randomImage, []);
title(['Random Low Res Image - Index ' num2str(randomIndex)]);
