close all
clear all
clc

%%
%Setup the image

A = imread('Thayer.jpg'); %load the original high-resolution image
A = double(max(A,[],3));

A = A - min(min(A));  %Modify the image to mostly be black and white
A = max(max(A))-A;
A = min(2*A,255);
A = A.*(A>100);

%%

x = 1:size(A,1); %Set up the image size
y = 1:size(A,2);

[xx, yy] = meshgrid(x,y);

figure;imagesc(y,x,A) %Display the image
colormap(gray)
axis image


%%

NImages = 2000;  %Number of images
PALM = zeros(size(A,1),size(A,2),NImages); % Allocate space

sigx = 11; %Width fo the PSF in the x and y directions
sigy = 11;

AvgMol = 4; %Average number of "on" molecules in each image

%Cycle through the number of images
for i = 1:NImages
    clc
    disp(['Image ' num2str(i) ' of ' num2str(NImages)]);
    numMol = ceil(AvgMol*2*rand(1)); %Calculate how many molecules are "on" in this image
    molCount = 0;
    
    while(molCount < numMol)  %Keep going until we get the right number of "on" molecules
        X = size(A,1)*rand(1)+.5;  %pick a random x-y coordinate
        Y = size(A,2)*rand(1)+.5;
        
        Xind = round(X);  %round to the nearest pixel
        Yind = round(Y);
        
        if(A(Yind,Xind) > 0)  %if the molecule corresponds to a white part of the image
            molCount = molCount+1;
            PALM(:,:,i) = PALM(:,:,i) + exp(-((X-xx).^2/(2*sigx^2) + (Y-yy).^2/(2*sigy^2))); %Add the PSF located at the exact location to the image
        end
    end
end
        

figure;  %display the conventional fluorscence image
imagesc(sum(PALM,3))
axis image
colormap(gray)

    

        