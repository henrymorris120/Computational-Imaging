close all
clear all
clc


%%
%Set up image parameters

x = 1:256; %x-locations of pixels
y = 1:256; %y-locations of pixels

[xx, yy] = meshgrid(x,y); %set up the image grid

NPoints = 1;

PointImage = zeros(size(xx));

for i = 1:NPoints
    xpoint = max(x)*rand(1);    %x location of the point source
    ypoint = max(x)*rand(1);    %y location of the point source
    sigx = 11;         %x-width of the Gaussian PSF
    sigy = 11;         %y-width of the Gaussian PSF
    
    PointImage = PointImage + exp(-((xpoint-xx).^2/(2*sigx^2) + (ypoint-yy).^2/(2*sigy^2))); %Make the image
end
figure;imagesc(y,x,PointImage) %Display the image
colormap(gray)
axis image

%%
%Add noise to the image and perform superresolution processing
NImages = 10; %number of images to evaluate
SNR = 0; %dB
xdata = cat(3,xx,yy);

noiselevel = 10^(-SNR/20); %assumes a signal value of 1
xest = [];
yest = [];
for i = 1:NImages
    clc
    disp(['Image ' num2str(i) ' of ' num2str(NImages)]);
    NoiseImage = PointImage + noiselevel*randn(size(PointImage));
   
    [y,x] = fittGauss(NoiseImage,xdata,SNR);
  
       yest = [yest,x(2)];
       xest = [xest,x(4)];
    
end

%%
%Perform analysis of your results. What is the standard deviation of your
%x- and y-locations? How close is the mean of these estimates to the actual
%values?