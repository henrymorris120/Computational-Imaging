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
figure;imagesc(PointImage) %Display the image
colormap(gray)
axis image

%Add noise to the image and perform superresolution processing
NImages = 1; %number of images to evaluate
SNR = 0; %dB
xdata = cat(3,xx,yy);

noiselevel = 10^(-SNR/20); %assumes a signal value of 1
xest = [];
yest = [];
for i = 1:NImages
    %disp(['Image ' num2str(i) ' of ' num2str(NImages)]);
    NoiseImage = PointImage + noiselevel*randn(size(PointImage));
    [y,x] = fittGauss(PointImage,xdata,SNR, PointImage);
  
     yest = [yest,y];
     xest = [xest,x];
    
end
%%

% Answer Question 2, use template of code above and just run 100 times for
% each db level

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
dbs = [0, 10, 20, 30, 40];
meansx = zeros(1, 5);
meansy = zeros(1, 5);
stdx = zeros(1, 5);
stdy= zeros(1, 5);
for k = 1:1:numel(dbs)
    NImages = 100; %number of images to evaluate
    SNR = dbs(k); %dB
    xdata = cat(3,xx,yy);
    noiselevel = 10^(-SNR/20); %assumes a signal value of 1
    xest = [];
    yest = [];
    for i = 1:NImages
        NoiseImage = PointImage + noiselevel*randn(size(PointImage));
        [y,x] = fittGauss2(NoiseImage,xdata,SNR);
        yest = [yest,y];
        xest = [xest,x];
    end
    meansx(k) = mean(xest);
    meansy(k) = mean(yest);
    stdx(k) = std(xest);
    stdy(k) = std(yest);
    if SNR == 20
        % Scatter plot of estimated points for dbs = 20
        figure;
        subplot(1,2,1);
        scatter(xest, yest, 10, 'filled');
        xlim([1, 260]);  % Set x-axis limits to 260
        ylim([1, 260]); 
        grid on;  % Display grid
        set(gca, 'YDir', 'reverse');  % Reverse y-axis direction
        title('Scatter Plot of Estimated Points for dbs = 20');
        xlabel('X');
        ylabel('Y')
        subplot(1,2,2);
        scatter(xest, yest, 10, 'filled');
        xlim([xpoint-3, xpoint+3]);
        ylim([ypoint-3, ypoint+3]); 
        grid on;  % Display grid
        set(gca, 'YDir', 'reverse');  % Reverse y-axis direction
        title('Zoomed in Scatter Plot of Estimated Points for dbs = 20');
        xlabel('X');
        ylabel('Y')
    end
end
fprintf('actual x is %f and actual y is %f \n',xpoint,ypoint);
for i = 1:1:numel(dbs)
    fprintf('when dbs = %f, x has mean %f and std %f, y has mean %f and std %f \n',dbs(i), meansx(i), stdx(i), meansy(i), stdy(i));
end


figure;
hold on;  % Hold the plot to overlay multiple plots
plot(dbs, stdx, 'o-', 'LineWidth', 1.5, 'DisplayName', 'stdx points');

% Plot the points represented by stdy and dbs
plot(dbs, stdy, 's-', 'LineWidth', 1.5, 'DisplayName', 'stdy points');

xlabel('SNR');
ylabel('Standard Deviation');
title('Comparison of Standard Deviation (stdx vs stdy) for different SNR');
legend('Location', 'Best');  % Show legend
grid on;  % Display grid


%%
% Part 2
load('PALM.mat');
frames = PALM;

frame_height = 280;
frame_width = 280;
frame_number = 2000;
xest=[];
yest=[];
conventionalImage= sum(frames,3);

% Initialize a matrix to store the reconstructed image
reconstructed_no_noise = zeros(size(conventionalImage));
for i = 1:frame_number
    pointIm = frames(:,:,i);
    [y,x] = fittGauss2(pointIm,xdata,0);
     yest = [yest,y];
     xest = [xest,x];
end

for i = 1:numel(yest)
    x = max(1, min(round(xest(i)), 280));
    y = max(1, min(round(yest(i)), 280));

    reconstructed_no_noise(x,y)=1;
end


% graph “conventional” fluorescence image
figure;
subplot(1,2,1)
imagesc(conventionalImage);
axis image
colormap(gray)
colorbar;
title('Conventional Fluorescence Image')

% graph the reconstructed
subplot(1,2,2);
imagesc(reconstructed_no_noise);
axis image;
colormap(gray);
colorbar;
title('Reconstructed Noisefree Image');



%%
% Question 2 of Part 2
SNR = 20; %dB
noiselevel = 10^(-SNR/20);
load('PALM.mat');
frames = PALM;
frame_height = 280;
frame_width = 280;
frame_number = 2000;
xest=[];
yest=[];
conventionalImage= sum(frames,3);



% Initialize a matrix to store the reconstructed image
reconstructedNoiseImage = zeros(size(conventionalImage));

for i = 1:frame_number
    pointIm = frames(:,:,i);
    NoiseImage = pointIm + noiselevel*randn(size(pointIm));
    [y,x] = fittGauss2(NoiseImage,xdata,SNR);
    yest = [yest,y];
    xest = [xest,x];
end

for i = 1:numel(yest)
    x = max(1, min(round(xest(i)), 280));
    y = max(1, min(round(yest(i)), 280));
    reconstructedNoiseImage(x,y)=1;
end


% graph “conventional” fluorescence image
figure;
subplot(1,2,1)
imagesc(conventionalImage);
axis image
colormap(gray)
colorbar;
title('Conventional Fluorescence Image')

% graph the reconstructed
subplot(1,2,2);
imagesc(reconstructedNoiseImage);
axis image;
colormap(gray);
colorbar;
title('Reconstructed Noise Image');
%%
% graph “conventional” fluorescence image
figure;
subplot(1,2,1)
imagesc(reconstructed_no_noise);
axis image
colormap(gray)
colorbar;
title('reconstruct no noise')

% graph the reconstructed
subplot(1,2,2);
imagesc(reconstructedNoiseImage);
axis image;
colormap(gray);
colorbar;
title('Reconstructed Noise Image');


%%

function [y_means,x_means] = fittGauss(NoiseImage, xdata,SNR, PointImage)
  
    % Step 1: Low-pass filter
    [x_size, y_size] = size(NoiseImage);
    radius = 26;
    [x, y] = meshgrid(1:x_size, 1:y_size);
    % Calculate the center of the image
    center_x = (x_size + 1) / 2;
    center_y = (y_size + 1) / 2;

    % Calculate the distance of each point from the center
    distance_from_center = sqrt((x - center_x).^2 + (y - center_y).^2);

    % Create a low pass circular mask and apply to NoiseImage in Fourier
    % domain
    mask = distance_from_center <= radius;
    filtered_fourier_image = fftshift(fft2(NoiseImage)) .* mask;
    lowPassNoiseImage = abs(ifft2(ifftshift(filtered_fourier_image)));
    
    
    % graph of the noisy image
    figure;
    subplot(2,3,1);
    imagesc(NoiseImage);
    colormap(gray);
    colorbar;
    axis image;
    title('Noisy Image');
    

    % graphing of the noisy image after low pass
     subplot(2,3,2);
    imagesc(lowPassNoiseImage);
    colormap(gray);
    axis image;
    colorbar;
    title('With Lowpass');
   
    
    
    % Step 2: Threshold
    windowSize = 6;  % Define the size of the local window for calculating the mean
    threshold = 0.85;  % Threshold value for the mean of the local window
    thresholdedNoiseImage = zeros(size(lowPassNoiseImage));

    for i = 1:x_size
        for j = 1:y_size
            % determine local window to look at for threshold
            windowStartX = max(1, i - floor(windowSize/2));
            windowEndX = min(x_size, i + floor(windowSize/2));
            windowStartY = max(1, j - floor(windowSize/2));
            windowEndY = min(y_size, j + floor(windowSize/2));
            localWindow = lowPassNoiseImage(windowStartX:windowEndX, windowStartY:windowEndY);
        
           % Calculate the mean of the local window
            localMean = mean(localWindow(:));
        
           % Threshold based on the local mean
            if localMean >= threshold
                thresholdedNoiseImage(i, j) = lowPassNoiseImage(i, j);
            end
        end
    end
    
    % graph after threshold 
    subplot(2,3,3);
    imagesc(thresholdedNoiseImage);
    colormap(gray);
    axis image;
    colorbar;
    title('Thresholded Image');
  
    
    % find all the local maxima, one per point source
    maximaImage = imregionalmax(thresholdedNoiseImage);
  
    % graph only local maxima, may be more than one per source
    subplot(2,3,4);
    imagesc(maximaImage);
    colormap(gray);
    axis image;
    colorbar;
    title('Local Maxima');
    
    % find all the local maxima, one per point source
    [x_indices, y_indices] = find(maximaImage);
    [x_max,y_max] = check_max_value_pixels(thresholdedNoiseImage,x_indices,y_indices,15);
    singlePoint = zeros(size(thresholdedNoiseImage));
    for i = 1:numel(x_max)
        singlePoint(x_max(i),y_max(i)) = 1;
    end
    
    % graph single local maximumum for each point source
    subplot(2,3,5);
    imagesc(singlePoint);
    colormap(gray);
    axis image;
    colorbar;
    title('Local Max per Point Source');

  
    
    % Step 4: Fit a 2d Gaussian function
    
    % Define the Gaussian model function
    gaussianModel = @(params, data) params(1) * exp(-((data(:,:,1) - params(2)).^2) / (2 * params(3)^2) - ((data(:,:,2) - params(4)).^2) / (2 * params(5)^2));
    x_means = [];
    y_means = [];
    % iterate through each local maxima for each point source
    for i = 1:numel(x_max)
        xStart = max(1,x_max(i) - 7);
        xEnd = min(256,x_max(i) + 7);
        yStart = max(1,y_max(i) - 7);
        yEnd = min(256,y_max(i) + 7);
        subimage = NoiseImage(xStart:xEnd, yStart:yEnd);
        initial_params = [NoiseImage(x_max(i),y_max(i)), y_max(i), 11, x_max(i), 11];
        options = optimoptions(@lsqcurvefit,'FunctionTolerance', 1e-12, 'StepTolerance', 1e-6, 'OptimalityTolerance', 1e-6, 'Display', 'Off');
        estimatedParams = lsqcurvefit(gaussianModel, initial_params, xdata(xStart:xEnd, yStart:yEnd,:), subimage,[.5 0 11 0 11], [1.5 300 11 300 11],options);
        

        % graph of zoomed in noiseimage, PointImage is just set as a
        % parameter here for debuggin purpose to make sure we zoom in
        % correctly
        sub = PointImage(xStart:xEnd, yStart:yEnd);
        figure;
        imagesc(sub);
        axis image;
        colormap(gray);
        colorbar;
        title('Zoomed in on PointImage');

    
        % Unpack the estimated parameters
        amplitude = estimatedParams(1);
        mean_x = estimatedParams(2);
        mean_y = estimatedParams(4);
        std_dev_x = estimatedParams(3);
        std_dev_y = estimatedParams(5);
        
    

        params = [amplitude, mean_x, std_dev_x,mean_y,std_dev_y];
        initialGuessImage = gaussianModel(initial_params,xdata);
        fitImage = gaussianModel(params,xdata);

        % Point Image used to verify the next 2 graphs
        figure;
        subplot(1,3,1);
        imagesc(PointImage);
        axis image;
        colormap(gray);
        colorbar;
        title('Point Image');

        % graph of initial gaussian fit guess
        subplot(1,3,2);
        imagesc(initialGuessImage);
        axis image;
        colormap(gray);
        colorbar;
        title('Initial Guess Gaussian');
        
        % graph of gaussian fit
        subplot(1,3,3);
        imagesc(fitImage);
        axis image;
        colormap(gray);
        colorbar;
        title('Fitted 2D Gaussian');

       

        % use residual to test whether what you concluded make sense
        % for example, you might get an incorrect local maxima if the two 
        % different point source are next to each other
        % below is also helpful for testing
        residuals = lowPassNoiseImage(xStart:xEnd, yStart:yEnd) - fitImage(xStart:xEnd, yStart:yEnd);
        residuals2 = lowPassNoiseImage(xStart:xEnd, yStart:yEnd) - initialGuessImage(xStart:xEnd, yStart:yEnd);
        rmse = sqrt(mean(residuals(:).^2));
        rmse2 = sqrt(mean(residuals2(:).^2));
        % these 2 for testing/debugging purposed
        disp(initial_params);
        disp(params);
        fprintf('initial guess RMSE error: %f. with fit: %f',rmse2, rmse);
       
        
       % good points RMSE should be less than this function
        if rmse < -0.0056*SNR + 0.23
            x_means = [x_means,mean_x];
            y_means = [y_means,mean_y];
        end
    end
end



%%
function [y_means,x_means] = fittGauss2(NoiseImage, xdata,SNR)
  
    % Step 1: Low-pass filter
    [x_size, y_size] = size(NoiseImage);
    radius = 26;
    [x, y] = meshgrid(1:x_size, 1:y_size);
    % Calculate the center of the image
    center_x = (x_size + 1) / 2;
    center_y = (y_size + 1) / 2;

    % Calculate the distance of each point from the center
    distance_from_center = sqrt((x - center_x).^2 + (y - center_y).^2);

    % Create a low pass circular mask and apply to NoiseImage in Fourier
    % domain
    mask = distance_from_center <= radius;
    filtered_fourier_image = fftshift(fft2(NoiseImage)) .* mask;
    lowPassNoiseImage = abs(ifft2(ifftshift(filtered_fourier_image)));
    
   

    
    
    % Step 2: Threshold
    windowSize = 6;  % Define the size of the local window for calculating the mean
    threshold = 0.85;  % Threshold value for the mean of the local window
    thresholdedNoiseImage = zeros(size(lowPassNoiseImage));

    for i = 1:x_size
        for j = 1:y_size
            % determine local window to look at for threshold
            windowStartX = max(1, i - floor(windowSize/2));
            windowEndX = min(x_size, i + floor(windowSize/2));
            windowStartY = max(1, j - floor(windowSize/2));
            windowEndY = min(y_size, j + floor(windowSize/2));
            localWindow = lowPassNoiseImage(windowStartX:windowEndX, windowStartY:windowEndY);
        
           % Calculate the mean of the local window
            localMean = mean(localWindow(:));
        
           % Threshold based on the local mean
            if localMean >= threshold
                thresholdedNoiseImage(i, j) = lowPassNoiseImage(i, j);
            end
        end
    end
    
   
  
    
    % find all the local maxima, one per point source
    maximaImage = imregionalmax(thresholdedNoiseImage);

    
    % find all the local maxima, one per point source
    [x_indices, y_indices] = find(maximaImage);
    [x_max,y_max] = check_max_value_pixels(thresholdedNoiseImage,x_indices,y_indices,15);
    singlePoint = zeros(size(thresholdedNoiseImage));
    for i = 1:numel(x_max)
        singlePoint(x_max(i),y_max(i)) = 1;
    end
  

  
    
    % Step 4: Fit a 2d Gaussian function
    
    % Define the Gaussian model function
    gaussianModel = @(params, data) params(1) * exp(-((data(:,:,1) - params(2)).^2) / (2 * params(3)^2) - ((data(:,:,2) - params(4)).^2) / (2 * params(5)^2));
    x_means = [];
    y_means = [];
    % iterate through each local maxima for each point source
    for i = 1:numel(x_max)
        xStart = max(1,x_max(i) - 7);
        xEnd = min(256,x_max(i) + 7);
        yStart = max(1,y_max(i) - 7);
        yEnd = min(256,y_max(i) + 7);
        subimage = NoiseImage(xStart:xEnd, yStart:yEnd);
        initial_params = [NoiseImage(x_max(i),y_max(i)), y_max(i), 11, x_max(i), 11];
        options = optimoptions(@lsqcurvefit,'FunctionTolerance', 1e-12, 'StepTolerance', 1e-6, 'OptimalityTolerance', 1e-6, 'Display', 'Off');
        estimatedParams = lsqcurvefit(gaussianModel, initial_params, xdata(xStart:xEnd, yStart:yEnd,:), subimage,[.5 0 11 0 11], [1.5 300 11 300 11],options);
       

   
    
        % Unpack the estimated parameters
        amplitude = estimatedParams(1);
        mean_x = estimatedParams(2);
        mean_y = estimatedParams(4);
        std_dev_x = estimatedParams(3);
        std_dev_y = estimatedParams(5);
        
    

        params = [amplitude, mean_x, std_dev_x,mean_y,std_dev_y];
        initialGuessImage = gaussianModel(initial_params,xdata);
        fitImage = gaussianModel(params,xdata);
    
        % use residual to test whether what you concluded make sense
        % for example, you might get an incorrect local maxima if the two 
        % different point source are next to each other
        % below is also helpful for testing
        residuals = lowPassNoiseImage(xStart:xEnd, yStart:yEnd) - fitImage(xStart:xEnd, yStart:yEnd);
        residuals2 = lowPassNoiseImage(xStart:xEnd, yStart:yEnd) - initialGuessImage(xStart:xEnd, yStart:yEnd);
        rmse = sqrt(mean(residuals(:).^2));
        rmse2 = sqrt(mean(residuals2(:).^2));
       
       
        
       
        % good points RMSE should be less than this function
        if rmse < -0.0056*SNR + 0.23
            x_means = [x_means,mean_x];
            y_means = [y_means,mean_y];
        end
    end
end

%%
function [y_means,x_means] = fittGauss3(NoiseImage, xdata,SNR)
  
    % Step 1: Low-pass filter
    [x_size, y_size] = size(NoiseImage);
    radius = 2;
    [x, y] = meshgrid(1:x_size, 1:y_size);
    % Calculate the center of the image
    center_x = (x_size + 1) / 2;
    center_y = (y_size + 1) / 2;

    % Calculate the distance of each point from the center
    distance_from_center = sqrt((x - center_x).^2 + (y - center_y).^2);

    % Create a low pass circular mask and apply to NoiseImage in Fourier
    % domain
    mask = distance_from_center <= radius;
    filtered_fourier_image = fftshift(fft2(NoiseImage)) .* mask;
    lowPassNoiseImage = abs(ifft2(ifftshift(filtered_fourier_image)));
    
   

    % graphing of the noisy image after low pass
    figure;
    subplot(1,2,1);
    imagesc(lowPassNoiseImage);
    colormap(gray);
    axis image;
    colorbar;
    title('With Lowpass');
   
    
    
    % Step 2: Threshold
    windowSize = 6;  % Define the size of the local window for calculating the mean
    threshold = 0.85;  % Threshold value for the mean of the local window
    thresholdedNoiseImage = zeros(size(lowPassNoiseImage));

    for i = 1:x_size
        for j = 1:y_size
            % determine local window to look at for threshold
            windowStartX = max(1, i - floor(windowSize/2));
            windowEndX = min(x_size, i + floor(windowSize/2));
            windowStartY = max(1, j - floor(windowSize/2));
            windowEndY = min(y_size, j + floor(windowSize/2));
            localWindow = lowPassNoiseImage(windowStartX:windowEndX, windowStartY:windowEndY);
        
           % Calculate the mean of the local window
            localMean = mean(localWindow(:));
        
           % Threshold based on the local mean
            if localMean >= threshold
                thresholdedNoiseImage(i, j) = lowPassNoiseImage(i, j);
            end
        end
    end
    
   
  
    
    % find all the local maxima, one per point source
    maximaImage = imregionalmax(thresholdedNoiseImage);

    
    % find all the local maxima, one per point source
    [x_indices, y_indices] = find(maximaImage);
    [x_max,y_max] = check_max_value_pixels(thresholdedNoiseImage,x_indices,y_indices,15);
    singlePoint = zeros(size(thresholdedNoiseImage));
    for i = 1:numel(x_max)
        singlePoint(x_max(i),y_max(i)) = 1;
    end
  

    % graph single local maximumum for each point source
    subplot(1,2,2);
    imagesc(singlePoint);
    colormap(gray);
    axis image;
    colorbar;
    title('Local Max per Point Source');
    
    % Step 4: Fit a 2d Gaussian function
    
    % Define the Gaussian model function
    gaussianModel = @(params, data) params(1) * exp(-((data(:,:,1) - params(2)).^2) / (2 * params(3)^2) - ((data(:,:,2) - params(4)).^2) / (2 * params(5)^2));
    x_means = [];
    y_means = [];
    % iterate through each local maxima for each point source
    for i = 1:numel(x_max)
        xStart = max(1,x_max(i) - 7);
        xEnd = min(32,x_max(i) + 7);
        yStart = max(1,y_max(i) - 7);
        yEnd = min(32,y_max(i) + 7);
        subimage = NoiseImage(xStart:xEnd, yStart:yEnd);
        initial_params = [NoiseImage(x_max(i),y_max(i)), y_max(i), 11, x_max(i), 11];
        options = optimoptions(@lsqcurvefit,'FunctionTolerance', 1e-12, 'StepTolerance', 1e-6, 'OptimalityTolerance', 1e-6, 'Display', 'Off');
        estimatedParams = lsqcurvefit(gaussianModel, initial_params, xdata(xStart:xEnd, yStart:yEnd,:), subimage,[.5 0 11 0 11], [1.5 300 11 300 11],options);
       

   
    
        % Unpack the estimated parameters
        amplitude = estimatedParams(1);
        mean_x = estimatedParams(2);
        mean_y = estimatedParams(4);
        std_dev_x = estimatedParams(3);
        std_dev_y = estimatedParams(5);
        
    

        params = [amplitude, mean_x, std_dev_x,mean_y,std_dev_y];
        initialGuessImage = gaussianModel(initial_params,xdata);
        fitImage = gaussianModel(params,xdata);
    
        % use residual to test whether what you concluded make sense
        % for example, you might get an incorrect local maxima if the two 
        % different point source are next to each other
        % below is also helpful for testing
        residuals = lowPassNoiseImage(xStart:xEnd, yStart:yEnd) - fitImage(xStart:xEnd, yStart:yEnd);
        residuals2 = lowPassNoiseImage(xStart:xEnd, yStart:yEnd) - initialGuessImage(xStart:xEnd, yStart:yEnd);
        rmse = sqrt(mean(residuals(:).^2));
        rmse2 = sqrt(mean(residuals2(:).^2));
       
       
        
       
        % good points RMSE should be less than this function
        if rmse < -0.0056*SNR + 0.23
            x_means = [x_means,mean_x];
            y_means = [y_means,mean_y];
        end
    end
end

%%
function [x_max,y_max] = check_max_value_pixels(image, x_indices,y_indices, distance)
    x_max = [];
    y_max = [];
    for k = 1:size(x_indices)
        x = x_indices(k);
        y = y_indices(k);
  
        current_value = image(x, y);
        
        % Define the region around the current pixel
        min_row = max(1, x - distance);
        max_row = min(size(image, 1), x + distance);
        min_col = max(1, y - distance);
        max_col = min(size(image, 2), y + distance);
        
        % Extract the region
        region = image(min_row:max_row, min_col:max_col);
        
        % Check if the current pixel has the maximum value in the region
        if current_value == max(region(:))
            x_max = [x_max, x];
            y_max = [y_max, y];
        end
    end
end
