close all
clear all
clc

% Load the data
load('proj3_data_test_corr_2D.mat');
load('proj3_data_test.mat');

% Get the number of images and size of each image
[image_number, x_size, y_size] = size(data_test_corr_2D);

num_samples = 1;  % Number of images to display
idx = randi([1, image_number], 1, num_samples);
disp(idx);
data = squeeze(data_test_corr_2D(idx,:,:));
data_compare = squeeze(data_test(idx,:,:));

% Create a random starting phase
start_phase = rand(x_size, y_size) * 2 * pi;

% No noise
% Compute the magnitude of the FFT, which is the power spectrum
power_spectrum_root = sqrt(abs((fft2(data))));
reconstructed = HIO_ErrorReduction(power_spectrum_root, start_phase);

figure;
subplot(1, 2, 1);
imagesc(reconstructed);
axis image;
colormap(gray);
colorbar;
title('Reconstructed Image');

subplot(1, 2, 2);
imagesc(data_compare);
axis image;
colormap(gray);
colorbar;
title('Comparison Image');

%%
% noisy
figure;
pos = 0;
max = max(data(:));
SNR_values = 100:-10:30; 
for SNR = SNR_values
    pos = pos + 1;
    noise_level = 10^(-SNR/20)* max;
    data = data + noise_level*randn(size(data));
    power_spectrum_root = sqrt(abs((fft2(data))));
    reconstructed = HIO_ErrorReduction(power_spectrum_root, start_phase);
     % Reconstructed Image
    subplot(2, 4, pos);
    imagesc(reconstructed);
    axis image;
    colormap(gray);
    colorbar;
    title(sprintf('Noisy Reconstructed Image with SNR = %.2f',SNR));
end
%%
function reconstructed = HIO_ErrorReduction(power_spectrum_root, start_phase)
    % Initialize
    inverse_gft = start_phase;
    updated_object_pattern = power_spectrum_root .* exp(1i * fft2(angle(start_phase)));

    % Parameters for the HIO and Error Reduction algorithms
    beta_values = 2:-0.04:0; 
    num_hio_iterations = 40;
    num_error_reduction_iterations = 40;
    for beta = beta_values
        % HIO iterations
        for i = 1:num_hio_iterations
            % inverse 2d FFT
            updated_inverse_gft = ifft2(updated_object_pattern);
            % Find the points violating the physical constraints
            violated_points = (real(updated_inverse_gft) < 0) | ~isreal(updated_inverse_gft);
            % Apply the update only for the violated points
            inverse_gft(~violated_points) = updated_inverse_gft(~violated_points);
            inverse_gft(violated_points) = inverse_gft(violated_points) - beta * updated_inverse_gft(violated_points);

            % 2D Fourier Transform
            object_pattern = fft2(inverse_gft);

            % Argument (Phase Calculation)
            modified_phase = angle(object_pattern);

            % Update the object pattern using modified phase
            updated_object_pattern = power_spectrum_root .* exp(1i * modified_phase);
        end
    end
    % Error Reduction iterations
    for i = 1:num_error_reduction_iterations
        % inverse 2d FFT
        updated_inverse_gft = ifft2(updated_object_pattern);
        % Find the points violating the physical constraints
        violated_points = (real(updated_inverse_gft) < 0) | ~isreal(updated_inverse_gft);
        % Apply the update only for the violated points
        inverse_gft(~violated_points) = updated_inverse_gft(~violated_points);
        inverse_gft(violated_points) = 0;

        % 2D Fourier Transform
        object_pattern = fft2(inverse_gft);

        % Argument (Phase Calculation)
        modified_phase = angle(object_pattern);

        % Update the object pattern using modified phase
        updated_object_pattern = power_spectrum_root .* exp(1i * modified_phase);
    end
    reconstructed = inverse_gft;
end
