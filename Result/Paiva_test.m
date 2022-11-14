clc
clear
close all

%% load ground_truth (HR_MSI)
load('pavia_cnn.mat');
HSI = HR_HSI;
MSI = HR_MSI;
%[Height,Width,Bands_H] = size(HSI);
[~,~,Bands_M] = size(MSI);
clear HR_HSI HR_MSI LR_HSI LR_MSI;


%% split training and testing samples 
Train_HSI = HSI(:,1:100,:);
Train_MSI = MSI(:,1:100,:);
Test_HSI_GT = HSI(:,101:end,:);
Test_MSI = MSI(:,101:end,:);
[Height_HR, Width_HR, Band_HR] = size(Test_HSI_GT);

%% load data
load('AllData_MY_Results_4.mat');

[num_patch, patch_size, ~, ~] = size(prob_map);

SR_HSI = zeros([Height_HR, Width_HR, Band_HR]);
count = 1;
for j = 1:patch_size:Width_HR-patch_size+1
    for i = 1:patch_size:Height_HR-patch_size+1
        temp = reshape(prob_map(count,:,:,:),[patch_size, patch_size, Band_HR]);
        SR_HSI(i:i+patch_size-1,j:j+patch_size-1,:) = temp;
        count = count + 1;
    end
end

addpath('quality_ass')
[psnr,rmse, ergas, sam, uiqi,ssim,DD,CC] = quality_assessment(double((255*Test_HSI_GT)),double((255*SR_HSI)), 0, 1.0/1);
disp(['PSNR  : ' num2str(psnr)]);
disp(['SAM   : ' num2str(sam)]);
disp(['RMSE  : ' num2str(rmse)]);
disp(['ERGAS : ' num2str(ergas)]);
disp(['UIQI  : ' num2str(uiqi)]);
disp(['SSIM  : ' num2str(ssim)]);
disp(['CC    : ' num2str(CC)]);
disp(['DD    : ' num2str(DD)]);