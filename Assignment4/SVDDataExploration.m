%% Read in data
close all; clear all;
[trainimages, trainlabels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');

%Make matrix to hold all num info
picheight = 28;
picwidth = 28;
num_rows = picheight * picwidth;
num_cols = length(trainlabels);
allNums = zeros(num_rows, num_cols);

for i = 1:num_cols
    %reshape into col vector, store with each col = different image
    allNums(:,i) = reshape(trainimages(:,:,i), num_rows, 1);
end

figure()
plotnum(allNums(:,1))

%SVD
[U,S,V] = svd(allNums, 'econ');
eigvals = diag(S);

%% Determine number of modes to incorperate
energyThreshold = .9;
normalized_eigvals = eigvals / sum(eigvals);
cumEnergy = cumsum(eigvals)/sum(eigvals);
energyIndex = find(cumEnergy >energyThreshold,1);
%% Plot First 4 PCA Modes
close all;
figure()
for j = 1:10
    subplot(2,5,j)
    imagesc(reshape(U(:, j), picheight, picwidth))
end
sgtitle('First 10 Principal Components of MNIST Training Data')

%% Plot singular values
figure(2)
subplot(2,1,1)
plot(diag(S), 'ko', 'Linewidth', 2) %X, Y plot
set(gca, 'Fontsize', 16, 'Xlim', [0, 400])

subplot(2,1,2)
semilogy(diag(S), 'ko', 'Linewidth', 2) %log plot
set(gca, 'Fontsize', 16, 'Xlim', [0,400])
%slow decay, first is most important
sgtitle('Singular Values of MNIST Training Data')

%% Plot Normalized Singular Values And Threshold
figure()
sline = diag(S) / sum(diag(S));
plot(sline*100, 'ko', 'Linewidth', 2) %X, Y plot
set(gca, 'Fontsize', 16, 'Xlim', [0, 400])
hold on
cumline = cumEnergy*100;
plot(cumline, 'Linewidth', 1)
hold on
xline(energyIndex)
yline(90)
title("Normalized Singular Values of MNIST Training Data")
ylabel('Percent Energy Represented')
xlabel('Principal Component')
%% Plot first 3 V modes in 3D
close all;
%# find out how many clusters you have
uClusters = unique(string(trainlabels));
nClusters = length(uClusters);

%# create colormap
%# distinguishable_colormap from the File Exchange 
%# is great for distinguishing groups instead of hsv
cmap = hsv(nClusters);

%# plot, set DisplayName so that the legend shows the right label
figure
for iCluster = 1:nClusters
    this_num = (iCluster-1);
    clustIdx = string(trainlabels)==uClusters(iCluster);
    plot3(V(clustIdx,4),V(clustIdx,2),...
        V(clustIdx,3), '.', 'MarkerSize', 5,...
       'DisplayName',sprintf('Number %i',(this_num)));
   hold on
end

legend('show');
title('MNIST Training Data Projected onto Second, Third, and Fourth V-Modes')

%% Make low rank approximations of data

%determine modes for 80 and 90 %
energyThreshold = .8;
modesfor80 = find(cumEnergy >energyThreshold,1);
modesfor90 = energyIndex;
modesfor100 = length(eigvals);


%make low rank approximations
rank80percent = U(:,1)*eigvals(1)*V(:,1)';
i = 1;
while i < modesfor80
    i = i + 1;
    rank80percent = rank80percent + (U(:,i)*eigvals(i)*V(:,i)');
end

rank90percent = rank80percent;
while i < modesfor90
    disp(i)
    i = i + 1;
    rank90percent = rank90percent + (U(:,i)*eigvals(i)*V(:,i)');
end
%% Make figure comparing numbers
figure()
subplot(3,5,4)
plotnum(rank80percent(:,12))
subplot(3,5,1)
plotnum(rank80percent(:,2))
ylabel("80% Energy")
subplot(3,5,3)
plotnum(rank80percent(:,3))
subplot(3,5,2)
plotnum(rank80percent(:,4))
subplot(3,5,5)
plotnum(rank80percent(:,5))

subplot(3,5,9)
plotnum(rank90percent(:,12))
subplot(3,5,6)
plotnum(rank90percent(:,2))
ylabel("90% Energy")
subplot(3,5,8)
plotnum(rank90percent(:,3))
subplot(3,5,7)
plotnum(rank90percent(:,4))
subplot(3,5,10)
plotnum(rank90percent(:,5))

subplot(3,5,14)
plotnum(allNums(:,12))
subplot(3,5,11)
plotnum(allNums(:,2))
ylabel("100% Energy")
subplot(3,5,13)
plotnum(allNums(:,3))
subplot(3,5,12)
plotnum(allNums(:,4))
subplot(3,5,15)
plotnum(allNums(:,5))

sgtitle('Low-Rank Approximations of Data')

%% Functions
function [] = plotnum(numvec)
    imagesc(reshape(numvec, 28, 28))
end

function [images, labels] = mnist_parse(path_to_digits, path_to_labels)

% The function is curtesy of stackoverflow user rayryeng from Sept. 20,
% 2016. Link: https://stackoverflow.com/questions/39580926/how-do-i-load-in-the-mnist-digits-and-label-data-in-matlab

% Open files
fid1 = fopen(path_to_digits, 'r');

% The labels file
fid2 = fopen(path_to_labels, 'r');

% Read in magic numbers for both files
A = fread(fid1, 1, 'uint32');
magicNumber1 = swapbytes(uint32(A)); % Should be 2051
fprintf('Magic Number - Images: %d\n', magicNumber1);

A = fread(fid2, 1, 'uint32');
magicNumber2 = swapbytes(uint32(A)); % Should be 2049
fprintf('Magic Number - Labels: %d\n', magicNumber2);

% Read in total number of images
% Ensure that this number matches with the labels file
A = fread(fid1, 1, 'uint32');
totalImages = swapbytes(uint32(A));
A = fread(fid2, 1, 'uint32');
if totalImages ~= swapbytes(uint32(A))
    error('Total number of images read from images and labels files are not the same');
end
fprintf('Total number of images: %d\n', totalImages);

% Read in number of rows
A = fread(fid1, 1, 'uint32');
numRows = swapbytes(uint32(A));

% Read in number of columns
A = fread(fid1, 1, 'uint32');
numCols = swapbytes(uint32(A));

fprintf('Dimensions of each digit: %d x %d\n', numRows, numCols);

% For each image, store into an individual slice
images = zeros(numRows, numCols, totalImages, 'uint8');
for k = 1 : totalImages
    % Read in numRows*numCols pixels at a time
    A = fread(fid1, numRows*numCols, 'uint8');
    
    % Reshape so that it becomes a matrix
    % We are actually reading this in column major format
    % so we need to transpose this at the end
    images(:,:,k) = reshape(uint8(A), numCols, numRows).';
end

% Read in the labels
labels = fread(fid2, totalImages, 'uint8');

% Close the files
fclose(fid1);
fclose(fid2);

end