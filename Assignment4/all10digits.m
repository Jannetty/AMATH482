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

%% read in test data
[testimages, testlabels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
%Make matrix to hold all test num info
test_num_cols = length(testlabels);
allTest = zeros(num_rows, test_num_cols);
for i = 1:test_num_cols
    %reshape into col vector, store with each col = different image
    allTest(:,i) = reshape(testimages(:,:,i), num_rows, 1);
end

%% Make total test and train matrices

totalmat = allNums;
totallabels = trainlabels;
totaltestmat = allTest;
totaltestlabels = testlabels;

%% Find Singular Values
[U,S,V] = svd(totalmat, 'econ');
eigvals = diag(S);

%% Determine number of modes to incorperate
energyThreshold = .9;
normalized_eigvals = eigvals / sum(eigvals);
cumEnergy = cumsum(eigvals)/sum(eigvals);
energyIndex = find(cumEnergy >energyThreshold,1);
energyIndex = 344;

%% Project onto determined number of modes
U = U(:, 1:energyIndex);
sample = U' * totaltestmat;
training = U' * totalmat;
group = totallabels;

%% Classify using tree
tree = fitctree(training', totallabels);
class = predict(tree, sample');

%% Assess training accuracy of tree train data
trainpercentaccuracytree = (1 - cvloss(tree));

%% Assess accuracy of test tree
numwrong = sum(totaltestlabels ~= class);
tree_percent_accuracy = 100*(1 - (numwrong / (length(totaltestlabels))));

% %% Determine number of modes to incorperate for LDA (fewer than tree)
% energyThreshold = .5;
% normalized_eigvals = eigvals / sum(eigvals);
% cumEnergy = cumsum(eigvals)/sum(eigvals);
% energyIndex = find(cumEnergy >energyThreshold,1);

%% Project onto determined number of modes
U = U(:, 1:energyIndex);
sample = U' * totaltestmat;
training = U' * totalmat;
group = totallabels;
%% Classify using SVM
sample = sample / max(training(:));
training = training / max(training(:));
Mdl = fitcecoc(training',totallabels);

%% Assess accuracy of training SVM
trainaccuracy_percents_SVM = 1 - (loss(Mdl, training', totallabels));

%% Assess accuracy of test SVM
class = predict(Mdl,sample');
numwrong = sum(totaltestlabels ~= class);
SVM_percent_accuracy = 1 - (numwrong / (length(totaltestlabels)));


