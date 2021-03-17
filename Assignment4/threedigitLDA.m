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

%% Parse out first, second, and third numbers from training and test data
firstnum = 1;
secondnum = 2;
thirdnum = 3;

firstidx = find(trainlabels == firstnum);
firstlabels = trainlabels(firstidx);
firstmat = allNums(:, firstidx);
numfirst = length(firstlabels);

firsttestidx = find(testlabels == firstnum);
firsttestlabels = testlabels(firsttestidx);
firsttestmat = allTest(:, firsttestidx);
numfirsttest = length(firsttestidx);

secondidx = find(trainlabels == secondnum);
secondlabels = trainlabels(secondidx);
secondmat = allNums(:, secondidx);
numsecond = length(secondlabels);

secondtestidx = find(testlabels == secondnum);
secondtestlabels = testlabels(secondtestidx);
secondtestmat = allTest(:, secondtestidx);
numsecondtest = length(secondtestidx);

thirdidx = find(trainlabels == thirdnum);
thirdlabels = trainlabels(thirdidx);
thirdmat = allNums(:, thirdidx);
numthird = length(thirdlabels);

thirdtestidx = find(testlabels == thirdnum);
thirdtestlabels = testlabels(thirdtestidx);
thirdtestmat = allTest(:, thirdtestidx);
numthirdtest = length(thirdtestidx);

%% Make total test and train matrices

totalmat = [firstmat, secondmat, thirdmat];
totallabels = [firstlabels', secondlabels', thirdlabels'];
totaltestmat = [firsttestmat, secondtestmat, thirdtestmat];
totaltestlabels = [firsttestlabels', secondtestlabels', thirdtestlabels'];

%% Find Singular Values
[U,S,V] = svd(totalmat, 'econ');
eigvals = diag(S);

%% Determine number of modes to incorperate
energyThreshold = .9;
normalized_eigvals = eigvals / sum(eigvals);
cumEnergy = cumsum(eigvals)/sum(eigvals);
energyIndex = find(cumEnergy >energyThreshold,1);
energyIndex = 344; %this was determined from running SVD on all test data,
                   % comment out this line to capture 90% of energy for
                   % difference between these two digits specifically
%% Project onto determined number of modes
U = U(:, 1:energyIndex);
sample = U' * totaltestmat;
training = U' * totalmat;
group = totallabels;

%% Classify using LDA
[class, error] = classify(sample', training', group);
%% Assess accuracy of LDA test
train_percent_accuracy = 1 - error;
numwrong = sum(totaltestlabels ~= class');
percent_accuracy = 1 - (numwrong / (length(totaltestlabels)));



%% EXTRAS
%% Classify using tree
tree = fitctree(training', totallabels);
class = predict(tree, sample');

%% Assess accuracy of tree test
numwrong = sum(totaltestlabels ~= class');
percent_accuracy = 1 - (numwrong / (length(totaltestlabels)));

%% Classify using SVM
sample = sample / max(eigvals);
training = training / max(eigvals);
Mdl = fitcecoc(training',totallabels);
%%
class = predict(Mdl,sample');
numwrong = sum(totaltestlabels ~= class');
percent_accuracy = 1 - (numwrong / (length(totaltestlabels)));