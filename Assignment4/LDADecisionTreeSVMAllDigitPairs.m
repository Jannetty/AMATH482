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

%% Iterate through all the digits
accuracy_labels = ["0,1", "0,2", "0,3", "0,4", "0,5", "0,6", "0,7", "0,8",...
    "0,9", "1,2", "1,3", "1,4", "1,5", "1,6", "1,7", "1,8", "1,9", "2,3", ...
    "2,4", "2,5", "2,6", "2,7", "2,8", "2,9", "3,4", "3,5", "3,6", "3,7", ...
    "3,8", "3,9", "4,5", "4,6", "4,7", "4,8", "4,9", "5,6", "5,7", "5,8", ...
    "5,9", "6,7", "6,8", "6,9", "7,8", "7,9", "8,9"];

trainaccuracy_percents = zeros(length(accuracy_labels), 1);
trainaccuracy_percents_tree = zeros(length(accuracy_labels), 1);
trainaccuracy_percents_svm = zeros(length(accuracy_labels), 1);
testaccuracy_percents = zeros(length(accuracy_labels), 1);
testaccuracy_percents_tree = zeros(length(accuracy_labels), 1);
testaccuracy_percents_SVM = zeros(length(accuracy_labels), 1);
modes_used = zeros(length(accuracy_labels), 1);
accuracy_percent_index = 1;

%% Run on all number combinations
for i = 0:1:8
    for j = i+1:9
        %seperate out each type of number
        firstidx = find(trainlabels == i);
        firstlabels = trainlabels(firstidx);
        firstmat = allNums(:, firstidx);
        numfirst = length(firstlabels);
        
        firsttestidx = find(testlabels == i);
        firsttestlabels = testlabels(firsttestidx);
        firsttestmat = allTest(:, firsttestidx);
        numfirsttest = length(firsttestidx);
        
        secondidx = find(trainlabels ==j);
        secondlabels = trainlabels(secondidx);
        secondmat = allNums(:, secondidx);
        numsecond = length(secondlabels);
        
        secondtestidx = find(testlabels == j);
        secondtestlabels = testlabels(secondtestidx);
        secondtestmat = allTest(:, secondtestidx);
        numsecondtest = length(secondtestidx);
        
        totalmat = [firstmat, secondmat];
        totallabels = [firstlabels', secondlabels'];
        
        totaltestmat = [firsttestmat, secondtestmat];
        totaltestlabels = [firsttestlabels', secondtestlabels'];
        
        [trainaccuracy_percents(accuracy_percent_index), ...
            trainaccuracy_percents_tree(accuracy_percent_index), ...
            trainaccuracy_percents_svm(accuracy_percent_index), ...
            testaccuracy_percents(accuracy_percent_index),...
            testaccuracy_percents_tree(accuracy_percent_index),...
            testaccuracy_percents_SVM(accuracy_percent_index),...
            modes_used(accuracy_percent_index)] = ...
            twodigitLDA(totalmat, totallabels, numfirst, numsecond,...
            totaltestmat, totaltestlabels,numfirsttest, numsecondtest, ...
            picheight, picwidth, false);
        
        accuracy_percent_index = accuracy_percent_index+1;
    end
end

%% Plot percents train LDA
figure()
plot (trainaccuracy_percents, '-o');
set(gca,'xaxisLocation','top')
xticks(1:length(accuracy_labels));
xticklabels(accuracy_labels);
hold on
%% 
%trainaccuracy_percents_tree = trainaccuracy_percents_tree* 100;
plot (trainaccuracy_percents_tree, '-o');
set(gca,'xaxisLocation','top')
xticks(1:length(accuracy_labels));
xticklabels(accuracy_labels);
hold on
%trainaccuracy_percents_svm = trainaccuracy_percents_svm * 100;
plot (trainaccuracy_percents_svm, '-o');
set(gca,'xaxisLocation','top')
xticks(1:length(accuracy_labels));
xticklabels(accuracy_labels);

ylabel('Percent Accuracy')
title("Training Data Accuracy")
legend('LDA','Decision Tree', 'SVM')

%% Plot percents test LDA
 testaccuracy_percents = testaccuracy_percents * 100;
 testaccuracy_percents_tree = testaccuracy_percents_tree * 100;
 testaccuracy_percents_SVM = testaccuracy_percents_SVM * 100;

figure()
plot (testaccuracy_percents, '-o');
set(gca,'xaxisLocation','top')
xticks(1:length(accuracy_labels));
xticklabels(accuracy_labels);
hold on
%worst 5 and 8, best 6 and 7

% Plot percents test Tree
plot (testaccuracy_percents_tree, '-o');
set(gca,'xaxisLocation','top')
xticks(1:length(accuracy_labels));
xticklabels(accuracy_labels);
hold on
%worst 4 and 9, best 0 and 1
% plot percents test SVM

plot (testaccuracy_percents_SVM, '-o');
set(gca,'xaxisLocation','top')
xticks(1:length(accuracy_labels));
xticklabels(accuracy_labels);
title("Test Data Accuracy")
legend('LDA','Decision Tree', 'SVM')
ylim([88,100])
ylabel('Percent Accuracy')
%% Functions

function [trainpercentaccuracylda, trainpercentaccuracytree,...
    trainpercentaccuracysvm, testpercentaccuracylda,...
    testpercentaccuracytree, testpercentaccuracysvm, modesused] = twodigitLDA(totalmat,...
    totallabels, numfirst, numsecond, testmat, testlabelmat, numfirsttest, numsecondtest,...
    picheight, picwidth, plot)

trainpercentaccuracylda = 0;
trainpercentaccuracytree = 0;
trainpercentaccuracysvm = 0;

testpercentaccuracylda = 0;
testpercentaccuracytree = 0;
testpercentaccuracysvm = 0;

% SVD on this new matrix
[U,S,V] = svd(totalmat, 'econ');
eigvals = diag(S);


%% Plot first four principal components
if plot == true
    for k = 1:4
        subplot(2,2,k)
        ut1 = reshape(U(:,k),picheight,picwidth);
        ut2 = rescale(ut1);
        imagesc(ut2)
    end
end
%% Plot singular values
if plot == true
    figure()
    subplot(2,1,1)
    plot(diag(S),'ko','Linewidth',2)
    set(gca,'Fontsize',16,'Xlim',[0 80])
    subplot(2,1,2)
    semilogy(diag(S),'ko','Linewidth',2)
    set(gca,'Fontsize',16,'Xlim',[0 80])
end
%% Plot right singular vectors
if plot == true
    figure()
    for k = 1:3
        subplot(3,2,2*k-1)
        plot(1:60,V(1:60,k),'ko-')
        subplot(3,2,2*k)
        plot(1:60,V(numfirst:(numfirst+59),k),'ko-')
    end
    subplot(3,2,1), set(gca,'Fontsize',12), title(num2str(totallabels(1)))
    subplot(3,2,2), set(gca,'Fontsize',12), title(num2str(totallabels(numfirst+1)))
    subplot(3,2,3), set(gca,'Fontsize',12)
    subplot(3,2,4), set(gca,'Fontsize',12)
    subplot(3,2,5), set(gca,'Fontsize',12)
    subplot(3,2,6), set(gca,'Fontsize',12)
end


%% Determine number of modes to incorperate
energyThreshold = .9;
normalized_eigvals = eigvals / sum(eigvals);
cumEnergy = cumsum(eigvals)/sum(eigvals);
energyIndex = find(cumEnergy >energyThreshold,1);
if plot == true
    figure()
    subplot(1,2,1), plot(normalized_eigvals, '.')
    %energySV = cumEnergy(energyIndex);
    subplot(1,2,2), plot(cumEnergy, '.')
end
energyIndex = 344; %this was determined from running SVD on all test data,
                   % comment out this line to capture 90% of energy for
                   % difference between these two digits specifically
modesused = energyIndex;
%% Project onto PCA modes
feature = energyIndex;
nf = numfirst;
ns = numsecond;
numbers = S*V'; % projection onto principal components: X = USV' --> U'X = SV'
first = numbers(1:feature,1:nf); %Projection of first numbers
second = numbers(1:feature,nf+1:nf+ns); %Projection of second numbers

%% Calculate scatter matrices

mf = mean(first,2);
md = mean(second,2);

Sw = 0; % within class variances
for k = 1:nf
    Sw = Sw + (first(:,k) - mf)*(first(:,k) - mf)';
end
for k = 1:ns
    Sw =  Sw + (second(:,k) - md)*(second(:,k) - md)';
end

Sb = (mf-md)*(mf-md)'; % between class

%% Find the best projection line

[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

%% Project onto w

vfirst = w'*first;
vsecond = w'*second;

%% Make first below the threshold

if mean(vfirst) > mean(vsecond)
    w = -w;
    vfirst = -vfirst;
    vsecond = -vsecond;
end

%% Plot first/second projections (not for function)
if plot == true
    figure()
    plot(vfirst,zeros(nf),'ob','Linewidth',2)
    hold on
    plot(vsecond,ones(ns),'dr','Linewidth',2)
    ylim([0 1.2])
end
%% Find the threshold value

sortfirst = sort(vfirst);
sortsecond = sort(vsecond);

t1 = length(sortfirst);
t2 = 1;
while sortfirst(t1) > sortsecond(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sortfirst(t1) + sortsecond(t2))/2;

%% Plot histogram of results

minxval = min([sortfirst, sortsecond]);
maxxval = max([sortfirst, sortsecond]);
if plot == true
    figure(5)
    subplot(2,1,1)
    histogram(sortfirst,30); hold on, plot([threshold threshold], [0 1600],'r')
    set(gca,'Fontsize',14)
    xlim([minxval maxxval]);
    title(num2str(totallabels(1)))
    subplot(2,1,2)
    histogram(sortsecond,30); hold on, plot([threshold threshold], [0 1600],'r')
    set(gca,'Fontsize',14)
    xlim([minxval maxxval]);
    title(num2str(totallabels(numfirst+1)))
end

%% Calculate training data percent accuracy
wrong_firsts = sum(sortfirst > threshold);
wrong_seconds = sum(sortsecond < threshold);
wrongtotal = wrong_firsts+wrong_seconds;
trainpercentaccuracylda = (1 - wrongtotal/(nf+ns)) * 100;

%% run test data
modesofU = U(:, 1:energyIndex);
svdprojection = modesofU' * testmat; %SVD projection
pval = w' * svdprojection; %LDA projection
resvecs = (pval > threshold);
%first below threshold
% 0 if first num, 1 if second num
label2 = totallabels(numfirst+1);
label2idx = find(testlabelmat == label2);
hiddenlabels = zeros(length(testlabelmat), 1);
hiddenlabels(label2idx) = 1;
errnum = sum(abs(resvecs' - hiddenlabels));
testpercentaccuracylda = (1 - (errnum/(numfirsttest + numsecondtest)));

% %% Classify using Tree
training = [first, second]; %these are training data projected onto correct number of principal components
tree = fitctree(training', totallabels);
class = predict(tree, svdprojection');
%% Assess Train accuracy of Tree
trainpercentaccuracytree = (1 - cvloss(tree));
%% Assess Test accuracy of Tree
numwrong = sum(testlabelmat ~= class');
testpercentaccuracytree = 1 - (numwrong / (length(testlabelmat)));

%% Classify using SVM
sample = svdprojection / max(training(:));
training = training / max(training(:));
Mdl = fitcsvm(training',totallabels);
%% Assess Train accuracy of SVM
trainpercentaccuracysvm = 1 - (loss(Mdl, training', totallabels));
%% Assess Test accuracy of SVM
class = predict(Mdl,sample');
numwrong = sum(testlabelmat ~= class');
testpercentaccuracysvm = 1 - (numwrong / (length(testlabelmat)));

end