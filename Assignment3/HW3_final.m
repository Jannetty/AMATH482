%% Ideal Case
clear all; close all;
%% Camera 1
load cam1_1.mat;
[height1_1 width1_1 rgb1_1 num_frames1_1] = size(vidFrames1_1);
% align timing with falling of the can
vidFrames1_1=vidFrames1_1(:,:,:,15:num_frames1_1);
[height1_1 width1_1 rgb1_1 num_frames1_1] = size(vidFrames1_1);
%search from half way through cols to 2/3 of wat through cols
colmin = floor(width1_1/2);
colmax = (width1_1 - floor(width1_1/3));
%look for white pixel
min_color_val = 245;
max_color_val = 255;
window_side_length = 20;
rowmin = 1;
rowmax = height1_1;
[x1loc, y1loc] = track_light(vidFrames1_1, height1_1, width1_1,...
    num_frames1_1, rowmin, rowmax, colmin, colmax, ...
    min_color_val, max_color_val, min_color_val, max_color_val,...
    min_color_val, max_color_val, window_side_length, false);

%plot_raw_data(x1loc, y1loc);

%% Camera 2
load cam2_1.mat;
[height2_1 width2_1 rgb2_1 num_frames2_1] = size(vidFrames2_1);
% align timing with falling of the can
vidFrames2_1=vidFrames2_1(:,:,:,25:num_frames2_1);
[height2_1 width2_1 rgb2_1 num_frames2_1] = size(vidFrames2_1);

rowmin2 = 1;
rowmax2 = height2_1;
%search only middle third of columns
colmin2 = floor(width2_1/3);
colmax2 = width2_1 - floor(width2_1/3);
%look for white pixel
min_color_val = 245;
max_color_val = 255;
window_side_length = 20;
[x2loc, y2loc] = track_light(vidFrames2_1, height2_1, width2_1,...
    num_frames2_1, rowmin2, rowmax2, colmin2,colmax2, ...
    min_color_val, max_color_val, min_color_val, max_color_val,...
    min_color_val, max_color_val, window_side_length, false);

%plot_raw_data(x2loc, y2loc)
%% Camera 3
%import data
load cam3_1.mat;
[height3_1 width3_1 rgb3_1 num_frames3_1] = size(vidFrames3_1);
% align timing with falling of the can
vidFrames3_1=vidFrames3_1(:,:,:,14:num_frames3_1);
[height3_1 width3_1 rgb3_1 num_frames3_1] = size(vidFrames3_1);

%For first search, search middle third of rows, middle third of cols
rowmin3 = floor(height3_1/3);
rowmax3 = height3_1 - floor(height3_1/3);
colmin3 = floor(width3_1/3);
colmax3 = width3_1 - floor(width3_1/3);
min_color_val = 235;
max_color_val = 255;
window_side_length = 20;
[x3loc, y3loc] = track_light(vidFrames3_1, height3_1, width3_1,...
    num_frames3_1, rowmin3, rowmax3, colmin3, colmax3,...
    min_color_val, max_color_val, min_color_val, max_color_val,...
    min_color_val, max_color_val, window_side_length, false);

%plot_raw_data(x3loc, y3loc)
%% make 6xtime Matrix
%matrix_width = min([num_frames1_1; num_frames2_1; num_frames3_1]);
matrix_width = 200;
case1_matrix = [x1loc(1:matrix_width);y1loc(1:matrix_width); ...
    x2loc(1:matrix_width); y2loc(1:matrix_width); x3loc(1:matrix_width)...
    ;y3loc(1:matrix_width)];
%% SVD
X = case1_matrix;
normalized = find_svd(X);

%% CASE 2
clear all; close all;

%% Camera 1
clear all; close all;
load cam1_2.mat;
[height1_2 width1_2 rgb1_2 num_frames1_2] = size(vidFrames1_2);
%align first trough
vidFrames1_2=vidFrames1_2(:,:,:,18:num_frames1_2);
[height1_2 width1_2 rgb1_2 num_frames1_2] = size(vidFrames1_2);

%search from half way through cols to 2/3 of wat through cols
colmin = floor(width1_2/2);
colmax = (width1_2 - floor(width1_2/3));
minc = 250;
maxc = 255;
rmin = minc;
rmax = maxc;
gmin = minc;
gmax = maxc;
bmin = minc;
bmax = maxc;


window_side_length = 40;
%look at bottom half of rows
rowmin = height1_2/2;
rowmax = height1_2;
[x1loc, y1loc] = track_light(vidFrames1_2, height1_2, width1_2,...
    num_frames1_2, rowmin, rowmax, colmin, colmax, ...
    rmin, rmax, gmin, gmax,...
    bmin, bmax, window_side_length, false);

%plot_raw_data(x1loc, y1loc)

%% Camera 2
load cam2_2.mat;
[height2_2 width2_2 rgb2_2 num_frames2_2] = size(vidFrames2_2);

rowmin2 = 1;
rowmax2 = height2_2;
%search only middle third of columns
colmin2 = floor(width2_2/3);
colmax2 = width2_2 - floor(width2_2/3);
%look for white pixel
minc = 240;
maxc = 255;
rmin = minc;
rmax = maxc;
gmin = minc;
gmax = maxc;
bmin = minc;
bmax = maxc;
window_side_length = 70;
[x2loc, y2loc] = track_light(vidFrames2_2, height2_2, width2_2,...
    num_frames2_2, rowmin2, rowmax2, colmin2,colmax2, ...
    rmin, rmax, gmin, gmax,...
    bmin, bmax, window_side_length, false);

%plot_raw_data(x2loc, y2loc)
%% Camera 3
load cam3_2.mat;
[height3_2 width3_2 rgb3_2 num_frames3_2] = size(vidFrames3_2);

%align first trough
vidFrames3_2=vidFrames3_2(:,:,:,20:num_frames3_2);
[height3_2 width3_2 rgb3_2 num_frames3_2] = size(vidFrames3_2);

%For first search, search middle third of rows, middle third of cols
rowmin3 = floor(height3_2/3);
rowmax3 = height3_2 - floor(height3_2/3);
colmin3 = floor(width3_2/3);
colmax3 = width3_2 - floor(width3_2/3);
%look for white pixeks
minc = 240;
maxc = 255;
rmin = minc;
rmax = maxc;
gmin = minc;
gmax = maxc;
bmin = minc;
bmax = maxc;
window_side_length = 70;
[x3loc, y3loc] = track_light(vidFrames3_2, height3_2, width3_2,...
    num_frames3_2, rowmin3, rowmax3, colmin3,colmax3, ...
    rmin, rmax, gmin, gmax,...
    bmin, bmax, window_side_length, false);

%plot_raw_data(x3loc, y3loc)
%% make 6xtime Matrix
%matrix_width = min([num_frames1_1; num_frames2_1; num_frames3_1]);
matrix_width = 200;
case2_matrix = [x1loc(1:matrix_width);y1loc(1:matrix_width); ...
    x2loc(1:matrix_width); y2loc(1:matrix_width); x3loc(1:matrix_width)...
    ;y3loc(1:matrix_width)];
%% SVD
X = case2_matrix;
normalized = find_svd(X);

%% CASE 3
clear all; close all;

%% Camera 1
clear all; close all;
load cam1_3.mat;
[height1_3 width1_3 rgb1_3 num_frames1_3] = size(vidFrames1_3);
%align first trough
vidFrames1_3=vidFrames1_3(:,:,:,9:num_frames1_3);
[height1_3 width1_3 rgb1_3 num_frames1_3] = size(vidFrames1_3);

%search from 1/3 way through cols to 1/2 of way through cols
colmin = floor(width1_3/3);
colmax = (width1_3 - floor(width1_3/2));

%look for pink pixel
rmin = 240;
rmax = 255;
gmin = 19;
gmax = 200;
bmin = 130;
bmax = 205;


window_side_length = 40;
%look at bottom half of rows
rowmin = height1_3/2;
rowmax = height1_3;
[x1loc, y1loc] = track_light(vidFrames1_3, height1_3, width1_3,...
    num_frames1_3, rowmin, rowmax, colmin, colmax, ...
    rmin, rmax, gmin, gmax,...
    bmin, bmax, window_side_length, false);

%plot_raw_data(x1loc, y1loc)

%% Camera 2
load cam2_3.mat;
[height2_3 width2_3 rgb2_3 num_frames2_3] = size(vidFrames2_3);

%search only bottom half of rows
rowmin2 = height2_3/2;
rowmax2 = height2_3;
%search only middle third of columns
colmin2 = floor(width2_3/3);
colmax2 = width2_3 - floor(width2_3/3);
%look for pink pixel
rmin = 240;
rmax = 255;
gmin = 19;
gmax = 200;
bmin = 130;
bmax = 205;
window_side_length = 40;
[x2loc, y2loc] = track_light(vidFrames2_3, height2_3, width2_3,...
    num_frames2_3, rowmin2, rowmax2, colmin2,colmax2, ...
    rmin, rmax, gmin, gmax,...
    bmin, bmax, window_side_length, false);

%plot_raw_data(x2loc, y2loc)
%% Camera 3
load cam3_3.mat;
[height3_3 width3_3 rgb3_3 num_frames3_3] = size(vidFrames3_3);

%align first trough
vidFrames3_3=vidFrames3_3(:,:,:,4:num_frames3_3);
[height3_3 width3_3 rgb3_3 num_frames3_3] = size(vidFrames3_3);

%For first search, search middle third of rows, middle third of cols
rowmin3 = floor(height3_3/3);
rowmax3 = height3_3 - floor(height3_3/3);
colmin3 = floor(width3_3/3);
colmax3 = width3_3 - floor(width3_3/3);
%look for pink pixel
rmin = 240;
rmax = 255;
gmin = 19;
gmax = 200;
bmin = 130;
window_side_length = 70;
[x3loc, y3loc] = track_light(vidFrames3_3, height3_3, width3_3,...
    num_frames3_3, rowmin3, rowmax3, colmin3,colmax3, ...
    rmin, rmax, gmin, gmax,...
    bmin, bmax, window_side_length, false);

%plot_raw_data(x3loc, y3loc)
%% make 6xtime Matrix
%matrix_width = min([num_frames1_1; num_frames2_1; num_frames3_1]);
matrix_width = 200;
case3_matrix = [x1loc(1:matrix_width);y1loc(1:matrix_width); ...
    x2loc(1:matrix_width); y2loc(1:matrix_width); x3loc(1:matrix_width)...
    ;y3loc(1:matrix_width)];
%% SVD
X = case3_matrix;
normalized = find_svd(X);

%% Case 4
close all;
clear all;
%% Camera 1
clear all; close all;
load cam1_4.mat;
[height1_4 width1_4 rgb1_4 num_frames1_4] = size(vidFrames1_4);

%search from 1/2 way through cols to 2/3 of way through cols
colmin = floor(width1_4/2);
colmax = (width1_4 - floor(width1_4/3));

%look for pink pixel
rmin = 240;
rmax = 255;
gmin = 19;
gmax = 200;
bmin = 130;
bmax = 205;


window_side_length = 40;
%look at bottom half of rows
rowmin = height1_4/2;
rowmax = height1_4;
[x1loc, y1loc] = track_light(vidFrames1_4, height1_4, width1_4,...
    num_frames1_4, rowmin, rowmax, colmin, colmax, ...
    rmin, rmax, gmin, gmax,...
    bmin, bmax, window_side_length, false);

%plot_raw_data(x1loc, y1loc)

%% Camera 2
load cam2_4.mat;
[height2_4 width2_4 rgb2_4 num_frames2_4] = size(vidFrames2_4);
%align first peak
vidFrames2_4=vidFrames2_4(:,:,:,10:num_frames2_4);
[height2_4 width2_4 rgb2_4 num_frames2_4] = size(vidFrames2_4);

%search only bottom half of rows
rowmin2 = height2_4/2;
rowmax2 = height2_4;
%search only middle third of columns
colmin2 = floor(width2_4/3);
colmax2 = width2_4 - floor(width2_4/3);
%look for pink pixel
rmin = 240;
rmax = 255;
gmin = 19;
gmax = 200;
bmin = 130;
bmax = 205;
window_side_length = 40;
[x2loc, y2loc] = track_light(vidFrames2_4, height2_4, width2_4,...
    num_frames2_4, rowmin2, rowmax2, colmin2,colmax2, ...
    rmin, rmax, gmin, gmax,...
    bmin, bmax, window_side_length, false);

%plot_raw_data(x2loc, y2loc)

%% Camera 3
load cam3_4.mat;
[height3_4 width3_4 rgb3_4 num_frames3_4] = size(vidFrames3_4);

%align first peak
vidFrames3_4=vidFrames3_4(:,:,:,4:num_frames3_4);
[height3_4 width3_4 rgb3_4 num_frames3_4] = size(vidFrames3_4);

%For first search, search middle third of rows, bottom half of cols
rowmin3 = floor(height3_4/3);
rowmax3 = height3_4 - floor(height3_4/3);
colmin3 = floor(width3_4/2);
colmax3 = width3_4;
% %look for white pixel
minc = 200;
maxc = 255;
rmin = minc;
rmax = maxc;
gmin = minc;
gmax = maxc;
bmin = minc;
bmax = maxc;
% rmin = 240;
% rmax = 255;
% gmin = 100;
% gmax = 200;
% bmin = 130;
% bmax = 205;
window_side_length = 70;
[x3loc, y3loc] = track_light(vidFrames3_4, height3_4, width3_4,...
    num_frames3_4, rowmin3, rowmax3, colmin3,colmax3, ...
    rmin, rmax, gmin, gmax,...
    bmin, bmax, window_side_length, false);

%plot_raw_data(x3loc, y3loc)

%% make 6xtime Matrix
%matrix_width = min([num_frames1_1; num_frames2_1; num_frames3_1]);
matrix_width = 200;
case4_matrix = [x1loc(1:matrix_width);y1loc(1:matrix_width); ...
    x2loc(1:matrix_width); y2loc(1:matrix_width); x3loc(1:matrix_width)...
    ;y3loc(1:matrix_width)];
%% SVD
X = case4_matrix;
normalized = find_svd(X);


%% Functions
function [pixel_color_in_range] = in_range(pixr, pixg, pixb, rmin, rmax,...
    gmin, gmax, bmin, bmax)
pixel_color_in_range = false;
if (pixr >= rmin) && (pixg >= gmin)&& (pixb >= bmin) && (pixr <= rmax)
    pixel_color_in_range = true;
end
end

function [] = plot_raw_data(xvec, yvec)
figure()
allvals = [xvec; yvec];
min_all = min(allvals);
min_y = min_all(1) - 10;
max_all = max(allvals);
max_y = max_all(1) + 10;
plot(1:length(xvec), xvec(:,1), "Linewidth", 1);
hold on;
plot(1:length(yvec), yvec(:,1), "Linewidth", 1);
set(gca, 'Ylim',[min_y, max_y], 'Fontsize', 16);
xlabel('Frame')
ylabel('Coordinate')
%legend()

legend('X Coordinate','Y Coordinate')
end

function [normalized_energy] = find_svd(X)
%% SVD
[m,n] = size(X); %data size
mn = mean(X,2); %row means
X = X - repmat(mn,1,n); %subtract mean

%u is principal directions, sigma is most important values, v is time
%series in each direction
[uhat,shat,v] = svd(X'/sqrt(n-1), 'econ'); %perform reduced svd


% Pull out eigen values
eigVals = diag(shat);


% Plot the eigen functions
figure;
subplot(1, 6, 1); plot(uhat(:,1));title('First Unit Basis Vector');
xlabel('Frame');
subplot(1, 6, 2); plot(uhat(:,2)); title('Second Unit Basis Vector');
xlabel('Frame');
subplot(1, 6, 3); plot(uhat(:,3)); title('Third Unit Basis Vector');
xlabel('Frame');
subplot(1, 6, 4); plot(uhat(:,4));title('Fourth Unit Basis Vector');
xlabel('Frame');
subplot(1, 6, 5); plot(uhat(:,5)); title('Fifth Unit Basis Vector');
xlabel('Frame');
subplot(1, 6, 6); plot(uhat(:,6)); title('Sixth Unit Basis Vector');
xlabel('Frame');


% Plot the amount of each mode in each dimension
figure;
subplot(2, 3, 1); bar(v(:,1));
title('Importance of Unit Vector 1 in each Input');
xticks(1:1:6)
ylabel('Dot Product Result');
xticklabels({'Cam1X','Cam1Y','Cam2X','Cam2Y','Cam3X','Cam3Y'})
subplot(2, 3, 2); bar(v(:,2)); 
title('Importance of Unit Vector 2 in each Input');
xticks(1:1:6)
xticklabels({'Cam1X','Cam1Y','Cam2X','Cam2Y','Cam3X','Cam3Y'})
ylabel('Dot Product Result');
subplot(2, 3, 3); bar(v(:,3)); 
title('Importance of Unit Vector 3 in each Input');
xticks(1:1:6)
xticklabels({'Cam1X','Cam1Y','Cam2X','Cam2Y','Cam3X','Cam3Y'})
ylabel('Dot Product Result');
subplot(2, 3, 4); bar(v(:,4));
title('Importance of Unit Vector 4 in each Input');
xticks(1:1:6)
xticklabels({'Cam1X','Cam1Y','Cam2X','Cam2Y','Cam3X','Cam3Y'})
ylabel('Dot Product Result');
subplot(2, 3, 5); bar(v(:,5)); 
title('Importance of Unit Vector 5 in each Input');
xticks(1:1:6)
xticklabels({'Cam1X','Cam1Y','Cam2X','Cam2Y','Cam3X','Cam3Y'})
ylabel('Dot Product Result');
subplot(2, 3, 6); bar(v(:,6)); 
title('Importance of Unit Vector 6 in each Input');
xticks(0:1:6)
xticklabels({'', 'Cam1X','Cam1Y','Cam2X','Cam2Y','Cam3X','Cam3Y', ''})
ylabel('Dot Product Result');


normalized_energy = eigVals/sum(eigVals);

% The cumulative energy content for the m'th eigenvector is the 
%sum of the energy content across eigenvalues 1:m
running_sum = 0;
num_modes_needed = 0;
i = 1;
while running_sum <= .80;
    running_sum = running_sum + normalized_energy(i);
    num_modes_needed = i;
    i = i+1;
end
fprintf('Number Modes Needed =%d \n', num_modes_needed)


bestrank1 = uhat(:,1)*eigVals(1)*v(:,1)';

figure;
subplot(3,1,1); plot(X'/sqrt(n-1)); title('Pre-Processed Input Data');
legend('cam1 X', 'cam1 Y', 'cam2 X', 'cam2 Y', 'cam3 X', 'cam3 Y')
bestrank = bestrank1;
i = 1;
while i < num_modes_needed
    i = i + 1;
    bestrank = bestrank + (uhat(:,2)*eigVals(i)*v(:,1)');
end
subplot(3, 1, 2); plot(bestrank1); title('Best Rank 1 Approximation of Data');
legend('cam1 X', 'cam1 Y', 'cam2 X', 'cam2 Y', 'cam3 X', 'cam3 Y')
subplot(3, 1, 3); plot(bestrank); title(sprintf('Best Rank %d Approximation of data', num_modes_needed));
xlabel('Frame');
legend('cam1 X', 'cam1 Y', 'cam2 X', 'cam2 Y', 'cam3 X', 'cam3 Y')

figure()
%eigenfunctions scaled to sigma
min_y = min(uhat(:,1)*eigVals(1));
max_y = max(uhat(:,1)*eigVals(1));
subplot(6, 1, 1); plot(uhat(:,1)*eigVals(1));title('First Unit Vector * First Sigma');
set(gca, 'Ylim',[min_y, max_y], 'Fontsize', 10);
subplot(6, 1, 2); plot(uhat(:,2)*eigVals(2)); title('Second Unit Vector * Second Sigma');
set(gca, 'Ylim',[min_y, max_y], 'Fontsize', 10); 
subplot(6, 1, 3); plot(uhat(:,3)*eigVals(3)); title('Third Unit Vector * Third Sigma');
set(gca, 'Ylim',[min_y, max_y], 'Fontsize', 10); 
subplot(6, 1, 4); plot(uhat(:,4)*eigVals(4));title('Fourth Unit Vector * Fourth Sigma');
set(gca, 'Ylim',[min_y, max_y], 'Fontsize', 10); 
subplot(6, 1, 5); plot(uhat(:,5)*eigVals(5)); title('Fifth Unit Vector * Fifth Sigma');
set(gca, 'Ylim',[min_y, max_y], 'Fontsize', 10);
subplot(6, 1, 6); plot(uhat(:,6)*eigVals(6)); title('Sixth Unit Vector * Sixth Sigma');
set(gca, 'Ylim',[min_y, max_y], 'Fontsize', 10);
xlabel('Frame');
end

function [xloc, yloc] = track_light(vidframes, height, width,...
    num_frames, rowmin, rowmax, colmin, colmax, rmin, rmax, gmin, gmax, ...
    bmin, bmax, window_side_length, parameter_optimize_mode)

%camera location vectors
xloc = zeros(num_frames);
yloc = zeros(num_frames);

%iterate through each frame
for frame = 1 : num_frames
    thisframe = vidframes(:,:,:,frame);
    %only find one location per frame
    location_found = false;
    
    %look at rows within selected range (window centered on prev location)
    for row = rowmin:rowmax
        %look at cols within selected range 
        %(window centered on prev location)
        for col = colmin:colmax
            %if the location hasn't already been found and the point is in
            %the window and close to white
            rpix = thisframe(row,col,1);
            gpix = thisframe(row,col,2);
            bpix = thisframe(row,col,3);
            if ((location_found) == false &&...
                    in_range(rpix, gpix, bpix, rmin, rmax, gmin, gmax,...
                    bmin,bmax) == true)
                %set this boolean to true (so take first white point found
                %in window to be the location)
                location_found = true;
                %save x and y locations in vectors
                xloc(frame) = col;
                yloc(frame) = row;
                %set window dimensions for next frame
                rowmin = row - window_side_length;
                rowmax = row + window_side_length;
                colmin = col - window_side_length;
                colmax = col + window_side_length;
                if rowmin < 1
                    rowmin = 1;
                end
                if colmin < 1
                    colmin = 1;
                end
                if rowmax > height
                    rowmax = height;
                end
                if colmax > width
                    colmax = width;
                end
            end
        end
    end
    
    %If optimize_parameters is true, show me the video of the overlayed
    %window in its proper location in each frame and tell me if any frames
    %didn't find a point in the window:
    if parameter_optimize_mode == true
        for row = 1:height
            for col = 1:width
                if (row > rowmin && (row < rowmax)...
                        && (col > colmin)) && (col < colmax)
                    thisframe(row,col,1) = 255;
                    thisframe(row,col,2) = 0;
                    thisframe(row,col,3) = 0;
                end
            end
        end
        if location_found == false
            fprintf("No location found in frame %d\n", frame)
        end
        imshow(thisframe); drawnow
    end
    
    %if no location found, set this location to be the same as the location
    %in the last frame
    if location_found == false
        xloc(frame) = xloc(frame-1);
        yloc(frame) = yloc(frame-1);
    end
end

%function end
end