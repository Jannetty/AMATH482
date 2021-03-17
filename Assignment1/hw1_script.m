%% Preparation and setup
%clean workspace
clear all; close all; clc
load subdata.mat %Imports data as 262144x49 (space by time) matrix

L = 10; %spatial domain (spatial resolution)
n = 64; %Fourier modes (spectral resolution)
x2 = linspace(-L,L,n+1); %x is from -10 to 10, generates lineraly spaced
% vector of n+1 points from -10 to 10
x = x2(1:n); %periodic boundary means first point is same as last
y =x; z = x;

%scale my data's domain to 2 pi periodic, define fourier wave numbers
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1];
ks = fftshift(k); %used for plotting in frequency domain

[X, Y, Z] = meshgrid(x,y,z); %time domain meshgrid, 3 3dimensional vectors
[Kx, Ky, Kz] = meshgrid(ks,ks,ks); %frequency domain meshgrid fftshifted 
[UKx, UKy, UKz] = meshgrid(k,k,k); %frequency domain meshgrid not shifted

%% Initial Data Investigation
%Create vectors for storing indicies of max magnitude point at each time
unfiltered_xvec = zeros(49,1);
unfiltered_yvec = zeros(49,1);
unfiltered_zvec = zeros(49,1);
for j=1:49 %for data at each time point (slice)
    %takes data from time point, reshape into 64x64x64 matrix
    Un(:,:,:) = reshape(subdata(:,j),n,n,n); 
    
    %find point with max magnitude at this time point
    [Mn, idx]=max(abs(Un(:)));
    %Get coordinates of this spatial domain max point
    % Matlab is column major, so int2sub returns y then x then z
    [yind, xind, zind] = ind2sub(size(Un), idx);
    unfiltered_xvec(j) = X(xind, xind, xind);
    unfiltered_yvec(j) = Y(yind, yind, yind);
    unfiltered_zvec(j) = Z(zind, zind, zind);
    
    %Plot all Unfiltered Data
    figure(1)
    isosurface(X,Y,Z,abs(Un)/Mn,0.7)
    title('All Unfiltered Submarine Data')
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    axis([-12 12 -12 12 -12 12]), grid on, drawnow
end

%Plot Max Point at each time point
figure(2)
plot3(unfiltered_xvec(),unfiltered_yvec(),unfiltered_zvec(), '-o',...
    'Color','r','MarkerSize',10,'MarkerFaceColor','cyan')
title('Unfiltered Submarine Location at Each Time Point')
xlabel('X Coordinate')
ylabel('Y Coordinate')
zlabel('Z Coordinate')
axis([-12 12 -12 12 -12 12]), grid on, drawnow

%% Averaging to find central frequencies in each dimension
Uave = zeros(64, 64, 64); %matrix for summing coefficients at each time
for j=1:49
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    Utn(:,:,:) = fftn(Un);% noisy data in frquency domain
    Uave = Uave + Utn; % add frequencies for each time point
end
Uave = Uave/49; % find average coefficient of each wavenumber
Mave=max(abs(Uave),[],'all'); %max coefficient

%Plot Max Fourier Coefficient Point- Coordinates of this point are central 
% frequencies in each dimension
figure(3)
%scale to max value so all magnitures between 0 and 1
isosurface(Kx,Ky,Kz, fftshift(abs(Uave))/Mave, .9)
title('Maximum Amplitude Point in Frequency Domain')
xlabel('X')
ylabel('Y')
zlabel('Z')

% maxvalue and its index for the averaged signal over all time slices
[mxv, idx] = max((abs(Uave(:))));
% Retrieve indeces for each cartesian dimension; yields index in each
%dimension's 3D k array corresponding to primary K number in that dimension
[firstind, secondind, thirdind] = ind2sub(size(Uave), idx);
xfreq = UKx(firstind, secondind, thirdind);
yfreq = UKy(firstind, secondind, thirdind);
zfreq = UKz(firstind, secondind, thirdind);



%% Make Filter
%a = width of filter, 1/a is variance of gausian filter:
% smaller a = wider, bigger = narrower,
a = .5;
filter = (exp(-a* ((UKx-xfreq) .^2))).* (exp(-a* ((UKy-yfreq) .^2))).* ...
    (exp(-a* ((UKz-zfreq) .^2)));

%Plot filter in frequency domain to make sure it is in the correct place
plotfilter = fftshift(filter);
figure(4)
isosurface(Kx,Ky,Kz,abs(plotfilter),.7)
title('Gaussian Filter Size and Location')
xlabel('X')
ylabel('Y')
zlabel('Z')


%% Apply filter

%Create vectors for storing indicies of max magnitude point at each time
%in spacial domain post filtering
xvec = zeros(49,1);
yvec = zeros(49,1);
zvec = zeros(49,1);

for j=1:49 %for data at each time point (slice)
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    
    Utn(:,:,:) = fftn(Un);%noisy data in frquency domain
    clean = Utn.*filter; %apply filter to data at this time point
    
    unf = ifftn(clean); %Move clean data back into time domain
    
    %find point with max magniture in filtered time domain at this time point
    [Mn, idx]=max(abs(unf(:)));
    % Get coordinates of this spatial domain max point (coordinates of sub)
    % Matlab is column major, so int2sub returns y then x then z
    [yind, xind, zind] = ind2sub(size(Un), idx);
    xvec(j) = X(xind, xind, xind);
    yvec(j) = Y(yind, yind, yind);
    zvec(j) = Z(zind, zind, zind);
    
    %Plot of all Filtered Submarine Data in Space Domain
    figure(5)
    isosurface(X,Y,Z,abs(unf)/Mn,0.9)
    title('All Filtered Submarine Data')
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    axis([-12 12 -12 12 -12 12]), grid on, drawnow
end

%% Plot Submarine Course
%Plots of max value at each time point
figure(6)
plot3(xvec(),yvec(),zvec(), '-o','Color','r','MarkerSize',10,...
    'MarkerFaceColor','cyan')
title('Filtered Submarine Location at Each Time Point')
xlabel('X Coordinate')
ylabel('Y Coordinate')
zlabel('Z Coordinate')
axis([-12 12 -12 12 -12 12]), grid on, drawnow

figure(7)
plot(xvec(),yvec(),'-o','Color','r','MarkerSize',10,...
    'MarkerFaceColor','cyan')
title('X and Y Submarine Location at Each Time Point')
xlabel('X Coordinate')
ylabel('Y Coordinate')
axis([-12 12 -12 12 -12 12]), grid on, drawnow

%Create character array of time labels
start = fix(now);
increment = 30/(60*24);% 30-Minute Increments
time_vct = 0:increment:(1);% Vector
time_str = datestr(start + time_vct, 'HH:MM:SS');% Time Strings
time_label_array = [time_str(1:49,:)];
time_label_array;

figure(8)
plot(xvec(),zvec(),'-o','Color','r','MarkerSize',10,'MarkerFaceColor',...
    'cyan')
title('X and Z Submarine Location at Each Time Point')
xlabel('X Coordinate')
ylabel('Z Coordinate')
%Apply time labels to each data point
%dx = 0.5; dz = 0.1; % displacement so the text isn't over data points
%text(xvec()+dx, zvec()+dz, c);
axis([-12 12 -12 12 -12 12]), grid on, drawnow