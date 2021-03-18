%% Monte Carlo
clear all; close all;
%% Read Video
mcv = VideoReader('monte_carlo.mov');
mcduration = mcv.Duration;
mcframerate = mcv.FrameRate;
mcnumframes = mcv.NumFrames;
mcframeheight = mcv.Height;
mcframewidth = mcv.Width;

%% Make Matrix to hold all frames, each col = frame, each row = timepoint
mcmat = zeros(mcframeheight*mcframewidth, mcnumframes);

%% Read All Frames
for i = 1:mcnumframes
    frame = readFrame(mcv);
    frame = rgb2gray(frame);
    %whos frame
    %imshow(frame); drawnow
    mcmat(:,i) = reshape(frame, mcframeheight*mcframewidth, 1);
end
%% Create Matricies X1 amd X2
X = mcmat;
X1 = X(:, 1:end-1);
X2 = X(:, 2:end);

%% Step 1, compute SVD of X1
[U, Sigma, V] = svd(X1, 'econ');
%r = mcnumframes-1; %start setting rank to number of frames in X1, full svd
r = 2;
Ur = U(:, 1:r);
Sigmar = Sigma(1:r, 1:r);
Vr = V(:, 1:r);

%% Step 2, Project A onto the POD modes of U 
%(get leading r eigenvalues and eigenvectors of A)
Atilde = Ur'*X2*Vr/Sigmar;

%% Step 3, Compute spectral decomposition of Atilde
[W, Lambda] = eig(Atilde);

%% Step 4, High-dimensional DMD modes Phi are reconstructed using eigenvectors
% W of reduced system and time-shifted snapshot matrix X2
Phi = X2*(Vr/Sigmar)*W;
alpha1 = Sigmar*Vr(1,:)';
b = (W*Lambda)\alpha1; %initial condition?

%% Define t vector and omega
dt = 1;
mu = diag(Lambda);
omega = log(mu)/dt;
t = 0:dt:mcnumframes;

%% plot absolute values of omegas
figure()
semilogx(abs(real(omega))*dt,abs(imag(omega))*dt,'r.','Markersize',15)
xlabel('abs(Real(\omega))')
ylabel('abs(Imaginary(\omega))')
set(gca,'FontSize',16)
xline(0);
yline(0);
title('Monte Carlo |Omega| Values Low Rank Method');
%% find umodes
u_modes = zeros(length(b), mcnumframes);
for iter = 1:mcnumframes
    u_modes(:,iter) = b.*exp(omega*t(iter));
end
%% Find DMD Solution
u_dmd = Phi*u_modes;

%% Watch Background Video
figure()
for i = 1:mcnumframes
    frame = reshape(u_dmd(:,i), mcframeheight, mcframewidth);
    imshow(uint8(frame)); drawnow
    disp(i);
end

%% Watch Foreground Video
foreground = X - abs(u_dmd);
brightness_to_add = abs(min(foreground(:)));
foreground = foreground + brightness_to_add;
figure()
for i = 1:mcnumframes
    frame = reshape(foreground(:,i), mcframeheight, mcframewidth);
    imshow(uint8(frame)); drawnow
    disp(i);
end

%% Make Figures
figure()
subplot(3,3,1)
frame = reshape(X(:,1), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
set(gca,'xaxisLocation','top')
ylabel("Original")
xlabel("Frame 1")

subplot(3,3,2)
frame = reshape(X(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
set(gca,'xaxisLocation','top')
xlabel("Frame 200")

subplot(3,3,3)
frame = reshape(X(:,250), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
set(gca,'xaxisLocation','top')
xlabel("Frame 250")

subplot(3,3,4)
frame = reshape(u_dmd(:,1), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
set(gca,'xaxisLocation','top')
ylabel("Background")

subplot(3,3,5)
frame = reshape(u_dmd(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow


subplot(3,3,6)
frame = reshape(u_dmd(:,250), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow

subplot(3,3,7)
frame = reshape(foreground(:,1), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
set(gca,'xaxisLocation','top')
ylabel("Foreground")

subplot(3,3,8)
frame = reshape(foreground(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow


subplot(3,3,9)
frame = reshape(foreground(:,250), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow


sgtitle('Monte Carlo Low-Rank Subtraction Results') 

%% Make big figure for Monte Carlo
figure()
subplot(1,3,1)
frame = reshape(X(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
title("Original")
xlabel("Frame 200")

subplot(1,3,2)
frame = reshape(u_dmd(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
title("Background")
xlabel("Frame 200")

subplot(1,3,3)
frame = reshape(foreground(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
title("Foreground")
xlabel("Frame 200")

sgtitle('Monte Carlo Low-Rank Subtraction Method') 

%% Skier
clear all; close all;
%% Read Video
mcv = VideoReader('ski_drop.mov');
mcduration = mcv.Duration;
mcframerate = mcv.FrameRate;
mcnumframes = mcv.NumFrames;
mcframeheight = mcv.Height;
mcframewidth = mcv.Width;

%% Make Matrix to hold all frames, each col = frame, each row = timepoint
mcmat = zeros(mcframeheight*mcframewidth, mcnumframes);

%% Read All Frames
for i = 1:mcnumframes
    frame = readFrame(mcv);
    frame = rgb2gray(frame);
    %whos frame
    %imshow(frame); drawnow
    mcmat(:,i) = reshape(frame, mcframeheight*mcframewidth, 1);
end
%% Create Matricies X_1^{M-1} amd X_2^M
X = mcmat;
X1 = X(:, 1:end-1);
X2 = X(:, 2:end);

%% Step 1, compute SVD of X1
[U, Sigma, V] = svd(X1, 'econ');
%r = mcnumframes-1; %start setting rank to number of frames in X1, full svd
r = 8;
Ur = U(:, 1:r);
Sigmar = Sigma(1:r, 1:r);
Vr = V(:, 1:r);

%% Step 2, Project A onto the POD modes of U (leading r eigenvalues and eigenvectors of A)
Atilde = Ur'*X2*Vr/Sigmar;

%% Step 3, Compute spectral decomposition of Atilde
[W, Lambda] = eig(Atilde);

%% Step 4, High-dimensional DMD modes Phi are reconstructed using eigenvectors
% W of reduced system and time-shifted snapshot matrix X2
Phi = X2*(Vr/Sigmar)*W;
alpha1 = Sigmar*Vr(1,:)';
b = (W*Lambda)\alpha1; %initial condition?

%% Define t vector and omega
dt = 1;
mu = diag(Lambda);
omega = log(mu)/(dt);
t = 1:dt:mcnumframes;

%% plot absolute values of omegas
figure()
semilogx(abs(real(omega))*dt,abs(imag(omega))*dt,'r.','Markersize',15)
xlabel('abs(Real(\omega))')
ylabel('abs(Imaginary(\omega))')
set(gca,'FontSize',16)
xline(0);
yline(0);
title('Skier |Omega| Values Low Rank Method');
%% find umodes
u_modes = zeros(length(b), mcnumframes);
for iter = 1:mcnumframes
    u_modes(:,iter) = b.*exp(omega*t(iter));
end
%% Find DMD Solution
u_dmd = Phi*u_modes;

%% Watch Background Video
figure()
for i = 1:mcnumframes
    frame = reshape(u_dmd(:,i), mcframeheight, mcframewidth);
    imshow(uint8(frame)); drawnow
    disp(i);
end

%% Watch Foreground Video
foreground = X - abs(u_dmd);
brightness_to_add = abs(min(foreground(:)));
foreground = foreground + brightness_to_add;

figure()
for i = 1:mcnumframes
    frame = reshape(foreground(:,i), mcframeheight, mcframewidth);
    imshow(uint8(frame)); drawnow
    disp(i);
end

%% Make Figures
figure()
subplot(3,3,1)
frame = reshape(X(:,1), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
set(gca,'xaxisLocation','top')
ylabel("Original")
xlabel("Frame 1")

subplot(3,3,2)
frame = reshape(X(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
set(gca,'xaxisLocation','top')
xlabel("Frame 200")

subplot(3,3,3)
frame = reshape(X(:,250), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
set(gca,'xaxisLocation','top')
xlabel("Frame 250")

subplot(3,3,4)
frame = reshape(u_dmd(:,1), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
set(gca,'xaxisLocation','top')
ylabel("Background")

subplot(3,3,5)
frame = reshape(u_dmd(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow


subplot(3,3,6)
frame = reshape(u_dmd(:,250), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow

subplot(3,3,7)
frame = reshape(foreground(:,1), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
set(gca,'xaxisLocation','top')
ylabel("Foreground")

subplot(3,3,8)
frame = reshape(foreground(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow


subplot(3,3,9)
frame = reshape(foreground(:,250), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow


sgtitle('Skier Low-Rank Subtraction Results') 

%% Make big figure for Skier
figure()
subplot(1,3,1)
frame = reshape(X(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
title("Original")
xlabel("Frame 200")

subplot(1,3,2)
frame = reshape(u_dmd(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
title("Background")
xlabel("Frame 200")

subplot(1,3,3)
frame = reshape(foreground(:,200), mcframeheight,mcframewidth);
imshow(uint8(frame)); drawnow
title("Foreground")
xlabel("Frame 200")

sgtitle('Skier Low-Rank Subtraction Method') 