%% Guns and Roses Guitar
clear all; close all; clc
[y, Fs] = audioread('GNR.m4a'); %Fs is sampling rate
tr_gnr = length(y)/Fs; % record time in seconds

%% Fourier Transform of whole song
L = tr_gnr; n = length(y); %L is time in seconds rounded up
%n isn't power of 2 -> relying on matlab to pad
%make linear vector of n points spanning time domain
t2 = linspace(0, L, n+1);
t = t2(1:n); %matlab fft assumes periodic domain so last point = first
%make frequency vector in Hz and shift for plotting
k = (1/L) * [0:n/2-1 -n/2:-1]; ks = fftshift(k);
sy = fft(y);
%% Bandpass Filter
% remove all wavenumbers higher and lower than guitar range
binary_filter = abs(k()) >= 800;
sy(binary_filter) = 0;
binary_filter = abs(k()) <= 200;
sy(binary_filter) = 0;
fy = ifft(sy);

%% Gabor Transform
a = 200; %width of gaussian, 1/a = variance of gaussian
fygt_spec=[]; %matrix for collecting spectrogram data

% Define tau vector
increment = tr_gnr/(2*57); %57 notes in section, sample twice per note
tslide=0:increment:tr_gnr;

for j = 1:length(tslide)
    g = exp(-a *(t-(tslide(j))).^2); %Define Gaussian shifted to tau
    %Apply Gaussian in Time Domain
    fyg = g'.*fy; %Invert filter so dimensions match data
    fygt = fft(fyg);
    
    %filter around central frequency
    mfygt = fygt;
    [max_vals, idxs] = maxk(mfygt, 5); %find 5 max power points
    central_freq = min(idxs); %assume central frequency is longest 
                              %wavelength of these
    
    % Make Gaussian Filter centered at central frequency
    a2 = .2;
    filter = (exp(-a2* ((k-k(central_freq)) .^2)));
    mfygt = mfygt'.*filter; %Apply Gaussian
    fygt_spec(:,j) = fftshift(abs(mfygt));
end
%% Plot
figure()
pcolor(tslide, ks, log(abs(fygt_spec)+1)), shading interp
set(gca, 'Ylim',[200, 800], 'Fontsize', 16)
colormap(hot)
xlabel('Time [sec]'); ylabel('Frequency [Hz]');
title('Sweet Child O'' Mine Guitar Solo');

%Label relevant frequencies
cl = yline(277.183,'-','C#4','LineWidth',.5);
cl.LabelVerticalAlignment = 'middle';
cl.Color = 'cyan';
ccl = yline(554.365,'-','C#5','LineWidth',.5);
ccl.LabelVerticalAlignment = 'middle';
ccl.Color = 'cyan';
cl.Color = 'cyan';
gl = yline(415.305,'-','G#4','LineWidth',.5);
gl.LabelVerticalAlignment = 'middle';
gl.Color = 'cyan';
dl = yline(311.127,'-','D#4','LineWidth',.5);
dl.LabelVerticalAlignment = 'middle';
dl.Color = 'cyan';
ddl = yline(698.456,'-','F5','LineWidth',.5);
ddl.LabelVerticalAlignment = 'middle';
ddl.Color = 'cyan';
ffl = yline(739.989,'-','F#5','LineWidth',.5);
ffl.LabelVerticalAlignment = 'middle';
ffl.Color = 'cyan';
ffl = yline(369.994,'-','F#4','LineWidth',.5);
ffl.LabelVerticalAlignment = 'middle';
ffl.Color = 'cyan';

%% Pink Floyd Bass Part
clear all; close all; clc

[y, Fs] = audioread('Floyd.m4a'); %Fs is sampling rate
y = y(1:(length(y)-1));% Subtract one because number of samples is odd
tr_gnr = length(y)/Fs; % record time in seconds

%% Split song into quarters, 4 4-measure sections, save in struct
fourth = (length(y) / 4);
full_song_1 = y(1:fourth);
full_song_2 = y(fourth+1:(2*fourth));
full_song_3 = y((2*fourth)+1:(3*fourth));
full_song_4 = y((3*fourth)+1:end);
B_full_song = struct('one', full_song_1, 'two', full_song_2, 'three', ...
    full_song_3, 'four', full_song_4);
tr_gnr = fourth/Fs; %length of each quarter in seconds
song_clips = fieldnames(B_full_song); %used to iterate through quarters

%% Bandpass Filter
L = tr_gnr; n = fourth; %L=length of each quarter in seconds
t2 = linspace(0, L, n+1); %linear vector of n points
t = t2(1:n);%matlab fft assumes periodic domain so last point = first
%make frequency vector in Hz and shift for plotting
k = (1/L) * [0:n/2-1 -n/2:-1]; ks = fftshift(k); 

for i = 1:numel(song_clips) %iterate through each quarter of the song
    sy = fft(B_full_song.(song_clips{i}));    
    % Threshold Filter by Frequency
    % remove all wavenumbers higher than my bass
    binary_filter = abs(k()) >= 140;
    sy(binary_filter) = 0;
    B_full_song.(song_clips{i}) = ifft(sy);
end

%% Gabor Transform
a = 10; %width of gaussian for gabor

%matrices for holding spectrogram data
bspecs = {'bspec1', 'bspec2', 'bspec3', 'bspec4'};
%matrices for holding central frequency data
cent_freq_vals = {'cfv1', 'cfv2', 'cfv3', 'cfv4'};

for i = 1:numel(bspecs) %iterate through each quarter of the song
    %Gabor filter
    fygt_spec=[]; %spectrogram matrix
    cfvs=[]; %central frequency matrix
    % Define tau vector
    tslide=0:.2:tr_gnr;
    fy = B_full_song.(song_clips{i}); %extract y vec for this quarter
    for j = 1:length(tslide)
        g = exp(-a *(t-(tslide(j))).^2); %Define Gaussian shifted to tau
        fyg = g'.*fy; %Apply in time domain
        mfygt = fft(fyg);
        
        %filter by power, gaussian in frequency domain centered at max power
        %frequency
        [max_vals, idxs] = maxk(mfygt, 5); %find 5 max power points
        central_freq = min(idxs);
        cfvs(j) = k(central_freq);
        
        
        % Make Gaussian Filter
        a2 = .2;
        % Shift to central frequency
        filter = (exp(-a2* ((k-k(central_freq)) .^2)));
        mfygt = mfygt'.*filter; %apply Gaussian
        
        fygt_spec(:,j) = fftshift(abs(mfygt)); %shift for plotting
    end
    B_full_song.(bspecs{i}) = fygt_spec; %save coefficients
    B_full_song.(cent_freq_vals{i}) = cfvs; %save central frequencies
end
%% Plot

for i = 1:numel(bspecs) %make a different plot for each quarter
    figure()
    fygt_spec = B_full_song.(bspecs{i});
    pcolor(tslide, ks, log(abs(fygt_spec)+1)), shading interp
    set(gca, 'Ylim',[60, 140], 'Fontsize', 16)
    colormap(hot)
    
    xlabel('Time [sec]'); ylabel('Frequency [Hz]');
    title_str = sprintf('Floyd Quarter %d Bass Spectogram', i);
    title(title_str); 
    
    %Label Relevant Notes
    
    bl = yline(123.471,'-','B2','LineWidth',.5);
    bl.LabelVerticalAlignment = 'middle';
    bl.Color = 'cyan';
    
    al = yline(110,'-','A2','LineWidth',.5);
    al.LabelVerticalAlignment = 'middle';
    al.Color = 'cyan';
    
    gl = yline(97.99,'-','G2','LineWidth',.5);
    gl.LabelVerticalAlignment = 'middle';
    gl.Color = 'cyan';
    
    el = yline(82.407,'-','E2','LineWidth',.5);
    el.LabelVerticalAlignment = 'middle';
    el.Color = 'cyan';
    
    fl = yline(92.499,'-','F#2','LineWidth',.5);
    fl.LabelVerticalAlignment = 'middle';
    fl.Color = 'cyan';
    
    fl = yline(87.307,'-','F2','LineWidth',.5);
    fl.LabelVerticalAlignment = 'middle';
    fl.Color = 'cyan';
end

%% Pink Floyd Guitar
[y, Fs] = audioread('Floyd.m4a'); %Fs is sampling rate
y = y(1:(length(y)-1)); %make n even
tr_gnr = length(y)/Fs; % record time of full clip in seconds

%% Split song into fourths
fourth = (length(y) / 4);
full_song_1 = y(1:fourth);
full_song_2 = y(fourth+1:(2*fourth));
full_song_3 = y((2*fourth)+1:(3*fourth));
full_song_4 = y((3*fourth)+1:end);
G_full_song = struct('one', full_song_1, 'two', full_song_2, 'three', ...
    full_song_3, 'four', full_song_4);
tr_gnr = fourth/Fs; %time of each quarter in seconds
song_clips = fieldnames(G_full_song); %used for iterating

%% Bandpass Filter
%make initial vectors to be used for whole thing
L = tr_gnr; n = fourth; %L=length of each quarter in seconds
%make linear vector of n points spanning duration of each quarter
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1/L) * [0:n/2-1 -n/2:-1]; ks = fftshift(k); %make wave number vector and shifted for plotting

for i = 1:numel(song_clips)%iterate through each quarter of the song
    sy = fft(G_full_song.(song_clips{i}));
    % Filter by Frequency
    minfreq = 440;
    maxfreq = 1100;
    binary_filter = abs(k()) >= maxfreq;
    sy(binary_filter) = 0;
    binary_filter = abs(k()) <= minfreq;
    sy(binary_filter) = 0;
    G_full_song.(song_clips{i}) = ifft(sy);
end

%% Gabor Transform
a = 200; %width of gaussian for gabor

gspecs = {'gspec1', 'gspec2', 'gspec3', 'gspec4'};
cent_freq_vals = {'cfv1', 'cfv2', 'cfv3', 'cfv4'};

for i = 1:numel(gspecs)%iterate through each quarter of the song
    fygt_spec=[];
    cfvs=[];
    tslide=0:.2:tr_gnr;
    fy = G_full_song.(song_clips{i}); %extract y vec for this quarter
    %Extract bass notes to filter out bass overtones
    bass_central_freqs = B_full_song.(cent_freq_vals{i});
    for j = 1:length(tslide)
        g = exp(-a *(t-(tslide(j))).^2); %Define Gaussian shifted to tau
        fyg = g'.*fy; %Apply Gaussian in time domain
        mfygt = fft(fyg);
        
        %first inverse gaussian around notes octaves up from central 
        %frequency emitted by bass at this time point (to filter out
        %rhythm guitar that is playing these notes)
        a3 = .2; %width of this gaussian
        bass_note = bass_central_freqs(j); %bass note out of relevant range
        bass_note2 = 2 * bass_note; %1 octave up
        bass_note3 = 2 * bass_note2; %2 octaves up
        bass_note4 = 2 * bass_note3; %3 octaves up
        %Define upside-down Gaussians centered at each of these frequencies
        guitar_rm_filter2 = 1 - (exp(-a3* ((k-bass_note2) .^2)));
        guitar_rm_filter3 = 1 - (exp(-a3* ((k-bass_note3) .^2)));
        guitar_rm_filter4 = 1 - (exp(-a3* ((k-bass_note4) .^2)));
        %Apply filters in frequency domain
        mfygt = mfygt'.* guitar_rm_filter2;
        mfygt = mfygt.* guitar_rm_filter3;
        mfygt = mfygt.* guitar_rm_filter4;
        
        
        %filter by power
        %find 5 max power points after filtering out rhythm guitar
        [max_vals, idxs] = maxk(mfygt, 5); 
        %assume central frequency is lowest pitch of this set
        central_freq = min(idxs);
        cfvs(j) = k(central_freq);
          
        % Make Gaussian Filter centered at central frequency
        a2 = .2;
        filter = (exp(-a2* ((k-k(central_freq)) .^2)));
        mfygt = mfygt.*filter; %apply Gaussian
        fygt_spec(:,j) = fftshift(abs(mfygt));
    end
    G_full_song.(gspecs{i}) = fygt_spec; %save coefficients
    G_full_song.(cent_freq_vals{i}) = cfvs;%save central frequencies
end


%% Plot
for i = 1:numel(gspecs) %make a different plot for each quarter
    fygt_spec =G_full_song.(gspecs{i});
    figure()
    pcolor(tslide, ks, log(abs(fygt_spec)+1)), shading interp
    set(gca, 'Ylim',[minfreq-10, maxfreq], 'Fontsize', 16)
    colormap(hot)
    xlabel('Time [sec]'); ylabel('Frequency [Hz]');
    title_str = sprintf('Floyd Quarter %d Guitar Solo Spectogram', i);
    title(title_str);
    %colorbar
    
    %Label Relevant Notes
    
    fsl = yline(739.989,'-','F#5','LineWidth',.5);
    fsl.LabelVerticalAlignment = 'middle';
    fsl.Color = 'cyan';
     
    gl = yline(783.991,'-','G5','LineWidth',.5);
    gl.LabelVerticalAlignment = 'middle';
    gl.Color = 'cyan';
    
    dl = yline(622.254,'-','D#5','LineWidth',.5);
    dl.LabelVerticalAlignment = 'middle';
    dl.Color = 'cyan';
    
    el = yline(659.255,'-','E5','LineWidth',.5);
    el.LabelVerticalAlignment = 'middle';
    el.Color = 'cyan';
    
    fl = yline(698.456,'-','F5','LineWidth',.5);
    fl.LabelVerticalAlignment = 'middle';
    fl.Color = 'cyan';
    
    al = yline(880,'-','A5','LineWidth',.5);
    al.LabelVerticalAlignment = 'middle';
    al.Color = 'cyan';
    
    al = yline(440,'-','A4','LineWidth',.5);
    al.LabelVerticalAlignment = 'middle';
    al.Color = 'cyan';
    
    dl = yline(587.33,'-','D5','LineWidth',.5);
    dl.LabelVerticalAlignment = 'middle';
    dl.Color = 'cyan';
    
    cl = yline(523.251,'-','C5','LineWidth',.5);
    cl.LabelVerticalAlignment = 'middle';
    cl.Color = 'cyan';
    
    cl = yline(554.365,'-','C#5','LineWidth',.5);
    cl.LabelVerticalAlignment = 'middle';
    cl.Color = 'cyan';
    
    bl = yline(987.767,'-','B5','LineWidth',.5);
    bl.LabelVerticalAlignment = 'middle';
    bl.Color = 'cyan'; 
    
    bl = yline(493.883,'-','B4','LineWidth',.5);
    bl.LabelVerticalAlignment = 'middle';
    bl.Color = 'cyan';
end