% How to plot experimental data directivity contours

expdir = '/projectnb/turbomac/arozman/projects/ULI/experimental_data/16degPitch_90degYaw_10mps_AutoSpec.mat';
micsdir = '/projectnb/turbomac/arozman/projects/ULI/experimental_data/Xc_U10.txt';

% Load data from MAT file directly into workspace
% Variables loaded: freq, broadband_spectra, rpm, tonal_spectra, total_spectra
load(expdir);

fq = squeeze(freq);
rpms = squeeze(rpm);
bbn = broadband_spectra;   % Size: [4097, 251, 21] (freq x mic x rpm)
tonal = tonal_spectra;     % Size: [4097, 251, 21]
total = total_spectra;     % Size: [4097, 251, 21]

xyz = readmatrix(micsdir);

%% Filter by BPF, remove NaNs
rpm_ix = length(rpms);   % 4000 RPM (last RPM entry)
BPF = rpms(rpm_ix) / 60 * 5;

% Index in fq array of BPF frequency
[~, ix] = min(abs(fq - BPF));

% Microphones with non-NaN values at BPF for the target RPM
mask = ~isnan(squeeze(tonal(ix, :, rpm_ix)));
if isrow(mask)
    mask = mask';
end

% Explicitly bad microphone indices (0-indexed [17, 155, 219] -> 1-indexed [18, 156, 220])
bad_mics = [18, 156, 220];
mask(bad_mics) = false;

% Remove invalid mic entries from xyz and spectral arrays
xyz = xyz(mask, :);
tonal = tonal(:, mask, :);
total = total(:, mask, :);
bbn = bbn(:, mask, :);

%% Interp PSD on grid & acoustic processing
[X, Y] = meshgrid(linspace(-2, 2, 100), linspace(-1, 1, 50));

tonal_bpf = squeeze(tonal(ix, :, rpm_ix));
Z = griddata(xyz(:,1), xyz(:,2), tonal_bpf, X, Y, 'linear');

df = fq(2) - fq(1);

% SPL (dB) is multiplied by bandwidth
Z_spectrum = 10 * log10( Z * df / (2e-5)^2 );

% PSD (dB/Hz) is not.
Z_PSD = 10 * log10( Z / (2e-5)^2 );

% Values for contour colorbar
levels = linspace(65, 75, 75);

%% Plotting

% 1. Plot SPL
figure;
ax1 = gca;
contourf(X, Y, clamp_levels(Z_spectrum, levels), levels, 'LineStyle', 'none');
colormap(jet);
clim([levels(1), levels(end)]);
format_fig(ax1, levels, 'SPL (dB)');
title(ax1, sprintf('Experiment 4000RPM Hover SPL, df=%g', fq(2)), 'FontSize', 14);

% 2. Plot PSD
figure;
ax2 = gca;
contourf(X, Y, clamp_levels(Z_PSD, levels), levels, 'LineStyle', 'none');
colormap(jet);
clim([levels(1), levels(end)]);
format_fig(ax2, levels, 'PSD (dB/sqrt(Hz))');
title(ax2, sprintf('Experiment 4000RPM Hover PSD, df=%g', fq(2)), 'FontSize', 14);

% 3. Plot PSD averaged within +/- 70Hz of BPF
freq_mask_70 = (fq >= (BPF - 70)) & (fq <= (BPF + 70));
tonal_last_rpm = squeeze(tonal(:, :, rpm_ix)); % Size: 4097 x N_mics
tonal_avg_70Hz = mean(tonal_last_rpm(freq_mask_70, :), 1)';
Z_avg_70Hz = griddata(xyz(:, 1), xyz(:, 2), tonal_avg_70Hz, X, Y, 'linear');
Z_PSD_70Hz = 10 * log10(Z_avg_70Hz / (2e-5)^2);

figure;
ax3 = gca;
contourf(X, Y, clamp_levels(Z_PSD_70Hz, levels), levels, 'LineStyle', 'none');
colormap(jet);
clim([levels(1), levels(end)]);
format_fig(ax3, levels, 'PSD (dB/Hz)');
title(ax3, sprintf('Experiment 4000RPM Hover PSD (Avg +/- 70Hz), df=%g', fq(2)), 'FontSize', 14);

% 4. Plot PSD averaged within +/- 10Hz of BPF
freq_mask_10 = (fq >= (BPF - 10)) & (fq <= (BPF + 10));
tonal_avg_10Hz = mean(tonal_last_rpm(freq_mask_10, :), 1)';
Z_avg_10Hz = griddata(xyz(:, 1), xyz(:, 2), tonal_avg_10Hz, X, Y, 'linear');
Z_PSD_10Hz = 10 * log10(Z_avg_10Hz / (2e-5)^2);

figure;
ax4 = gca;
contourf(X, Y, clamp_levels(Z_PSD_10Hz, levels), levels, 'LineStyle', 'none');
colormap(jet);
clim([levels(1), levels(end)]);
format_fig(ax4, levels, 'PSD (dB/Hz)');
title(ax4, sprintf('Experiment 4000RPM Hover PSD (Avg +/- 10Hz), df=%g', fq(2)), 'FontSize', 14);

% 5. Plot broadband noise (BBN) integrated from 500 to 20000 Hz
bbn_mask = (fq >= 500) & (fq <= 20000);
bbn_last_rpm = squeeze(bbn(:, :, rpm_ix)); % Size: 4097 x N_mics
bbn_int = sum(bbn_last_rpm(bbn_mask, :) * df, 1)';
Z_bbn = griddata(xyz(:, 1), xyz(:, 2), bbn_int, X, Y, 'linear');
Z_bbn_SPL = 10 * log10(Z_bbn / (2e-5)^2);

levels_bbn = linspace(90, 100, 75);
figure;
ax5 = gca;
contourf(X, Y, clamp_levels(Z_bbn_SPL, levels_bbn), levels_bbn, 'LineStyle', 'none');
colormap(jet);
clim([levels_bbn(1), levels_bbn(end)]);
format_fig(ax5, levels_bbn, 'BBN SPL (dB)');
title(ax5, 'Experiment 4000RPM Broadband Noise SPL (500-20000 Hz)', 'FontSize', 14);

%% Helper Functions

% Clamp data values to level bounds so contourf fills out-of-bounds regions (extend='both')
function Zc = clamp_levels(Z, levels)
    Zc = Z;
    Zc(Z < levels(1)) = levels(1);
    Zc(Z > levels(end)) = levels(end);
end

% Figure formatting helper
function format_fig(ax, levels, cb_title)
    xlabel(ax, 'X (m)', 'FontSize', 14);
    ylabel(ax, 'Y (m)', 'FontSize', 14);
    set(ax, 'FontSize', 10);
    axis(ax, 'equal');
    
    cbar = colorbar(ax);
    ylabel(cbar, cb_title, 'FontSize', 14);
    ticks = linspace(levels(1), levels(end), 10);
    cbar.Ticks = ticks;
    cbar.TickLabels = arrayfun(@(x) sprintf('%.2f', x), ticks, 'UniformOutput', false);

    ylim(ax, [-1, 1]);
    xlim(ax, [-2, 2]);
end
