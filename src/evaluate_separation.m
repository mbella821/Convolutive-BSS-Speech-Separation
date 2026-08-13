function [SDR, ISR, SIR, SAR, perm] = evaluate_separation(se_img, s_img)
% EVALUATE_SEPARATION BSS performance evaluation using BSSeval.
%   [SDR, ISR, SIR, SAR, perm] = evaluate_separation(se_img, s_img)
%   wraps the bss_eval_images function to compute standard BSS metrics.
%
%   Input:
%     se_img - N x L x M tensor of estimated source spatial images
%              (N sources, L samples, M microphones)
%     s_img  - N x L x M tensor of true source spatial images
%   Output:
%     SDR    - N x 1 Signal-to-Distortion Ratio (dB)
%     ISR    - N x 1 Image-to-Spatial distortion Ratio (dB)
%     SIR    - N x 1 Signal-to-Interference Ratio (dB)
%     SAR    - N x 1 Signal-to-Artifact Ratio (dB)
%     perm   - N x 1 optimal permutation vector
%
%   Note: This function requires the BSSeval toolbox to be in the MATLAB
%   path. Download it from: https://bass-db.gforge.inria.fr/bss_eval/

if ~exist('bss_eval_images', 'file')
    error('BSSeval toolbox not found. Please add it to the MATLAB path.');
end

[SDR, ISR, SIR, SAR, perm] = bss_eval_images(se_img, s_img);

fprintf('\n========== Separation Performance ==========\n');
fprintf('Mean SDR : %.2f dB\n', mean(SDR));
fprintf('Mean SIR : %.2f dB\n', mean(SIR));
fprintf('Mean SAR : %.2f dB\n', mean(SAR));
fprintf('============================================\n');
end
