% phase_vec is the vector to display on the SLM of size (340,320)
% values are between 0 and 255
% Here is an example with random values
% phase_vec = random('unif', 0, 255, 340, 320);
% phase_vec = kron(phase_vec, ones(2));
% phase_vec = zeros(340, 320);
load('phase_vec.mat')
% phase_vec = rand(50000, 1);
% src.TriggerMode = 'On'; % sometime trigermode automatically set to off

% Beam axis position on the SLM pitch
x_0 = 960; %BeamSLM__20_07_2018__09_33 // 202; %BeamSLM__12_07_2018__11_37 //  230; % 230 BeamSLM__09_07_2018__15_52  // 226 BeamSLM__18_05_2018__09_53
y_0 = 900; %BeamSLM__20_07_2018__09_33 //  254; %BeamSLM__12_07_2018__11_37 //  242; % 242 BeamSLM__09_07_2018__15_52  // 236 BeamSLM__18_05_2018__09_53
delta = ceil(sqrt(length(phase_vec))); 
%  delta_y = 265; %BeamSLM__20_07_2018__09_33 // 340 %BeamSLM__12_07_2018__11_37 // 400; %BeamSLM__09_07_2018__15_52  // 350 % BeamSLM__18_05_2018__09_53
% delta_y = 320; %BeamSLM__20_07_2018__09_33 // 320 %BeamSLM__12_07_2018__11_37 // 400;  %BeamSLM__09_07_2018__15_52   // 385 % BeamSLM__18_05_2018__09_53

new_phase_vec = zeros(delta^2, 1);
new_phase_vec(1:length(phase_vec)) = phase_vec;

if exist('slm_size', 'var')==0
    slm_size = [1920, 1152];
else
    slm_size = double(slm_size);
end
SLM_shape = zeros([1920, 1152]); % slm_size is given from reservoir.py
macropix_size = min(floor(1920/slm_size(1)), floor(1152/slm_size(2)));
slm_center = reshape(new_phase_vec, delta, delta);
slm_center = kron(slm_center, ones(macropix_size));
SLM_shape(x_0-macropix_size*floor(delta/2):x_0+macropix_size*ceil(delta/2)-1,y_0-macropix_size*floor(delta/2):y_0+macropix_size*ceil(delta/2)-1) = slm_center;
% SLM_shape(x_0-floor(delta/2):x_0+ceil(delta/2)-1,y_0-floor(delta/2):y_0+ceil(delta/2)-1) = reshape(new_phase_vec,delta,delta);
% SLM_shape(265-170:264, :) = random('unif', 0, 255, 170, 512);

if exist('SLM_shape', 'var')
    calllib('Blink_C_wrapper', 'Write_image', board_number, SLM_shape, 1920*1152, wait_For_Trigger, external_Pulse, timeout_ms);
    t0 = tic;
%     trigger(vid); % added for manual triggering
%     pause(0.1); % added for manual triggering
%     orig_frames = vid.FramesAcquired;        
%     while vid.FramesAcquired ~= orig_frames + 1
%         t1 = toc(t0);
%         if t1 > 2  % in seconds
%             error('Missing trigger, no frame captured by the camera.');
%         end
%     end
%     data = getsnapshot(vid);
    data = getsnapshot(vid);
%     data = conv2(double(data),[1,1;1,1],'valid');
%     data = Y(1:2:end,1:2:end)/4;

%      figure(1)
%      imagesc(data)
%     frame = getframe(gcf);
%     writeVideo(slm_video,frame)
%     
%      figure(2)
%      imagesc(SLM_shape)
%     frame = getframe(gcf);
%     writeVideo(camera_video,frame)

    

    

else
    disp('No phase mask in variable "phase_vec" to send on the SLM.');
end