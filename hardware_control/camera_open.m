original_path = pwd;

%% Camera

disp('Initializing the camera...')

% Opening CCD
% The detection of the camera is made by the Toolbox of matlab, Image acquisition tool.
% Finding the camera gives us the informations on the CCD camer, as
% exposition time, dimensions of display, ect
% vid = videoinput('avtmatlabadaptor_r2009b', 1, 'Mono8_780x580'); % name of the camera
% vid = videoinput('gige', 1, 'Mono8');
% src = getselectedsource(vid);
% src.PacketSize = 1500;
% vid.FramesPerTrigger = 1;  
% vid.ROIPosition = [0 0 780 580];

%start(vid)
% triggerconfig(vid, 'immediate');  % Self trigger of the camera 
% vid.TriggerRepeat = Inf;
% ROI creation (definition of quadrant of the camera)
% Fixed exposition time
% src.ExtendedShutterMode = 'on';
% src.ExtendedShutter = 2000; %1000
% src.NetworkPacketSize = 1500;

% % au 22 mai 2013 :
% vid = videoinput('gige', 1, 'Mono8');
% src = getselectedsource(vid);
% src.TriggerMode = 'Off';
% src.PacketSize = 2000 ;
% vid.FramesPerTrigger = 1;
% vid.TriggerRepeat = Inf;
% src.AcquisitionFrameRateAbs = 25;

% function opening manta ccd including an hardware external trigger from
% the SLM
if exist('vid', 'var')==0
    vid = videoinput('gentl', 1, 'Mono8');
    src = getselectedsource(vid);
    triggerconfig(vid, 'hardware');  % hardware trigger
    % triggerconfig(vid, 'manual');  % manual trigger (use "trigger(vid)")
    % triggerconfig(vid, 'immediate');  % continuous trigger
end

src.TriggerSelector = 'FrameStart';
src.TriggerSource = 'Line3';
src.TriggerActivation = 'RisingEdge';
src.TriggerMode = 'On';
src.ExposureMode = 'Timed';
vid.FramesPerTrigger = 1;
% src.PacketSize = 1500;
vid.TriggerRepeat = Inf;
src.ExposureTime = 1100;
% imaqmex('feature', '-gigeDisablePacketResend', true);
src.TriggerDelay = 3000;   % 3000

% imaqmem(100000000);

if exist('cam_roi', 'var')==0
    cam_roi = [200 200];  % roi is modified when values are not multiples of 4 (strange bug)
else
    cam_roi = double(cam_roi);
end
default_roi = [0 0 1936 1216];
if strcmp(vid.Running, 'on')
    stop(vid)
end
vid.ROIPosition = [
    round((default_roi(3)-cam_roi(1))/2) round((default_roi(4)-cam_roi(2))/2) cam_roi(1) cam_roi(2)];

start(vid);

% starting to record slm and camera videos
% slm_video = VideoWriter('slm_video.avi');
% open(slm_video);
% camera_video = VideoWriter('camera_video.avi');
% open(camera_video);

%% Conclusion
fprintf('Successfully opened the camera. \n');
cd(original_path)