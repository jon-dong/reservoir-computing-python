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

vid = videoinput('gentl', 1, 'Mono8');
src = getselectedsource(vid);
triggerconfig(vid, 'hardware', 'DeviceSpecific', 'DeviceSpecific'); % comment for manual triggering
triggerconfig(vid, 'immediate'); % add "manual" for manual triggering
src.TriggerSelector = 'FrameStart';
src.TriggerMode = 'On';
src.TriggerSource = 'Line1';
src.TriggerActivation = 'RisingEdge';
src.ExposureMode = 'Timed';
vid.FramesPerTrigger = 1;
% src.PacketSize = 1500;
vid.TriggerRepeat = Inf;
src.ExposureTime = 50000;
% imaqmex('feature', '-gigeDisablePacketResend', true);
src.TriggerDelay = 6000;   % 6000

% imaqmem(100000000);

if exist('cam_roi', 'var')==0
    cam_roi = [350 350];
else
    cam_roi = double(cam_roi);
end

current_roi = vid.ROIPosition;
vid.ROIPosition = [
    round((current_roi(3)-cam_roi(1))/2) round((current_roi(4)-cam_roi(2))/2) cam_roi(1) cam_roi(2)];

start(vid);

% starting to record slm and camera videos
% slm_video = VideoWriter('slm_video.avi');
% open(slm_video);
% camera_video = VideoWriter('camera_video.avi');
% open(camera_video);

%% Conclusion
fprintf('Successfully opened the camera. \n');
cd(original_path)