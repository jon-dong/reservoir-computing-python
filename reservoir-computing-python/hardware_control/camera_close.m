%% Camera
disp('Closing the camera...')

stop(vid)
delete(vid);
clear vid;

% in case of bug (if the vector is not closed and we started the preview image) 
imaqreset

% close recorded videos
% close(slm_video);
% close(camera_video);
% clear slm_video
% clear camera_video

disp('Successfully closed the camera')