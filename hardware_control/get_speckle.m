load('phase_vecs.mat')
% phase_vecs = 256*rand(1,100*100);   
% phase_vecs = zeros(1,100*100);   

slm_size = [1920, 1152];
x_0=floor(slm_size(1)/2);
y_0=floor(slm_size(2)/2);

[num_images, len_images] = size(phase_vecs);
dim = ceil(sqrt(len_images));

new_phase_vecs = zeros(num_images, dim^2);
new_phase_vecs(:, 1:len_images) = phase_vecs;
slm_images = reshape(new_phase_vecs, num_images, dim, dim);
if (exist('macropix_size1', 'var')==0) || (exist('macropix_size2', 'var')==0)
    macropix_size1 = floor(slm_size(1)/dim);
    macropix_size2 = floor(slm_size(2)/dim);
else
    macropix_size1 = min([floor(slm_size(1)/dim), macropix_size1]);
    macropix_size2 = min([floor(slm_size(2)/dim), macropix_size2]);
end

slm_shape = 128.*ones([slm_size(1), slm_size(2)]);
cam_images = zeros(num_images, cam_roi(1), cam_roi(2));
% t00 = tic;
for i=1:num_images
    if (macropix_size1 == 1) && (macropix_size2 == 1)
        slm_image = reshape(slm_images(i, :, :),dim,dim);
    elseif (macropix_size1 == 0) || (macropix_size2 == 0)
        disp('the data dimensions are larger than the size of the SLM');
    else
        slm_image = kron(reshape(slm_images(i, :, :),dim,dim), ones(macropix_size1, macropix_size2));
    end
    slm_shape(x_0-macropix_size1*floor(dim/2)+1:x_0+macropix_size1*ceil(dim/2),...
    y_0-macropix_size2*floor(dim/2)+1:y_0+macropix_size2*ceil(dim/2)) = slm_image;
    calllib('Blink_C_wrapper', 'Write_image', board_number, slm_shape, 1920*1152, wait_For_Trigger, external_Pulse, timeout_ms);
    orig_frames = vid.FramesAcquired;
    t0 = tic;
    while vid.FramesAcquired ~= orig_frames + 1
        t1 = toc(t0);
        if t1 > 10  % in seconds
%             missing_trigs = missing_trigs + 1;
            disp('Missing trigger, no frame captured by the camera.');
%             continue
            error('Missing trigger, no frame captured by the camera.');
        end
    end
    cam_images(i,:,:) = peekdata(vid, 1);
    flushdata(vid);
%     trigger(vid); % added for manual triggering
% 
%      figure(1)
%      imagesc(reshape(cam_images(i,:,:), cam_roi(1), cam_roi(2)))
%      colorbar
%     frame = getframe(gcf);
%     writeVideo(slm_video,frame)
% 
%      figure(2)
%      imagesc(slm_shape)
%      colorbar
%     frame = getframe(gcf);
%     writeVideo(camera_video,frame)
end
% toc(t00)
