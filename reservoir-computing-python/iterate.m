% Created on 2019-06-20
% Author: Yongqi Tan
% Version 1.0
% Title: Iterate
% Description: This code is a matlab version of iterate function. It can be
% used to do fitting or predicting (with or without concatenate).
% The iterate type is choosed through variable
% 'iterate_type':
%
% iterate_type = 0: fitting mode
% iterate_type = 1: predicting mode without concatenate
% iterate_type = 2: predicting mode with concatenate but loss function of
% standardization
% _________________________________________________________________________
addpath(fullfile('D:\Users\Comedia\Desktop\reservoir-computing-python\hardware_control'));
% load the iteration type
load('iterate_type.mat')
% load the mat file according to its iteration type
if iterate_type == 0
    load('parameters_fit.mat')
elseif iterate_type >= 1
    load('parameters_pred.mat')
    spatial_points = cast(size(raw_input_data, 3), 'int32');
    pred_output = zeros(parallel, spatial_points*self_pred_horizon*self_rec_pred_steps);
end
% If it's fitting or predicting without concat, then we don't need
% self_rec_pred_steps, which is only needed when getting pred_output
if iterate_type < 2
    self_rec_pred_steps = 1;
end
% This 'for' loop is for the calculation of pred_output
for ii = 1:self_rec_pred_steps
    cam_sampling_range = cast(linspace(1, (cam_roi(1)-1)*(cam_roi(2)-1)-1, self_n_res), 'int32');
    % cam_sampling_range = cast(self_cam_sampling_range, 'int32');
    if size(size(raw_input_data), 2) == 2
        % if input data is 2D (previous pred_output) means it is in refreshing phase
        input_data = encode(raw_input_data);
        input_data = reshape(input_data,[size(input_data,1) 1 size(input_data,2)]);
        self_forget = 0;
        iteration_state = 'update';
    else
        input_data = encode(raw_input_data);
        iteration_state = 'construction';
    end

    [n_sequence, sequence_length, ~] = size(input_data);
    res_states = zeros([n_sequence, sequence_length - self_forget, self_n_res], 'double');
    res_states = complex(res_states, 0);
    concat_res_states = zeros([n_sequence, sequence_length - self_forget, self_n_res*self_concat_res], 'double');
    concat_res_states = complex(concat_res_states, 0);

%         % this is temporary, giving the shape of data_all (which we use to
%         % save each frame from the camera), most time it's not needed.
%         data_all = zeros(1, 340^2, sequence_length);%___________
%         data_reference_all = zeros(1, 340^2, sequence_length);%___________

%         % this part is to save the frames to a video
%         data_ = zeros(100,sequence_length);
% 
%         slm_video = VideoWriter('slm_video_matlab3.avi');
%         open(slm_video);
%         camera_video = VideoWriter('camera_video_matlab3.avi');
%         open(camera_video);
    for res_num = 1:self_concat_res
        for time_step = 1:sequence_length
            selfstate = zeros(self_n_res, size(self_state, 2));
            for i = 1:size(self_state, 2)
                input_data_generate = reshape(input_data(i, time_step, :), [], size(input_data, 3));
                reservoir = encode(self_state(:, i))';
                generate_slm_imgs;
                phase_vec = slm_imgs(1,:);

                % This part is 'get_speckle'
                %__________________________________________
                % Beam axis position on the SLM pitch
                if exist('delta', 'var')==0
                    x_0 = 960;
                    y_0 = 576;
                    delta = ceil(sqrt(length(phase_vec)));
                    new_phase_vec = zeros(delta^2, 1);
                    % new_phase_vec = reshape(new_phase_vec, delta, delta);
                    if exist('slm_size', 'var')==0
                        slm_size = [1920, 1152];
                    else
                        slm_size = double(slm_size);
                    end
                    SLM_shape = zeros([1920, 1152]); % slm_size is given from reservoir.py
                    macropix_size = min(floor(1920/slm_size(1)), floor(1152/slm_size(2)));
                end
                new_phase_vec(1:length(phase_vec)) = phase_vec;
                slm_center = reshape(new_phase_vec, delta, delta);
                % slm_center = kron(slm_center, ones(macropix_size));
                SLM_shape(x_0-floor(delta/2):x_0+ceil(delta/2)-1,y_0-floor(delta/2):y_0+ceil(delta/2)-1) = slm_center;
    %             SLM_shape(x_0-floor(delta/2):x_0+ceil(delta/2)-1,y_0-floor(delta/2):y_0+ceil(delta/2)-1) = zeros(delta, delta);
                if exist('SLM_shape', 'var')
                    calllib('Blink_C_wrapper', 'Write_image', board_number, SLM_shape, 1920*1152, wait_For_Trigger, external_Pulse, timeout_ms);
                    % pause to wait for the camera to finish loading the image
                    % and get data ready, otherwise we'll get wrong frame
%                     pause(0.03);
%                    % another approach to get right frame
                   while vid.FramesAvailable==0
                   end
                    data = peekdata(vid, 1);
                    flushdata(vid);
                else
                    disp('No phase mask in variable "phase_vec" to send on the SLM.');
                    return;
                end

%                 % plotting the data and SLM_shape, to check if they work
%                 if ii == 1 && time_step == 1
%                 figure(1)
%                 imagesc(data)
%                 colorbar
%                 figure(2)
%                 imagesc(SLM_shape)
%                 colorbar
%                 end

                % noise_normalization part
                if self_slm_noise_normalization
                    SLM_shape(x_0-floor(delta/2):x_0+ceil(delta/2)-1,y_0-floor(delta/2):y_0+ceil(delta/2)-1) = self_slm_reference;
                    % give a flatten pattern to the SLM
%                     SLM_shape(:, :) = self_slm_reference;
                    if exist('SLM_shape', 'var')
                        calllib('Blink_C_wrapper', 'Write_image', board_number, SLM_shape, 1920*1152, wait_For_Trigger, external_Pulse, timeout_ms);
%                         pause(0.03);
%                         % another approach to get right frame
                        while vid.FramesAvailable==0
                        end
                        % taking the data from camera memory
                        data_reference = peekdata(vid, 1);
                        % clear camera memory
                        flushdata(vid);
                        % normalization
                        data = data_reference - data;

%                         % this is temporary, giving the shape of data_all (which we use to
%                         % save each frame from the camera), most time it's not needed.
%                         if iterate_type == 0%___________
%                             data_all(:, :, time_step) = reshape(data, 1, []);
%                         end

                    else
                        disp('No phase mask in variable "phase_vec" to send on the SLM.');
                        return;
                    end
                end
                %__________________________________________
                % 'get_speckle' ends


        %             cam_data_matlab = data;
                data = reshape(data, 1, []);
%                     % Making the image more sparse, which might help
%                     data = data.^(2)./max(data);
                selfstate(:, i) = (cast((1-self_leak_rate)*data(cam_sampling_range),'double') + self_leak_rate*reshape(reservoir, 1, []));
            end
            self_state = selfstate;
            if time_step > self_forget
                res_states(:, time_step - self_forget, :) = selfstate';
            end
        end
        concat_res_states(:, :, (res_num-1) * self_n_res + 1:res_num * self_n_res) = res_states;
    end
%   close(slm_video);
% 	close(camera_video);


    if iterate_type <= 1 && self_concat_res > 1
        save('res_states.mat','concat_res_states')
    elseif iterate_type <= 1
        save('res_states.mat','res_states')
    elseif iterate_type >= 2
        if size(size(raw_input_data), 2) == 2
            % if input data is 2D (previous pred_output) means it is in refreshing phase
            input_data = encode(raw_input_data);
            input_data = reshape(input_data, size(input_data, 1), 1, size(input_data, 2));
            self_forget = 0;
            self_parallel_runs = size(input_data, 1);
        else
            input_data = encode(raw_input_data);
        end

        n_sequence = size(input_data, 1);
        res_states = real(res_states); % Should be checked if it's proper
        if size(res_states, 2) == 1 && size(res_states, 1) ~= 1
            self_state = reshape(res_states, size(res_states, 1), size(res_states, 3))';
        else
            self_state = reshape(res_states, size(res_states, 2), size(res_states, 3))'; % will be used in update equation if recursive prediction is active
        end

        % construct the concatenated states
        concat_states = concat_res_states;
        % print('res_states'+str(res_states.shape))
        if self_raw_input_feature
            concat_states = cat(3, concat_states, raw_input_data(:, self_forget+1:end, :));
        end
        if self_enc_input_feature
            concat_states = cat(3, concat_states, input_data(:, self_forget+1:end, :));
        end
        raw_input_data = reshape(concat_states, [], self_concat_res*self_n_res+spatial_points) * self_output_w; % reccurently iterate the input
        pred_output(:, (ii-1)*spatial_points+1:ii*spatial_points) = raw_input_data;  %%%_____________
    end
end
if iterate_type == 2
    save('pred_output.mat', 'pred_output')
end