% Created on 2019-07-12
% Author: Yongqi Tan
% Version 1.0
% Title: stability_test
% Description: This code is to do the stability test. In this script we
% will do the prediction for several times and save all the pred_output.
% And in another script 'compare_pred_output.m' we will load these
% pred_output and see how different they are. Also we save the data of
% every frame we got from the camera so that we can do the Fourier
% analysis to see more properties of the pixels.
%
% _________________________________________________________________________
addpath(fullfile('D:\Users\Comedia\Desktop\reservoir-computing-python\hardware_control'));
% load the mat file according to its iteration type
load('parameters_pred.mat')
spatial_points = cast(size(raw_input_data, 3), 'int32');
pred_output = zeros(parallel, spatial_points*self_pred_horizon*self_rec_pred_steps);
% the pred_output should be 3-dimensional. Also we define para_steps here, 
% which means how many times we want to do the prediction, to compare with 
% the first prediction.
para_steps = 20;
pred_output_ = zeros(parallel, spatial_points*self_pred_horizon*self_rec_pred_steps, para_steps);
% saving a few parameters which will be changed in the iteration
self_forget_ = self_forget;
raw_input_data_ = raw_input_data;
self_parallel_runs_ = self_parallel_runs;
self_state_ = self_state;
for para = 1:para_steps
    % reload the parameters to keep the initial conditions the same
    self_forget = self_forget_;
    raw_input_data = raw_input_data_;
    self_parallel_runs = self_parallel_runs_;
    self_state = self_state_;
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
        data_all = zeros(1, 340^2, sequence_length);
%         data_reference_all = zeros(1, 340^2, sequence_length);%___________

        t0 = tic;
        
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
                            data_all(:, :, time_step) = reshape(data, 1, []);

                        else
                            disp('No phase mask in variable "phase_vec" to send on the SLM.');
                            return;
                        end
                    end

                    data = reshape(data, 1, []);
                    selfstate(:, i) = (cast((1-self_leak_rate)*data(cam_sampling_range),'double') + self_leak_rate*reshape(reservoir, 1, []));
                end
                self_state = selfstate;
                if time_step > self_forget
                    res_states(:, time_step - self_forget, :) = selfstate';
                end
            end
            concat_res_states(:, :, (res_num-1) * self_n_res + 1:res_num * self_n_res) = res_states;
        end
        t1 = toc(t0);
        frequency = sequence_length / t1;
        save('data_all.mat','data_all','frequency')
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
        pred_output(:, (ii-1)*spatial_points+1:ii*spatial_points) = raw_input_data; 
    end
    pred_output_(:,:,para) = pred_output;
end
save('pred_output_compare.mat', 'pred_output_')