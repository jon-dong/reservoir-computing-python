original_path = pwd;

%% SLM
disp('Initializing the SLM...')

% SLM on (for further details on the functions, see the script Blink_SDK_example.m)
cd 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\'
%SLM SDK parameters
bit_depth = 12; %For the 512L bit depth is 16, for the small 512 bit depth is 8
num_boards_found = libpointer('uint32Ptr', 0);
constructed_okay = libpointer('int32Ptr', 0);
is_nematic_type = 1; RAM_write_enable = 1; use_GPU = 0; max_transients = 10;
wait_For_Trigger = 0; % put 1 to enable external trigger control on SLM display
external_Pulse = 0; % put 1 to enable SLM controller box to send a trigger pulse when mask is displayed
timeout_ms = 5000;
true_frames = 5;
if ~libisloaded('Blink_C_wrapper')
    loadlibrary('SDK\Blink_C_wrapper.dll', 'SDK\Blink_C_wrapper.h');
end
lut_file = 'LUT Files\linear.LUT';
reg_lut = libpointer('string');
WFC = imread('WFC Files\1920black.bmp');
sdk = calllib('Blink_C_wrapper', 'Create_SDK', bit_depth, num_boards_found, constructed_okay, is_nematic_type, RAM_write_enable, use_GPU, max_transients, reg_lut);
if constructed_okay.value ~= 0  % Convention follows that of C function return values: 0 is success, nonzero integer is an error
    disp('Blink SDK was not successfully constructed');
    disp(calllib('Blink_C_wrapper', 'Get_last_error_message', sdk));
    calllib('Blink_C_wrapper', 'Delete_SDK', sdk);
    if libisloaded('Blink_C_wrapper');unloadlibrary('Blink_C_wrapper');end
    return
else
    board_number = 1;
    disp('Blink SDK was successfully constructed');
    fprintf('Found %u SLM controller(s)\n', num_boards_found.value);
    calllib('Blink_C_wrapper', 'Load_LUT_file',board_number, lut_file);
    calllib('Blink_C_wrapper', 'Set_true_frames', true_frames);% Set the basic SLM parameters
    calllib('Blink_C_wrapper', 'SLM_power', board_number); % Turn the SLM power on
end

% cd 'D:\Users\Mickael-manip\Desktop\Manip'
%% Conclusion
fprintf('Successfully opened the SLM. \n');
cd(original_path)