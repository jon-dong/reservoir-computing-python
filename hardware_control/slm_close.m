%% SLM
disp('Closing the SLM')
original_path = pwd;

cd 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\'

% SLM off
calllib('Blink_C_wrapper', 'Delete_SDK');
if libisloaded('Blink_C_wrapper')
    unloadlibrary('Blink_C_wrapper');
end

disp('Successfully closed the slm')
cd(original_path)