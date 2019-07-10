function [] = construct()
  load('a0.mat', 'a0');
  load('L.mat', 'L');
  load('h.mat', 'h');
  load('nstp.mat', 'nstp');
  [tt, da] = ksfmstp(a0, L, h, nstp, 1);
  save('tt.mat', 'tt');
  save('da.mat', 'da');
  [xx, ii] = ksfm2real(da, L);
  save('xx.mat', 'xx');
  save('ii.mat', 'ii');
    