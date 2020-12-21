%% This file is to generate the matrix for generating colored noise.

clear;
% 用于产生对应码长的噪声矩阵，只关心总码长，不关心码率。
% eta = 0.8;
% eta = 0.5
eta = 0.0;
% N = 576;  % (576,432)
% N = 240;
% N = 6;  % (6,3)
N = 16;
% N = 10; %(10, 5)
% N = 96;  % (96, 48)
% N = 63;
%N = 128
cov = zeros(N, N);
for ii=1:N
    for jj=ii:N
        cov(ii,jj) = eta^(abs(ii-jj));
        cov(jj,ii) = cov(ii,jj);
    end
end
transfer_mat = cov^(1/2);

% 灏嗕骇鐢熺殑鏁版嵁鍐欏叆鍒版枃浠朵腑锛屽啓鎴?2杩涘埗鏂囦欢锛屽彲浠ュ仛鍒伴?氱敤鎬с??
fout = fopen(sprintf('cov_1_2_corr_para%.2f.dat', eta),'wb');
fwrite(fout, transfer_mat, 'single');
fclose(fout);
