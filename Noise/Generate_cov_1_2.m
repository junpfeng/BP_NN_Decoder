%% This file is to generate the matrix for generating colored noise.

clear;

% eta = 0.8;
% eta = 0.5
eta = 0.0;
% N = 576;  % (576,432)
N = 6;  % (6,3)
% N = 10; %(10, 5)
% N = 96;  % (96, 48)
cov = zeros(N, N);
for ii=1:N
    for jj=ii:N
        cov(ii,jj) = eta^(abs(ii-jj));
        cov(jj,ii) = cov(ii,jj);
    end
end
transfer_mat = cov^(1/2);

% 将产生的数据写入到文件中，写�?2进制文件，可以做到�?�用性�??
fout = fopen(sprintf('cov_1_2_corr_para%.2f.dat', eta),'wb');
fwrite(fout, transfer_mat, 'single');
fclose(fout);
