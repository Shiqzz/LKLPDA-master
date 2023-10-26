%[Wrr1,Wmm1] = Rbf_kernel1(w);


function [Gaussian_D,Gaussian_R] = Rbf_kernel(adj)
%Gaussian_D 行相关的相似度
%Gaussian_R 列相关的相似度

A=adj';
length_D = size(A,1);
length_R = size(A,2);
Gaussian_Dis = zeros(length_D,length_D);
pare_a=0; 
sum=0;
temp=0;

for i=1:length_D  
    temp=norm(A(i,:));
    sum=sum+temp^2;
end
pare_a=1/(sum/length_D);

for i=1:length_D
    for j=1:length_D
        Gaussian_Dis(i,j)=exp(-pare_a*(norm(A(i,:)-A(j,:))^2));
    end
end
Gaussian_D = Gaussian_Dis;

Gaussian_miR = zeros(length_R,length_R);
pare_b=0;
sum=0; 
temp=0;

for i=1:length_R
    temp=norm(A(:,i));
    sum=sum+temp^2;
end
pare_b=1/(sum/length_R);

for i=1:length_R
    for j=1:length_R
        Gaussian_miR(i,j)=exp(-pare_b*(norm(A(:,i)-A(:,j))^2));
    end
end
Gaussian_R = Gaussian_miR;
