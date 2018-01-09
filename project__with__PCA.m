%% PROJECT WITH PCA

close all; clear; clc; 

load vvoice.mat; 

v1=v1(:);
v2=v2(:);
v3=v3(:);
v4=v4(:);
v5=v5(:);

V = [v1 v2 v3 v4 v5];

Rx = cov(V);

[v, d] = eig(Rx);

[dd id] = sort(diag(d), 'descend');

w(:,1) = abs(v(:,id(1)));
w(:,2) = abs(v(:,id(2)));


yv1 = w'*reshape(v1,5,10);
yv2 = w'*reshape(v2,5,10);
yv3 = w'*reshape(v3,5,10);
yv4 = w'*reshape(v4,5,10);
yv5 = w'*reshape(v5,5,10);

figure(1),plot(yv1(1,:),yv1(2,:),'r*') %voice 1
hold on
plot(yv2(1,:),yv2(2,:),'g*') %voice 2
hold on
plot(yv3(1,:),yv3(2,:),'b*') %voice 3
hold on
plot(yv4(1,:),yv4(2,:),'m*') %voice 4
hold on
plot(yv5(1,:),yv5(2,:),'c*') %voice 5
title('Representation of the vectors in the 2-D space');


