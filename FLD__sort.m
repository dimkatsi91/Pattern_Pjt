%% Fisher Linear Discriminant (FLD) 
% 5 Faces - 10 vectors for each face - 5D space 
clear ; close all ; clc;

load vvoice.mat;

%mean values , mass centers of 5-D space
m1=mean(v1');
m2=mean(v2');
m3=mean(v3');
m4=mean(v4');
m5=mean(v5');


%within scatter matrix
cov1=cov(v1');
cov2=cov(v2');
cov3=cov(v3');
cov4=cov(v4');
cov5=cov(v5');

Sw = cov1 + cov2 + cov3 + cov4 + cov5;

%between scatter matrix
a=zeros(5,5);

a(1,:)=m1;
a(2,:)=m2;
a(3,:)=m3;
a(4,:)=m4;
a(5,:)=m5;

Sb = cov(a);    

A = Sw\Sb;          % A=inv(Sb)*Sw;                 

%Generalized eigenvalues equation     
[v, d] = eig(A);

dd = diag(d);

[z, id] = sort(dd,'descend');

% v:eigenvectors/ w:transformation matrix
w(:,1) = abs(v(:,id(1)));
w(:,2) = abs(v(:,id(2)));


% transformation from 5_D space to  2-D space
yv1=w'*v1;
yv2=w'*v2;
yv3=w'*v3;
yv4=w'*v4;
yv5=w'*v5;


%% REPRESENTATION 
%mass centers of the 2-D space
m2_1=mean(yv1');
m2_2=mean(yv2');
m2_3=mean(yv3');
m2_4=mean(yv4');
m2_5=mean(yv5');

%coordinates of the 2_D space mass centers
x1=(m2_1(1,1));y1=(m2_1(1,2));
x2=(m2_2(1,1));y2=(m2_2(1,2));
x3=(m2_3(1,1));y3=(m2_3(1,2));
x4=(m2_4(1,1));y4=(m2_4(1,2));
x5=(m2_5(1,1));y5=(m2_5(1,2));

% Represantation of the vectors in the 2-D space

figure(1),plot(yv1(1,:),yv1(2,:),'r*') %voice 1
hold on
plot(yv2(1,:),yv2(2,:),'g*') %voice 2
hold on
plot(yv3(1,:),yv3(2,:),'b*') %voice 3
hold on
plot(yv4(1,:),yv4(2,:),'m*') %voice 4
hold on
plot(yv5(1,:),yv5(2,:),'c*') %voice 5
title('Representation of the vectors in the 2-D space','Fontname','Courier','fontweight','bold','fontsize',14);


%% Space Segmentation using Euclidian distance
% step 4
% euclidian distance
figure(2)
title('Space Segmentation using Euclidian distance','Fontname','Courier','fontweight','bold','fontsize',14);
hold on

for i=0.25:0.0008:0.65
    for j=0.42:0.0009:0.58
        d1=dist(m2_1,[i;j]);
        d2=dist(m2_2,[i;j]);
        d3=dist(m2_3,[i;j]);
        d4=dist(m2_4,[i;j]);
        d5=dist(m2_5,[i;j]);
        
        dd(1,1)=d1;
        dd(2,1)=d2;
        dd(3,1)=d3;
        dd(4,1)=d4;
        dd(5,1)=d5;
        
        [y,s]=min(dd);
        
        if s==1
            plot(i,j,'r.');
        elseif s==2
            plot(i,j,'g.');
        elseif s==3
            plot(i,j,'b.');
        elseif s==4
            plot(i,j,'m.');
        else
            plot(i,j,'c.');      
        end
    end
end



hold on

plot(x1,y1,'black*');
hold on
plot(x2,y2,'black*');
hold on
plot(x3,y3,'black*');
hold on
plot(x4,y4,'black*');
hold on
plot(x5,y5,'black*');

axis off
%% Space Segmantation using Mahalanobis Distance



%using Mahalanobis distance

s1=cov(yv1');
s2=cov(yv2');
s3=cov(yv3');
s4=cov(yv4');
s5=cov(yv5');

s1_2=inv(s1);
s2_2=inv(s2);
s3_2=inv(s3);
s4_2=inv(s4);
s5_2=inv(s5);

dets1=det(s1);
dets2=det(s2);
dets3=det(s3);
dets4=det(s4);
dets5=det(s5);

figure(3)
title('Space Segmentation using Mahalanobis distance','Fontname','Courier','fontweight','bold','fontsize',14);
hold on

for i=0.25:0.0008:0.65
    for j=0.42:0.001:0.58
        x=[i;j];
        t1=(x-m2_1');
        t2=(x-m2_2');
        t3=(x-m2_3');
        t4=(x-m2_4');
        t5=(x-m2_5');
        
        dm1=-(t1'*s1_2*t1)-log10(dets1);
        dm2=-(t2'*s2_2*t2)-log10(dets2);
        dm3=-(t3'*s3_2*t3)-log10(dets3);
        dm4=-(t4'*s4_2*t4)-log10(dets4);
        dm5=-(t5'*s5_2*t5)-log10(dets5);
        
        dmah(1,1)=dm1;
        dmah(2,1)=dm2;
        dmah(3,1)=dm3;
        dmah(4,1)=dm4;
        dmah(5,1)=dm5;
        
        [k,l]=max(dmah);
        
        if l==1
            plot(i,j,'r.');
        elseif l==2
            plot(i,j,'g.');
        elseif l==3
            plot(i,j,'b.');
        elseif l==4
            plot(i,j,'m.');
        else
            plot(i,j,'c.');
        end
        
    end
end

hold on

plot(x1,y1,'black*');
hold on
plot(x2,y2,'black*');
hold on
plot(x3,y3,'black*');
hold on
plot(x4,y4,'black*');
hold on
plot(x5,y5,'black*');

axis off