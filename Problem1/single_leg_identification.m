clear all;clc;close all
addpath('./mr/')
% load simulation data
load("time.mat")
load("q_qd_torque.mat")
load("qdd.mat")
q = q_qd_torque(1:3,:);
qd = q_qd_torque(13:15,:);
qdd = qdd(1:3,:);
torque = q_qd_torque(25:27,:);
inertial_parameters = zeros(30,1);
% m hx hy hz Ixx Ixy Ixz Iyy Iyz Izz
W = 0.05;
A_half = zeros(30,30);
f_T = zeros(30,1);
A_half = A_half + W*eye(30);
inertial_0 = zeros(1,10);
inertial_0 = add_inertial_parameters(0.54,[0.19,0.049,0],[0.0,0.036,0.], ...
    [0.000381,0.000058,0.00000045,0.000560,0.00000095,0.000444],inertial_0);
inertial_0 = add_inertial_parameters(0.634,[0.19,0.049+0.062,0],[0.0,0.016,-0.02], ...
    [0.001983,0.000245,0.000013,0.002103,0.0000015,0.000508],inertial_0);
inertial_0 = add_inertial_parameters(0.064,[0.19,0.049+0.062,-0.029],[0.0,0.,-0.029], ...
    [0.000245,0.0,0.0,0.000248,0.0,0.000006],inertial_0);

f_T = f_T - 2*W*inertial_0';

for i=1:size(q,2)
    % for each point use some skill to calculate Ym
    Ym = zeros(3,30);
    for j=1:30
        temp_inertial = zeros(30,1);
        temp_inertial(j) = 1;
        Ym(:,j) = rnse(q(:,i),qd(:,i),qdd(:,i),temp_inertial);
    end
    A_half = A_half+Ym'*Ym;
    f_T = f_T-2*torque(:,i)'*Ym;
end
A = 2*A_half;
f = f_T';






function tau = rnse(q_list,qd_list,qdd_list,inertial_parameters)
%{
q_list = joint angle list
qd_list = joint angle velocity list
qdd_list = joint angle accel list
g = [0;0;-9.8]
Mlist = list of link frame {i} relative to {i-1} at home position
Glist = Spatial inertial matrices Gi of links
Slist = Screw axis 
%}
g = [0;0;-9.8];
% from urdf
M01 = [[1,0,0,0.19];
       [0,1,0,0.049];
       [0,0,1,0];
       [0,0,0,1]];
M12 = [[1,0,0,0.00];
       [0,1,0,0.062];
       [0,0,1,0];
       [0,0,0,1]];
M23 = [[1,0,0,0];
       [0,1,0,0];
       [0,0,1,-0.209];
       [0,0,0,1]];
M34 = [[1,0,0,0];
       [0,1,0,0];
       [0,0,1,0];
       [0,0,0,1]];
Ftip = zeros(6,1);
G1 = getG(inertial_parameters(1:10));
G2 = getG(inertial_parameters(11:20));
G3 = getG(inertial_parameters(21:30));
Glist = cat(3,G1,G2,G3);
Mlist = cat(3,M01,M12,M23,M34);
S1 = ScrewToAxis([0.19;0.049;0],[1;0;0],0);
S2 = ScrewToAxis([0.19;0.111;0],[0;-1;0],0);
S3 = ScrewToAxis([0.19;0.111;-0.209],[0;-1;0],0);
Slist = [S1,S2,S3];
tau = InverseDynamics(q_list,qd_list,qdd_list, ...
    g,Ftip,Mlist,Glist,Slist);
end
function G = getG(inertial_parameters)
   m = inertial_parameters(1);
   hx = inertial_parameters(2);
   hy = inertial_parameters(3);
   hz = inertial_parameters(4);
   Ixx = inertial_parameters(5);
   Ixy = inertial_parameters(6);
   Ixz = inertial_parameters(7);
   Iyy = inertial_parameters(8);
   Iyz = inertial_parameters(9);
   Izz = inertial_parameters(10);
   G =  zeros(6,6);
   G(1:3,1:3) = [Ixx,Ixy,Ixz;Ixy,Iyy,Iyz;Ixz,Iyz,Izz];
   G(4:6,1:3) = VecToso3([hx,hy,hz]);
   G(1:3,4:6) = G(4:6,1:3)';
   G(4:6,4:6) = m*eye(3);
end
function para = add_inertial_parameters(m,origin_xyz,cop_xyz,inertial,para)
    if para(1)==0
        para(1) = m;
        para(2:4) = m*origin_xyz+m*cop_xyz;
        para(5:10) = inertial;
    else
        para = [para,m];
        para = [para,m*origin_xyz+m*cop_xyz];
        para = [para,inertial];
    end
end