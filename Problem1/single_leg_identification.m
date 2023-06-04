clear all;clc;close all
addpath('./mr/')
% load simulation data
load("time.mat")
load("q_qd_torque.mat")
load("qdd.mat")
num_points = 2000;
t = t(1001:1000+num_points);
q = q_qd_torque(1:3,1001:1000+num_points);
qd = q_qd_torque(13:15,1001:1000+num_points);
qdd = qdd(1:3,1001:1000+num_points);
torque = q_qd_torque(25:27,1001:1000+num_points);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global m1 xyz_joint1_world xyz_com1_joint1 I1_com
global m2 xyz_joint2_joint1 xyz_joint2_world xyz_com2_joint2 I2_com
global m3 xyz_joint3_joint2 xyz_joint3_world xyz_com3_joint3 I3_com
m1 = 0.54;
xyz_joint1_world = [0.19,0.049,0];
xyz_com1_joint1 = [0,0.036,0];
I1_com = [0.000381,0.000058,0.00000045,0.000560,0.00000095,0.000444];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m2 = 0.634;
xyz_joint2_joint1 = [0,0.062,0.0];
xyz_com2_joint2 = [0.0,0.016,-0.02];
I2_com = [0.001983,0.000245,0.000013,0.002103,0.0000015,0.000508];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m3 = 0.064;
xyz_joint3_joint2 = [0,0,-0.209];
xyz_com3_joint3 = [0,0,-0.209];
I3_com = [0.000245,0.0,0.0,0.000248,0.0,0.000006];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xyz_joint2_world = xyz_joint2_joint1+xyz_joint1_world;
xyz_joint3_world = xyz_joint3_joint2+xyz_joint2_world;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inertial_parameters_default = [change_Ic_coordinate([m1,xyz_com1_joint1,I1_com]),change_Ic_coordinate([m2,xyz_com2_joint2,I2_com]),change_Ic_coordinate([m3,xyz_com3_joint3,I3_com])];
inertial_parameters_default = inertial_parameters_default';

% m hx hy hz Ixx Ixy Ixz Iyy Iyz Izz
num_points = size(q,2);
tau_cal_list = zeros(3,num_points);
pi_hat = sdpvar(30,1);
e = 0;


A = zeros(30,30);
f = zeros(1,30);
for i=1:num_points
    % for each point use some skill to calculate Ym
    Ym = zeros(3,30);
    for j=1:30
        temp_inertial = zeros(30,1);
        temp_inertial(j) = 1;
        Ym(:,j) = rnse(q(:,i),qd(:,i),qdd(:,i),temp_inertial);
    end
    tau_cal_list(:,i) = Ym*inertial_parameters_default;
    e = e+(Ym*pi_hat-torque(:,i))'*(Ym*pi_hat-torque(:,i));
    A = A+Ym'*Ym/num_points;
    f = f-2*torque(:,i)'*Ym/num_points;
end
e = e/num_points+0.2*(pi_hat-inertial_parameters_default)'*(pi_hat-inertial_parameters_default);
A = A+2*eye(30);
f = f-2*inertial_parameters_default';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % use quadprog
pi_ = quadprog(A*2,f');
for i=1:num_points
    % for each point use some skill to calculate Ym
    Ym = zeros(3,30);
    for j=1:30
        temp_inertial = zeros(30,1);
        temp_inertial(j) = 1;
        Ym(:,j) = rnse(q(:,i),qd(:,i),qdd(:,i),temp_inertial);
    end
    tau_cal_list(:,i) = Ym*pi_;
end
plot(t,torque(2,:))
hold on
plot(t,tau_cal_list(2,:))
J1 = cal_Pseudo_Inertial_Matrix(pi_(1:10));
J2 = cal_Pseudo_Inertial_Matrix(pi_(11:20));
J3 = cal_Pseudo_Inertial_Matrix(pi_(21:30));
try chol(J1)
    disp('J1 is symmetric positive definite.')
catch ME
    disp('J1 is not symmetric positive definite')
end

try chol(J2)
    disp('J2 is symmetric positive definite.')
catch ME
    disp('J2 is not symmetric positive definite')
end

try chol(J3)
    disp('J3 is symmetric positive definite.')
catch ME
    disp('J3 is not symmetric positive definite')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J1 = cal_Pseudo_Inertial_Matrix(pi_hat(1:10));
J2 = cal_Pseudo_Inertial_Matrix(pi_hat(11:20));
J3 = cal_Pseudo_Inertial_Matrix(pi_hat(21:30));
Constraints = [J1>=eye(4)*1e-8,J2>=eye(4)*1e-8,J3>=eye(4)*1e-8];
sol = optimize(Constraints,e);
if sol.problem == 0
 disp('x should have a value')
 value(pi_hat)
else
 yalmiperror(sol.problem)
end
pi_hat = value(pi_hat);
for i=1:num_points
    Ym = zeros(3,30);
    for j=1:30
        temp_inertial = zeros(30,1);
        temp_inertial(j) = 1;
        Ym(:,j) = rnse(q(:,i),qd(:,i),qdd(:,i),temp_inertial);
    end
    tau_cal_list(:,i) = Ym*pi_hat;
end
plot(t,tau_cal_list(2,:));
legend('real','use quadprog','use sdp')
J1 = cal_Pseudo_Inertial_Matrix(pi_hat(1:10));
J2 = cal_Pseudo_Inertial_Matrix(pi_hat(11:20));
J3 = cal_Pseudo_Inertial_Matrix(pi_hat(21:30));
try chol(J1)
    disp('J1 is symmetric positive definite.')
catch ME
    disp('J1 is not symmetric positive definite')
end

try chol(J2)
    disp('J2 is symmetric positive definite.')
catch ME
    disp('J2 is not symmetric positive definite')
end

try chol(J3)
    disp('J3 is symmetric positive definite.')
catch ME
    disp('J3 is not symmetric positive definite')
end




function tau = rnse(q_list,qd_list,qdd_list,inertial_parameters)
%{
q_list = joint angle list
qd_list = joint angle velocity list
qdd_list = joint angle accel list
g = [0;0;-9.8]
Mlist = list of link frame {i} relative to {i-1} at home position
Glist = Spatial inertial matrices Gi of links in link i frame(not com frame)
Slist = Screw axis of joint i in world frame at home position.
%}
global m1 xyz_joint1_world xyz_com1_joint1 I1_com
global m2 xyz_joint2_joint1 xyz_joint2_world xyz_com2_joint2 I2_com
global m3 xyz_joint3_joint2 xyz_joint3_world xyz_com3_joint3 I3_com
g = [0;0;-9.8];
% from urdf
M01 = [[1,0,0,xyz_joint1_world(1)];
       [0,1,0,xyz_joint1_world(2)];
       [0,0,1,xyz_joint1_world(3)];
       [0,0,0,1]];
M12 = [[1,0,0,xyz_joint2_joint1(1)];
       [0,1,0,xyz_joint2_joint1(2)];
       [0,0,1,xyz_joint2_joint1(3)];
       [0,0,0,1]];
M23 = [[1,0,0,xyz_joint3_joint2(1)];
       [0,1,0,xyz_joint3_joint2(2)];
       [0,0,1,xyz_joint3_joint2(3)];
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
S1 = ScrewToAxis(xyz_joint1_world',[1;0;0],0);
S2 = ScrewToAxis(xyz_joint2_world',[0;-1;0],0);
S3 = ScrewToAxis(xyz_joint3_world',[0;-1;0],0);
Slist = [S1,S2,S3];
tau = InverseDynamics(q_list,qd_list,qdd_list, ...
    g,Ftip,Mlist,Glist,Slist);
end
function G = getG(inertial_parameters)
   m = inertial_parameters(1);% mass
   hx = inertial_parameters(2);% h=mc
   hy = inertial_parameters(3);
   hz = inertial_parameters(4);
   Ixx = inertial_parameters(5);%I
   Ixy = inertial_parameters(6);
   Ixz = inertial_parameters(7);
   Iyy = inertial_parameters(8);
   Iyz = inertial_parameters(9);
   Izz = inertial_parameters(10);
   G =  zeros(6,6);
   skew_h = VecToso3([hx;hy;hz]);
   G(1:3,1:3) = [Ixx,Ixy,Ixz;Ixy,Iyy,Iyz;Ixz,Iyz,Izz];
   G(4:6,1:3) = skew_h';
   G(1:3,4:6) = skew_h;
   G(4:6,4:6) = m*eye(3);
end
function J = cal_Pseudo_Inertial_Matrix(inertial_parameters)
    m = inertial_parameters(1);% mass
   hx = inertial_parameters(2);%mc
   hy = inertial_parameters(3);
   hz = inertial_parameters(4);
   Ixx = inertial_parameters(5);%I
   Ixy = inertial_parameters(6);
   Ixz = inertial_parameters(7);
   Iyy = inertial_parameters(8);
   Iyz = inertial_parameters(9);
   Izz = inertial_parameters(10); 
   h = [hx;hy;hz];
   I = [Ixx,Ixy,Ixz;Ixy,Iyy,Iyz;Ixz,Iyz,Izz];
   Sigma = [-1/2*Ixx+1/2*Iyy+1/2*Izz,Ixy,Ixz;
           Ixy, -1/2*Iyy+1/2*Ixx+1/2*Izz,Iyz;
           Ixz,Iyz,-1/2*Izz+1/2*Ixx+1/2*Iyy];
   J = sdpvar(4,4);
   J(1:3,1:3) = Sigma;
   J(1:3,4) = h;
   J(4,1:3) = h';
   J(4,4) = m;
end
function inertial_joint = change_Ic_coordinate(inertial_parameters)
   m = inertial_parameters(1);% mass
   cx = inertial_parameters(2);%c
   cy = inertial_parameters(3);
   cz = inertial_parameters(4);
   Ixx = inertial_parameters(5);%Ic
   Ixy = inertial_parameters(6);
   Ixz = inertial_parameters(7);
   Iyy = inertial_parameters(8);
   Iyz = inertial_parameters(9);
   Izz = inertial_parameters(10);
   G =  zeros(6,6);
   skew_c = VecToso3([cx;cy;cz]);
   Ic = [Ixx,Ixy,Ixz;Ixy,Iyy,Iyz;Ixz,Iyz,Izz]+m*skew_c*skew_c';
   I = [Ic(1,1),Ic(1,2),Ic(1,3),Ic(2,2),Ic(2,3),Ic(3,3)];
   inertial_joint = [m,m*cx,m*cy,m*cz,I];
end