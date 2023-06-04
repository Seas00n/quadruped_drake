pi_hat = sdpvar(30,1);
J1 = cal_Pseudo_Inertial_Matrix(pi_hat(1:10));
function J = cal_Pseudo_Inertial_Matrix(inertial_parameters)
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
   Ic = [Ixx,Ixy,Ixz;Ixy,Iyy,Iyz;Ixz,Iyz,Izz];
   Sigma_c = 0.5*trace(Ic)*eye(3)-Ic;
   c = [cx;cy;cz];
   Sigma = Sigma_c+m*c*c';
   J = sdpvar(4,4);
   J(1:3,1:3) = Sigma;
   J(1:3,4) = m*c';
   J(4,1:3) = m*c;
   J(4,4) = m;
end