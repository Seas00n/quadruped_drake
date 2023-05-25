#include <cmath>
#include <iostream>

#include <towr/terrain/examples/height_map_examples.h>
#include <towr/nlp_formulation.h>
#include <ifopt/ipopt_solver.h>

#include <towr/initialization/gait_generator.h>
#include <towr/models/endeffector_mappings.h>
#include <Eigen/Dense>

#include <lcm/lcm-cpp.hpp>
#include "../lcm_types/trunklcm/trunk_state_t.hpp"

using namespace towr;

void publish_trunk_state(SplineHolder solution, double t, bool finished=false) {
    
    lcm::LCM lcm;
    trunklcm::trunk_state_t state;
        
    state.timestamp = t;
    state.finished = finished;

    // Base linear position/vel/accel
    Eigen::Map<Eigen::VectorXd>(&state.base_p[0], 3) = solution.base_linear_->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.base_pd[0], 3) = solution.base_linear_->GetPoint(t).v();
    Eigen::Map<Eigen::VectorXd>(&state.base_pdd[0], 3) = solution.base_linear_->GetPoint(t).a();

    // Base angular position/vel/accel
    Eigen::Map<Eigen::VectorXd>(&state.base_rpy[0], 3) = solution.base_angular_->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.base_rpyd[0], 3) = solution.base_angular_->GetPoint(t).v();
    Eigen::Map<Eigen::VectorXd>(&state.base_rpydd[0], 3) = solution.base_angular_->GetPoint(t).a();

    // Foot positions
    Eigen::Map<Eigen::VectorXd>(&state.lf_p[0], 3) = solution.ee_motion_.at(LF)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.rf_p[0], 3) = solution.ee_motion_.at(RF)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.lh_p[0], 3) = solution.ee_motion_.at(LH)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.rh_p[0], 3) = solution.ee_motion_.at(RH)->GetPoint(t).p();

    // Foot velocities
    Eigen::Map<Eigen::VectorXd>(&state.lf_pd[0], 3) = solution.ee_motion_.at(LF)->GetPoint(t).v();
    Eigen::Map<Eigen::VectorXd>(&state.rf_pd[0], 3) = solution.ee_motion_.at(RF)->GetPoint(t).v();
    Eigen::Map<Eigen::VectorXd>(&state.lh_pd[0], 3) = solution.ee_motion_.at(LH)->GetPoint(t).v();
    Eigen::Map<Eigen::VectorXd>(&state.rh_pd[0], 3) = solution.ee_motion_.at(RH)->GetPoint(t).v();

    // Foot accelerations
    Eigen::Map<Eigen::VectorXd>(&state.lf_pdd[0], 3) = solution.ee_motion_.at(LF)->GetPoint(t).a();
    Eigen::Map<Eigen::VectorXd>(&state.rf_pdd[0], 3) = solution.ee_motion_.at(RF)->GetPoint(t).a();
    Eigen::Map<Eigen::VectorXd>(&state.lh_pdd[0], 3) = solution.ee_motion_.at(LH)->GetPoint(t).a();
    Eigen::Map<Eigen::VectorXd>(&state.rh_pdd[0], 3) = solution.ee_motion_.at(RH)->GetPoint(t).a();

    // Foot contact states
    state.lf_contact = solution.phase_durations_.at(LF)->IsContactPhase(t);
    state.rf_contact = solution.phase_durations_.at(RF)->IsContactPhase(t);
    state.lh_contact = solution.phase_durations_.at(LH)->IsContactPhase(t);
    state.rh_contact = solution.phase_durations_.at(RH)->IsContactPhase(t);

    // Foot contact forces
    Eigen::Map<Eigen::VectorXd>(&state.lf_f[0], 3) = solution.ee_force_.at(LF)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.rf_f[0], 3) = solution.ee_force_.at(RF)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.lh_f[0], 3) = solution.ee_force_.at(LH)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.rh_f[0], 3) = solution.ee_force_.at(RH)->GetPoint(t).p();

    lcm.publish("trunk_state", &state);
}


class Obstacle_Block:public HeightMap{
    private:
        double block_start = 0.3;
        double length_ = 0.15;
        double height_ = 0.12;
        double eps_ = 0.03;
        const double slope_ = height_/eps_;
    public:
        double GetHeight(double x, double y) const{
            double h = 0.0;
            if(block_start<=x && x<=block_start+eps_){
                h = slope_*(x-block_start);
            }
            if (block_start+eps_ <= x && x <= block_start+length_){
                h = height_;
            }
            return h;
        }
        double GetHeightDerivWrtX(double x, double y) const{
            double dhdx = 0.0;
            // very steep ramp leading up to block
            if (block_start <= x && x <=block_start+eps_)
                dhdx = slope_;
            return dhdx;
        }
};

int main(int argc, char* argv[]){
    char usage_message[] = "";
    int gait_type;
    if (!strcmp(argv[1],"walk")) {
        gait_type = 0;
    } else if (!strcmp(argv[1],"trot")) {
        gait_type = 1;
    } else if (!strcmp(argv[1],"pace")) {
        gait_type = 2;
    } else if (!strcmp(argv[1],"bound")) {
        gait_type = 3;
    } else if (!strcmp(argv[1],"gallop")) {
        gait_type = 4;
    } else {
        std::cout << "Invalid gait_type " << argv[1] << std::endl;
        std::cout << usage_message << std::endl;
        return 1;
    }

    bool optimize_gait = !strcmp(argv[2],"1");

    float dist_x = std::stof(argv[3]);
    float dist_y = std::stof(argv[4]);

    NlpFormulation formulation;
    formulation.terrain_ = std::make_shared<Obstacle_Block>();
    formulation.model_ = RobotModel(RobotModel::MiniCheetah);
    
    /*
    初始的四足位置
    (LF)(0.2,0.11,0)-----(RF)(0.2,-0.11,0)
    |                                    |
    |         （Base)(0,0,0.3)           |
    |                                    |
    |                                    |
    (LH)(-0.2,0.11,0)----(RH)(-0.2,-0.11,0)
    */
    auto nominal_stance_B = formulation.model_.kinematic_model_->GetNominalStanceInBase();
    double z_ground = 0;
    formulation.initial_ee_W_ = nominal_stance_B;
    std::for_each(formulation.initial_ee_W_.begin(), formulation.initial_ee_W_.end(),
        [&](Eigen::Vector3d& p){p.z()=z_ground;});

    formulation.initial_base_.lin.at(kPos).z() = -nominal_stance_B.front().z()+z_ground;
    std::cout<<nominal_stance_B.front()<<std::endl;
    formulation.final_base_.lin.at(towr::kPos)<<dist_x, dist_y, 0.2;
    
    
    
    double totoal_duration = 4.0;

    auto gait_gen_= GaitGenerator::MakeGaitGenerator(4);
    auto id_gait   = static_cast<GaitGenerator::Combos>(gait_type); // 0=walk, 1=flying trot, 2=pace, 3=bound, 4=gallop
    gait_gen_->SetCombo(id_gait);
    for (int ee=0; ee<4; ++ee) {
        formulation.params_.ee_phase_durations_.push_back(gait_gen_->GetPhaseDurations(totoal_duration, ee));
        formulation.params_.ee_in_contact_at_start_.push_back(gait_gen_->IsInContactAtStart(ee));
    }
    if (optimize_gait) {
        formulation.params_.OptimizePhaseDurations();
    }
    ifopt::Problem nlp;
    SplineHolder solution;
    for (auto c : formulation.GetVariableSets(solution))
        nlp.AddVariableSet(c);
    for (auto c : formulation.GetConstraints(solution))
        nlp.AddConstraintSet(c);
    for (auto c : formulation.GetCosts())
        nlp.AddCostSet(c);
    
    auto solver = std::make_shared<ifopt::IpoptSolver>();
    solver->SetOption("jacobian_approximation","exact");
    solver->SetOption("derivative_test","none");
    solver->SetOption("max_cpu_time",50.0);
    solver->Solve(nlp);

    lcm::LCM lcm;
    trunklcm::trunk_state_t state;

    double dt = 0.01;
    for(double t=0; t<totoal_duration; t=t+dt){
        publish_trunk_state(solution, t);
        std::cout<<state.base_p<<std::endl;
    }
    publish_trunk_state(solution, totoal_duration, true);
}