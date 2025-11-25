#include <iostream>
#include <cmath>
#include <Highs.h>

int main() {
    std::cout << "--- HiGHS Library Sanity Check ---" << std::endl;

    // 1. Define a Model: Maximize f(x,y) = 1*x + 1*y
    // Subject to simple variable bounds: [0, 1]
    HighsModel model;
    model.lp_.num_col_ = 2;
    model.lp_.num_row_ = 0; // No matrix constraints, just bounds
    model.lp_.sense_ = ObjSense::kMaximize;
    model.lp_.offset_ = 0;

    // Define Column 0 (Variable x)
    model.lp_.col_cost_.push_back(1.0);
    model.lp_.col_lower_.push_back(0.0);
    model.lp_.col_upper_.push_back(1.0);

    // Define Column 1 (Variable y)
    model.lp_.col_cost_.push_back(1.0);
    model.lp_.col_lower_.push_back(0.0);
    model.lp_.col_upper_.push_back(1.0);

    // Define Matrix (Empty, but required format)
    // HiGHS expects a Compressed Sparse Column (CSC) format even if empty
    model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model.lp_.a_matrix_.start_ = {0, 0, 0}; // Start index for col 0, col 1, and end

    // 2. Initialize Solver
    Highs highs;
    // Turn off console output to verify we can silence it
    highs.setOptionValue("output_flag", false);

    // 3. Pass Model
    HighsStatus pass_status = highs.passModel(model);
    if (pass_status != HighsStatus::kOk) {
        std::cerr << "[Error] Failed to pass model to HiGHS." << std::endl;
        return 1;
    }

    // 4. Solve
    HighsStatus run_status = highs.run();
    if (run_status != HighsStatus::kOk) {
        std::cerr << "[Error] Failed to run HiGHS." << std::endl;
        return 1;
    }

    // 5. Check Results
    HighsModelStatus model_status = highs.getModelStatus();
    double obj_value = highs.getObjectiveValue();

    std::cout << "Solver Status: " << highs.modelStatusToString(model_status) << std::endl;
    std::cout << "Objective Value: " << obj_value << std::endl;

    if (model_status == HighsModelStatus::kOptimal && std::abs(obj_value - 2.0) < 1e-6) {
        std::cout << ">> SUCCESS: HiGHS found the correct solution!" << std::endl;
        return 0;
    } else {
        std::cout << ">> FAILURE: Incorrect result. Expected 2.0." << std::endl;
        return 1;
    }
}