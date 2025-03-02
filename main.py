import os
import argparse
import matplotlib.pyplot as plt
from src.config import FOLDER_PATHS, TRAIN_SET, TEST_SET, PARAMS
from src.data_loader import load_data
from src.fitting import initialize_params, generate_init_params, mpl_adam_fit
from src.evaluation import evaluate_mpl
from src.optimization import optimize_lr_schedule

def main():
    parser = argparse.ArgumentParser(description="Learning Rate Scheduler Fitting")
    parser.add_argument("--folder_path", "-f", type=str, default="400", choices=["25", "100", "400"],
                        help="Model size folder path")
    parser.add_argument("--opt_only", "-o", action="store_true", help="Optimize learning rate schedule only")
    args = parser.parse_args()
    
    folder_path = FOLDER_PATHS[args.folder_path]
    fig_folder = f"./{args.folder_path}M/fit/"
    os.makedirs(fig_folder, exist_ok=True)
    
    # Load and visualize data
    data = load_data(folder_path)
    
    if args.opt_only:
        best_params = PARAMS[args.folder_path]
    else:
        # Fit model
        init_param = initialize_params(data, TRAIN_SET)
        init_params = generate_init_params(init_param)
        best_params, best_loss = mpl_adam_fit(data, TRAIN_SET, TEST_SET, init_params, fig_folder)
        
        # Evaluate
        print("Train Set Evaluation:")
        evaluate_mpl(data, TRAIN_SET, best_params, fig_folder)
        print("Test Set Evaluation:")
        evaluate_mpl(data, TEST_SET, best_params, fig_folder)
        
        print(f"Best Loss: {best_loss}")

    print(f"Best Parameters: {best_params}")
    # Optimize learning rate schedule
    print("\nOptimizing Learning Rate Schedule:")
    opt_eta = optimize_lr_schedule(best_params, name=args.folder_path)
    print("Optimized Learning Rate Schedule:")
    print(opt_eta[:5], " ... ", opt_eta[-5:])
    
if __name__ == "__main__":
    main()