import re
import pandas as pd

# Define patterns
DECIMAL_PATTERN = r"(-?\d+\.\d+)"
INTEGER_PATTERN = r"(\d+)"
WHITESPACE = r"\s+"

# Configure table structure 
DECIMAL_POSITIONS = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16]  # positions with decimal numbers
INTEGER_POSITIONS = [10]  # positions with integer numbers
TOTAL_COLUMNS = 16

columns = [
    "test_in",
    "test_out",
    "train_in",
    "train_out",
    "tr_outloss",
    "data1",
    "data2",
    "data3",
    "data4",
    "data5",
    "step",
    "epoch",
    "loss",
    "loss1",
    "step_time",
    "eval_time",
]

# Build pattern dynamically
patterns = []
for i in range(TOTAL_COLUMNS):
    if i in INTEGER_POSITIONS:
        patterns.append(INTEGER_PATTERN)
    else:
        patterns.append(DECIMAL_PATTERN)

# Combine patterns with whitespace
table_pattern = WHITESPACE.join(patterns)

def analyze_domain_generalization_log(log_filepath):
    """
    Analyzes a domain generalization log file.

    Args:
        log_filepath: Path to the log file.

    Returns:
        A dictionary containing:
            - dataframes: A dictionary of pandas DataFrames, one for each test environment.
            - optimal_steps: A dictionary with optimal steps for each test environment and overall.
    """
    with open(log_filepath, 'r') as f:
        log_content = f.read()
        table_data = re.findall(table_pattern, log_content)
        # print(table_data)

        # parse number of steps by looking for the pattern n_steps = some number
        steps = int(re.search(r"n_steps = (\d+)", log_content).group(1))
        test_envs = [0, 1, 2, 3]
        steps_per_row = 200


    # convert table data to panadas df
    all_data = pd.DataFrame(table_data, columns=columns)
    all_data = all_data.apply(pd.to_numeric)
    test_env_dfs = []
    cur_start_row = 0
    for test_env in test_envs:
        cur_end_row = cur_start_row + steps//steps_per_row + 1
        # get a new df from start to end and append to list
        test_env_df = all_data.iloc[cur_start_row:cur_end_row]
        test_env_dfs.append(test_env_df.reset_index(drop=True))
        cur_start_row = cur_end_row


    best_avg_train_out = 0
    best_step_train_out = None
    best_avg_tr_out_loss = 0
    best_step_tr_out_loss = None
    for stopping_step in range(2000, steps+1, steps_per_row):
        choosen_vals_train_out = []
        choose_vals_tr_out_loss = []
        for test_env, test_env_df in enumerate(test_env_dfs):
            env_name = f"te_{test_env}"
            # get rows that have step less than stopping step
            test_env_df_tmp = test_env_df[test_env_df["step"] <= stopping_step]
            # get the test_in value that corresponds to the highest train_out value
            choosen_vals_train_out.append(test_env_df_tmp.loc[test_env_df_tmp["train_out"].idxmax()]["test_in"])
            choose_vals_tr_out_loss.append(test_env_df_tmp.loc[test_env_df_tmp["tr_outloss"].idxmin()]["test_in"])
        choosen_avg_train_out = sum(choosen_vals_train_out)/len(choosen_vals_train_out)
        choosen_avg_tr_out_loss = sum(choose_vals_tr_out_loss)/len(choose_vals_tr_out_loss)
        # print at stopping step x, the average of choosen vals is y
        #print(f"At step {stopping_step}, the avg_test_in is based on train_out:{choosen_avg_train_out}, based on tr_outloss:{choosen_avg_tr_out_loss}")

        if choosen_avg_train_out > best_avg_train_out:
            best_avg_train_out = choosen_avg_train_out
            best_step_train_out = stopping_step
        if choosen_avg_tr_out_loss > best_avg_tr_out_loss:
            best_avg_tr_out_loss = choosen_avg_tr_out_loss
            best_step_tr_out_loss = stopping_step
    print(f"Best step (Based on max train_out) is {best_step_train_out} with average of {best_avg_train_out * 100:.4f}%")
    print(f"Best step (Based on min tr_outloss) is {best_step_tr_out_loss} with average of {best_avg_tr_out_loss* 100:.4f}%")

if __name__ == "__main__":
    base_dir = "/home/ahmedm04/projects/MoA/train_output/OfficeHome/"
    exp_name = "250202_15-01-59_moe_squared_v3_experts:3, div_loss:0.1,abs"
    log_filepath = "log.txt"
    analyze_domain_generalization_log(log_filepath)