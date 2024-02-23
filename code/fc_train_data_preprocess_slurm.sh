#!/bin/bash

# This file is dedicated to prepare the training data.

# 1) we need a complete rewrite of this process.
# 2) separate the training data preparation and testing data preparation.
# 3) All the share functions should go to the util process. 

echo "start to run test_data_slurm_generated.sh"
pwd

# Specify the name of the script you want to submit
SCRIPT_NAME="fc_model_data_preprocess_slurm_generated.sh"
echo "write the slurm script into ${SCRIPT_NAME}"
cat > ${SCRIPT_NAME} << EOF
#!/bin/bash
#SBATCH -J fc_model_data_preprocessing       # Job name
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file`
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file`
#SBATCH -n 1               # Number of tasks
#SBATCH -c 12               # Number of CPUs per task (threads)
#SBATCH --mem=50G          # Memory per node (use units like G for gigabytes) - this job must need 200GB lol
#SBATCH -t 0-01:00         # Runtime in D-HH:MM format
## Slurm can send you updates via email
#SBATCH --mail-type=FAIL  # BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=zsun@gmu.edu     # Put your GMU email address here

# Activate your customized virtual environment
source /home/zsun/anaconda3/bin/activate

python -u << INNER_EOF

from fc_train_data_preprocess import prepare_training_data

if __name__ == "__main__":
  # this is today, and we want to use all the meteo data of today and FRP data of day -7 - yesterday to predict today's FRP. 
  training_end_date = "20200715" # the last day of the 7 day history
  training_data_folder = "/groups/ESS3/zsun/firecasting/data/train/all_cells/"
  prepare_training_data(training_end_date, training_data_folder)

INNER_EOF

EOF

# Submit the Slurm job and wait for it to finish
echo "sbatch ${SCRIPT_NAME}"

# Submit the Slurm job
job_id=$(sbatch ${SCRIPT_NAME} | awk '{print $4}')
echo "job_id="${job_id}

if [ -z "${job_id}" ]; then
    echo "job id is empty. something wrong with the slurm job submission."
    exit 1
fi

# Wait for the Slurm job to finish
file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
previous_content=$(<"${file_name}")
while true; do
    # Capture the current content
    file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
    current_content=$(<"${file_name}")

    # Compare current content with previous content
    diff_result=$(diff <(echo "$previous_content") <(echo "$current_content"))
    # Check if there is new content
    if [ -n "$diff_result" ]; then
        echo "$diff_result"
    fi
    # Update previous content
    previous_content="$current_content"


    job_status=$(scontrol show job ${job_id} | awk '/JobState=/{print $1}')
    if [[ $job_status == *"COMPLETED"* || $job_status == *"CANCELLED"* || $job_status == *"FAILED"* || $job_status == *"TIMEOUT"* || $job_status == *"NODE_FAIL"* || $job_status == *"PREEMPTED"* || $job_status == *"OUT_OF_MEMORY"* ]]; then
        echo "Job $job_id has finished with state: $job_status"
        break;
    fi
    sleep 10  # Adjust the sleep interval as needed
done

echo "Slurm job ($job_id) has finished."

echo "Print the job's output logs"
sacct --format=JobID,JobName,State,ExitCode,MaxRSS,Start,End -j $job_id
#find /scratch/zsun/ -type f -name "*${job_id}.out" -exec cat {} \;

echo "All slurm job for ${SCRIPT_NAME} finishes."

