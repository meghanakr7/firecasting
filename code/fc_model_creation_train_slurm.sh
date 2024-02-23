#!/bin/bash

# Specify the name of the script you want to submit
SCRIPT_NAME="fc_model_creation_train_slurm_generated.sh"
echo "write the slurm script into ${SCRIPT_NAME}"
cat > ${SCRIPT_NAME} << EOF
#!/bin/bash
#SBATCH -J fc_model_creation_train       # Job name
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file`
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file`
#SBATCH -n 1               # Number of tasks
#SBATCH -c 12               # Number of CPUs per task (threads)
#SBATCH --mem=50G          # Memory per node (use units like G for gigabytes) - this job must need 200GB lol
#SBATCH -t 0-03:00         # Runtime in D-HH:MM format
## Slurm can send you updates via email
#SBATCH --mail-type=FAIL  # BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=zsun@gmu.edu     # Put your GMU email address here

# Activate your customized virtual environment
source /home/zsun/anaconda3/bin/activate

python -u << INNER_EOF

from fc_model_creation import train_model

# Define your start and end dates as strings
start_date_str = "20200701"
end_date_str = "20200703"
training_data_folder = "/groups/ESS3/zsun/firecasting/data/train/"
train_model(start_date_str, end_date_str, training_data_folder)
print("all training on {training_data_folder} is done")


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



