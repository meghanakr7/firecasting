#!/bin/bash

# move the generated PNG images and metrics to the public website folders
echo "Copying FRP predicted png files to public server.."
#scp -i /home/zsun/.ssh/id_geobrain_no.pem /groups/ESS3/zsun/cmaq/ai_results/evaluation/* zsun@129.174.131.229:/var/www/html/cmaq_site/evaluation/
rsync -u -e "ssh -i /home/zsun/.ssh/id_geobrain_no.pem" -avz /groups/ESS3/zsun/firecasting/data/output/output_xgboost_window_15_days_forecasting/20210714/* zsun@129.174.131.229:/var/www/html/wildfire_site/data/

rsync -u -e "ssh -i /home/zsun/.ssh/id_geobrain_no.pem" -avz  /groups/ESS3/zsun/firecasting/data/output/importance_summary_plot_* zsun@129.174.131.229:/var/www/html/wildfire_site/model/


