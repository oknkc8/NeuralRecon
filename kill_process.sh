ps ax|grep "/home/changho/anaconda3/envs/neucon/" |awk '{print $1}' |xargs kill