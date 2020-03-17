rm profile_log
for thread_num in 1 2 4 8 16
do
    $PYTHONROOT/bin/python benchmark.py --thread 4 --model ifaw_conf_model/ifaw_client_conf/serving_client_conf.prototxt  --request http > profile 2>&1
    echo "========================================"
    $PYTHONROOT/bin/python ../util/show_profile.py profile $thread_num >> profile_log
    tail -n 3 profile >> profile_log
done
