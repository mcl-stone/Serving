rm profile_log
for thread_num in 1 2
do
    $PYTHONROOT/bin/python2.7 benchmark.py --thread 4 --model ifaw_conf_model/ifaw_client_conf/serving_client_conf.prototxt  --request http > profile 2>&1
    echo "========================================" >> profile_log
    tail -n 3 profile >> profile_log
done
