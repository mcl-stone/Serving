rm profile_log
for thread_num in 1 2 4
do
    for batch_size in 1 2 4 8
    do
        $PYTHONROOT/bin/python2.7 benchmark.py --thread $thread_num --batch_size $batch_size --model ifaw_conf_model/ifaw_client_conf/serving_client_conf.prototxt  --request http > profile 2>&1
        echo "========================================" >> profile_log
        tail -n 3 profile >> profile_log
    done
done
