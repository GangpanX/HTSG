export HTSG_HOME=$(pwd)
export LOG_DIR="$HTSG_HOME/logs"
export PYTHONPATH="$HTSG_HOME:${PYTHONPATH:-}"
export DATAPATH="./data_path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
