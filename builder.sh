export LD_LIBRARY_PATH=./LayerNormPlugin/:$LD_LIBRARY_PATH
python builder.py -x bert-base-uncased/model.onnx -c bert-base-uncased/ -o bert-base-uncased/model.plan | tee log.txt
# python builder.py -x bert-base-uncased/model.onnx -c bert-base-uncased/ -o bert-base-uncased/model.plan -p calibration-cache -i| tee log.txt