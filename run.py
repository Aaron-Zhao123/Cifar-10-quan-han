import os
import quantized_training
import Clustering_weights_cifar10
# os.system('python training_v3.py -p0')
# os.system('python training_v3.py -p1')
# os.system('python training_v3.py -p2')
# os.system('python training_v3.py -p3')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p5')

acc_list = []
pt_acc_list = []
count = 0
pcov = 0
pfc = 0
retrain = 0
cluster = [2,4,8,16,32,64]
while (count < len(cluster)):
    param = [
        ('-pcov',pcov),
        ('-pfc',pfc),
        ('-t', 0),
        ('-cluster',cluster[count])
        ]
    Clustering_weights_cifar10.main(cluster[count])
    pre_train_acc = quantized_training.main(param)
    param = [
        ('-pcov',pcov),
        ('-pfc',pfc),
        ('-t', 0),
        ('-cluster',cluster[count])
        ]
    train_acc = quantized_training.main(param)

    pt_acc_list.append(pre_train_acc)
    acc_list.append(train_acc)
    count = count + 1
    print (acc)
print('accuracy summary: {}'.format(acc_list))
# acc_list = [0.82349998, 0.8233, 0.82319999, 0.81870002, 0.82050002, 0.80400002, 0.74940002, 0.66060001, 0.5011]
with open("ptacc_cifar_quantize_han.txt", "w") as f:
    for item in pt_acc_list:
        f.write("%s\n"%item)

with open("ptacc_cifar_quantize_han.txt", "w") as f:
    for item in acc_list:
        f.write("%s\n"%item)
