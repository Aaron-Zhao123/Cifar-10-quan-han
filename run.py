import os
import quantized_training
# os.system('python training_v3.py -p0')
# os.system('python training_v3.py -p1')
# os.system('python training_v3.py -p2')
# os.system('python training_v3.py -p3')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p5')

acc_list = []
count = 0
pcov = 0
pfc = 0
retrain = 0
while (count < 1):
    param = [
        ('-pcov',pcov),
        ('-pfc',pfc)
        ]
    acc = quantized_training.main(param)
    acc_list.append(acc)
    retrain = 0
    count = count + 1
    print (acc)
print('accuracy summary: {}'.format(acc_list))
# acc_list = [0.82349998, 0.8233, 0.82319999, 0.81870002, 0.82050002, 0.80400002, 0.74940002, 0.66060001, 0.5011]
# with open("acc_cifar.txt", "w") as f:
#     for item in acc_list:
#         f.write("%s\n"%item)
