import rapidjson

json_data = open('whichtest.json')

'''
emnist en mode cifar109 lasses 
100 img en test (par classe)
300 img en train (par classe)


dict_keys(['store', 'client_names', 'dtest', 'Client0', 'Client1', 'Client2'])
x = images
y = étiquettes (0 ou 1 ici vu que 2 classes)
'''


readed = rapidjson.load(json_data)

print("go")
dtest_y_0 = 0
dtest_y_1 = 0
for yolo in range(len(readed['dtest']['y'])):
    if readed['dtest']['y'][yolo] == 0:
        dtest_y_0 += 1
    else:
        dtest_y_1 += 1
print("len dtest:",len(readed['dtest']['y']))
print("nbr dtest à 0 : ", dtest_y_0)
print("nbr dtest à 1 : ", dtest_y_1)
print("--------------------------")

for i in range(5):
    dtrain_y_0 = 0
    dtrain_y_1 = 0
    for yolo in range(len(readed['Client'+str(i)]['dtrain']['y'])):
        if readed['Client'+str(i)]['dtrain']['y'][yolo] == 0:
            dtrain_y_0 += 1
        else:
            dtrain_y_1 += 1

    print("nbr dtrain à 0 client",str(i), ": ", dtrain_y_0)
    print("nbr dtrain à 1 client",str(i), ": ", dtrain_y_1)
    for yolo in range(len(readed['Client'+str(i)]['dvalid']['y'])):
        if readed['Client'+str(i)]['dvalid']['y'][yolo] == 0:
            dtrain_y_0 += 1
        else:
            dtrain_y_1 += 1

    print("len dtrain+valid client",str(i), ": ", len(readed['Client'+str(i)]['dtrain']['x']) + len(readed['Client'+str(i)]['dvalid']['x']))
    print("nbr dtrain+valid à 0 client",str(i), ": ", dtrain_y_0)
    print("nbr dtrain+valid à 1 client",str(i), ": ", dtrain_y_1)
    print("--------------------------")
