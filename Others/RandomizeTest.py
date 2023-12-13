import random



candidate = ["P_1", "P_2", "P_3", "P_4"]
is_print=True
for i in range(200):
    random.shuffle(candidate)
    for j in range(len(candidate)-2):
        is_print = True
        if len(candidate[j]) == 3:
            if (candidate[j] in candidate[j+1])& (candidate[j] in candidate[j+2]):
                is_print = False
                break
        else:
            first = candidate[j].split("_")[0]
            second = candidate[j].split("_")[1]
            if ((first in candidate[j+1])& (first in candidate[j+2])) | ((second in candidate[j+1]) & (second in candidate[j+2])):
                is_print = False
                break
    if is_print:
        print(candidate)


