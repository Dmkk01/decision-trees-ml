from formatter import test
import monkdata as m
import dtree as d
from drawtree_qt5 import drawTree
import random
import matplotlib.pyplot as plt
from statistics import stdev

monk_1 = d.entropy(m.monk1)
monk_2 = d.entropy(m.monk2)
monk_3 = d.entropy(m.monk3)
print(monk_1, monk_2, monk_3)



monk_1_a_1 = d.averageGain(m.monk1, m.attributes[0])
monk_1_a_2 = d.averageGain(m.monk1, m.attributes[1])
monk_1_a_3 = d.averageGain(m.monk1, m.attributes[2])
monk_1_a_4 = d.averageGain(m.monk1, m.attributes[3])
monk_1_a_5 = d.averageGain(m.monk1, m.attributes[4])
monk_1_a_6 = d.averageGain(m.monk1, m.attributes[5])
print(monk_1_a_1, monk_1_a_2, monk_1_a_3, monk_1_a_4, monk_1_a_5, monk_1_a_6)

monk_2_a_1 = d.averageGain(m.monk2, m.attributes[0])
monk_2_a_2 = d.averageGain(m.monk2, m.attributes[1])
monk_2_a_3 = d.averageGain(m.monk2, m.attributes[2])
monk_2_a_4 = d.averageGain(m.monk2, m.attributes[3])
monk_2_a_5 = d.averageGain(m.monk2, m.attributes[4])
monk_2_a_6 = d.averageGain(m.monk2, m.attributes[5])
print(monk_2_a_1, monk_2_a_2, monk_2_a_3, monk_2_a_4, monk_2_a_5, monk_2_a_6)

monk_3_a_1 = d.averageGain(m.monk3, m.attributes[0])
monk_3_a_2 = d.averageGain(m.monk3, m.attributes[1])
monk_3_a_3 = d.averageGain(m.monk3, m.attributes[2])
monk_3_a_4 = d.averageGain(m.monk3, m.attributes[3])
monk_3_a_5 = d.averageGain(m.monk3, m.attributes[4])
monk_3_a_6 = d.averageGain(m.monk3, m.attributes[5])
print(monk_3_a_1, monk_3_a_2, monk_3_a_3, monk_3_a_4, monk_3_a_5, monk_3_a_6)

t_1 = d.buildTree(m.monk1, m.attributes);
print(d.check(t_1, m.monk1))
print(d.check(t_1, m.monk1test))

t_2 = d.buildTree(m.monk2, m.attributes);
print(d.check(t_2, m.monk2))
print(d.check(t_2, m.monk2test))

t_3 = d.buildTree(m.monk3, m.attributes);
print(d.check(t_3, m.monk3))
print(d.check(t_3, m.monk3test))


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]



def prune(dataset, fraction, testset):
    train_set, valid_set = partition(dataset, fraction)
    newTree = d.buildTree(train_set, m.attributes)   
    smallestError = 1000000 
    
    while True:
        alternativeTrees = d.allPruned(newTree)
        if len(alternativeTrees) == 1:
            break
        
        bestTree = 0
        minVal = 10000000000
        
        for i in range(1, len(alternativeTrees)):
            x = 1 - d.check(alternativeTrees[i], valid_set)
            if x < minVal:
                minVal = x
                bestTree = i
                
        if minVal <= smallestError:
            smallestError = minVal
        else:
            break
        
        newTree = alternativeTrees[bestTree]
    
    return 1 - d.check(newTree, testset)        

def assignment_seven():
    monk1_std = [0,0,0,0,0,0]
    # monk2_std = [0,0,0,0,0,0]
    monk3_std = [0,0,0,0,0,0]

    monk1_mean = [0,0,0,0,0,0]
    # monk2_mean = [0,0,0,0,0,0]
    monk3_mean = [0,0,0,0,0,0]

    max_iterations = 100
    
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for i in range(len(fractions)):
        monk1_error_rates = []
        # monk2_error_rates = []
        monk3_error_rates = []
        monk1_total_error = 0
        # monk2_total_error = 0
        monk3_total_error = 0
        
        for j in range(max_iterations):
            error_1 = prune(m.monk1, fractions[i], m.monk1test)
            monk1_total_error += error_1
            monk1_error_rates.append(error_1)
            
            # error_2 = prune(m.monk2, fractions[i], m.monk2test)
            # monk2_total_error += error_2
            # monk2_error_rates.append(error_2)
            
            error_3 = prune(m.monk3, fractions[i], m.monk3test)
            monk3_total_error += error_3
            monk3_error_rates.append(error_3)
            
        monk1_mean[i] = monk1_total_error / max_iterations
        # monk2_mean[i] = monk2_total_error / max_iterations
        monk3_mean[i] = monk3_total_error / max_iterations
        
        monk1_std[i] = stdev(monk1_error_rates)
        # monk2_std[i] = stdev(monk2_error_rates)
        monk3_std[i] = stdev(monk3_error_rates)
        
    plt.subplot(2, 2, 1)    
    plt.plot(fractions, monk1_mean, marker=".", label='mean of errors')
    plt.title('MONK-1')
    plt.xlabel('fractions')
    plt.ylabel('error mean')
    plt.legend(loc="upper right")
    
    # plt.subplot(2, 3, 2)  
    # plt.plot(fractions, monk2_mean, marker=".", label='mean of errors')
    # plt.title('MONK-2')
    # plt.xlabel('fractions')
    # plt.ylabel('error mean')
    # plt.legend(loc="upper right")
    
    plt.subplot(2, 2, 2)  
    plt.plot(fractions, monk3_mean, marker=".", label='mean of errors')
    plt.title('MONK-3')
    plt.xlabel('fractions')
    plt.ylabel('error mean')
    plt.legend(loc="upper right")
    
    plt.subplot(2, 2, 3)  
    plt.plot(fractions, monk1_std, marker=".", label='std of errors')
    plt.title('MONK-1')
    plt.xlabel('fractions')
    plt.ylabel('error std')
    plt.legend(loc="upper right")
    
    # plt.subplot(2, 3, 5)  
    # plt.plot(fractions, monk2_std, marker=".", label='std of errors')
    # plt.title('MONK-2')
    # plt.xlabel('fractions')
    # plt.ylabel('error std')
    # plt.legend(loc="upper right")
    
    plt.subplot(2, 2, 4)  
    plt.plot(fractions, monk3_std, marker=".", label='std of errors')
    plt.title('MONK-3')
    plt.xlabel('fractions')
    plt.ylabel('error std')
    plt.legend(loc="upper right")
    
    plt.show()
    
assignment_seven()

