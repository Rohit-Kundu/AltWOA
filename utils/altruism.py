import numpy as np
import sklearn.svm

def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features

def classification_accuracy(labels, predictions):
    correct = np.where(labels == predictions)[0]
    accuracy = correct.shape[0]/labels.shape[0]
    return accuracy

def altruism_fitness(pop, train_datas, train_labels, test_datas, test_labels):
    accuracies = np.zeros(pop.shape[0])
    idx = 0

    for curr_solution in pop:
        num_feat = np.where(curr_solution==1)[0].shape[0]
        reduced_train_features = reduce_features(curr_solution, train_datas)
        reduced_test_features = reduce_features(curr_solution, test_datas)
        X=reduced_train_features
        y=train_labels

        SVM_classifier = sklearn.svm.SVC(kernel='rbf',gamma='scale',C=5000)
        SVM_classifier.fit(X, y)
        predictions = SVM_classifier.predict(reduced_test_features)
        accuracies[idx] = classification_accuracy(test_labels, predictions)
        
        idx = idx + 1
    return accuracies

def hamming_distance(b1,b2):
  ans = 0
  for i in range(len(b1)):
    ans += not(b1[i]==b2[i])
  return ans

def similarity(beta, chromosome1, chromosome2, acc1, acc2):
  H_d = hamming_distance(chromosome1,chromosome2)
  D_a = abs(acc1-acc2)
  if (H_d !=0):
    if (D_a !=0): 
      S = beta/H_d + (1-beta)/D_a
    else :
      S = 99999
  else :
    S = 99999
  return S

def get_closest_ind(pop, acc, beta=0.3):
  ind1 = pop[0]
  acc1 = acc[0]
  similarity_list = []
  for i in range(1,len(pop)):
    ind2 = pop[i]
    acc2 = acc[i]
    similarity_list.append(similarity(beta, ind1, ind2, acc1, acc2))
  max_sim_index = similarity_list.index(max(similarity_list))+1
  # 1 is added to the index since the 1st item in similarity_index_list corresponds to individual number 2 in array "pop"

  ind2 = pop[max_sim_index]
  return ind1, ind2, max_sim_index

def group_population(pop, acc):
  grouped_pop = np.zeros(shape=pop.shape)
  count = 0
  while (len(pop)>0):
    grouped_pop[count], grouped_pop[count+1], pos2 = get_closest_ind(pop,acc,beta=0.3)
    count+=2
    pop = np.delete(pop,[0,pos2],axis=0)
    acc = np.delete(acc,[0,pos2],axis=0)
  return grouped_pop

from Py_FS.filter import SCC,PCC
def generate_scc(train_data,train_label):
  scc = SCC(train_data,train_label)
  return scc.scores
def generate_pcc(train_data,train_label):
  pcc = PCC(train_data,train_label)
  return pcc.scores

def Altruism(new_pop, train_data, train_label, test_data, test_label, scc_score, pcc_score, altruism_indi = 8, pop_size=20, alpha = 0.5):
  """
  new_pop: The whole population out of which half will be selected to be the population of next generation (of size 2*pop_size)
  altruism_indi: Number of individuals that will be sacrificed (altruism)
  pop_size = The original size of the population that will be used in the next generation
  alpha, beta: weights for asserting the dominance of one individual over another

  returns: final_pop- a population of size = 'pop_size', where 'altruism_indi' number of mediocre solutions were sacrificed
  """
  if (new_pop.shape[0]-altruism_indi)<pop_size:
    altruism_indi = new_pop.shape[0]-pop_size
    print("Switching to maximum possible number of altruistic individuals = {}.".format(altruism_indi))

  fit = altruism_fitness(new_pop, train_data, train_label, test_data, test_label)

  #Sort the population according to decreasing order of fitness (higher fitness means better solution)
  fit_indices = fit.argsort()
  fit = fit[fit_indices[::-1]]
  new_pop = new_pop[fit_indices[::-1]]

  #Calculate how many best solutions need to be intact
  num_pop_to_keep = int(pop_size-altruism_indi)

  #Select the best (pop_size-altruism_indi) in the final population
  final_pop = new_pop[0:num_pop_to_keep,:]

  #Select 'altruism_indi*2' number of mediocre solutions for the altruism operation (half of these will finally be selected)
  new_pop = new_pop[num_pop_to_keep:num_pop_to_keep+2*altruism_indi,:]

  #Group population according to similarity indices
  grouped_pop = group_population(new_pop, fit)

  #Calculate the fitness of these grouped population
  group_fit = altruism_fitness(grouped_pop, train_data, train_label, test_data, test_label)

  #Initialize the population who will finally survive the altruism operation
  altruism_pop = np.zeros(shape=(altruism_indi, new_pop.shape[1]))
  count = 0
  while (count/2)<altruism_indi:
    player1 = grouped_pop[count]
    player2 = grouped_pop[count+1]

    idx1 = np.where(player1==1)[0]
    idx2 = np.where(player2==1)[0]
    scc1 = np.average(scc_score[idx1])
    scc2 = np.average(scc_score[idx2])
    pcc1 = np.average(pcc_score[idx1])
    pcc2 = np.average(pcc_score[idx2])

    player1_score = alpha*scc1 + (1-alpha)*pcc1
    player2_score = alpha*scc2 + (1-alpha)*pcc2

    if player1_score <= player2_score:
      altruism_pop[int(count/2)] = player1
    else:
      altruism_pop[int(count/2)] = player2

    count+=2
  
  #Merge the population that was kept intact and the altruistic individuals to form the final population
  final_pop = np.concatenate((final_pop, altruism_pop), axis=0)
  
  return final_pop.astype(int)
