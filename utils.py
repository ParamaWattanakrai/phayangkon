import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Data():
    def __init__(self, label, one_hot, ipa, vector):
        self.label = label
        self.one_hot = one_hot
        self.ipa = ipa
        self.vector = vector
    
    def getInformation(self):
        return f'Label: {self.label}\nOne Hot: {self.one_hot}\nIPA: {self.ipa}\nVector: {self.vector}'

class DataLoader():
    def __init__(self):
        character_embedded = np.genfromtxt('onehot_vectors.csv', delimiter=",", dtype='str', skip_header=1, encoding='utf-8')
        data_vectors = np.genfromtxt('ipa_vectors.csv', delimiter=",", dtype='str', encoding='utf-8')

        self.label = character_embedded[:, 0]
        self.one_hot = torch.from_numpy(character_embedded[:, 2:70].astype(np.float32))
        self.ipa = torch.from_numpy(character_embedded[:, 70:86].astype(np.float32))
        self.vector = torch.from_numpy(data_vectors[:, 1:].astype(np.float32))
        self.list_of_data = []
        for index, char in enumerate(self.label):
            self.list_of_data.append(Data(self.label[index], self.one_hot[index], self.ipa[index], self.vector[index]))
    
    def searchData(self, label):
        target_label = label

        selected_data = None
        for data in self.list_of_data:
            if data.label == target_label:
                selected_data = data
                return selected_data

        if selected_data is None:
            return f'No object found with label: {target_label}'

def embed(type, word, data):
    list_of_tensor = []
    for char in word:
        if type == "one_hot":
            list_of_tensor.append(data.searchData(char).one_hot)
        elif type == "ipa":
            list_of_tensor.append(data.searchData(char).vector)    
    final_tensor = torch.stack(list_of_tensor, dim=0)
    return final_tensor

# -----------------------------Trial-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    data = DataLoader()

    while True:
        sentence = input("\n>>> ")
        print(embed("one_hot", sentence, data))