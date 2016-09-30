

def test():
    # We have a sequence labeling task
    x = [[0, 1, 2]]
    y = [[1, -1, 1]]
    
    crf = CRF()
    crf.fit(x, y)
    crf.predict(x)

# We need a feature function

def feature_function(seq, i, y_cur, y_prev):
    # A form of this kind is a linear chain crf because it 
    # only depends on y_cur and y_prev
    return y_cur == 1


def gradient(sentence, labeling):
    grad = 0
    for j, l in enumerate(sentence):
        grad += f(sentence, j, labeling[j], labeling[j - 1])
    
    for labeling_prime in possible_labelings:
        term = prob(sentence, labeling_prime)
        term2 = 0
        for j, l in enumerate(labeling_prime):
            term2 += f(sentence, j, labeling_prime[j], labeling_prime[j - 1])
    
        grad -= term * term2
    
    return grad
    
class CRF:
    
    def __init__(self):
        self.feature_functions = [feature_function]
        self.weights = [1.0]
    
    def score(self, sentence, labeling):        
        score = 0
        for i, w in enumerate(sentence):
            
            # Need to skip the first one
            if i == 0:
                continue
            
            for l, f in zip(self.weights, self.feature_functions):
                score += l * f(sentence, i, labeling[i], labeling[i - 1])

        return score
    
    def possible_labelings(self, length, prefix=[]):
        if length == 1:
            for i in self.labels:
                yield prefix + [i] 
        else:
            for p2 in possible_labelings(length - 1, prefix):
                for i in self.labels:
                    yield p2 + [i]
                    
        
        
    def fit(self, X, y):
        pass
        
    def predict(self,X):
        pass
