import numpy as np

def parse_log(filename):
    with open(filename, 'r') as f:
        on_test = False
        
        train_stats = []
        test_stats = []
        
        for i, line in enumerate(f.readlines()[2:]):
            if line[0].isalpha():
                on_test = True
                continue
            spl = line.split("\t")
            if on_test:
                test_stats.extend([float(x) for x in spl])
            else:
                train_stats.extend([float(x) for x in spl])

        return np.array(train_stats).reshape((-1, 3)), np.array(test_stats).reshape((-1, 2))