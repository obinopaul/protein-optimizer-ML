import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler 

class TimeSeriesDataGenerator:
    def __init__(self, w, t, a, b, mod, normalize_abnormal=False):
        self.w = w
        self.t = t
        self.a = a
        self.b = b
        self.mod = mod
        self.normalize_abnormal = normalize_abnormal 
        self.data = None
        self.labels = None
        self.weights = None
        self.binary = False

    def generate_binary_class_data(self, abtype):
        s = np.arange(1, self.w + 1)
        data1 = np.random.randn(self.a, self.w)

        data2 = np.zeros((self.b, self.w))
        for i in range(self.b):
            if abtype == 1:
                data2[i] = np.random.randn(self.w) + self.t * s
            elif abtype == 2:
                data2[i] = np.random.randn(self.w) - self.t * s
            elif abtype == 3:
                data2[i] = np.random.randn(self.w) + self.t * np.ones(self.w)
            elif abtype == 4:
                data2[i] = np.random.randn(self.w) - self.t * np.ones(self.w)
            elif abtype == 5:
                data2[i] = np.random.randn(self.w) + self.t * (-1) ** s
            elif abtype == 6:
                data2[i] = np.random.randn(self.w) + self.t * np.cos(2 * np.pi * s / 8)
            elif abtype == 7:
                data2[i] = self.t * np.random.randn(self.w)

        # Normalize abnormal data if needed
        if self.normalize_abnormal:
            scaler = StandardScaler()
            data2 = scaler.fit_transform(data2)
            
        data = np.vstack((data1, data2))
        labels = np.hstack((np.ones(self.a), np.zeros(self.b)))
        data = np.hstack((data, labels.reshape(-1, 1)))

        if self.mod == 2:
            weights = np.hstack((np.ones(self.a) / self.a, np.ones(self.b) / self.b))
        else:
            weights = np.ones(self.a + self.b)

        data = np.hstack((data, weights.reshape(-1, 1)))

        self.data = data
        self.labels = labels
        self.weights = weights
        self.binary = True

    def generate_multiclass_data(self, abtypes):
        s = np.arange(1, self.w + 1)
        num_classes = len(abtypes) + 1
        data1 = np.random.randn(self.a, self.w)

        data2 = []
        labels = []

        for idx, abtype in enumerate(abtypes, start=1):
            for _ in range(self.b):
                if abtype == 1:
                    data = np.random.randn(self.w) + self.t * s
                elif abtype == 2:
                    data = np.random.randn(self.w) - self.t * s
                elif abtype == 3:
                    data = np.random.randn(self.w) + self.t * np.ones(self.w)
                elif abtype == 4:
                    data = np.random.randn(self.w) - self.t * np.ones(self.w)
                elif abtype == 5:
                    data = np.random.randn(self.w) + self.t * (-1) ** s
                elif abtype == 6:
                    data = np.random.randn(self.w) + self.t * np.cos(2 * np.pi * s / 8)
                elif abtype == 7:
                    data = self.t * np.random.randn(self.w)
                data2.append(data)
                labels.append(idx)

        data2 = np.array(data2)

        # Normalize abnormal data if needed
        if self.normalize_abnormal:
            scaler = StandardScaler()
            data2 = scaler.fit_transform(data2)
            
        data = np.vstack((data1, data2))
        labels = np.hstack((np.zeros(self.a), np.array(labels)))
        data = np.hstack((data, labels.reshape(-1, 1)))

        if self.mod == 2:
            weights_normal = np.ones(self.a) / self.a
            weights_abnormal = np.ones(len(labels) - self.a) / self.b
            weights = np.hstack((weights_normal, weights_abnormal))
        else:
            weights = np.ones(self.a + len(labels) - self.a)

        data = np.hstack((data, weights.reshape(-1, 1)))

        self.data = data
        self.labels = labels
        self.weights = weights
        self.binary = False

    def get_data(self):
        if self.data is None:
            raise ValueError("Data not generated. Please call generate_data() first.")
        return self.data[:, :-2]

    def get_labels(self):
        if self.labels is None:
            raise ValueError("Data not generated. Please call generate_data() first.")
        return self.labels

    def get_weights(self):
        if self.weights is None:
            raise ValueError("Data not generated. Please call generate_data() first.")
        return self.weights

    def visualize_data(self, abtypes_to_visualize=None):
        if self.data is None:
            raise ValueError("Data not generated. Please call generate_data() first.")
        
        plt.figure(figsize=(10, 6))

        abtype_names = {
            1: 'Uptrend',
            2: 'Downtrend',
            3: 'Upshift',
            4: 'Downshift',
            5: 'Systematic',
            6: 'Cyclic',
            7: 'Stratification'
        }

        if self.binary:
            label_names = {0: 'Abnormal', 1: 'Normal'}
        else:
            if abtypes_to_visualize is None:
                abtypes_to_visualize = list(abtype_names.keys())
            label_names = {0: 'Normal'}
            for idx in abtypes_to_visualize:
                label_names[idx] = f'Abnormal {abtype_names[idx]}'

        colors = plt.cm.get_cmap('tab10', len(label_names))

        for i in range(len(self.labels)):
            label = int(self.data[i, -2])
            if label in label_names:
                plt.plot(self.data[i, :-2], color=colors(label), alpha=0.3, label=label_names[label] if i == 0 else "")

        handles = []
        for label, name in label_names.items():
            handles.append(plt.Line2D([0], [0], color=colors(label), lw=2, label=name))
        plt.legend(handles=handles)

        plt.title('Time Series Data Visualization')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()


def save_libsvm_format(data, labels, filename):
    with open(filename, 'w') as f:
        for i in range(data.shape[0]):
            label = int(labels[i])
            features = ' '.join(f"{j+1}:{data[i,j]:.6f}" for j in range(data.shape[1]))
            f.write(f"{label} {features}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic time series data.')
    parser.add_argument('-t', '--type', choices=['bc', 'mc'], required=True, help='Type of classification: bc for binary class, mc for multiclass.')
    parser.add_argument('-d', '--data_path', required=True, help='Path to save the generated data.')
    parser.add_argument('-w', '--window_length', type=int, default=48, help='Window length for time series data.')
    parser.add_argument('--t', type=float, default=0.5, help='Parameter of abnormal pattern.')
    parser.add_argument('-a', type=int, default=20, help='Size of Normal class.')
    parser.add_argument('-b', type=int, default=10, help='Size of abnormal class.')
    parser.add_argument('-m', '--mod', type=int, choices=[1, 2], default=1, help='Mode: 1 for SVM, 2 for WSVM.')

    args = parser.parse_args()

    generator = TimeSeriesDataGenerator(w=args.window_length, t=args.t, a=args.a, b=args.b, mod=args.mod)

    if args.type == 'bc':
        generator.generate_binary_class_data(abtype=1)
        data = generator.get_data()
        labels = generator.get_labels()
        save_libsvm_format(data, labels, args.data_path)
    elif args.type == 'mc':
        abtypes = [1, 2, 3, 4, 5, 6, 7]
        generator.generate_multiclass_data(abtypes=abtypes)
        data = generator.get_data()
        labels = generator.get_labels()
        save_libsvm_format(data, labels, args.data_path)

if __name__ == "__main__":
    main()
