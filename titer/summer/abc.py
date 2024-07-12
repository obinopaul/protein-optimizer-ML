import numpy as np
import matplotlib.pyplot as plt
import re
import time
from matplotlib.animation import FuncAnimation

class BinaryControlChartPatterns:
    """
    This class generates a set of simulated datasets for the problem of control chart pattern recognition.
    """

    def __init__(self, name, dataset_distribution, time_window=10, imbalanced_ratio=0.95,
                 abnormality_parameters=None, visualization=False):
        if abnormality_parameters is None:
            abnormality_parameters = {'pattern_name': 'uptrend', 'parameter_value': 0.05}
        self.name = name
        self.dataset_distribution = dataset_distribution
        self.time_window = time_window
        self.imbalanced_ratio = imbalanced_ratio
        self.abnormality_parameters = abnormality_parameters
        
        self.abnormality_rates = [1, 0.8, 0.6, 0.4, 0.2, 0]
        
        self.raw_data = self.data_generator()
        
        if visualization:
            self.control_chart_pattern_visualizer('train')

    def raw_data_initializer(self):
        distribution = self.dataset_distribution
        no_samples = sum([sum(distribution[set_name].values()) for set_name in distribution.keys()])
        normal_term = np.random.normal(size=(no_samples, self.time_window))
        np.random.shuffle(normal_term)
        
        initial_data = dict()
        start_index = 0
        for set_name in distribution.keys():
            initial_data[set_name] = dict()
            for class_name in distribution[set_name].keys():
                initial_data[set_name][class_name] = normal_term[start_index:
                                                                 start_index + distribution[set_name][class_name]]
                start_index += distribution[set_name][class_name]

        return initial_data

    def data_generator(self):
        def abnormality_appender(set_name, class_name, abnormality_rate):
            pattern_name = self.abnormality_parameters["pattern_name"]
            parameter_value = self.abnormality_parameters["parameter_value"]
            time_window = self.time_window
            normal_term = initial_data[set_name][class_name]
            
            no_normal_steps = round((1 - abnormality_rate) * time_window)
            no_abnormal_steps = time_window - no_normal_steps
            
            normal_steps = np.zeros(no_normal_steps)
            if pattern_name == 'uptrend':
                abnormal_steps = np.array(list(map(lambda t: parameter_value * t, range(no_abnormal_steps))))
            elif pattern_name == 'downtrend':
                abnormal_steps = - np.array(list(map(lambda t: parameter_value * t, range(no_abnormal_steps))))
            elif pattern_name == 'upshift':
                abnormal_steps = np.array(list(map(lambda t: parameter_value, range(no_abnormal_steps))))
            elif pattern_name == 'downshift':
                abnormal_steps = - np.array(list(map(lambda t: parameter_value, range(no_abnormal_steps))))
            elif pattern_name == "systematic":
                abnormal_steps = np.array(list(map(lambda t: parameter_value if t % 2 == 0 else -1 * parameter_value,
                                                   range(no_abnormal_steps))))
            elif pattern_name == "stratification":
                abnormal_steps = parameter_value * np.random.normal(size=no_abnormal_steps)
            elif pattern_name == "cyclic":
                period = 8
                abnormal_steps = np.array(list(map(lambda t: parameter_value * np.sin(2 * np.pi * t / period),
                                                   range(no_abnormal_steps))))
            else:
                raise ValueError("pattern name NOT found!")
            
            trend_term = np.concatenate((normal_steps, abnormal_steps))
            
            features = normal_term + trend_term
            labels = abnormality_rate * np.ones(features.shape[0])
            
            return features, labels
        
        initial_data = self.raw_data_initializer()
        raw_data = dict()
        for set_name in initial_data.keys():
            raw_data[set_name] = dict()
            for class_name in initial_data[set_name].keys():
                abnormality_rate = self.abnormality_rate_specifier(class_name)
                raw_data[set_name][class_name] = abnormality_appender(set_name, class_name, abnormality_rate)
        
        return raw_data

    def abnormality_rate_specifier(self, subclass_name):
        if subclass_name == 'normal':
            abnormality_rate = 0
        else:
            abnormality_rate = float(re.findall(r"[-+]?\d*\.\d+|\d+", subclass_name)[0])
        
        assert abnormality_rate in self.abnormality_rates
        
        return abnormality_rate

    def control_chart_pattern_visualizer(self, dataset_name='train'):
        if dataset_name not in self.raw_data:
            raise ValueError(f"Dataset {dataset_name} not found. Choose from: train, valid, imbalanced_test, balanced_test.")
        
        dataset = self.raw_data[dataset_name]
        
        figure, axes = plt.subplots(2, 3, figsize=(18, 12))
        figure.tight_layout(pad=4.0)
        
        plot_id = 0
        for class_name in dataset.keys():
            timeseries, labels = dataset[class_name]
            eta = self.abnormality_rate_specifier(class_name)
            index = int(np.random.choice(range(len(timeseries)), 1, replace=False))
            ax = axes[divmod(plot_id, 3)]
            
            ax.plot(timeseries[index], color="#000032", linewidth=2.2, label='Normal')
            if eta > 0:
                time_window = self.time_window
                abnormality_start_point = round(time_window * (1 - eta))
                ax.plot(list(range(abnormality_start_point, time_window)),
                        timeseries[index][abnormality_start_point:],
                        color='#5c0000', linewidth=2.4, label="Abnormal")
            
            pattern_name = self.abnormality_parameters["pattern_name"]
            ax.set_title(f"Class: {class_name} ({pattern_name})", fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.legend(loc='best')
            
            plot_id += 1

        figure.suptitle(f"Control Chart Patterns for {self.name} - {dataset_name}", fontsize=16)
        figure.savefig(f'{self.name}_{dataset_name}.png', dpi=300)
        plt.show()

    def stream_data_generator(self, batch_size=10):
        all_data = self.data_generator()
        set_names = list(all_data.keys())
        
        while True:
            for set_name in set_names:
                for class_name, (features, labels) in all_data[set_name].items():
                    for i in range(0, len(features), batch_size):
                        yield {
                            'features': features[i:i + batch_size],
                            'labels': labels[i:i + batch_size],
                            'set_name': set_name,
                            'class_name': class_name
                        }
            time.sleep(1)

    def dynamic_control_chart_visualizer(self, dataset_name='train', batch_interval=1, total_batches=10):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.tight_layout(pad=4.0)
        dataset = self.raw_data[dataset_name]
        
        lines = []
        for ax in axes.flatten():
            line, = ax.plot([], [], lw=2)
            lines.append(line)
        
        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            batch = next(data_stream)
            for ax in axes.flatten():
                ax.clear()
                
            for i, class_name in enumerate(dataset.keys()):
                features, labels = batch['features'], batch['labels']
                line = lines[i]
                ax = axes.flatten()[i]
                eta = self.abnormality_rate_specifier(class_name)
                
                line.set_data(range(self.time_window), features[frame % len(features)])
                ax.plot(features[frame % len(features)], color="#000032", linewidth=2.2, label='Normal')
                if eta > 0:
                    abnormality_start_point = round(self.time_window * (1 - eta))
                    ax.plot(list(range(abnormality_start_point, self.time_window)),
                            features[frame % len(features)][abnormality_start_point:], color='#5c0000', linewidth=2.4, label="Abnormal")
                ax.set_title(f"Class: {class_name} ({self.abnormality_parameters['pattern_name']})", fontsize=14)
                ax.set_xlabel("Time", fontsize=12)
                ax.set_ylabel("Value", fontsize=12)
                ax.legend(loc='best')

            fig.suptitle(f"Control Chart Patterns for {self.name} - {dataset_name}", fontsize=16)
            return lines
        
        ani = FuncAnimation(fig, update, frames=total_batches, init_func=init, interval=batch_interval * 1000, blit=False)
        plt.show()

# Example Usage:

# Define the dataset distribution
dataset_distribution = {
    'train': {'normal': 570, 'abnormal_0.2': 6, 'abnormal_0.4': 6, 'abnormal_0.6': 6, 'abnormal_0.8': 6, 'abnormal_1': 6},
    'valid': {'normal': 190, 'abnormal_0.2': 2, 'abnormal_0.4': 2, 'abnormal_0.6': 2, 'abnormal_0.8': 2, 'abnormal_1': 2},
    'imbalanced_test': {'normal': 190, 'abnormal_0.2': 2, 'abnormal_0.4': 2, 'abnormal_0.6': 2, 'abnormal_0.8': 2, 'abnormal_1': 2},
    'balanced_test': {'normal': 100, 'abnormal_0.2': 20, 'abnormal_0.4': 20, 'abnormal_0.6': 20, 'abnormal_0.8': 20, 'abnormal_1': 20}
}

# Create an instance of the class
control_chart_patterns = BinaryControlChartPatterns(
    name='example_dataset',
    dataset_distribution=dataset_distribution,
    time_window=48,
    imbalanced_ratio=0.95,
    abnormality_parameters={'pattern_name': 'systematic', 'parameter_value': 0.05},
    visualization=False
)

# Visualize the training data
control_chart_patterns.control_chart_pattern_visualizer('train')

# Stream data in batches
batch_size = 5  # Adjust the batch size as needed
total_batches = 50  # Adjust the total number of batches for the animation
data_stream = control_chart_patterns.stream_data_generator(batch_size=batch_size)
control_chart_patterns.dynamic_control_chart_visualizer(dataset_name='train', batch_interval=1, total_batches=total_batches)
