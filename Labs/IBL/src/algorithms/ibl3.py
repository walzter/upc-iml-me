import numpy as np

from src.algorithms.i_ibl import IIBL


class IBL3(IIBL):
    def __init__(self, sim_func):
        self.sim_func = sim_func
        self.x_train = None
        self.y_train = None
        self.min_threshold = -5
        self.max_threshold = 5
        self.CD = None
        self.CD_Y = None
        self.CD_times_used = None
        self.CD_times_correct = None
        self.class_freq = None

    def fit(self, x_train, y_train):
        self.CD = [x_train.iloc[0]]
        self.CD_Y = [y_train[0]]
        self.CD_times_used = [1]
        self.CD_times_correct = [1]
        self.class_freq = np.zeros(len(np.unique(y_train)))
        self.class_freq[y_train[0]] += 1

        for i in range(1, len(x_train)):
            self.class_freq[y_train[i]] += 1
            sim = self._get_sim(self.CD, x_train.iloc[i])
            accepted_index = self._get_accepted_instances()
            max_sim_ind = self._get_max_sim_index(sim, accepted_index)
            cd_pred_class = self.CD_Y[max_sim_ind]

            ## If not found in CD, add it as a sample
            if not cd_pred_class == y_train[i]:
                # Store the new instance
                self.CD.append(x_train.iloc[i])
                self.CD_Y.append(y_train[i])
                self.CD_times_used.append(1)
                self.CD_times_correct.append(1)

            self._update_CD(sim, max_sim_ind, y_train[i])

    def _update_CD(self, all_sim, max_accepted_sim, y_train):
        to_remove = []

        for i in range(len(all_sim)):
            if all_sim[i] <= all_sim[max_accepted_sim]:
                self.CD_times_used[i] += 1

                if self.CD_Y[i] == y_train:
                    self.CD_times_correct[i] += 1

                if self._is_rejected(i):
                    to_remove.append(i)

        self.CD_times_correct = [self.CD_times_correct[i] for i in range(len(self.CD_times_correct)) if i not in to_remove]
        self.CD_times_used = [self.CD_times_used[i] for i in range(len(self.CD_times_used)) if i not in to_remove]
        self.CD = [self.CD[i] for i in range(len(self.CD)) if i not in to_remove]
        self.CD_Y = [self.CD_Y[i] for i in range(len(self.CD_Y)) if i not in to_remove]

    def _get_max_sim_index(self, sim, accepted_index):
        if len(accepted_index) > 0:
            accepted_instances = [sim[i] for i in range(len(sim)) if i in accepted_index]
            return np.argmin(accepted_instances)

        return np.random.randint(len(sim))

    def _get_sim(self, sample_pool, to_match):
        sim = np.zeros(len(sample_pool))
        for i in range(0, len(sample_pool)):
            sim[i] = self.sim_func(to_match, sample_pool[i])

        return sim

    def _get_accepted_instances(self):
        accepted = []

        for i in range(len(self.CD)):
            if self._is_accepted(i):
                accepted.append(i)

        return accepted

    def _is_accepted(self, index):
        total_samples = np.sum(self.class_freq)
        observed_frequency = self.CD_times_correct[index] / total_samples
        instance_accuracy = self.CD_times_correct[index] / self.CD_times_used[index]

        class_high_point, _ = self._get_interval(0.9, observed_frequency, total_samples)
        _, instance_low_point = self._get_interval(0.9, instance_accuracy, self.CD_times_used[index])

        return instance_low_point > class_high_point

    def _is_rejected(self, index):
        total_samples = np.sum(self.class_freq)
        observed_frequency = self.CD_times_correct[index] / total_samples
        instance_accuracy = self.CD_times_correct[index] / self.CD_times_used[index]

        _, class_low_point = self._get_interval(0.7, observed_frequency, total_samples)
        instance_high_point, _ = self._get_interval(0.7, instance_accuracy, self.CD_times_used[index])

        return class_low_point > instance_high_point

    def _get_interval(self, z, proportion, sample_size):
        z_calculation = z * np.sqrt(proportion * (1 - proportion) / sample_size)
        upper_value = proportion + z_calculation
        lower_value = proportion - z_calculation

        return upper_value, lower_value

    def predict(self, x_pred):
        y_index = np.argmin(self._get_sim(self.CD, x_pred))

        return self.CD_Y[y_index]
