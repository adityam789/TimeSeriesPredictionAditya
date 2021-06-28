from .ddm import DDM
from .eddm import EDDM
from .hddm_w import HDDM_W
from skmultiflow.drift_detection.detector import ADWIN
adwin = ADWIN()

class CustomClass:
  def __init__(self, detector):
    self.detector = detector
  def update(self, x):
    self.detector.add_element(x)
    return (self.detector.detected_change(), None)

def detector(df, training_size, detector_type):

    isAdwin = True if detector_type == "ADWIN" else False

    if detector_type == "ADWIN":
        detector = CustomClass(adwin)
    elif detector_type == "DDM":
        detector = DDM()
    elif detector_type == "EDDM":
        detector = EDDM()
    else:
        detector = HDDM_W()

    print(training_size)
    change_detected = []
    train_data_change_detected = [0]
    test_data_change_detected = [training_size]

    for i in range(df.size):
        if i < training_size:
            in_drift, in_warning = detector.update(df.iat[i, 0])
            # if detector.detected_warning_zone():
            #     print('Warning detected in data: ' + str(df.iat[i,0]) + ' - at index: ' + str(i))
            if(in_drift):
                train_data_change_detected.append(i)
                change_detected.append(i)
                print('Change detected in data: ' +
                      str(df.iat[i, 0]) + ' - at index: ' + str(i))
        else:
            in_drift, in_warning = detector.update(df.iat[i, 0])
            # if detector.detected_warning_zone():
            #     print('Warning detected in data: ' + str(df.iat[i,0]) + ' - at index: ' + str(i))
            if(in_drift):
                test_data_change_detected.append(i)
                change_detected.append(i)
                print('Change detected in data: ' +
                      str(df.iat[i, 0]) + ' - at index: ' + str(i))

    if train_data_change_detected[-1] != training_size:
        train_data_change_detected += [training_size]
    test_data_change_detected += [df.size]

    return train_data_change_detected, test_data_change_detected, change_detected
