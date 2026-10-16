import mne
import numpy as np
import os
import scipy.io as sio
from scipy.io import savemat



def preprocess_and_save_subjects(dir_path):
    for subject_id in range(1, 10):
        print(f"Processing subject {subject_id}...")

        train_gdf_path = dir_path + f'lyx_WorkSpace/Datasets/BCI4_2A_Original/A0{subject_id}T.gdf'
        raw_train = mne.io.read_raw_gdf(train_gdf_path, preload=True)

        events_train, event_id_dict_train = mne.events_from_annotations(raw_train)
        raw_train.load_data()

        raw_train.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

        eeg_picks_train = mne.pick_types(
            raw_train.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads'
        )

        event_id_train = {
            'Left': event_id_dict_train['769'],
            'Right': event_id_dict_train['770'],
            'Foot': event_id_dict_train['771'],
            'Tongue': event_id_dict_train['772']
        }

        selected_events_train = events_train[
            np.isin(events_train[:, 2], list(event_id_train.values()))
        ]

        epochs_train = mne.Epochs(
            raw_train, selected_events_train, event_id_train,
            tmin=0, tmax=3.996, proj=False, picks=eeg_picks_train,
            preload=True, baseline=None
        )

        train_data_uv = epochs_train.get_data() * 1e6  

        train_label_path = dir_path + f'lyx_WorkSpace/Datasets/BCI4_2A_Original/A0{subject_id}T.mat'
        train_labels = sio.loadmat(train_label_path)['classlabel']  # shape: (N, 1)



        test_gdf_path = dir_path + f'lyx_WorkSpace/Datasets/BCI4_2A_Original/A0{subject_id}E.gdf'
        raw_test = mne.io.read_raw_gdf(test_gdf_path, preload=True)

        events_test, event_id_dict_test = mne.events_from_annotations(raw_test)
        raw_test.load_data()

        raw_test.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

        eeg_picks_test = mne.pick_types(
            raw_test.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads'
        )

        event_id_test = {'Unknown': event_id_dict_test['783']}
        selected_events_test = events_test[
            np.isin(events_test[:, 2], list(event_id_test.values()))
        ]

        epochs_test = mne.Epochs(
            raw_test, selected_events_test, event_id_test,
            tmin=0, tmax=3.996, proj=False, picks=eeg_picks_test,
            preload=True, baseline=None
        )

        test_data_uv = epochs_test.get_data() * 1e6

        test_label_path = dir_path + f'lyx_WorkSpace/Datasets/BCI4_2A_Original/A0{subject_id}E.mat'
        test_labels = sio.loadmat(test_label_path)['classlabel']  # shape: (N, 1）

        output_path = dir_path + f'lyx_WorkSpace/Datasets/Mydata2/A0{subject_id}.mat'
        savemat(output_path, {
            'train_data': train_data_uv,        # (N_train, n_channels, 1000)
            'train_labels': train_labels,       # (N_train, 1)
            'test_data': test_data_uv,          # (N_test, n_channels, 1000)
            'test_labels': test_labels          # (N_test, 1)
        })
        print(f"Saved processed data to: {output_path}")


if __name__ == "__main__":
    base_dir = './'
    preprocess_and_save_subjects(base_dir)

    mkpath = os.path.join('/home/lyx/lyx_WorkSpace/Datasets/Mydata2')
    print(mkpath)

    mat_data = sio.loadmat(mkpath + '/A0%d.mat' % (1))
    train_x = mat_data['train_data']
    test_x  = mat_data['test_data']
    train_y = mat_data['train_labels']
    test_y = mat_data['test_labels']

    print("train_x-shape", train_x.shape)
    print("test_x-shape", test_x.shape)
    print("train_y-shape", train_y.shape)
    print("test_y-shape", test_y.shape)


    