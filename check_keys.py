import h5py

dataset_path = 'd:/Python学习/DSR-RL-master/datasets/eccv16_dataset_summe_google_pool5.h5'
try:
    with h5py.File(dataset_path, 'r') as f:
        print("Keys:", list(f.keys())[:3])
        for key in list(f.keys())[:3]:
            print(f"-- {key} --")
            # Check datasets inside
            print("  Datasets:", list(f[key].keys()))
            # Check attributes on the group
            print("  Group Attrs:", dict(f[key].attrs))
            # Check if there is a 'video_name' dataset or attribute
            if 'video_name' in f[key]:
                print(f"  video_name (dataset): {f[key]['video_name'][()]}")
except Exception as e:
    print(f"Error: {e}")
