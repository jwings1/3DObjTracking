import os
import cv2
import numpy as np
import motmetrics as mm
import glob

def calculate_iou(box1, box2):
    if None in [item for sublist in (box1 + box2) for item in sublist]:
        return 0
    x1, y1, w1, h1 = box1[0][0], box1[0][1], box1[0][2], box1[0][3]
    x2, y2, w2, h2 = box2[0][0], box2[0][1], box2[0][2], box2[0][3]
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = np.float32(w1) * np.float32(h1)
    box2Area = np.float32(w2) * np.float32(h2)
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

def extract_bounding_box(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = np.where(image > 0, 1, 0)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return [[None, None, None, None]]
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return [[x_min, y_max, x_max-x_min, y_max-y_min]]

def calculate_distances(candidate_path, ground_path, method):
    trace_detection_list = []
    ground_detection_list = []
    data = np.load(candidate_path, allow_pickle=True)
    tracking_data = data['tracking'].item()
    for frame, frame_data in tracking_data.items():
        for i, track_id in enumerate(frame_data['track_ids']):
            bbox = frame_data['track_bbox'][i]
            dict = {}
            dict[track_id]= bbox
            trace_detection_list.append(dict)

    camera = ground_path.split('.')[-1]
    ground_path = '.'.join(ground_path.split('.')[:-1])
    pattern = os.path.join(ground_path, '**', 't*.000', 'k' + camera + '.' + 'person_mask.jpg')
    matching_files = sorted(glob.glob(pattern, recursive=True))

    for filename in matching_files:
        id = 1
        bbox = extract_bounding_box(filename)
        dict={}
        dict[id]=bbox
        ground_detection_list.append(dict)

    acc = mm.MOTAccumulator(auto_id=False)

    frame_num = min(len(trace_detection_list),len(ground_detection_list))
    for frameid in range(frame_num - 1, -1, -1):
        for key in (trace_detection_list[frameid].keys()):
            dist = np.array(1 - calculate_iou([trace_detection_list[frameid][key]], ground_detection_list[frameid][1]))          
            acc.update([key], [1], dist, frameid=frameid)

    mh = mm.metrics.create()
    result = {}
    result[candidate_path] = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
    return result

def generate_directories_dict(method):
    sequences_dir = '/scratch-second/lgermano/behave/sequences'
    trace_ev_dir = '/scratch/lgermano/TRACE_results/tracks_1fps'
    directories = {}
    for root, dirs, _ in os.walk(sequences_dir):
        for dir_name in dirs:
            for cam in range(4):
                directories[os.path.join(root, dir_name + f'.{cam}')] = []
        break
    for file in os.listdir(trace_ev_dir):
        if file.endswith('_tracking.npz'):
            file_name_root = '.'.join(file.split('.')[:2])
            for key in directories.keys():
                if key.endswith(file_name_root):
                    directories[key].append(os.path.join(trace_ev_dir,file))
    return directories    

def process_dict(directories, calculate_distances, method):
    new_dict = {}
    for key, values in directories.items():
        for value in values:
            result = calculate_distances(value, key, method)
            new_key = value.split('/')[-1]
            new_dict[new_key] = result
    return new_dict

method = 'trace'
directories = generate_directories_dict(method)
results = process_dict(directories, calculate_distances, method)

print(results)