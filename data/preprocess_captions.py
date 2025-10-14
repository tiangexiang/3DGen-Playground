import json
import argparse
import csv
from tqdm import tqdm


def main(args):
    # load the complete 3DTopia captions
    with open(args.caption_path, 'r') as f:
        # a list of dicts
        captions = json.load(f)
    captions = {cap['obj_id']: cap['3dtopia'] for cap in captions}

    # load the complete cap3d captions
    csv_reader = csv.reader(open(args.cap3d_caption_path, 'r'))
    cap3d_captions = {row[0]: row[1] for row in csv_reader}

    # load the combined object list
    with open(args.json_path, 'r') as f:
        obj_list = json.load(f)

    captions_dict = {}
    cap3d_count = 0
    tdtopia_count = 0
    for obj_id in tqdm(obj_list.keys()):
        obj_path = obj_list[obj_id].split('.tar.gz')[0]
        if obj_id in captions:
            captions_dict[obj_path] = captions[obj_id]
            tdtopia_count += 1
        elif obj_id in cap3d_captions:
            captions_dict[obj_path] = cap3d_captions[obj_id]
            cap3d_count += 1
    
    print(f"Total captions: {len(captions_dict)}")
    print(f"Used 3DTopia captions: {tdtopia_count}")
    print(f"Used Cap3D captions: {cap3d_count}")
    
    with open(args.output_path, 'w') as f:
        json.dump(captions_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, required=True)
    parser.add_argument('--cap3d_caption_path', type=str, required=True)
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    main(args)