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

    # load the aesthetic list
    with open(args.aesthetic_json, 'r') as f:
        aesthetic_list = json.load(f)
    
    # load the non-aesthetic list
    with open(args.non_aesthetic_json, 'r') as f:
        non_aesthetic_list = json.load(f)

    # Combine both lists
    combined_list = aesthetic_list.copy()
    combined_list.update(non_aesthetic_list)

    # Build captions dict and track missing IDs
    captions_dict = {}
    missing_ids = []
    for obj_id in tqdm(combined_list.keys(), desc="Processing captions"):
        if obj_id in captions:
            captions_dict[obj_id] = captions[obj_id]
        elif obj_id in cap3d_captions:
            captions_dict[obj_id] = cap3d_captions[obj_id]
        else:
            missing_ids.append(obj_id)

    print(f"\nOriginal combined list: {len(combined_list)} items")
    print(f"Captions found: {len(captions_dict)} items")
    print(f"Missing captions: {len(missing_ids)} items")
    
    # Remove missing IDs from aesthetic_list
    cleaned_aesthetic_list = {k: v for k, v in aesthetic_list.items() if k not in missing_ids}
    print(f"\nAesthetic list: {len(aesthetic_list)} -> {len(cleaned_aesthetic_list)} items (removed {len(aesthetic_list) - len(cleaned_aesthetic_list)})")
    
    # Remove missing IDs from non_aesthetic_list
    cleaned_non_aesthetic_list = {k: v for k, v in non_aesthetic_list.items() if k not in missing_ids}
    print(f"Non-aesthetic list: {len(non_aesthetic_list)} -> {len(cleaned_non_aesthetic_list)} items (removed {len(non_aesthetic_list) - len(cleaned_non_aesthetic_list)})")
    
    # Save cleaned lists to current directory
    with open('aesthetic_list_cleaned.json', 'w') as f:
        json.dump(cleaned_aesthetic_list, f, indent=2)
    print("\nSaved cleaned aesthetic list to: aesthetic_list_cleaned.json")
    
    with open('non_aesthetic_list_cleaned.json', 'w') as f:
        json.dump(cleaned_non_aesthetic_list, f, indent=2)
    print("Saved cleaned non-aesthetic list to: non_aesthetic_list_cleaned.json")
    
    # Save list of missing IDs for reference
    with open('missing_caption_ids.json', 'w') as f:
        json.dump(missing_ids, f, indent=2)
    print(f"Saved {len(missing_ids)} missing IDs to: missing_caption_ids.json")
    
    # Save the captions dict
    with open(args.output_path, 'w') as f:
        json.dump(captions_dict, f)
    print(f"\nSaved captions to: {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, required=True)
    parser.add_argument('--cap3d_caption_path', type=str, required=True)
    parser.add_argument('--aesthetic_json', type=str, required=True)
    parser.add_argument('--non_aesthetic_json', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    main(args)