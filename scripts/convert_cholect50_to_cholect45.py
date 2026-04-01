"""Convert CholecT50 JSON labels to CholecT45 text format expected by the dataset class."""
import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Convert CholecT50 JSON labels to CholecT45 text format")
    parser.add_argument("--src", default="/home/ubuntu/datasets/vlm/CholecT50/CholecT50",
                        help="Path to CholecT50 directory")
    parser.add_argument("--dst", default="/home/ubuntu/datasets/vlm/CholecT45",
                        help="Path to output CholecT45 directory")
    args = parser.parse_args()

    src = args.src
    dst = args.dst

    os.makedirs(f"{dst}/dict", exist_ok=True)
    os.makedirs(f"{dst}/triplet", exist_ok=True)

    # Read one JSON to get categories (they're the same across all)
    with open(f"{src}/labels/VID01.json") as f:
        meta = json.load(f)
    cats = meta["categories"]

    # Write dict files
    # instrument.txt: "idx:name"
    with open(f"{dst}/dict/instrument.txt", "w") as f:
        for k, v in sorted(cats["instrument"].items(), key=lambda x: int(x[0])):
            f.write(f"{k}:{v}\n")
        f.write("6:null_instrument\n")

    # verb.txt
    with open(f"{dst}/dict/verb.txt", "w") as f:
        for k, v in sorted(cats["verb"].items(), key=lambda x: int(x[0])):
            f.write(f"{k}:{v}\n")

    # target.txt
    with open(f"{dst}/dict/target.txt", "w") as f:
        for k, v in sorted(cats["target"].items(), key=lambda x: int(x[0])):
            f.write(f"{k}:{v}\n")

    # triplet.txt
    with open(f"{dst}/dict/triplet.txt", "w") as f:
        for k, v in sorted(cats["triplet"].items(), key=lambda x: int(x[0])):
            f.write(f"{k}:{v}\n")

    # maps.txt: triplet_id -> instrument_id, verb_id, target_id
    inst_map = {v: int(k) for k, v in cats["instrument"].items()}
    verb_map = {v: int(k) for k, v in cats["verb"].items()}
    target_map = {v: int(k) for k, v in cats["target"].items()}

    with open(f"{dst}/dict/maps.txt", "w") as f:
        f.write("triplet_id,instrument_id,verb_id,target_id\n")
        for k, v in sorted(cats["triplet"].items(), key=lambda x: int(x[0])):
            parts = v.split(",")
            inst_id = inst_map.get(parts[0], 6)
            verb_id = verb_map.get(parts[1], 9)
            target_id = target_map.get(parts[2], 14)
            f.write(f"{k},{inst_id},{verb_id},{target_id}\n")

    print("Dict files created.")

    # Convert each video's JSON labels to triplet txt format
    # The triplet txt expected: frame_number, triplet_0_present, ..., triplet_99_present
    num_triplets = len(cats["triplet"])
    label_files = sorted(os.listdir(f"{src}/labels/"))

    for lf in label_files:
        vid = lf.replace(".json", "")
        with open(f"{src}/labels/{lf}") as f:
            data = json.load(f)
        ann = data["annotations"]

        with open(f"{dst}/triplet/{vid}.txt", "w") as f:
            for frame_str in sorted(ann.keys(), key=int):
                frame_num = int(frame_str)
                triplet_vec = [0] * num_triplets
                for instance in ann[frame_str]:
                    triplet_id = instance[0]  # first element is triplet_id
                    if 0 <= triplet_id < num_triplets:
                        triplet_vec[triplet_id] = 1
                line = f"{frame_num}," + ",".join(str(x) for x in triplet_vec)
                f.write(line + "\n")

        print(f"  {vid}: {len(ann)} frames")

    # Create symlink for rgb -> CholecT50 videos directory (frames are already extracted there)
    rgb_link = f"{dst}/rgb"
    if not os.path.exists(rgb_link):
        os.symlink(f"{src}/videos", rgb_link)
        print(f"Symlinked {rgb_link} -> {src}/videos")
    else:
        print(f"Symlink {rgb_link} already exists, skipping.")

    # Create component-level label directories (verb, target, instrument)
    for component in ["verb", "target", "instrument"]:
        comp_dir = f"{dst}/{component}"
        os.makedirs(comp_dir, exist_ok=True)
        comp_cats = cats[component]
        num_classes = len(comp_cats)

        for lf in label_files:
            vid = lf.replace(".json", "")
            with open(f"{src}/labels/{lf}") as f:
                data = json.load(f)
            ann = data["annotations"]

            with open(f"{comp_dir}/{vid}.txt", "w") as f:
                for frame_str in sorted(ann.keys(), key=int):
                    frame_num = int(frame_str)
                    comp_vec = [0] * num_classes
                    for instance in ann[frame_str]:
                        if component == "instrument":
                            comp_id = instance[1]  # instrument_id is index 1
                        elif component == "verb":
                            triplet_id = instance[0]
                            trip_str = cats["triplet"].get(str(triplet_id), "")
                            if trip_str:
                                verb_name = trip_str.split(",")[1]
                                comp_id = verb_map.get(verb_name, -1)
                            else:
                                comp_id = -1
                        elif component == "target":
                            triplet_id = instance[0]
                            trip_str = cats["triplet"].get(str(triplet_id), "")
                            if trip_str:
                                target_name = trip_str.split(",")[2]
                                comp_id = target_map.get(target_name, -1)
                            else:
                                comp_id = -1
                        if 0 <= comp_id < num_classes:
                            comp_vec[comp_id] = 1
                    line = f"{frame_num}," + ",".join(str(x) for x in comp_vec)
                    f.write(line + "\n")

    print("\nComponent label dirs created (verb, target, instrument)")
    print("Done!")


if __name__ == "__main__":
    main()
