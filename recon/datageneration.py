from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
from BIDS import BIDS_FILE, NII, Centroids, BIDS_Global_info, run_docker, load_centroids
import random
import numpy as np
from preprocessing import extract_vertebra

np.set_printoptions(precision=4, suppress=True)
import shutil
import random

# Dataset 1
# File Docker unscaled to iso -> GT
# File generation iso-ct with GT -> any-ct + docker -> upscale segmentation to iso -> extract each labels
# Folder structure:
# [Base]/[train|val|test]/[1-28]/*_[gt|00-99].nii.gz
# Req: all iso, ("P", "I", "R");


def findFiles(
    path: str | Path = "/media/data/robert/datasets/verse19/v19_BIDS_structure/dataset-verse19training/",
    filter_non_iso=True,
):

    out_list: list[dict[str, BIDS_FILE]] = []
    gi = BIDS_Global_info(
        datasets=[str(path)],
        parents=["rawdata", "derivatives"],
        clear=True,
        additional_key=["sequ", "seg", "ovl", "snp"],
        verbose=False,
    )
    ignored = 0
    for subject_name, sub in gi.enumerate_subjects(sort=True):
        query = sub.new_query()
        query.flatten()
        query.filter("format", lambda x: x != "snapshot")
        query.filter("format", lambda x: x != "snp")
        query.unflatten()
        query.filter("format", "ct")
        query.filter("sub", lambda x: x != "ctfu00522")
        query.filter("sub", lambda x: x != "ctfu00514")

        # query.filter("format", "vert")
        for i in query.loop_dict():
            try:
                ct_nii = NII.load_bids(i["ct"][0])
            except Exception as e:
                continue
            if filter_non_iso and any(ct_nii.zoom[i] > 2.01 for i in range(3)):
                ignored += 1
                continue
            out_list.append(i)  # type: ignore
    print(ignored, "/", len(gi.enumerate_subjects(sort=True)), Path(path).name)
    return out_list


roots = {
    0: "/media/data/robert/datasets/verse19/v19_BIDS_structure/dataset-verse19training/",
    99: "/media/data/robert/datasets/verse19/v19_BIDS_structure/dataset-verse19validation/",
    1: "/media/data/robert/datasets/CT_TRAINING2_org/dataset-ATL",
    2: "/media/data/robert/datasets/CT_TRAINING2_org/dataset-fxclass",
    3: "/media/data/robert/datasets/CT_TRAINING2_org/dataset-gl",
    4: "/media/data/robert/datasets/CT_TRAINING2_org/dataset-verse19",
    5: "/media/data/robert/datasets/CT_TRAINING2_org/dataset-verse20",
}

skip_counter = 0


def generate(i, out="/media/data/robert/datasets/verse19/"):
    global skip_counter
    bbox_size = 128

    out_tmp_path = Path(out, "tmp" + str(i))
    out_path = Path(out, "docker_ds2")
    if not Path(roots[i]).exists():
        return
    l = findFiles(roots[i])
    # downscale
    for family_lr in l:
        try:
            ct, vert, subreg = family_lr["ct"], family_lr["msk_vert"], family_lr["ctd_subreg"]
        except Exception as ex:
            continue

        path = ct.get_changed_path(file_type="nii.gz", parent="rawdata", dataset_path=str(out_tmp_path))
        path_cdt = ct.get_changed_path(
            file_type="json", dataset_path=str(out_tmp_path), info={"seg": "subreg"}, format="ctd"
        )

        if path.exists() and path_cdt.exists():
            continue
        ct_nii = NII.load_bids(ct)
        cdt: Centroids = load_centroids(subreg)
        print("[ ]", ct_nii)
        # (random.random() * 2.8 + 0.7, random.random() * 2.8 + 0.7, random.random() * 2.8 + 0.7)
        cdt.zoom = ct_nii.zoom
        ct_nii.reorient_to_(("P", "I", "R"))
        cdt.reorient_centroids_to_(ct_nii)
        ct_nii.rescale_nib_(voxel_spacing=(1.0, 1.0, 3.0))
        cdt.rescale_centroids_((1, 1, 3))
        print("[ ]", ct_nii)
        ct_nii.save(path, make_parents=True, verbose=True)
        cdt.save(path_cdt, make_parents=True, verbose=True)
    run_docker(out_tmp_path)
    l2 = findFiles(out_tmp_path, filter_non_iso=False)

    max_size = np.zeros(3, np.int32)
    avg_size = np.zeros(3, np.float32)
    count = 0
    l2_f = []
    # if len(l) != len(l2):
    #    l = l[: len(l2)]
    for i in l:
        end = True
        for j in l2:
            if "msk_vert" not in j:
                l2.remove(j)
            elif "msk_vert" not in i:
                break
            elif i["msk_vert"].get("sub") == j["msk_vert"].get("sub"):
                l2.remove(j)
                l2_f.append(j)
                end = False
                break
        if end:
            l2_f.append(None)
    Parallel(n_jobs=10)(
        delayed(__helper)(family_lr, family, out_path, skip_counter, family_counter, l, bbox_size)
        for family_counter, (family, family_lr) in enumerate(zip(l, l2_f))
    )


def __helper(family_lr, family, out_path, skip_counter, family_counter, l, bbox_size):
    try:
        ct, vert, subreg = family_lr["ct"], family_lr["msk_vert"], family_lr["ctd_subreg"]
        ct_org, vert_org, subreg_org = family["ct"], family["msk_vert"], family["ctd_subreg"]
    except Exception as ex:
        return
    cdt = load_centroids(subreg)
    cdt.zoom = (1, 1, 3)
    cdt_iso = cdt.rescale_centroids((1, 1, 1))
    # ct_nii_sm = NII.load_bids(ct)
    print(f'{vert_org.get("sub")} {vert.get("sub")}')
    assert vert_org.get("sub") == vert.get("sub"), f'{vert_org.get("sub")} {vert.get("sub")}'
    try:
        vert_nii = NII.load_bids(vert)
    except Exception:
        return
    # vert_nii.rescale_nib_((1, 1, 1), verbose=True)
    vert_nii_iso = None
    arr_org = None
    for id, center in cdt_iso.items():
        train = random.random() < 0.9
        p1 = Path(out_path, "train", str(id), ct.get("sub") + f"_{0:04}.npz")
        p2 = Path(out_path, "val", str(id), ct.get("sub") + f"_{0:04}.npz")
        if p1.exists() or p2.exists():
            print("[ ] SKIP", skip_counter, "\t", family_counter, "/", len(l))
            skip_counter += 1
            continue
        if vert_nii_iso is None:
            vert_nii_iso = NII.load_bids(vert_org).rescale_and_reorient_(("P", "I", "R"), (1, 1, 1))
            arr_org = vert_nii_iso.get_seg_array()

        p = p1 if train else p2
        vert_nii.seg = True
        mask = np.equal(vert_nii.get_seg_array(), id).astype(np.uint8)
        vert_nii.seg = False
        mask = vert_nii.set_array(mask.astype(float), inplace=False).rescale_nib_((1, 1, 1), c_val=0).get_array()
        assert arr_org is not None
        mask_org = np.equal(arr_org, id).astype(np.uint8)
        size, center_pos = extract_vertebra.get_fg_size_and_position(mask.astype(np.uint8))
        # max_size = np.maximum(size, max_size)
        # avg_size += size
        # count += 1
        if np.any(size > bbox_size - 2):
            raise ValueError(f"Vertebra type {id} in case {ct} is larger than " f"bbox - 2: {size} vs {bbox_size - 2}.")
        cropped_vert = extract_vertebra.crop_and_pad(mask, center_pos, bbox_size)
        cropped_vert_org = extract_vertebra.crop_and_pad(mask_org, center_pos, bbox_size)
        p.parent.mkdir(exist_ok=True, parents=True)
        # vert_nii_iso.seg = False
        # vert_nii_iso.set_array(cropped_vert).save(str(p).replace(".npz", ".nii.gz"))
        # vert_nii_iso.set_array(cropped_vert_org).save(str(p).replace(".npz", "_gt.nii.gz"))
        # vert_nii_iso.seg = True

        np.savez_compressed(p, low_res=cropped_vert, high_res=cropped_vert_org)
    # Make image per vertebra.
    # print(max_size, avg_size / count)
    # shutil.rmtree(out_tmp_path)


if __name__ == "__main__":
    from joblib import Parallel, delayed

    n_jobs = 10
    if n_jobs > 1:
        print("[*] Running {} parallel jobs. Note that stdout will not be sequential".format(n_jobs))
    out = (
        "/media/data/robert/datasets/CT_TRAINING2_org"
        if Path("/media/data/robert/datasets/CT_TRAINING2_org").exists()
        else "/media/data/robert/datasets/verse19/"
    )

    # generate(4, out)
    Parallel(n_jobs=n_jobs)(delayed(generate)(i, out) for i in roots.keys())
