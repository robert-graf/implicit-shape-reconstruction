import re
from pathlib import Path

import nibabel as nib
import numpy as np


def find_closest_image_axes(ijk_to_lps: np.ndarray) -> tuple[list[int], list[int]]:
    """For each global axis, find a closest image axis."""
    # To get the inverse transformation, remove the scaling to obtain the rotation matrix by
    # normalizing each column. Here we assume that there is no shearing involved (use polar
    # decomposition otherwise). Technically speaking it's an "improper rotation", potentially
    # including reflections (which is what we want here).
    ijk_to_lps = ijk_to_lps / np.linalg.norm(ijk_to_lps, axis=0)
    # Inverted matrix translates global coordinates into image coordinates.
    lps_to_ijk = ijk_to_lps.T
    # Each column this matrix corresponds to a transformed global axis x, y, z in image coordinates.
    # For each transformed global axis, we want to find the closest image axis.
    rotated_x = lps_to_ijk[:, 0]
    rotated_y = lps_to_ijk[:, 1]
    rotated_z = lps_to_ijk[:, 2]
    # For each rotated axis, find closest image axis
    closest_axis_to_x = int(np.argmax(np.abs(rotated_x)))
    closest_axis_to_y = int(np.argmax(np.abs(rotated_y)))
    closest_axis_to_z = int(np.argmax(np.abs(rotated_z)))
    if closest_axis_to_x == closest_axis_to_y or closest_axis_to_y == closest_axis_to_z or closest_axis_to_z == closest_axis_to_x:
        raise ValueError(
            f"Ambiguous closest axes after rotation: " f"{closest_axis_to_x}, {closest_axis_to_y}, {closest_axis_to_z}\n" f"{lps_to_ijk}"
        )
    # Find closest image axis sign
    sign_closest_axis_to_x = -1 if rotated_x[closest_axis_to_x] < 0 else 1
    sign_closest_axis_to_y = -1 if rotated_y[closest_axis_to_y] < 0 else 1
    sign_closest_axis_to_z = -1 if rotated_z[closest_axis_to_z] < 0 else 1

    return [closest_axis_to_x, closest_axis_to_y, closest_axis_to_z], [
        sign_closest_axis_to_x,
        sign_closest_axis_to_y,
        sign_closest_axis_to_z,
    ]


def normalize_coordinates(image_data: np.ndarray, ijk_to_ras: np.ndarray, verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    """Tranpose and/or flip the image array so that it is close to LPS coordinate system without any
    transformation matrix. Return the resulting image array and spacing.
    """
    # Convert the transformation matrix from nibabel's RAS coordinates to DICOM's LPS coordinates
    ijk_to_lps = ijk_to_ras.copy()
    ijk_to_lps[0] *= -1
    ijk_to_lps[1] *= -1
    assert np.allclose(ijk_to_lps[3, 3], 1.0)
    linear = ijk_to_lps[:3, :3]
    # Find the closest axes
    new_axes, axes_signs = find_closest_image_axes(linear)

    # First transpose, then flip
    # Rearrange the columns
    linear_fixed = linear[:, new_axes]
    img_data_fixed = np.transpose(image_data, new_axes)
    # Flip axes
    linear_fixed = linear_fixed @ np.diag(axes_signs)
    for i, sign in enumerate(axes_signs):
        if sign == -1:
            img_data_fixed = np.flip(img_data_fixed, axis=i)

    # Get the scale w.r.t. new axes
    spacing = np.linalg.norm(linear_fixed, axis=0)

    # Sanity check -- rotation matrix
    rotation = linear_fixed / spacing
    det = np.linalg.det(rotation)
    prod_with_t = rotation.T @ rotation
    if not np.isclose(det, 1.0) or not np.allclose(prod_with_t, np.eye(3), atol=1e-5):
        raise ValueError(f"Remaining matrix is not a rotation matrix. Determinant is {det}, " f"R^T @ R:\n{prod_with_t}")

    if verbose:
        print("closest signed image axes:", [ax * sign for ax, sign in zip(new_axes, axes_signs)])
        print(f"shape: {image_data.shape} -> {img_data_fixed.shape}")
        print("spacing:", spacing)

    return img_data_fixed, spacing


def load_nifti(file: Path, verbose: bool, max_zms) -> tuple[np.ndarray, np.ndarray]:
    if not file.exists():
        raise ValueError(f"Nifti file does not exist:\n{file}")
    img = nib.load(file, mmap=False)
    zms = img.header.get_zooms()
    if max_zms is not None:
        if any([i > max_zms for i in zms]):
            print("[X] Segmentation to coarse; zooms are", zms)
            return None, None  # type: ignore
    img = resample_nib(img)
    img_data = img.get_fdata(caching="unchanged")
    img_data = img_data.astype(img.get_data_dtype())
    img_data, spacing = normalize_coordinates(img_data, img.affine, verbose)
    img_data = np.ascontiguousarray(img_data)
    return img_data, spacing


import nibabel.processing as nip
import nibabel.orientations as nio


def resample_nib(img, voxel_spacing=(1, 1, 1), order=0, c_val=0, verbose=True) -> nib.Nifti1Image:

    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()

    # Calculate new shape
    new_shp = tuple(
        np.rint([shp[0] * zms[0] / voxel_spacing[0], shp[1] * zms[1] / voxel_spacing[1], shp[2] * zms[2] / voxel_spacing[2]]).astype(int)
    )
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)  # type: ignore
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=c_val)
    if verbose:
        print(
            f"[*] Image resampled to voxel size: {voxel_spacing} from {tuple(zms)}",
        )
    return new_img


def nii2arr(img: nib.Nifti1Image) -> np.ndarray:
    return np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)


def reorient_same_as(img: nib.Nifti1Image, img_as: nib.Nifti1Image, verbose=True, return_arr=False) -> nib.Nifti1Image:
    axcodes_to = nio.ornt2axcodes(nio.io_orientation(img_as.affine))
    return reorient_to(img, axcodes_to, verbose, return_arr)


def reorient_to(img: nib.Nifti1Image, axcodes_to=("P", "I", "R"), verbose=False, return_arr=False) -> nib.Nifti1Image:
    # Note: nibabel axes codes describe the direction not origin of axes
    # direction PIR+ = origin ASL
    img = img
    aff = img.affine
    arr = nii2arr(img)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    new_aff = np.matmul(aff, aff_trans)
    new_img = nib.Nifti1Image(arr, new_aff)
    if verbose:
        print("[*] Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    if return_arr:
        return new_img, arr.copy()  # type: ignore
    return new_img


def save_nifti(volume: np.ndarray, target_file: Path, spacing: tuple[int]):
    affine_diag = [*spacing, 1.0]
    affine = np.diag(affine_diag)
    # Make sure matrix is stored in RAS instead of LPS (this is what nibabel expects)
    affine[0] *= -1
    affine[1] *= -1
    img = nib.Nifti1Image(volume, affine)
    target_file.parent.mkdir(exist_ok=True)
    nib.save(img, target_file)


class DataGenerator:
    def __init__(self, labels_dir: Path, labels_pattern: str, verbose: bool, max_zms: float | None = None, return_other=False):
        if not labels_dir.exists():
            raise ValueError(f"Labels directory does not exist:\n{labels_dir}")

        self.label_paths = list(labels_dir.glob(labels_pattern))
        self.verbose = verbose
        # self.return_other = return_other
        self.max_zms = max_zms

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, item) -> tuple[np.ndarray, np.ndarray, str]:
        lbl_path = self.label_paths[item]
        # casename_match = re.match(r".*verse\d{3}.*", lbl_path.name)

        # if casename_match is None:
        #    raise ValueError(f"Filename does not match with casename pattern:\n{lbl_path.name}")
        casename = lbl_path.name

        lbl, spacing_lbl = load_nifti(lbl_path, self.verbose, self.max_zms)
        if lbl is None:
            return None, None, None  # type: ignore
        # if self.return_other:
        #    lbl, spacing_lbl, lbl_path
        return lbl, spacing_lbl, casename


def get_fg_size_and_position(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = np.zeros(3, np.int32)
    center_pos = -np.ones(3, np.int32)
    if not np.any(mask):
        # Empty image
        return size, center_pos

    for i in range(3):
        if i == 0:  # sag
            proj = np.max(mask, axis=(1, 2))
        elif i == 1:  # cor
            proj = np.max(mask, axis=(0, 2))
        else:  # ax
            proj = np.max(mask, axis=(0, 1))
        proj_list = proj.tolist()
        assert isinstance(proj_list, list)
        first_fg_pos = -1
        for j, val in enumerate(proj_list):
            if val == 1:
                first_fg_pos = j
                break
        last_fg_pos = -1
        for j, val in enumerate(reversed(proj_list)):
            if val == 1:
                last_fg_pos = len(proj_list) - j
                break
        if first_fg_pos == -1 or last_fg_pos == -1:
            raise ValueError("No ones found -- this shouldn't happen!")
        size[i] = last_fg_pos - first_fg_pos
        center_pos[i] = first_fg_pos + round(size[i] / 2)
    return size, center_pos


def crop_and_pad(image: np.ndarray, position: np.ndarray, bbox_size: int) -> np.ndarray:
    start_crop = np.round(position - bbox_size / 2).astype(np.int32)
    end_crop = np.round(position + bbox_size / 2).astype(np.int32)
    start_crop = np.maximum(start_crop, 0)
    end_crop = np.minimum(end_crop, np.array(image.shape))
    crop = image[start_crop[0] : end_crop[0], start_crop[1] : end_crop[1], start_crop[2] : end_crop[2]]
    # Pad to desired bb size
    pad_sizes = bbox_size - np.array(crop.shape)
    pad_beginning = np.floor(pad_sizes / 2).astype(np.int32)
    pad_end = np.ceil(pad_sizes / 2).astype(np.int32)
    crop = np.pad(crop, list(zip(pad_beginning, pad_end)))
    assert np.all(np.equal(crop.shape, bbox_size))
    return crop


def invert_crop_and_pad(image: np.ndarray, size: np.ndarray | tuple[int, ...]) -> np.ndarray:
    size = np.array(size)
    bbox_size: int = image.shape[0]
    assert image.shape[0] == image.shape[1]
    assert image.shape[0] == image.shape[2]
    center = bbox_size / 2

    start_crop = np.round(center - np.floor(size / 2)).astype(np.int32)
    end_crop = np.round(center + np.ceil(size / 2)).astype(np.int32)
    # start_crop = np.maximum(start_crop, 0)
    # end_crop = np.minimum(end_crop, np.array(image.shape))
    # print(start_crop, end_crop)
    crop = image[start_crop[0] : end_crop[0], start_crop[1] : end_crop[1], start_crop[2] : end_crop[2]]
    # Pad to desired bb size
    # pad_sizes = bbox_size - np.array(crop.shape)
    # pad_beginning = np.floor(pad_sizes / 2).astype(np.int32)
    # pad_end = np.ceil(pad_sizes / 2).astype(np.int32)
    # crop = np.pad(crop, list(zip(pad_beginning, pad_end)))
    # print(crop.shape, size)
    assert crop.shape == tuple(size), (crop.shape, size)
    return crop


from typing import overload, Literal


@overload
def save_individual_vertebrae(
    labels: np.ndarray,
    spacing_lbl: np.ndarray,
    bbox_size: int,
    target_labels_dir: Path,
    casename: str,
    return_pos: Literal[True],
    label_range=[20, 21, 22, 23, 24, 25],
) -> tuple[np.ndarray, np.ndarray, list[tuple[Path, np.ndarray, np.ndarray]]]:
    ...


@overload
def save_individual_vertebrae(
    labels: np.ndarray,
    spacing_lbl: np.ndarray,
    bbox_size: int,
    target_labels_dir: Path,
    casename: str,
    return_pos: Literal[False] = False,
    label_range=[20, 21, 22, 23, 24, 25],
) -> tuple[np.ndarray, np.ndarray]:
    ...


def save_individual_vertebrae(
    labels: np.ndarray,
    spacing_lbl: np.ndarray,
    bbox_size: int,
    target_labels_dir: Path,
    casename: str,
    return_pos: bool = False,
    label_range=[20, 21, 22, 23, 24, 25],
):
    """Process single case with multiple vertebra. Return the average and max vertebra size (from
    labels)."""
    max_size = np.zeros(3, np.int32)
    avg_size = np.zeros(3, np.float32)

    vrtbr_types = np.unique(labels)[1:]  # skip background label 0

    # vertebra_selection = vrtbr_types >= 20  # lumbar only
    types_masked = [i for i in vrtbr_types if int(i) in label_range]
    # label_range
    # if np.sum(vertebra_selection) == 0:
    if np.sum(types_masked) == 0:
        if return_pos:
            return max_size, avg_size, []
        return max_size, avg_size

    # types_masked = vrtbr_types[vertebra_selection]
    info = []
    for vrtbr_type in types_masked:
        labels_binary = np.equal(labels, vrtbr_type).astype(np.uint8)
        assert np.any(labels_binary)
        # Extract the vertebra size and position from binary mask
        vertebra_size, position = get_fg_size_and_position(labels_binary)
        max_size = np.maximum(vertebra_size, max_size)
        avg_size += vertebra_size
        if np.any(vertebra_size > bbox_size - 2):
            print(position)
            raise ValueError(
                f"Vertebra type {vrtbr_type} in case {casename} is larger than " f"bbox - 2: {vertebra_size} vs {bbox_size - 2}."
            )
        crop_lbl = crop_and_pad(labels_binary, position, bbox_size)
        out_file = target_labels_dir / f"{casename}_{vrtbr_type:02d}.nii.gz"
        save_nifti(crop_lbl, out_file, tuple(spacing_lbl))
        info.append((Path(out_file), vertebra_size, position))
    if return_pos:
        return max_size, avg_size / len(types_masked), info
    return max_size, avg_size / len(types_masked)


def main(
    labels_dir=Path("/media/data/robert/datasets/verse19/test_seg/"),
    target_labels_dir=Path("/media/data/robert/datasets/verse19/test_seg/vert/"),
    pattern="*.nii*",
    label_range=[20, 21, 22, 23, 24, 25],
    max_zms=1.3,
):
    """Transpose+flip label arrays so that indices (i,j,k) are aligned with anatomical axes (L,P,S).
    Then extract volumes with individual vertebra.
    """
    np.set_printoptions(precision=4, suppress=True)
    # labels_dir = Path("/media/data/robert/datasets/verse19/v19_BIDS_structure/dataset-verse19training/derivatives/")
    # target_labels_dir = Path("/media/data/robert/datasets/verse19/v19_BIDS_structure/dataset-verse19training/derivatives2/")
    # pattern = "*/*.nii*"

    verbose_loading = False
    bbox_size = 128

    data_gen = DataGenerator(labels_dir, pattern, verbose_loading, max_zms=max_zms, return_other=True)

    max_size = np.zeros(3, np.int32)
    avg_size = np.zeros(3, np.float32)
    num_avgs = 0
    num_processed = 0
    info: list[tuple[Path, np.ndarray, np.ndarray, Path, tuple[int, ...]]] = []
    for i, (labels, spacing_lbl, casename_path) in enumerate(data_gen):  # type: ignore
        if labels is None:
            continue
        casename = Path(casename_path).name
        print(f"[ ] Case {i + 1}/{len(data_gen)}: {casename}")
        # Skip non-isotropic spacings
        if not np.allclose(spacing_lbl, spacing_lbl[0]):
            print("Non-isotropic voxel size, skipping")
            continue
        num_processed += 1
        labels = labels.astype(np.uint8)
        max_curr, avg_curr, i = save_individual_vertebrae(
            labels,
            spacing_lbl,
            bbox_size,
            target_labels_dir,
            str(casename).replace(".nii.gz", ""),
            return_pos=True,
            label_range=label_range,
        )
        max_size = np.maximum(max_curr, max_size)
        if not np.all(avg_curr == 0.0):
            avg_size += avg_curr
            num_avgs += 1
        for j in i:
            info.append((*j, Path(casename_path), labels.shape))
    print(f"\n\nMax vertebra size: {max_size}\nAvg vertebra size:{avg_size / num_avgs}")
    print(f"Processed {num_processed}/{len(data_gen)} cases.")
    return info


if __name__ == "__main__":
    # main(
    #    labels_dir=Path("/media/data/robert/datasets/verse19/v19_BIDS_structure/dataset-verse19training/derivatives/"),
    #    target_labels_dir=Path("/media/data/robert/datasets/verse19/v19_BIDS_structure/dataset-verse19training/derivatives2/"),
    #    pattern="*/*_vert_msk.nii.gz",
    # )
    main(
        labels_dir=Path("/media/data/robert/datasets/verse19/v19_BIDS_structure/dataset-verse19training/derivatives/"),
        target_labels_dir=Path("/media/data/robert/datasets/verse19/v19_BIDS_structure/dataset-verse19training/thorakal/"),
        pattern="*/*-vert_msk.nii.gz",
        label_range=[i for i in range(8, 20)],
    )
    main(
        labels_dir=Path("/media/data/robert/datasets/verse19/v19_BIDS_structure/dataset-verse19training/derivatives/"),
        target_labels_dir=Path("/media/data/robert/datasets/verse19/v19_BIDS_structure/dataset-verse19training/hals/"),
        pattern="*/*-vert_msk.nii.gz",
        label_range=[i for i in range(1, 8)],
    )
