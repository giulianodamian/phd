# %%
# -*- coding: utf-8 -*-

from astropy.io import fits
import glob
import numpy as np
import os
import traceback

# Input/output configuration
# --- Change according to your needs ---
megacube_path_pattern = '/data/public/sdss/manga/mcubes/*MEGACUBE.fits'
emission_cube_path_pattern = '/scratch/users/giuliano.damian/data/emission_data/'

if not os.path.exists(emission_cube_path_pattern):
    os.makedirs(emission_cube_path_pattern)

# Output patter name
# Saves the output file in the different directory as the input file, with "MEGACUBE" replaced by "EMISSION"
def get_output_filename(input_name):
    base_name = os.path.basename(input_name)
    new_name = base_name.replace('MEGACUBE', 'EMISSION')

    return os.path.join(emission_cube_path_pattern, new_name)

# H-alpha index in the HDU 'EQW_M' in MEGACUBE data:
IDX_HA_EW = 6

# Spectral window settings for SNR calculation (lambda_min, lambda_max, solution_index)
LINE_CONFIGS = {
    "hb":        (4822, 4902, 0),
    "o3_4959":   (4920, 5000, 3),
    "o3_5007":   (4967, 5047, 6),
    "he1_5876":  (5836, 5916, 9),
    "o1_6300":   (6260, 6340, 12),
    "n2_6548":   (6508, 6588, 15),
    "ha":        (6524, 6604, 18),
    "n2_6583":   (6543, 6623, 21),
    "s2_6716":   (6676, 6756, 24),
    "s2_6731":   (6691, 6771, 27)
}

# --- Functions for Calculating SNR ---
def _line_snr(hdus_list, min_lambda, max_lambda, index):
    """ Calculates the 2D SNR map for a specific emission line. """
    try:
        solutions = hdus_list['SOLUTION'].data
        fitspec = hdus_list['FITSPEC'].data
        model = hdus_list['MODEL'].data
        _sn_mask_hdu = hdus_list['SN_MASKS_5']
        _sn_mask = _sn_mask_hdu.data

        # Treating Mask
        if _sn_mask.ndim != 2:
            _sn_mask_factor = 1.0
        elif not np.issubdtype(_sn_mask.dtype, np.number) and not np.issubdtype(_sn_mask.dtype, np.bool):
            _sn_mask_factor = 1.0
        else:
            _sn_mask_binary = (_sn_mask != 0).astype(float)
            _sn_mask_factor = 1.0 - _sn_mask_binary
        
        z_size, y_size, x_size = fitspec.shape
        header = hdus_list['FITSPEC'].header
        crval = header['CRVAL3']; crpix = header['CRPIX3'] - 1; cdelt = header['CD3_3']

        if cdelt == 0:
            return None
        
        init_lambda = crval - crpix * cdelt
        min_ind = int(round((min_lambda - init_lambda) / cdelt))
        max_ind = int(round((max_lambda - init_lambda) / cdelt))
        min_ind = max(0, min_ind); max_ind = min(z_size - 1, max_ind)

        if min_ind > max_ind:
            return np.zeros((y_size, x_size))
        
        if fitspec.shape != model.shape:
            return None

        residuals_slice = fitspec[min_ind:max_ind+1, :, :] - model[min_ind:max_ind+1, :, :]
        std_map = np.full((y_size, x_size), np.nan)

        if residuals_slice.shape[0] < 2:
            std_map[:] = np.inf
        else:
            try:
                with np.errstate(invalid='ignore'):
                    std_map = np.nanstd(residuals_slice, axis=0)
                std_map[std_map == 0] = np.inf
            except Exception:
                std_map[:] = np.inf
        
        amp_line = solutions[index]
        valid_std = np.isfinite(std_map) & (std_map > 0)
        snr_map  =np.where(valid_std, amp_line / std_map, 0.0)

        if isinstance(_sn_mask_factor, np.ndarray) and snr_map.shape == _sn_mask_factor.shape:
            snr_map = snr_map * _sn_mask_factor
        
        return snr_map
    
    except Exception:
        return None

# --- Wrapper for SNR functions ---
def get_all_snrs(hdus_list):
    results = []
    """The order here is defined by the LINE_CONFIGS dictionary."""
    for key in ["hb", "o3_4959", "o3_5007", "he1_5876", "o1_6300", "n2_6548", "ha", "n2_6583", "s2_6716", "s2_6731"]:
        conf = LINE_CONFIGS[key]
        results.append(_line_snr(hdus_list, conf[0], conf[1], conf[2]))
    return results

# --- Main Processing Loop ---
print(f"Searching for MEGACUBE files in: {megacube_path_pattern}")

processed_files = 0
failed_files = 0

for megacube_filename in glob.iglob(megacube_path_pattern):
    print(f"\nProcessing file: {megacube_filename}")
    output_filename = get_output_filename(megacube_filename)

    try:
        with fits.open(megacube_filename) as hdul_megacube:
            print("  Calculating SNR maps for emission lines...")
            all_line_snr_list = get_all_snrs(hdul_megacube)

            """Validating results"""
            if any(res is None for res in all_line_snr_list):
                print("  ERROR: Failed to compute SNR maps for one or more lines.")
                failed_files += 1
                continue

            all_line_snr_data = np.array(all_line_snr_list)

            """Creating output HDUs"""
            image_hdu0 = fits.PrimaryHDU(header = hdul_megacube[0].header)

            """HDU FLUX_M"""
            try:
                flux_hdu_in = hdul_megacube['FLUX_M']
                image_hdu1 = fits.ImageHDU(data = flux_hdu_in.data, header = flux_hdu_in.header, name = 'FLUX_M')
            except KeyError:
                image_hdu1 = fits.ImageHDU(name = 'FLUX_M')

            """HDU EQW_M"""
            image_hdu2 = None
            try:
                ew_hdu_in = hdul_megacube['EQW_M']
                if ew_hdu_in.data.ndim != 3:
                    ew_ha_data_2d = ew_hdu_in.data
                else:
                    ew_ha_data_2d = ew_hdu_in.data[IDX_HA_EW, :, :]

                image_hdu2 = fits.ImageHDU(data = ew_ha_data_2d, header = ew_hdu_in.header, name = 'EQW_M')
            except Exception:
                image_hdu2 = fits.ImageHDU(name = 'EQW_M')

            """HDU LSNR_M"""
            hdr_lsnr = fits.Header()
            hdr_lsnr['COMMENT'] = 'Order: Hb,OIII4959,OIII5007,HeI5876,OI6300,NII6548,Ha,NII6583,SII6716,SII6731'
            image_hdu3 = fits.ImageHDU(data = all_line_snr_data, header = hdr_lsnr, name = 'LSNR_M')

            hdul_output = fits.HDUList([image_hdu0, image_hdu1, image_hdu2, image_hdu3])
            hdul_output.writeto(output_filename, overwrite=True)
            processed_files += 1

    except Exception as e:
        print(f'Error processing file {os.path.basename(megacube_filename)}: {e}')
        failed_files += 1

print(f"\nProcessing completed. Successfully processed files: {processed_files}, Failed files: {failed_files}")




