# %%
import os
from astropy.io import fits
import glob
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# Generates .txt tables from drpAll.

'''Configuration'''
file_directory = '/data/public/sdss/manga/mcubes'
drpall_file_path = '/scratch/users/giuliano.damian/data/other_data/drpall-v3_1_1.fits'

drpall_hdu_name = 'MANGA'

galaxy_id_column_drpall = 'PLATEIFU'
redshift_column_drpall = 'z'

output_max_sum_filename_txt = '/scratch/users/giuliano.damian/text/3-maiores_somas_spaxels.txt'
output_distances_txt_folder = '/scratch/users/giuliano.damian/text/3-distancias_txt'

arcsec_per_pixel_scale = 0.5
ARCSEC_TO_RADIANS = np.pi / (180.0 * 3600.0)
cosmo_model = FlatLambdaCDM(H0=70, Om0=0.3)
#####################################################################################################

global_redshift_map = {}
print(f'# Starting DRPall loading: {drpall_file_path}, HDU: {drpall_hdu_name}')
drpall_loaded_successfully = False
try:
    with fits.open(drpall_file_path) as hdul_drpall:
        if drpall_hdu_name not in hdul_drpall:
            print(f"CRITICAL ERROR: HDU '{drpall_hdu_name}' not found in DRPall file '{drpall_file_path}'.")
            print(f"Available HDUs: {[hdu.name for hdu in hdul_drpall]}")
        else:
            drpall_data = hdul_drpall[drpall_hdu_name].data
            print(f"HDU '{drpall_hdu_name}' successfully loaded from DRPall file.")
            print(f"Attempting to read IDs from column '{galaxy_id_column_drpall}' and redshifts from column '{redshift_column_drpall}'.")

            drpall_column_names = [name.upper() for name in drpall_data.columns.names]
            id_column_ok = galaxy_id_column_drpall.upper() in drpall_column_names
            z_column_ok = redshift_column_drpall.upper() in drpall_column_names

            if not id_column_ok:
                print(f"CRITICAL ERROR: ID column '{galaxy_id_column_drpall}' not found in HDU '{drpall_hdu_name}'. Available columns: {drpall_data.columns.names}")
            if not z_column_ok:
                print(f"CRITICAL ERROR: Redshift column '{redshift_column_drpall}' not found in HDU '{drpall_hdu_name}'. Available columns: {drpall_data.columns.names}")
            
            if id_column_ok and z_column_ok:
                read_id_examples = []
                for i, row in enumerate(drpall_data):
                    try:
                        galaxy_id_in_drpall = str(row[galaxy_id_column_drpall]).strip()
                        redshift = float(row[redshift_column_drpall])

                        if i < 5:
                            read_id_examples.append(f"'{galaxy_id_in_drpall}' (z = {redshift:.5f})")
                        
                        if np.isfinite(redshift):
                            global_redshift_map[galaxy_id_in_drpall] = redshift
                        else:
                            print(f"WARNING: Invalid redshift (NaN or Inf) for ID '{galaxy_id_in_drpall}' (row {i}) in DRPall. It will be ignored.")
                    except KeyError as e:
                        print(f"Key error processing row {i} of HDU '{drpall_hdu_name}': {e}.")
                        if i < 5: print(f"Row {i} data (problematic): {row}")
                        continue

                if read_id_examples:
                    print(f"# DEBUG: Examples of IDs read from DRPall (HDU '{drpall_hdu_name}', ID column '{galaxy_id_column_drpall}'): {', '.join(read_id_examples)}")
                
                print(f"{len(global_redshift_map)} galaxies with valid redshifts loaded from DRPall (HDU '{drpall_hdu_name}').")
                if not global_redshift_map and len(drpall_data) > 0:
                    print(f"# DEBUG: The redshift map is empty, but HDU '{drpall_hdu_name}' contains data. Check the ID column ('{galaxy_id_column_drpall}') and content.")
                drpall_loaded_successfully = True
            else:
                print("# DEBUG: Could not proceed with DRPall data reading due to missing columns.")
                
except FileNotFoundError:
    print(f"CRITICAL ERROR: DRPall file '{drpall_file_path}' not found.")
except Exception as e:
    print(f"CRITICAL ERROR opening or processing DRPall file: {e}")

if not global_redshift_map and not drpall_loaded_successfully:
    print("GENERAL WARNING: The redshift map could not be populated due to previous critical error. Dist_kpc will be 'N/A'.")
elif not global_redshift_map and drpall_loaded_successfully:
     print("GENERAL WARNING: The redshift map is empty after reading DRPall. Check column names and data in the file. Dist_kpc will be 'N/A'.")

'''Megacube file processing'''
print(f"\nSearching for MEGACUBE files in: {os.path.abspath(file_directory)}")
file_pattern = os.path.join(file_directory, '*-MEGACUBE.fits')
fits_file_list = glob.glob(file_pattern)

if not fits_file_list:
    print(f"No files ending with '-MEGACUBE.fits' found in '{file_directory}'.")
else:
    print(f"MEGACUBE files found: {len(fits_file_list)}")
    for filename in fits_file_list:
        print(f' - {os.path.basename(filename)}')

integrated_flux_maps_all_files = {}
max_sum_spaxel_results = []

for full_file_path in fits_file_list:
    file_basename = os.path.basename(full_file_path)
    print(f"\nProcessing file: {file_basename}...")
    try:
        with fits.open(full_file_path, memmap = False) as hdul:
            if 'FLUX' in hdul:
                flux_data = hdul['FLUX'].data
                if flux_data is not None and isinstance(flux_data, np.ndarray):
                    current_integrated_flux_map = None
                    if flux_data.ndim == 3:
                        current_integrated_flux_map = np.sum(flux_data, axis = 0)
                    elif flux_data.ndim == 2:
                        current_integrated_flux_map = flux_data
                    else:
                        print(f"  WARNING: FLUX data is neither 3D nor 2D for {file_basename}. Ignoring.")
                        continue 

                    integrated_flux_maps_all_files[file_basename] = current_integrated_flux_map

                    if current_integrated_flux_map is not None and current_integrated_flux_map.size > 0:
                        max_value = np.max(current_integrated_flux_map)
                        max_linear_idx = np.argmax(current_integrated_flux_map)
                        max_y_x_coords = np.unravel_index(max_linear_idx, current_integrated_flux_map.shape)

                        standard_suffix = '-MEGACUBE.fits'
                        if file_basename.endswith(standard_suffix):
                            original_cube_prefix = file_basename[:-len(standard_suffix)]
                        else:
                            original_cube_prefix = os.path.splitext(file_basename)[0]

                        '''Stores coordinates as (1,1)-based for output:'''
                        max_sum_spaxel_results.append({
                            "prefix": original_cube_prefix,
                            "file_basename": file_basename,
                            "spaxel_y": max_y_x_coords[0] + 1,
                            "spaxel_x": max_y_x_coords[1] + 1,
                            "max_sum": max_value,
                            "id_for_drpall": original_cube_prefix
                        })
                    else:
                         print(f"  WARNING: Integrated flux map for {file_basename} is empty or invalid.")
                else:
                    print(f"  WARNING: 'FLUX' data in '{file_basename}' is None or not a NumPy array.")
            else:
                print(f"  WARNING: HDU 'FLUX' not found in '{file_basename}'.")
    except Exception as e:
        print(f"  Unexpected error processing '{file_basename}': {e}")

if max_sum_spaxel_results:
    print(f"\nSaving highest sum spaxel information to: {output_max_sum_filename_txt}")
    try:
        with open(output_max_sum_filename_txt, "w") as f_out:
            f_out.write("CubePrefix\tSpaxel_Y\tSpaxel_X\tMaxFluxSum\n")
            for res in max_sum_spaxel_results:
                f_out.write(f"{res['prefix']}\t{res['spaxel_y']}\t{res['spaxel_x']}\t{res['max_sum']:.4e}\n")
        print(f"Max sum results successfully saved.")
    except IOError as e:
        print(f"ERROR saving TXT file '{output_max_sum_filename_txt}': {e}")

if integrated_flux_maps_all_files:
    print(f"\nCalculating and saving distance files (X Y Dist_arcsec Dist_kpc) in folder: '{output_distances_txt_folder}'")
    if not os.path.exists(output_distances_txt_folder):
        try: os.makedirs(output_distances_txt_folder, exist_ok=True)
        except OSError as e: 
            print(f"ERROR creating folder '{output_distances_txt_folder}': {e}")
    
    central_info = {res['file_basename']: res for res in max_sum_spaxel_results}

    for map_basename, flux_map in integrated_flux_maps_all_files.items():
        if map_basename in central_info:
            central_info_item = central_info[map_basename]
            
            y_central_0based = central_info_item['spaxel_y'] - 1
            x_central_0based = central_info_item['spaxel_x'] - 1

            full_megacube_id = central_info_item['id_for_drpall'] 

            print(f"  Processing distances for: {full_megacube_id} (Central (1,1)-based Y,X: {central_info_item['spaxel_y']},{central_info_item['spaxel_x']})")
            map_height, map_width = flux_map.shape
            pixel_distance_matrix = np.zeros((map_height, map_width), dtype=float)
            
            for y_spaxel in range(map_height):
                for x_spaxel in range(map_width):
                    dist_pixels = np.sqrt((y_spaxel - y_central_0based)**2 + (x_spaxel - x_central_0based)**2)
                    pixel_distance_matrix[y_spaxel, x_spaxel] = dist_pixels
            
            dist_txt_filename = os.path.join(output_distances_txt_folder, f"{full_megacube_id}_distancias_fisicas.txt")
            
            object_distance_D_kpc = None
            current_object_redshift = None

            id_for_drpall_lookup = full_megacube_id
            if full_megacube_id.startswith("manga-"):
                id_for_drpall_lookup = full_megacube_id.split('-', 1)[-1] 
                print(f"# DEBUG: ID for DRPall lookup adjusted from '{full_megacube_id}' to '{id_for_drpall_lookup}'")
            else:
                print(f"# DEBUG: ID for DRPall lookup (did not require 'manga-' adjustment): '{id_for_drpall_lookup}'")

            if global_redshift_map:
                if id_for_drpall_lookup in global_redshift_map:
                    current_object_redshift = global_redshift_map[id_for_drpall_lookup]
                    print(f"# DEBUG: ID '{id_for_drpall_lookup}' (originated from '{full_megacube_id}') FOUND in DRPall. Redshift = {current_object_redshift:.5f}")
                    
                    if current_object_redshift > 0: 
                        try:
                            dist_D_mpc = cosmo_model.angular_diameter_distance(current_object_redshift)
                            object_distance_D_kpc = dist_D_mpc.to(u.kpc).value
                            print(f"    Redshift (z={current_object_redshift:.5f}). Angular Diameter Distance (Da) = {object_distance_D_kpc:.2f} kpc.")
                        except Exception as e_cosmo:
                            print(f"    ERROR calculating cosmological distance for z={current_object_redshift}: {e_cosmo}")
                    elif current_object_redshift == 0:
                        print(f"    WARNING: Redshift for '{id_for_drpall_lookup}' is 0. Dist_kpc will be 'N/A'.")
                    else: 
                        print(f"    WARNING: Redshift for '{id_for_drpall_lookup}' is negative (blueshift, z={current_object_redshift:.5f}). Dist_kpc will be 'N/A'.")
                else: 
                    print(f"    CRITICAL WARNING: ID '{id_for_drpall_lookup}' (originated from '{full_megacube_id}') NOT FOUND in DRPall redshift map. Dist_kpc will not be calculated.")
            else: 
                 print(f"    CRITICAL WARNING: The redshift map (global_redshift_map) is not available or is empty. Cannot search for ID '{id_for_drpall_lookup}'.")
            
            try:
                with open(dist_txt_filename, "w") as f_dist:
                    f_dist.write("x_pixel\ty_pixel\tdistance_arcsec\tdistance_kpc\n")
                    for y_idx in range(map_height):
                        for x_idx in range(map_width):
                            dist_pixels_val = pixel_distance_matrix[y_idx, x_idx]
                            dist_arcsec_val = dist_pixels_val * arcsec_per_pixel_scale
                            
                            dist_kpc_val_str = "N/A"
                            if object_distance_D_kpc is not None and current_object_redshift is not None and current_object_redshift > 0:
                                theta_radians = dist_arcsec_val * ARCSEC_TO_RADIANS
                                dist_kpc_val = object_distance_D_kpc * theta_radians
                                dist_kpc_val_str = f"{dist_kpc_val:.4f}"
                            
                            f_dist.write(f"{x_idx + 1}\t{y_idx + 1}\t{dist_arcsec_val:.4f}\t{dist_kpc_val_str}\n")
                print(f"    Distance file saved to: {dist_txt_filename}")
            except Exception as e_save_dist:
                print(f"    ERROR saving distance list for '{full_megacube_id}': {e_save_dist}")
        else:
            print(f"  WARNING: Central spaxel information not found for '{map_basename}'.")

print("\n--- End of script ---")
   


