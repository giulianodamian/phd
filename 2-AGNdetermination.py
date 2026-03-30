# %%
# -*- coding: utf-8 -*-
import glob
import numpy as np
from astropy.io import fits
import os
import traceback
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

# Determines which spaxel cells are ionized from AGN using the BPT and WHAN diagrams with the data generated in the .fits file of code 1.

EPS = 1e-9

warnings.filterwarnings("ignore", category=RuntimeWarning)

config = {
    'input_pattern': '/scratch/users/giuliano.damian/data/emission_data/*EMISSION.fits',
    'flux_hdu_name': 'FLUX_M',
    'output_list_dir': '/scratch/users/giuliano.damian/text/2_agn_output_lists',
    'output_plot_dir': '/scratch/users/giuliano.damian/images/2_agn_output_plots',
    'output_diagram_dir': '/scratch/users/giuliano.damian/images/2_agn_output_diagrams',
    'output_highlighted_diagram_dir': '/scratch/users/giuliano.damian/images/2_agn_output_highlighted',
    'output_consolidated_diagram_dir': '/scratch/users/giuliano.damian/images/2_agn_output_consolidated_diagrams',
    'thresholds': {
        'snr': 3.0,
        'ew': -1000
    },
    'line_indices': {
        'HB': 0, 'OIII': 2, "HA": 6, 'NII': 7, 'SII': None, "OI": None
    }
}

for dir_path in [config['output_list_dir'], config['output_plot_dir'], config['output_diagram_dir'], config['output_highlighted_diagram_dir'], config['output_consolidated_diagram_dir']]:
    os.makedirs(dir_path, exist_ok=True)

input_pattern_glob = config['input_pattern']
emission_files = sorted(glob.glob(input_pattern_glob))
if not emission_files:
    print(f'WARNING: No files were found matching the pattern: {input_pattern_glob}')
else:
    print(f'Found {len(emission_files)} files to process.')

total_agn_pixels_overall = 0
total_masked_pixels_overall = 0
total_non_agn_pixels_overall = 0

all_files_log_nii_ha = []
all_files_log_oiii_hb = []
all_files_log_ew_ha = []
all_files_bpt_agn_condition = []
all_files_whan_agn_condition = [] 
all_files_bpt_snr_passed = []    

# --- Main Loop ---
if emission_files:
    for filename in emission_files:
        print(f'\nProcessing file: {os.path.basename(filename)}')
        current_line_indices_config_loop = config['line_indices'].copy()
        flux_hdu_name_loop = config['flux_hdu_name']

        print(f'Loading and validating: {os.path.basename(filename)}')
        flux_loop, lsnr_loop, ew_loop, mask_data_loop = None, None, None, None
        has_mask_loop = False
        line_indices_validated_loop = current_line_indices_config_loop.copy()
        data_valid_loop = False

        try:
            with fits.open(filename) as hdul:
                required_hdus = [flux_hdu_name_loop, 'LSNR_M', 'EQW_M']
                missing_hdus = [hdu for hdu in required_hdus if hdu not in hdul]
                if missing_hdus:
                    print(f'ERROR: Missing required HDUs in {os.path.basename(filename)}: {", ".join(missing_hdus)}. Skipping this file.')
                    continue
            
                flux_loop = hdul[flux_hdu_name_loop].data
                lsnr_loop = hdul['LSNR_M'].data
                ew_loop = hdul['EQW_M'].data

                if 'MASK' in hdul and hdul['MASK'].data.shape == ew_loop.shape:
                    mask_data_loop = hdul['MASK'].data
                    has_mask_loop = True
                    print(f'Found MASK HDU in {os.path.basename(filename)}. It will be used for masking pixels.')

                if not (ew_loop.ndim == 2 and flux_loop.ndim ==3 and lsnr_loop.ndim == 3):
                    print(f'ERROR: Unexpected data shapes in {os.path.basename(filename)}. Expected EQW_M to be 2D and FLUX_M, LSNR_M to be 3D. Skipping this file.')
                    continue
                if not (flux_loop.shape[1:] == ew_loop.shape and lsnr_loop.ndim == 3):
                    print(f'ERROR: Mismatched spatial dimensions in {os.path.basename(filename)}. FLUX_M and LSNR_M should have the same spatial dimensions as EQW_M. Skipping this file.')
                    continue
                if flux_loop.shape != lsnr_loop.shape:
                    print(f'ERROR: Mismatched shapes between FLUX_M and LSNR_M in {os.path.basename(filename)}. They should have the same shape. Skipping this file.')
                    continue

                num_lines_loop =  flux_loop.shape[0]

                base_keys_loop = ['HB', 'OIII', 'HA', 'NII']
                all_indices_valid_loop = True
                for key in base_keys_loop:
                    idx = line_indices_validated_loop.get(key)
                    if not (isinstance(idx, int) and 0 <= idx < num_lines_loop):
                        print(f'ERROR: Invalid line index for {key} in {os.path.basename(filename)}. Expected an integer between 0 and {num_lines_loop - 1}. Skipping this file.')
                        all_indices_valid_loop = False
                        break
                if not all_indices_valid_loop:
                    continue
                data_valid_loop = True
        except Exception as e:
            print(f'ERROR: Failed to load or validate {os.path.basename(filename)}. Exception: {str(e)}. Skipping this file.')
            traceback.print_exc()
            continue
        if not data_valid_loop:
            print(f'ERROR: Data validation failed for {os.path.basename(filename)}. Skipping this file.')
            continue
        else:
            print(f'Successfully loaded and validated {os.path.basename(filename)}. Proceeding with analysis.')

        """Processing Spaxels"""
        y_size_loop, x_size_loop = ew_loop.shape
        agn_map_loop = np.full((y_size_loop, x_size_loop), -1, dtype=int)

        plot_data_loop = {key: [] for key in ['log_nii_ha', 'log_oiii_hb', 'log_ew_ha']}
        plot_final_code_loop = []
        plot_bpt_agn_condition_loop = []
        plot_whan_agn_condition_loop = []
        plot_bpt_snr_filter_passed_loop = []

        snr_thresh_loop = config['thresholds']['snr']
        ew_thresh_loop = config['thresholds']['ew']
        idx_hb_loop, idx_oiii_loop = line_indices_validated_loop['HB'], line_indices_validated_loop['OIII']
        idx_ha_loop, idx_nii_loop = line_indices_validated_loop['HA'], line_indices_validated_loop['NII']

        print(f'Processing spaxels in {os.path.basename(filename)}...')

        for y in range(y_size_loop):
            for x in range(x_size_loop):
                final_map_code_loop = -1
                current_spaxel_satisfies_bpt_agn = 0
                current_spaxel_satisfies_whan_agn = 0
                current_spaxel_passes_bpt_snr_filter = False

                flux_hb_val_loop = flux_loop[idx_hb_loop, y, x]
                flux_oiii_val_loop = flux_loop[idx_oiii_loop, y, x]
                flux_ha_val_loop = flux_loop[idx_ha_loop, y, x]
                flux_nii_val_loop = flux_loop[idx_nii_loop, y, x]

                snr_hb_val_loop = lsnr_loop[idx_hb_loop, y, x]
                snr_oiii_val_loop = lsnr_loop[idx_oiii_loop, y, x]
                snr_ha_val_loop = lsnr_loop[idx_ha_loop, y, x]
                snr_nii_val_loop = lsnr_loop[idx_nii_loop, y, x]

                ew_ha_val_loop = ew_loop[y, x]
                is_masked_loop = has_mask_loop and mask_data_loop[y,x] != 0 
                base_fluxes_loop = [flux_hb_val_loop, flux_oiii_val_loop, flux_ha_val_loop, flux_nii_val_loop]
                basic_data_valid_loop = np.isfinite(ew_ha_val_loop) and all(np.isfinite(f) for f in base_fluxes_loop)

                snrs_for_bpt_check = [snr_hb_val_loop, snr_oiii_val_loop, snr_ha_val_loop, snr_thresh_loop]
                all_bpt_snrs_pass_filter = True
                for s_val in snrs_for_bpt_check:
                    if not (np.isfinite(s_val) and s_val >= snr_thresh_loop):
                        all_bpt_snrs_pass_filter = False
                        break
                if all_bpt_snrs_pass_filter:
                    current_spaxel_passes_bpt_snr_filter = True

                if is_masked_loop:
                    final_map_code_loop = -1
                elif not basic_data_valid_loop:
                    final_map_code_loop = -1
                else:
                    if np.isfinite(ew_ha_val_loop) and np.abs(ew_ha_val_loop) > 3.0:
                        current_spaxel_satisfies_whan_agn = 1
                    
# True se AGN pelo critério BPT
                    required_snrs_for_map_classification = [snr_hb_val_loop, snr_oiii_val_loop, snr_ha_val_loop, snr_nii_val_loop]
                    valid_snrs_for_map_classification = all(np.isfinite(snr) and (snr) >= (snr_thresh_loop) for snr in required_snrs_for_map_classification)
                    valid_ew_initial_threshold_loop = np.abs(ew_ha_val_loop) > ew_thresh_loop 

                    if valid_snrs_for_map_classification and valid_ew_initial_threshold_loop:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', category=RuntimeWarning)
                            is_above_kewley = False
                            if flux_ha_val_loop < 0 or flux_hb_val_loop <= 0:
                                final_map_code_loop = 0
                            else: 
                                log_nii_ha_loop_bpt = np.log10((flux_nii_val_loop / flux_ha_val_loop) + EPS)
                                log_oiii_hb_loop_bpt = np.log10((flux_oiii_val_loop / flux_hb_val_loop) + EPS)
                                
                                log_oiii_hb_kewley01_loop = np.where(log_nii_ha_loop_bpt < 0.47 - EPS, (0.61 / (log_nii_ha_loop_bpt - 0.47 - EPS)) + 1.19, np.inf)

                                if log_oiii_hb_loop_bpt > log_oiii_hb_kewley01_loop:
                                    is_above_kewley = True
                                    current_spaxel_satisfies_bpt_agn = 1

                                    """Classification map for AGN if Kewley 2001 is satisfied"""
                                if is_above_kewley and (np.abs(ew_ha_val_loop) > 3.0):
                                    final_map_code_loop = 1
                                else:
                                    final_map_code_loop = 0
                    else:
                        final_map_code_loop = 0

                agn_map_loop[y, x] = final_map_code_loop
                        
                """Collects data for plotting diagrams (even if final_map_code_loop is 0 or -1). The final filtering for plotting (valid_plot_indices_loop) uses plot_final_code_loop >= 0"""

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    plot_data_loop['log_nii_ha'].append(np.log10((flux_nii_val_loop/ flux_ha_val_loop) + EPS) if np.isfinite(flux_nii_val_loop) and np.isfinite(flux_ha_val_loop) and flux_ha_val_loop > 0 else np.nan)
                    plot_data_loop['log_oiii_hb'].append(np.log10((flux_oiii_val_loop / flux_hb_val_loop) + EPS) if np.isfinite(flux_oiii_val_loop) and np.isfinite(flux_hb_val_loop) and flux_hb_val_loop > 0 else np.nan)
                    plot_data_loop['log_ew_ha'].append(np.log10(np.abs(ew_ha_val_loop) + EPS) if np.isfinite(ew_ha_val_loop) and ew_ha_val_loop != 0 else np.nan)

                plot_final_code_loop.append(final_map_code_loop)
                plot_bpt_agn_condition_loop.append(current_spaxel_satisfies_bpt_agn)
                plot_whan_agn_condition_loop.append(current_spaxel_satisfies_whan_agn)
                plot_bpt_snr_filter_passed_loop.append(current_spaxel_passes_bpt_snr_filter)

        for key in plot_data_loop: plot_data_loop[key] = np.array(plot_data_loop[key])
        plot_final_code_loop = np.array(plot_final_code_loop)
        plot_bpt_agn_condition_loop = np.array(plot_bpt_agn_condition_loop, dtype = bool)
        plot_whan_agn_condition_loop = np.array(plot_whan_agn_condition_loop, dtype = bool)
        plot_bpt_snr_filter_passed_loop = np.array(plot_bpt_snr_filter_passed_loop, dtype = bool)

        current_file_agn_pixels = np.sum(agn_map_loop == 1)
        current_file_non_agn_pixels = np.sum(agn_map_loop == 0)
        current_file_masked_pixels = np.sum(agn_map_loop == -1)
        print(f" Processing completed. -> AGN (Code 1): {current_file_agn_pixels}, Non-AGN (Code 0): {current_file_non_agn_pixels}, Invalid/Masked (Code -1): {current_file_masked_pixels}")
        total_agn_pixels_overall += current_file_agn_pixels
        total_non_agn_pixels_overall += current_file_non_agn_pixels
        total_masked_pixels_overall += current_file_masked_pixels
        """Ending spaxels processing"""

        """Saving lists and plotting maps"""

        file_id_loop = os.path.basename(filename).replace('_EMISSION.fits', '').replace('.fits', '')
        output_txt_file_loop = os.path.join(config['output_list_dir'], f'{file_id_loop}_AGNxNonAGN_list.txt')
        try:# True se AGN pelo critério BPT
            header_loop = "# y x classification (-1=Mask/Inv, 0=Non-AGN, 1=AGN)"
            y_coords_loop, x_coords_loop = np.indices(agn_map_loop.shape)
            output_data_loop = np.column_stack((y_coords_loop.flatten(), x_coords_loop.flatten(), agn_map_loop.flatten()))
            np.savetxt(output_txt_file_loop, output_data_loop, fmt = '%d %d %d', header = header_loop)
            print(f'  List AGN x NonAGN saved in: {output_txt_file_loop}')
        except Exception as e:
            print(f'  ERROR saving TXT file in {output_txt_file_loop} : {e}')

        """Plotting Map"""
        try:
            output_plot_file_loop = os.path.join(config['output_plot_dir'], f'{file_id_loop}_AGNxNonAGN_plot.png')
            cmap_map_loop = ListedColormap(['grey', 'blue', 'red']) 
            bounds_map_loop = [-1.5, -0.5, 0.5, 1.5]
            norm_map_loop = plt.cm.colors.BoundaryNorm(bounds_map_loop, cmap_map_loop.N)
            map_labels_loop = ['Masked/Invalid', 'Non-AGN', 'AGN']
            fig_map_loop, ax_map_loop = plt.subplots(figsize=(5.5, 4.5))
            im_map_loop = ax_map_loop.imshow(agn_map_loop, cmap=cmap_map_loop, norm=norm_map_loop, origin='lower', interpolation='nearest')
            cbar_map_loop = fig_map_loop.colorbar(im_map_loop, ax=ax_map_loop, ticks=[-1, 0, 1], spacing='proportional')
            cbar_map_loop.ax.set_yticklabels(map_labels_loop)
            ax_map_loop.set_xlabel("X Coordinate (pixel)"); ax_map_loop.set_ylabel("Y Coordinate (pixel)")
            fig_map_loop.tight_layout()
            plt.savefig(output_plot_file_loop, dpi=300, bbox_inches='tight')
            plt.close(fig_map_loop)
            print(f"  MAP saved in: {output_plot_file_loop}")

        except Exception as e:
            print(f'  ERROR creating or saving AGN Map {output_plot_file_loop}: {e}')
            if 'fig_map_loop' in locals() and plt.fignum_exists(fig_map_loop.number): plt.close(fig_map_loop)

        """Preparing data for diagrams"""
        valid_plot_indices_loop = plot_final_code_loop >= 0

        if not np.any(valid_plot_indices_loop):
            print('WARNING: No valid points (final score 0 or 1) can be plotted on the diagrams in this file.')     
        else:
            current_file_log_nii_ha_valid = plot_data_loop['log_nii_ha'][valid_plot_indices_loop]
            current_file_log_oiii_hb_valid= plot_data_loop['log_oiii_hb'][valid_plot_indices_loop]
            current_file_log_ew_ha_valid = plot_data_loop['log_ew_ha'][valid_plot_indices_loop]

            current_file_bpt_agn_cond_valid = plot_bpt_agn_condition_loop[valid_plot_indices_loop] 
            current_file_whan_agn_cond_valid = plot_whan_agn_condition_loop[valid_plot_indices_loop]
            current_file_bpt_snr_passed_valid = plot_bpt_snr_filter_passed_loop[valid_plot_indices_loop]

            all_files_log_nii_ha.extend(current_file_log_nii_ha_valid.tolist())
            all_files_log_oiii_hb.extend(current_file_log_oiii_hb_valid.tolist())
            all_files_log_ew_ha.extend(current_file_log_ew_ha_valid.tolist())
            all_files_bpt_agn_condition.extend(current_file_bpt_agn_cond_valid.tolist())
            all_files_whan_agn_condition.extend(current_file_whan_agn_cond_valid.tolist())
            all_files_bpt_snr_passed.extend(current_file_bpt_snr_passed_valid.tolist())

            output_diagram_file_loop = os.path.join(config["output_diagram_dir"], f"{file_id_loop}_DIAGRAMS_AGNvsNonAGN.png")
            cmap_condition_diag_loop = ListedColormap(['blue', 'red']) 
            bounds_condition_diag_loop = [-0.5, 0.5, 1.5] 
            norm_condition_diag_loop = plt.cm.colors.BoundaryNorm(bounds_condition_diag_loop, cmap_condition_diag_loop.N)
            condition_labels_loop = ['Non AGN', 'AGN']
            plot_kwargs_loop = {'s': 5, 'alpha': 0.4, 'rasterized': True}
            try:
                fig_diag_loop, axes_diag_loop = plt.subplots(1, 2, figsize=(11, 4.5))
                
                
                ax_bpt_loop = axes_diag_loop[0]
                valid_bpt_points_for_plot = (np.isfinite(current_file_log_nii_ha_valid) & 
                                            np.isfinite(current_file_log_oiii_hb_valid) & 
                                            current_file_bpt_snr_passed_valid)
                
                if np.any(valid_bpt_points_for_plot):
                    ax_bpt_loop.scatter(
                        current_file_log_nii_ha_valid[valid_bpt_points_for_plot], 
                        current_file_log_oiii_hb_valid[valid_bpt_points_for_plot],
                        c=current_file_bpt_agn_cond_valid[valid_bpt_points_for_plot], 
                        cmap=cmap_condition_diag_loop, norm=norm_condition_diag_loop, **plot_kwargs_loop)
                
                
                x_vals_nii_loop = np.linspace(-2.5, 0.4, 200)
                y_kewley01_loop = np.where(x_vals_nii_loop < 0.47 - EPS, (0.61 / (x_vals_nii_loop - 0.47 - EPS)) + 1.19, np.inf)
                ax_bpt_loop.plot(x_vals_nii_loop, y_kewley01_loop, 'k--', label='Kewley+01', lw=1)
                
               
                ax_bpt_loop.set_xlim(-1.5, 0.5)
                ax_bpt_loop.set_ylim(-1.2, 1.5)
                ax_bpt_loop.set_xlabel(r'$\log_{10}($ Flux([NII]) / Flux(H$\alpha$) $)$')
                ax_bpt_loop.set_ylabel(r'$\log_{10}($ Flux([OIII]) / Flux(H$\beta$) $)$')
                ax_bpt_loop.grid(True, linestyle=':', alpha=0.5)
                
               
                ax_bpt_loop.legend(loc='lower left', fontsize='small', frameon=True)

              
                ax_whan_loop = axes_diag_loop[1]
                valid_whan_points_for_plot = np.isfinite(current_file_log_nii_ha_valid) & np.isfinite(current_file_log_ew_ha_valid)
                
                if np.any(valid_whan_points_for_plot):
                    ax_whan_loop.scatter(
                        current_file_log_nii_ha_valid[valid_whan_points_for_plot], 
                        current_file_log_ew_ha_valid[valid_whan_points_for_plot],
                        c=current_file_whan_agn_cond_valid[valid_whan_points_for_plot], 
                        cmap=cmap_condition_diag_loop, norm=norm_condition_diag_loop, **plot_kwargs_loop)   
            
                ew_min_log_loop = np.log10(0.1 + EPS)
                ew_max_log_loop = np.log10(100 + EPS) 
                ax_whan_loop.hlines([np.log10(3.0)], -2.0, 1.0, colors='k', linestyles='-', lw=1.5, alpha=0.7, label=r'$|EW(H\alpha)| = 3\AA$')
                ax_whan_loop.vlines(-0.4, ew_min_log_loop - 0.5, ew_max_log_loop + 0.5, colors='k', linestyles=':', lw=0.8, alpha=0.7, label=r'$\log([NII]/H\alpha) = -0.4$')
            
                ax_whan_loop.set_xlim(-1.5, 0.5)
                ax_whan_loop.set_ylim(ew_min_log_loop, ew_max_log_loop)
                ax_whan_loop.set_xlabel(r'$\log_{10}($ Flux([NII]) / Flux(H$\alpha$) $)$')
                ax_whan_loop.set_ylabel(r'$\log_{10}($ $|EW(H\alpha)|$ / $\AA$ $)$')
                ax_whan_loop.grid(True, linestyle=':', alpha=0.5)
            
                ax_whan_loop.legend(loc='upper left', fontsize='small', frameon=True)

            
                handles_loop = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=cmap_condition_diag_loop(norm_condition_diag_loop(i)), 
                                        markersize=8, label=label) for i, label in enumerate(condition_labels_loop)] 
         
                fig_diag_loop.legend(handles=handles_loop, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                                    ncol=len(handles_loop), frameon=False, fontsize='medium')

                
                fig_diag_loop.tight_layout(rect=[0, 0, 1, 0.95]) 
                
                plt.savefig(output_diagram_file_loop, dpi=100, bbox_inches='tight')                
                plt.close(fig_diag_loop)
                print(f"  Original Diagrams saved in: {output_diagram_file_loop}")

            except Exception as e:
                print(f"  ERROR creating Original Diagrams {output_diagram_file_loop}: {e}")
                import traceback
                traceback.print_exc()
                if 'fig_diag_loop' in locals() and plt.fignum_exists(fig_diag_loop.number): 
                    plt.close(fig_diag_loop)   

            """Plotting Diagnostic Diagrams with Emphasis on AGN (BPT & WHAN)"""  
            output_highlighted_diagram_file_loop = os.path.join(config["output_highlighted_diagram_dir"], f"{file_id_loop}_DIAGRAMS_HIGHLIGHTED_AGN.png")
            agn_in_both_bpt_and_whan_condition = current_file_bpt_agn_cond_valid & current_file_whan_agn_cond_valid
            color_codes_highlight_loop = agn_in_both_bpt_and_whan_condition.astype(int)
            cmap_highlight = ListedColormap(['blue', 'red']) 
            bounds_highlight = [-0.5, 0.5, 1.5]
            norm_highlight = plt.cm.colors.BoundaryNorm(bounds_highlight, cmap_highlight.N)
            highlight_labels = ['Others', 'AGN (BPT & WHAN)']

            try:
                fig_diag_highlight_loop, axes_diag_highlight_loop = plt.subplots(1, 2, figsize=(11, 4.5))

                ax_bpt_highlight_loop = axes_diag_highlight_loop[0]
                if np.any(valid_bpt_points_for_plot):
                    ax_bpt_highlight_loop.scatter(
                        current_file_log_nii_ha_valid[valid_bpt_points_for_plot], 
                        current_file_log_oiii_hb_valid[valid_bpt_points_for_plot],
                        c=color_codes_highlight_loop[valid_bpt_points_for_plot], 
                        cmap=cmap_highlight, norm=norm_highlight, **plot_kwargs_loop)
                
                ax_bpt_highlight_loop.plot(x_vals_nii_loop, y_kewley01_loop, 'k--', label='Kewley+01', lw=1)
                
                ax_bpt_highlight_loop.set_xlim(-1.5, 0.5)
                ax_bpt_highlight_loop.set_ylim(-1.2, 1.5)
                ax_bpt_highlight_loop.set_xlabel(r'$\log_{10}($ Flux([NII]) / Flux(H$\alpha$) $)$')
                ax_bpt_highlight_loop.set_ylabel(r'$\log_{10}($ Flux([OIII]) / Flux(H$\beta$) $)$')
                ax_bpt_highlight_loop.grid(True, linestyle=':', alpha=0.5)
                
                ax_bpt_highlight_loop.legend(loc='lower left', fontsize='small')

                ax_whan_highlight_loop = axes_diag_highlight_loop[1]
                if np.any(valid_whan_points_for_plot):
                    ax_whan_highlight_loop.scatter(
                        current_file_log_nii_ha_valid[valid_whan_points_for_plot], 
                        current_file_log_ew_ha_valid[valid_whan_points_for_plot],
                        c=color_codes_highlight_loop[valid_whan_points_for_plot], 
                        cmap=cmap_highlight, norm=norm_highlight, **plot_kwargs_loop)
                        
                ax_whan_highlight_loop.hlines([np.log10(3.0)], -2.0, 1.0, colors='k', linestyles='-', lw=1.5, alpha=0.7, label=r'$|EW(H\alpha)| = 3\AA$')
                ax_whan_highlight_loop.vlines(-0.4, ew_min_log_loop - 0.5, ew_max_log_loop + 0.5, colors='k', linestyles=':', lw=0.8, alpha=0.7, label=r'$\log([NII]/H\alpha) = -0.4$')
                
                ax_whan_highlight_loop.set_xlim(-1.5, 0.5)
                ax_whan_highlight_loop.set_ylim(ew_min_log_loop, ew_max_log_loop)
                ax_whan_highlight_loop.set_xlabel(r'$\log_{10}($ Flux([NII]) / Flux(H$\alpha$) $)$')
                ax_whan_highlight_loop.set_ylabel(r'$\log_{10}($ $|EW(H\alpha)|$ / $\AA$ $)$')
                ax_whan_highlight_loop.grid(True, linestyle=':', alpha=0.5)
                plt.savefig(output_diagram_file_loop, dpi=150, bbox_inches='tight')

                ax_whan_highlight_loop.legend(loc='upper left', fontsize='small')

                handles_highlight_loop = [
                    plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=cmap_highlight(norm_highlight(i)), 
                            markersize=10, label=label) 
                    for i, label in enumerate(highlight_labels)
                ] 
                
                fig_diag_highlight_loop.legend(
                    handles=handles_highlight_loop, 
                    loc='upper center', 
                    bbox_to_anchor=(0.5, 1.0), 
                    ncol=2, 
                    fontsize='medium', 
                    title_fontsize='large', 
                    frameon=False
                )

                fig_diag_highlight_loop.tight_layout(rect=[0, 0, 1, 0.91]) 
                
                plt.savefig(output_highlighted_diagram_file_loop, dpi=150, bbox_inches='tight')
                plt.close(fig_diag_highlight_loop)
                print(f"  Diagrams saved in: {output_highlighted_diagram_file_loop}")

            except Exception as e:
                print(f"  ERROR creating highlighted diagrams {output_highlighted_diagram_file_loop}: {e}")
                import traceback
                traceback.print_exc()
                if 'fig_diag_highlight_loop' in locals() and plt.fignum_exists(fig_diag_highlight_loop.number):
                    plt.close(fig_diag_highlight_loop)
   
    print("\n--- Processing of all files completed! ---")

    print("\n--- General Processing Summary (All Files) ---")

    print(f"Total AGN pixels: {total_agn_pixels_overall}")

    print(f"Total Non-AGN pixels: {total_non_agn_pixels_overall}")

    print(f"Total Masked/Invalid pixels: {total_masked_pixels_overall}")

    """Plotting Consolidated Diagrams (all files)"""
if all_files_log_nii_ha: 
        print("\n--- Generating Consolidated Diagrams (All Files) ---")
        
        cf_log_nii_ha = np.array(all_files_log_nii_ha)
        cf_log_oiii_hb = np.array(all_files_log_oiii_hb)
        cf_log_ew_ha = np.array(all_files_log_ew_ha)
        cf_is_bpt_agn = np.array(all_files_bpt_agn_condition, dtype=bool)
        cf_is_whan_agn = np.array(all_files_whan_agn_condition, dtype=bool)
        cf_bpt_snr_ok = np.array(all_files_bpt_snr_passed, dtype=bool)

        cf_highlight_condition = cf_is_bpt_agn & cf_is_whan_agn

        cmap_consolidated = ListedColormap(['blue', 'red']) 
        consolidated_labels = ['Others', 'AGN (BPT & WHAN)'] 
        
        s_other = 1
        s_highlight = 1
        alpha_other = 0.5
        alpha_highlight = 0.7 

        output_consolidated_file = os.path.join(config["output_consolidated_diagram_dir"], "CONSOLIDATED_DIAGRAMS_BPT_WHAN.png")

        try:
            # figsize levemente maior na altura para acomodar a legenda global
            fig_consolidated, axes_consolidated = plt.subplots(1, 2, figsize=(11, 5.0))

            ax_bpt_cf = axes_consolidated[0]
            valid_bpt_points_cf = np.isfinite(cf_log_nii_ha) & np.isfinite(cf_log_oiii_hb) & cf_bpt_snr_ok
            
            if np.any(valid_bpt_points_cf):
                log_nii_ha_bpt_cf = cf_log_nii_ha[valid_bpt_points_cf]
                log_oiii_hb_bpt_cf = cf_log_oiii_hb[valid_bpt_points_cf]
                highlight_cond_bpt_cf = cf_highlight_condition[valid_bpt_points_cf]

                ax_bpt_cf.scatter(
                    log_nii_ha_bpt_cf[~highlight_cond_bpt_cf], log_oiii_hb_bpt_cf[~highlight_cond_bpt_cf],
                    color=cmap_consolidated.colors[0], s=s_other, alpha=alpha_other, rasterized=True
                )
                ax_bpt_cf.scatter(
                    log_nii_ha_bpt_cf[highlight_cond_bpt_cf], log_oiii_hb_bpt_cf[highlight_cond_bpt_cf],
                    color=cmap_consolidated.colors[1], s=s_highlight, alpha=alpha_highlight, rasterized=True
                )
            else:
                print("  Consolidated BPT Notice: No valid points for plotting.")

            x_vals_nii_cf = np.linspace(-2.5, 0.4, 200) 
            y_kewley01_cf = np.where(x_vals_nii_cf < 0.47 - EPS, (0.61 / (x_vals_nii_cf - 0.47 - EPS)) + 1.19, np.inf)
            ax_bpt_cf.plot(x_vals_nii_cf, y_kewley01_cf, 'k--', label='Kewley+01 (AGN Separation)', lw=1) 
            ax_bpt_cf.set_xlim(-1.5, 0.5); ax_bpt_cf.set_ylim(-1.2, 1.5)
            ax_bpt_cf.set_xlabel('$log_{10}$( Flux([NII]) / Flux(Hα) )')
            ax_bpt_cf.set_ylabel('$log_{10}$( Flux([OIII]) / Flux(Hβ) )')
            ax_bpt_cf.grid(True, linestyle=':', alpha=0.5)
            
            # Legenda local BPT
            ax_bpt_cf.legend(loc='lower left', fontsize='small')

            ax_whan_cf = axes_consolidated[1]
            valid_whan_points_cf = np.isfinite(cf_log_nii_ha) & np.isfinite(cf_log_ew_ha)

            if np.any(valid_whan_points_cf):
                log_nii_ha_whan_cf = cf_log_nii_ha[valid_whan_points_cf]
                log_ew_ha_whan_cf = cf_log_ew_ha[valid_whan_points_cf]
                highlight_cond_whan_cf = cf_highlight_condition[valid_whan_points_cf]

                ax_whan_cf.scatter(
                    log_nii_ha_whan_cf[~highlight_cond_whan_cf], log_ew_ha_whan_cf[~highlight_cond_whan_cf],
                    color=cmap_consolidated.colors[0], s=s_other, alpha=alpha_other, rasterized=True
                )
                ax_whan_cf.scatter(
                    log_nii_ha_whan_cf[highlight_cond_whan_cf], log_ew_ha_whan_cf[highlight_cond_whan_cf],
                    color=cmap_consolidated.colors[1], s=s_highlight, alpha=alpha_highlight, rasterized=True
                )
            else:
                print("  Consolidated WHAN Notice: No valid points to plot.")
            
            ew_min_log_cf = np.log10(0.1+EPS); ew_max_log_cf = np.log10(100+EPS) 
            ax_whan_cf.hlines([np.log10(3.0)], -2.0, 1.0, colors='k', linestyles='-', lw=1.5, alpha=0.7, label='|EW(Hα)| = 3Å')
            ax_whan_cf.vlines(-0.4, ew_min_log_cf - 0.5 , ew_max_log_cf + 0.5, colors='k', linestyles=':', lw=0.8, alpha=0.7, label='log([NII]/Hα) = -0.4')
            ax_whan_cf.set_xlim(-1.5, 0.5); ax_whan_cf.set_ylim(ew_min_log_cf, ew_max_log_cf)
            ax_whan_cf.set_xlabel('$log_{10}$( Flux([NII]) / Flux(Hα) )')
            ax_whan_cf.set_ylabel('$log_{10}$( |EW(Hα)| / Å )')
            ax_whan_cf.grid(True, linestyle=':', alpha=0.5)
            
            # Legenda local WHAN
            ax_whan_cf.legend(loc='upper left', fontsize='small')

            consolidated_legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w', label=consolidated_labels[0],
                           markerfacecolor=cmap_consolidated.colors[0], markersize=8), 
                plt.Line2D([0], [0], marker='o', color='w', label=consolidated_labels[1],
                           markerfacecolor=cmap_consolidated.colors[1], markersize=8) 
            ]

            # Legenda Global para cores
            fig_consolidated.legend(handles=consolidated_legend_handles, loc='upper center', 
                                    bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
            
            # Ajuste do layout para acomodar a legenda no topo (rect top=0.92)
            fig_consolidated.tight_layout(rect=[0, 0, 1, 0.92]) 
            
            plt.savefig(output_consolidated_file, dpi=300, bbox_inches='tight')
            plt.close(fig_consolidated)
            print(f"  Consolidated diagrams saved in: {output_consolidated_file}")

        except Exception as e:
            print(f"  ERROR creating/saving consolidated diagrams {output_consolidated_file}: {e}"); traceback.print_exc()
            if 'fig_consolidated' in locals() and plt.fignum_exists(fig_consolidated.number): plt.close(fig_consolidated)




