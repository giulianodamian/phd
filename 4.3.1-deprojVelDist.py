# %%
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import numpy as np

SOURCE_FILE = '/scratch/users/giuliano.damian/text/4.3-df_code.txt'
BACKUP_FILE = '/scratch/users/giuliano.damian/text/4.3-df_code.txt.bak'
OUTPUT_DIR = '/scratch/users/giuliano.damian/text'
OUTPUT_GRAPH_DIR = '/scratch/users/giuliano.damian/images'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, '4.3.1-df_code.txt')
OUTPUT_GRAPH = os.path.join(OUTPUT_GRAPH_DIR, 'deprojected_distances.png')
ROTATION_CURVE_DIR = '/scratch/users/giuliano.damian/images/4.3.1-rotation_curves'
os.makedirs(ROTATION_CURVE_DIR, exist_ok = True)
VEL_MAP_DIR = '/scratch/users/giuliano.damian/images/4.3.1-velocity_map'
os.makedirs(VEL_MAP_DIR, exist_ok = True)

# %%
try:
    #print(f"\nCreating backup of original file at '{BACKUP_FILE}'...\n")
    #shutil.copy(SOURCE_FILE, BACKUP_FILE)

    print(f"\nReading data from '{SOURCE_FILE}'...\n")

    df = pd.read_csv(SOURCE_FILE, sep = '\t', header = 0)

    original_columns = df.columns.tolist()

    print(f"\nDefining variable names for columns in '{SOURCE_FILE}'...\n")

    file = df['source_file']
    x = df['x']
    y = df['y']
    b_over_a = df['nsa_sersic_ba']
    dist_kpc = df['distance_kpc_finded']
    yc = df['center_y']
    xc = df['center_x']
    angle_phi = df['angle_phi']
    angle_alpha = df['angle_alpha']
    angle_theta = df['angle_theta']
    angle_i = df['angle_i']
    vrot_star = df['vrot_star']
    y_rot = df['y_rot']
    x_rot = df['x_rot']
    yc_rot = df['center_y_rot']
    xc_rot = df['center_x_rot']
    v_real = df['v_real']
    dist_real = df['real_dist']

    df['dist_pixel']  = np.sqrt((x - xc)**2 + (y - yc)**2)
    dist_pixel = df['dist_pixel']

    angle_theta_rad = np.deg2rad(angle_theta)
    cos_part = (np.cos((angle_theta_rad)))**2
    sin_part = (np.sin((angle_theta_rad)) / b_over_a)**2
    deproj_scale = np.sqrt(sin_part + cos_part)

    df['dist_real'] = dist_pixel * deproj_scale
    dist_real = df['dist_real']
  

    # Visualize the relation between distance real and deprojected:

    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(df['dist_pixel'], df['dist_real'], alpha = 0.5)
    plt.xlabel('Non-deprojected')
    plt.ylabel('Deprojected')
    plt.xlim(left = 0)
    plt.ylim(bottom = 0)
    plt.axis('equal')
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5, axis = 'both', alpha = 0.7)
    plt.savefig('/scratch/users/giuliano.damian/images/4.3.1-distances.png')
    plt.close()

    dist_real_kpc = dist_kpc * dist_real/dist_pixel
    df['dist_real_kpc'] = dist_real_kpc
    
    x_relative = df['x'] - df['center_x']
    y_relative = df['y'] - df['center_y']

    df['x_rot'] = (x_relative * np.cos(angle_theta_rad) + y_relative * np.sin(angle_theta_rad)) + df['center_x']
    df['y_rot'] = (-x_relative * np.sin(angle_theta_rad) + y_relative * np.cos(angle_theta_rad)) + df['center_y']

    angle_phi_rad =  np.deg2rad(angle_phi)
    angle_i_rad = np.deg2rad(angle_i)

    m = np.tan(angle_phi_rad)
    reta = (x_relative * m) + yc
    tol = np.sqrt(2) - 1.0
    cond = np.isclose(y, reta, atol=tol)
    v_calculated = vrot_star / np.cos(angle_i_rad)
    df['v_real'] = np.where(cond, v_calculated, np.nan)

    graph_data = {'vel_pre_graph':[], 'name_pre_graph':[], 'position_x':[], 'position_y':[], 'dist_real':[]}

    # Velocity Scatter:

    for i in range(len(df)):
        if not np.isnan(df.loc[i, 'v_real']):
            graph_data['vel_pre_graph'].append(df.loc[i, 'v_real'])
            graph_data['name_pre_graph'].append(df.loc[i, 'source_file'])
            graph_data['position_x'].append(df.loc[i, 'x_rot'])
            graph_data['position_y'].append(df.loc[i, 'y_rot'])
            graph_data['dist_real'].append(df.loc[i, 'dist_real'])

    graph_data = pd.DataFrame(graph_data)
    
    print("\nStarting smoothing process and graph generation...")

    smoothed_galaxies = []

    for name, complete_group in df.groupby('source_file'):
        group_with_vel = complete_group.dropna(subset = ['v_real'])

        if len(group_with_vel) < 10:
            print(f"  -> Ignoring '{name}': has only {len(group_with_vel)} points with velocity (minimum 10).")
            continue

        print(f"  -> Processing and generating graph for: {name}")

        idx_central_point = group_with_vel['dist_real'].idxmin()
        central_vel = group_with_vel.loc[idx_central_point, 'v_real']

        relative_vel = np.abs(group_with_vel['v_real'] - central_vel)

        ordered_group = group_with_vel.copy()
        ordered_group['relative_vel'] = relative_vel
        ordered_group = ordered_group.sort_values(by = 'dist_real')

        softned = sm.nonparametric.lowess(endog = ordered_group['relative_vel'], exog = ordered_group['dist_real'], frac = 0.7)

        plt.figure(figsize = (5.5, 4.5))
        plt.scatter(ordered_group['dist_real'], ordered_group['relative_vel'], alpha = 0.7, s = 20, label = 'Observational Data')
        plt.plot(softned[:, 0], softned[:, 1], color = 'red', lw = 2, label = 'LOWESS Trend')
        plt.xlabel('Real Distance from Center (deprojected pixels)')
        plt.ylabel('Relative Velocity Modulus (km/s)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend()
        file_name = f"{name}_curva_rotacao.png"
        complete_path = os.path.join(ROTATION_CURVE_DIR, file_name)
        plt.savefig(complete_path, dpi=150)
        plt.close()

        x_tendency = softned[:, 0]
        y_relative_tendency = softned[:, 1]

        target_distances = complete_group['dist_real']
        interpoled_relative_vel = np.interp(target_distances, x_tendency, y_relative_tendency)

        df.loc[complete_group.index, 'v_real'] = interpoled_relative_vel

        smoothed_galaxies.append(name)

    print("\nProcess finished! The column 'v_real' was updated with smoothed and interpolated values.")
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['v_real'] = pd.to_numeric(df['v_real'], errors='coerce')
    print("\nStarting velocity map generation...")

    for name, group in df.groupby('source_file'):

        print(f"  -> Generating velocity map for: {name}")

        # Forçando conversão numérica segura
        x_vals = pd.to_numeric(group['x'], errors='coerce')
        y_vals = pd.to_numeric(group['y'], errors='coerce')
        v_vals = pd.to_numeric(group['v_real'], errors='coerce')

        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(v_vals)

        if valid_mask.sum() == 0:
            print(f"     Skipping {name}: no valid numeric velocity data.")
            continue

        plt.figure(figsize=(5.5, 4.5))

        scatter = plt.scatter(
            x_vals[valid_mask],
            y_vals[valid_mask],
            c=v_vals[valid_mask],
            cmap='jet',
            s=10
        )

        plt.colorbar(scatter, label='Velocity (km/s)')
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')
        plt.gca().set_aspect('equal', adjustable='box')

        output_path = os.path.join(VEL_MAP_DIR, f"{name}_velocity_map.png")
        plt.savefig(output_path, dpi=150)
        plt.close()

        print("\nVelocity maps generated successfully for eligible galaxies!")

    print("\nRemoving galaxies that were not smoothed from the final file...")

    total_galaxies  =df['source_file'].nunique()
    galaxies_to_keep = len(smoothed_galaxies)
    print(f"  -> {total_galaxies} total galaxies at the start.")
    print(f"  -> {galaxies_to_keep} galaxies were successfully processed and will be kept.")
    print(f"  -> {total_galaxies - galaxies_to_keep} galaxies will be excluded from the output file.")

    df_final = df[df['source_file'].isin(smoothed_galaxies)].copy()

    print("\nStarting smoothing process and graph generation...")

    smoothed_galaxies = []
    smoothed_results = {}

    for galaxy in smoothed_galaxies:
        smoothed_curve = smoothed_results[galaxy]

        vel_peak = smoothed_curve[:, 1].max()

        galaxy_index = df_final[df_final['source_file'] == galaxy].index

        print(f"  -> For '{galaxy}', the smoothed curve peak is {vel_peak:.2f}. Assigning to {len(galaxy_index)} rows.")

        df_final.loc[galaxy_index, 'v_real'] = vel_peak

    print("\nUpdate of 'v_real' completed!")

    print("\n--- Preparing to save final file ---")

    columns_to_remove = ['center_x_rot', 'center_y_rot', 'distance_kpc_finded', 'real_dist']
    base_list = original_columns + ['dist_pixel', 'dist_real', 'dist_real_kpc']    
    final_columns = [column for column in base_list if column not in columns_to_remove]

    print(f"Saving modified columns to '{OUTPUT_FILE}'...")

    df_final.to_csv(
        OUTPUT_FILE,
        columns = final_columns,
        sep = '\t',
        index = False,
        na_rep = 'NaN'
    )

    print("File saved successfully!")

except FileNotFoundError:
    print(f"\nCRITICAL ERROR: Source file '{SOURCE_FILE}' was not found. Process aborted.")
except Exception as e:
    print(f"\nUNEXPECTED ERROR: {e}. Check backup at '{BACKUP_FILE}'.")

