import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os # For checking file existence

def create_us_state_map(state_color_map,
                         output_filename="us_state_colors.png",
                         shapefile_path="cb_2018_us_state_500k.shp", # Default path relative to script
                         default_color='lightgrey',
                         exclude_territories=True,
                         title="US State Map"):
    """
    Generates a PNG map of the US with states colored according to input.

    Args:
        state_color_map (dict): A dictionary where keys are state abbreviations
                                (e.g., 'CA', 'TX') and values are color strings
                                acceptable by matplotlib (e.g., 'red', '#FF0000').
        output_filename (str): The name of the output PNG file.
        shapefile_path (str): The path to the US states .shp file.
        default_color (str): Color for states not included in state_color_map.
        exclude_territories (bool): If True, filters out territories like PR, VI.
        title (str): The title for the map.
    """
    print(f"Attempting to load shapefile from: {shapefile_path}")
    if not os.path.exists(shapefile_path):
        print(f"ERROR: Shapefile not found at '{shapefile_path}'")
        print("Please download the US States Cartographic Boundary shapefile (e.g., cb_2018_us_state_500k.shp)")
        print("from the US Census Bureau or provide the correct path.")
        return

    try:
        # 1. Load US States Geometry
        states_gdf = gpd.read_file(shapefile_path)
        print(f"Shapefile loaded successfully. Columns: {states_gdf.columns.tolist()}")

        # Check for common state abbreviation columns ('STUSPS' is typical for Census data)
        if 'STUSPS' not in states_gdf.columns:
             # Try 'postal' or 'iso_a2' if using Natural Earth or other sources
             possible_abbr_cols = ['STUSPS', 'postal', 'iso_a2', 'STATE_ABBR']
             abbr_col = None
             for col in possible_abbr_cols:
                  if col in states_gdf.columns:
                       abbr_col = col
                       print(f"Using column '{abbr_col}' for state abbreviations.")
                       break
             if abbr_col is None:
                  print(f"ERROR: Could not find a suitable state abbreviation column in the shapefile.")
                  print(f"       Available columns: {states_gdf.columns.tolist()}")
                  return
        else:
             abbr_col = 'STUSPS'

        # 2. Filter out territories if requested (based on state FIPS codes < 60)
        if exclude_territories and 'STATEFP' in states_gdf.columns:
            print("Excluding territories (STATEFP >= 60)...")
            # Ensure STATEFP is numeric
            states_gdf['STATEFP'] = pd.to_numeric(states_gdf['STATEFP'], errors='coerce')
            states_gdf = states_gdf[states_gdf['STATEFP'] < 60]
            print(f"Remaining states/districts: {len(states_gdf)}")


        # 3. Prepare Input Data
        # Ensure keys in the input map are uppercase for matching standard abbreviations
        state_color_map_upper = {k.upper(): v for k, v in state_color_map.items()}
        data_df = pd.DataFrame(list(state_color_map_upper.items()), columns=[abbr_col, 'color'])

        # 4. Merge Data with Geometries
        # Ensure the merge column in states_gdf is uppercase string if needed
        states_gdf[abbr_col] = states_gdf[abbr_col].astype(str).str.upper()
        # Use left join to keep all states
        merged_gdf = states_gdf.merge(data_df, on=abbr_col, how='left')

        # 5. Handle Missing Colors
        merged_gdf['plot_color'] = merged_gdf['color'].fillna(default_color)

        # 6. Create the Plot
        # Use Albers Equal Area projection for continental US - requires filtering
        # For simplicity first, plot all loaded states without special projection
        # Or filter to CONUS approx range: STATEFP < 60 excludes territories,
        # but AK and HI need special handling for typical CONUS+Insets maps.
        # Let's plot everything together first, then suggest CONUS filtering.

        # Plotting directly (includes AK, HI potentially far away)
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        merged_gdf.plot(color=merged_gdf['plot_color'], linewidth=0.5, ax=ax, edgecolor='0.8')

        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

        # Add title
        #ax.set_title(title, fontdict={'fontsize': '16', 'fontweight' : '3'})

        # Optional: Plot only CONUS (Continental US) - adjust plot bounds
        # Example rough bounding box for CONUS
        # ax.set_xlim(-125, -66.5)
        # ax.set_ylim(24, 49.5)
        # You might need to filter merged_gdf *before* plotting for this to work well.
        # Common CONUS states have STATEFP codes excluding 02 (AK) and 15 (HI).

        # 7. Save Output
        plt.savefig(output_filename, dpi=1000, bbox_inches='tight')
        print(f"Map saved to {output_filename}")
        plt.close(fig) # Close the plot figure

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define your state colors
    # Example: Color states based on some category
    color_data = {
        'CA': 'blue',
        'TX': 'red',
        'NY': 'blue',
        'FL': 'red',
        'IL': 'blue',
        'WA': '#00FF00', # Green using hex
        'OR': 'green',
        'NV': 'orange',
        'AZ': 'orange',
        'CO': 'purple',
        'ME': '#AAAAAA' # Grey using hex
        # Add more states and colors as needed
    }

    # 2. Specify path to your downloaded and unzipped shapefile
    #    (Make sure the .shp, .dbf, .shx etc. files are in this directory)
    #    This might be relative to where you run the script, or an absolute path.
    shapefile = "cb_2018_us_state_500k.shp"
    # If shapefile is in the same directory as the script:
    # shapefile = "cb_2018_us_state_500k.shp"

    # 3. Specify output filename
    output_file = "my_colored_us_map.png"

    # 4. Create the map
    create_us_state_map(
        state_color_map=color_data,
        output_filename=output_file,
        shapefile_path=shapefile,
        title="Example Colored US States Map"
    )

    # Example 2: Different colors, different output name
    color_data_2 = {
        'MN': 'darkblue', 'WI': 'darkblue', 'MI': 'darkblue', # Midwest cluster
        'GA': 'darkred', 'SC': 'darkred', 'NC': 'darkred',  # Southeast cluster
        'PA': 'darkgreen'
    }
    create_us_state_map(
        state_color_map=color_data_2,
        output_filename="regional_clusters_map.png",
        shapefile_path=shapefile,
        title="Regional Clusters Example"
    )
