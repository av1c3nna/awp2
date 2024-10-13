import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xarray as xr
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from windrose import WindroseAxes, plot_windrose
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import matplotlib.ticker as mtick
import argparse
import math
from matplotlib import cm


def convert_xr_to_df(xr):
    if "ref_datetime" and "valid_datetime" in xr.dims:

        df = xr.to_dataframe().reset_index()
        df["ref_datetime"] = df["ref_datetime"].dt.tz_localize("UTC")
        df["valid_datetime"] = df["ref_datetime"] + pd.TimedeltaIndex(df["valid_datetime"],unit="hours")
        df.rename(columns={'valid_datetime': 'valid_time', 'ref_datetime':'reference_time'}, inplace=True)

    else:
        df = xr.to_dataframe().reset_index()
        df["reference_time"] = df["reference_time"].dt.tz_localize("UTC")
        df["valid_time"] = df["reference_time"] + pd.TimedeltaIndex(df["valid_time"],unit="hours")

    return df

def concat_dfs(df_list):

    df_base = df_list[0]
    for i in range (1, len(df_list)):
        df_base = pd.concat([df_base, df_list[i]], axis=0)
    return df_base

def groupby_time(df):
    df = df.groupby(["reference_time", "valid_time"], as_index=False).mean()

    return df.drop(["latitude", "longitude", "point"], axis=1, errors="ignore")

def groupby_coordinates(df):
    df = df.groupby(["latitude", "longitude", "valid_time"], as_index=False).agg(['mean', 'std'])
    df = df.drop(["reference_time", "point"], axis=1, errors="ignore")
    #df.columns = ['_'.join(col).strip() for col in df.columns.values]
    

    return df#df.rename(columns={'latitude_': 'latitude', 'longitude_':'longitude'})

def combine_files(file_paths, agg=None):
    dataframes = []
    
    # Lade jede Datei, konvertiere sie in einen DataFrame und füge zur Liste hinzu
    for file_path in file_paths:
        xr_dataset = xr.open_dataset(file_path)  # NetCDF-Datei laden
        df = convert_xr_to_df(xr_dataset)       # Konvertieren in DataFrame
        dataframes.append(df)                   # Hinzufügen zur Liste
    
    # DataFrames zusammenfügen
    df_full = concat_dfs(dataframes)
    
    if not agg:
        df_full_agg = df_full

    if agg == "time":
        df_full_agg = groupby_time(df_full)

    if agg == "coordinates":
        df_full_agg = groupby_coordinates(df_full)
    
    
    return df_full_agg

def show_histograms(plot):
    # Initialize a 3x3 charts
    # plot = df.iloc[:, 2:]

    fig, axes = plt.subplots(nrows=1, ncols=len(plot.columns), figsize=(20, 6))

    # Flatten the axes array (makes it easier to iterate over)
    axes = axes.flatten()

    # Loop through each column and plot a histogram
    for i, column in enumerate(plot.columns):
        
        # Add the histogram
        plot[column].hist(ax=axes[i], # Define on which ax we're working on
                        edgecolor='white', # Color of the border
                        color='#69b3a2', # Color of the bins
                    )
        
        # Add title and axis label
        axes[i].set_title(f'{column} distribution') 
        axes[i].set_xlabel(column) 
        axes[i].set_ylabel('Frequency') 

    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def combine_histograms(dfs, names):
    """
    Funktion, die Histogramme der gleichen Spalten aus mehreren DataFrames auf dem gleichen Plot kombiniert.
    :param dfs: Liste von DataFrames
    """
    # Wähle die Spalten, die verglichen werden sollen (in diesem Fall alle nach der zweiten Spalte)
    # columns = dfs[0].columns[2:]
    columns = dfs[0].columns

    # Erstelle die Subplots für jede Spalte
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(20, 6))

    # Überprüfe, ob nur eine Spalte ausgewählt wurde
    if len(columns) == 1:
        axes = [axes]  # Stelle sicher, dass es sich um eine Liste handelt

    # Für jede Spalte
    for i, column in enumerate(columns):
        # Zeichne jedes DataFrame auf derselben Achse
        for df, name, color in zip(dfs, names, ["blue", "orange"]):
            df[column].hist(ax=axes[i],alpha=0.6, edgecolor='white', label=name, color=color)

        # Beschriftungen und Titel hinzufügen
        axes[i].set_title(f'{column} distribution')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

    # Layout anpassen und alle Plots anzeigen
    plt.tight_layout()
    plt.show()

def show_lineplot(plot, freq):
    plot = plot.iloc[:, 1:]
    plot = plot.resample(freq, on='valid_time').mean().reset_index()

    fig, axes = plt.subplots(nrows=len(plot.columns)-1, ncols=1, figsize=(20, 20))

    # Flatten the axes array (makes it easier to iterate over)
    axes = axes.flatten()

    # Loop through each column and plot a histogram
    for i, column in enumerate(plot.iloc[:, 1:].columns):
        
        # Add the histogram
        axes[i].plot(plot["valid_time"], plot[column], marker='o', color='#69b3a2')
        
        # Add title and axis label
        axes[i].set_title(f'{column} over time') 
        axes[i].set_xlabel('time') 
        axes[i].set_ylabel(column) 
        axes[i].grid(True)

    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def show_lineplot_energy(plot, freq):
    # plot = plot.iloc[:, 1:]
    plot = plot.resample(freq, on='dtm').mean().reset_index()

    fig, axes = plt.subplots(nrows=len(plot.columns)-1, ncols=1, figsize=(20, 20))

    # Flatten the axes array (makes it easier to iterate over)
    axes = axes.flatten()

    # Loop through each column and plot a histogram
    for i, column in enumerate(plot.iloc[:, 1:].columns):
        
        # Add the histogram
        axes[i].plot(plot["dtm"], plot[column], marker='o', color='#69b3a2')
        
        # Add title and axis label
        axes[i].set_title(f'{column} over time') 
        axes[i].set_xlabel('time') 
        axes[i].set_ylabel(column) 
        axes[i].grid(True)

    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def compare_lineplots(plots, names, freq):


    fig, axes = plt.subplots(nrows=len(plots[0].columns)-2, ncols=1, figsize=(20, 20))

    # Flatten the axes array (makes it easier to iterate over)
    axes = axes.flatten()

    # Loop through each column and plot a histogram
    for i, column in enumerate(plots[0].iloc[:, 2:].columns):

        for plot, name, color in zip(plots, names, ["blue", "orange"]):
            plot = plot.iloc[:, 1:]
            plot = plot.resample(freq, on='valid_time').mean().reset_index()
            axes[i].plot(plot["valid_time"], plot[column], marker='o', label = name, color=color, alpha=0.5)
        
        # Add title and axis label
        axes[i].set_title(f'{column} over time') 
        axes[i].set_xlabel('time') 
        axes[i].set_ylabel(column) 
        axes[i].legend()
        axes[i].grid(True)

    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def compare_lineplots_energy(plots, names, freq):


    fig, axes = plt.subplots(nrows=len(plots[0].columns)-1, ncols=1, figsize=(20, 20))

    # Flatten the axes array (makes it easier to iterate over)
    axes = axes.flatten()

    # Loop through each column and plot a histogram
    for i, column in enumerate(plots[0].iloc[:, 1:].columns):

        for plot, name, color in zip(plots, names, ["blue", "orange"]):
            # plot = plot.iloc[:, 1:]
            plot = plot.resample(freq, on='dtm').mean().reset_index()
            axes[i].plot(plot["dtm"], plot[column], marker='o', label = name, color=color, alpha=0.5)
        
        # Add title and axis label
        axes[i].set_title(f'{column} over time') 
        axes[i].set_xlabel('time') 
        axes[i].set_ylabel(column) 
        axes[i].legend()
        axes[i].grid(True)

    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def show_scatter(df,name, color_variable=None):

    if not color_variable:
        scatter = plt.scatter(df['latitude'], df['longitude'],
                        cmap='viridis', label=name, alpha=0.6)
        plt.title(f'Scatterplot of Latitude vs Longitude')
    else:
        color_variable = df[color_variable]

        scatter = plt.scatter(df['latitude'], df['longitude'], c=color_variable,
                        cmap='viridis', label=name, alpha=0.6)

        # Farbskala (Colorbar) hinzufügen
        plt.colorbar(scatter, label=name)

        # Titel und Achsenbeschriftungen hinzufügen
        plt.title(f'Scatterplot of Latitude vs Longitude with {color_variable.name} colored')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')

    # Raster aktivieren
    plt.grid(True)

    # Legende hinzufügen
    plt.legend()

    # Plot anzeigen
    plt.show()

def compare_scatter(dfs, names):
    
    plt.figure(figsize=(10,6))

    for df, name in zip(dfs, names):
        plt.scatter(df['latitude'], df['longitude'], label=name, alpha=0.6)
        
    plt.title(f'Scatterplot of Latitude vs Longitude')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')

    # Raster aktivieren
    plt.grid(True)

    # Legende hinzufügen
    plt.legend()

    # Plot anzeigen
    plt.show()

def plot_stations_on_map(dfs, names):
    """
    Zeigt die Stationen aus mehreren DataFrames auf einer Karte von Großbritannien an.
    
    :param dfs: Liste der DataFrames, die jeweils 'latitude' und 'longitude' Spalten enthalten
    :param names: Liste der Namen für jeden DataFrame
    """
    fig = px.scatter_geo()
    
    # Farben für die einzelnen DataFrames
    colors = px.colors.qualitative.Plotly  # Plotly Standard-Farbpalette

    for i, (df, name) in enumerate(zip(dfs, names)):
        fig.add_scattergeo(
            lat=df['latitude'],
            lon=df['longitude'],
            mode='markers',  # Marker und Namen anzeigen
            marker=dict(color=colors[i % len(colors)], size=8),
            name=name  # Legendenname für den jeweiligen DataFrame
        )

    # Karteneinstellungen, um Großbritannien zu fokussieren
    fig.update_geos(
        #projection_type="natural earth",  # Kartentyp
        showcountries=True, 
        scope='world',
        fitbounds = 'locations',
        countrycolor="Black",  # Ländergrenzen
        lataxis_showgrid=True, 
        lonaxis_showgrid=True
    )
    
    fig.update_layout(
        title="Messstationen in England",
        legend_title="Dataset",
        height = 500,
        showlegend=True
    )
    
    # Karte anzeigen
    fig.show()

def show_errorbar(plot):
    plot = plot.iloc[:, 2:]

    count = len(plot.columns)
    features = []
    for i in range(0, count, 2):
    
        features.append(plot.iloc[:, i:i+2].columns[0][0])
    

    fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(20, 4))

    # Flatten the axes array (makes it easier to iterate over)
    axes = axes.flatten()

    # Loop through each column and plot a histogram
    for i, column in enumerate(features):
        
        # Add the histogram
        axes[i].errorbar(plot.index, plot[column]['mean'], yerr=plot[column]['std'], fmt='o', label=column, elinewidth=3, capsize=0)
        
        # Add title and axis label
        axes[i].set_title(f'{column} mit STD') 
        axes[i].set_xlabel('Punkt') 
        axes[i].set_ylabel(column) 
        axes[i].grid(True)

    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def energy_vs_feature(plot, energy, color):
    

    fig, axes = plt.subplots(nrows=len(plot.columns)-1, ncols=1, figsize=(20, 10))

    # Flatten the axes array (makes it easier to iterate over)
    axes = axes.flatten()
    axes2 = [None] * len(plot.columns)
    # Loop through each column and plot a histogram
    for i, column in enumerate(plot.columns):
        if column == energy:
            break
        else:

            axes[i].plot(plot.index, plot[column], color, label=column)
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel(column, color=color)
            axes[i].tick_params(axis='y', labelcolor=color)

            axes2[i] = axes[i].twinx()
            axes2[i].plot(plot.index, plot[energy], 'orange', label=energy, alpha=0.7)
            axes2[i].set_ylabel(energy, color='orange')
            axes2[i].tick_params(axis='y', labelcolor='orange')
            

            # x-Achse für beide Plots als Datumsformat setzen
            axes[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            axes[i].grid(True)
            axes[i].set_title(f'{column} vs {energy} with Time Axis')

            # Legenden erstellen
            axes[i].legend(loc='upper left')
            axes2[i].legend(loc='upper right')
        # Rotieren der x-Achse Labels für bessere Lesbarkeit
        # axes[i].xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def energy_vs_feature_hourly(plot, energy, color):
    
    plot["hour"] = plot.index.hour

    plot = plot.groupby("hour").mean()
    fig, axes = plt.subplots(nrows=len(plot.columns)-1, ncols=1, figsize=(20, 10))

    # Flatten the axes array (makes it easier to iterate over)
    axes = axes.flatten()
    axes2 = [None] * len(plot.columns)
    # Loop through each column and plot a histogram
    for i, column in enumerate(plot.columns):
        if column == energy:
            break
        else:

            axes[i].plot(plot.index, plot[column], color, label=column)
            axes[i].set_xlabel('hour')
            axes[i].set_ylabel(column, color=color)
            axes[i].tick_params(axis='y', labelcolor=color)

            axes2[i] = axes[i].twinx()
            axes2[i].plot(plot.index, plot[energy], 'orange', label=energy, alpha=0.7)
            axes2[i].set_ylabel(energy, color='orange')
            axes2[i].tick_params(axis='y', labelcolor='orange')
            

            # x-Achse für beide Plots als Datumsformat setzen
            axes[i].grid(True)
            axes[i].set_title(f'{column} vs {energy} daily')

            # Legenden erstellen
            axes[i].legend(loc='upper left')
            axes2[i].legend(loc='upper right')
        # Rotieren der x-Achse Labels für bessere Lesbarkeit
        # axes[i].xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def truncate_colormap(color, minval=0.0, maxval=1.0, n=100):
    cmap = plt.colormaps.get_cmap(color)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_wind_rose(df, title, ax=None):

    cmap = cm.viridis
    ws = df['WindSpeed:100']["mean"].to_numpy()
    wd = df['WindDirection:100']["mean"].to_numpy()

    std = np.nanstd(ws)
    u = np.nanmean(ws)

    width = 3

    maxVal = round(u+4*std)
    num = math.ceil((maxVal)/width)
    maxVal = width*num
    if maxVal > 24:
        maxVal = 24

    # Form bin ranges

    windRange = np.arange(0, maxVal , width)

    # windRange = np.array([0, 3, 6, 9, 12, 15])

    # Magically rounds the triangles (triangles to pizza slices if you will)
    plt.hist([0, 1])
    plt.close()

    
    ax = WindroseAxes.from_ax()
    
    ax.contourf(wd, ws, bins = windRange, normed=True, linewidth=0.5, cmap=cmap)
    ax.contour(wd, ws, bins = windRange, normed=True, linewidth=0.5, colors="black")

    
    ax.set_legend(title = 'Wind Speed (m/s)', loc='best')
    ax.set_title(title, fontsize = 16)

    # Format radius axis to percentages
    fmt = '%.0f%%' 
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)


