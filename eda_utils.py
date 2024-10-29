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
    """creates pandas dataframes"""

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
    """concat dfs"""

    df_base = df_list[0]
    for i in range (1, len(df_list)):
        df_base = pd.concat([df_base, df_list[i]], axis=0)
    return df_base

def groupby_time(df):
    """time point of view"""

    df = df.groupby(["reference_time", "valid_time"], as_index=False).mean()

    return df.drop(["latitude", "longitude", "point"], axis=1, errors="ignore")

def groupby_coordinates(df):
    """geographic point of view"""

    df = df.groupby(["latitude", "longitude", "valid_time"], as_index=False).agg(['mean', 'std'])
    df = df.drop(["reference_time", "point"], axis=1, errors="ignore")
    

    return df

def combine_files(file_paths, agg=None):
    """combines given filepaths to one dataframe"""
    dataframes = []
    
    for file_path in file_paths:
        xr_dataset = xr.open_dataset(file_path)  
        df = convert_xr_to_df(xr_dataset)       
        dataframes.append(df)                   
    
    df_full = concat_dfs(dataframes)
    
    if not agg:
        df_full_agg = df_full
    
    # choose aggregation method to reduce data 
    if agg == "time":
        df_full_agg = groupby_time(df_full)

    if agg == "coordinates":
        df_full_agg = groupby_coordinates(df_full)
    
    
    return df_full_agg

def show_histograms(plot):
    """creates histogram for every feature in df"""
    fig, axes = plt.subplots(nrows=1, ncols=len(plot.columns), figsize=(20, 6))

    axes = axes.flatten()

    # Loop through each column and plot a histogram
    for i, column in enumerate(plot.columns):
        
        # Add the histogram
        plot[column].hist(ax=axes[i], # Define on which ax we're working on
                        edgecolor='white', 
                        color='#69b3a2', 
                    )
        
        # Add title and axis label
        axes[i].set_title(f'{column} distribution') 
        axes[i].set_xlabel(column) 
        axes[i].set_ylabel('Frequency') 

    
    plt.tight_layout()
    plt.show()


def combine_histograms(dfs, names, figsize=(15, 4), title="Vergleich der Histogramme"):
    """compares histograms of given dataframes"""
    columns = dfs[0].columns[2:] #time columns skipped

    # create subplot for every feature
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=figsize)

    if len(columns) == 1:
        axes = [axes]  

    # iterate over every feature
    for i, column in enumerate(columns):
        for df, name, color in zip(dfs, names, ["blue", "orange"]):
            df[column].hist(ax=axes[i],alpha=0.6, edgecolor='white', label=name, color=color, density=True)

        axes[i].set_title(f'{column} distribution')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.show()

def show_lineplot(plot, freq):
    """lineplot weather data"""

    plot = plot.iloc[:, 1:]
    plot = plot.resample(freq, on='valid_time').mean().reset_index()

    fig, axes = plt.subplots(nrows=len(plot.columns)-1, ncols=1, figsize=(20, 20))

    axes = axes.flatten()

    # Loop through each column and plot a histogram
    for i, column in enumerate(plot.iloc[:, 1:].columns):
        
        # Add the lineplot
        axes[i].plot(plot["valid_time"], plot[column], marker='o', color='#69b3a2')
        
        # Add title and axis label
        axes[i].set_title(f'{column} over time') 
        axes[i].set_xlabel('time') 
        axes[i].set_ylabel(column) 
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

def show_lineplot_energy(plot, freq):
    """lineplot energy_dataset"""
    # plot = plot.iloc[:, 1:]
    plot = plot.resample(freq, on='dtm').mean().reset_index()

    fig, axes = plt.subplots(nrows=len(plot.columns)-1, ncols=1, figsize=(20, 20))

    axes = axes.flatten()

    # Loop through each column and plot a lineplot
    for i, column in enumerate(plot.iloc[:, 1:].columns):
        
        # Add the lineplot
        axes[i].plot(plot["dtm"], plot[column], marker='o', color='#69b3a2')
        
        # Add title and axis label
        axes[i].set_title(f'{column} over time') 
        axes[i].set_xlabel('time') 
        axes[i].set_ylabel(column) 
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

def compare_lineplots(plots, names, freq):
    """compare lineplots weather data"""

    fig, axes = plt.subplots(nrows=len(plots[0].columns)-2, ncols=1, figsize=(20, 20))
    axes = axes.flatten()

    # Loop through each column and plot a histogram
    for i, column in enumerate(plots[0].iloc[:, 2:].columns):

        for plot, name, color in zip(plots, names, ["blue", "orange"]):
            plot = plot.iloc[:, 1:]
            plot = plot.resample(freq, on='valid_time').mean().reset_index() # resamples on given frequency
            axes[i].plot(plot["valid_time"], plot[column], marker='o', label = name, color=color, alpha=0.5)
        
        # Add title and axis label
        axes[i].set_title(f'{column} over time') 
        axes[i].set_xlabel('time') 
        axes[i].set_ylabel(column) 
        axes[i].legend()
        axes[i].grid(True)


    plt.tight_layout()
    plt.show()

def compare_lineplots_energy(plots, names, freq):
    """lineplot for energy dataset"""

    fig, axes = plt.subplots(nrows=len(plots[0].columns)-1, ncols=1, figsize=(20, 20))
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

    plt.tight_layout()
    plt.show()

def show_scatter(df,name, color_variable=None):
    """shows coordinates as scatter"""

    if not color_variable:
        scatter = plt.scatter(df['latitude'], df['longitude'],
                        cmap='viridis', label=name, alpha=0.6)
        plt.title(f'Scatterplot of Latitude vs Longitude')
    else:
        color_variable = df[color_variable]

        scatter = plt.scatter(df['latitude'], df['longitude'], c=color_variable,
                        cmap='viridis', label=name, alpha=0.6)

        plt.colorbar(scatter, label=name)

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
    """plots coordinates of multiple dfs"""
    plt.figure(figsize=(10,6))

    for df, name in zip(dfs, names):
        plt.scatter(df['latitude'], df['longitude'], label=name, alpha=0.6)
        
    plt.title(f'Scatterplot of Latitude vs Longitude')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')

    plt.grid(True)
    plt.legend()
    plt.show()

def plot_stations_on_map(dfs, names):
    """
    Zeigt die Stationen aus mehreren DataFrames auf einer Karte von Großbritannien an.
    
    :param dfs: Liste der DataFrames, die jeweils 'latitude' und 'longitude' Spalten enthalten
    :param names: Liste der Namen für jeden DataFrame
    """
    fig = px.scatter_geo()
    
    colors = px.colors.qualitative.Plotly  

    for i, (df, name) in enumerate(zip(dfs, names)):
        fig.add_scattergeo(
            lat=df['latitude'],
            lon=df['longitude'],
            mode='markers',  
            marker=dict(color=colors[i % len(colors)], size=8),
            name=name  # Legendenname für den jeweiligen DataFrame
        )

    # Karteneinstellungen, um Großbritannien zu fokussieren
    fig.update_geos(
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

    axes = axes.flatten()
    for i, column in enumerate(features):
        
        # Add the errorbar plot
        axes[i].errorbar(plot.index, plot[column]['mean'], yerr=plot[column]['std'], fmt='o', label=column, elinewidth=3, capsize=0)
        
        # Add title and axis label
        axes[i].set_title(f'{column} mit STD') 
        axes[i].set_xlabel('Punkt') 
        axes[i].set_ylabel(column) 
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


def energy_vs_feature(plot, energy, color, figsize=(20, 10)):
    """compares energy production to weather feature over time"""

    fig, axes = plt.subplots(nrows=len(plot.columns)-1, ncols=1, figsize=figsize)

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
            

            axes[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            axes[i].grid(True)
            axes[i].set_title(f'{column} vs {energy} mit Zeitachse', fontweight="bold")

            
            axes[i].legend(loc='upper left')
            axes2[i].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def energy_vs_feature_hourly(plot, energy, color):
    """compares energy production to weather feature over hourly"""
    plot["hour"] = plot.index.hour

    plot = plot.groupby("hour").mean()
    fig, axes = plt.subplots(nrows=len(plot.columns)-1, ncols=1, figsize=(20, 10))

    
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
            

            
            axes[i].grid(True)
            axes[i].set_title(f'{column} vs {energy} daily')

            
            axes[i].legend(loc='upper left')
            axes2[i].legend(loc='upper right')
    
    plt.tight_layout()
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

    
    ax = WindroseAxes.from_ax(figsize=(7,7))
    
    ax.contourf(wd, ws, bins = windRange, normed=True, linewidth=0.5, cmap=cmap)
    ax.contour(wd, ws, bins = windRange, normed=True, linewidth=0.5, colors="black")

    
    ax.set_legend(title = 'Wind Speed (m/s)', loc='best')
    ax.set_title(title, fontweight="bold")

    # Format radius axis to percentages
    fmt = '%.0f%%' 
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)


