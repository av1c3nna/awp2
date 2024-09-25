import xarray as xr
import pandas as pd
import numpy as np
        

class XarrayPreprocessing:

    def __init__(self, dataset):
        self.dataset = dataset
    
    def remove_duplicates(self, dim):
        """Removes duplicate values along a given dimension."""
        self.dataset = self.dataset.drop_duplicates(dim=dim)
        
        return self
    
    def remove_nans(self, dim, how):
        """Removes missing values along a given dimension."""
        self.dataset = self.dataset.dropna(dim=dim, how=how)

        return self
    
    def fill_nans(self, value):
        """Fills missing values."""
        self.dataset = self.dataset.fillna(value)

        return self
    
    def remove_outliers(self, variables, dim):
        """Removes outliers of one or more variables along a given dimension."""
        for var in variables:

            q_1 = self.dataset[var].quantile(0.25, dim=dim)
            q_3 = self.dataset[var].quantile(0.75, dim=dim)
            iqr = q_3 - q_1

            lower_bound = q_1 - 1.5*iqr
            upper_bound = q_3 + 1.5*iqr

            mask = self.dataset[var].where((self.dataset[var] >= lower_bound) & (self.dataset[var] <= upper_bound), 
                                            self.dataset[var].mean(dim=["ref_time", "val_time"]))
            self.dataset[var] = mask

        return self

    def convert_val_time(self, ref_time, val_time):
        """Converts validation times to actual datetimes and converts reference time to a variable."""
        val_timedeltas = np.array(self.dataset[val_time].values, dtype="timedelta64[h]")
        ref_times = np.array(self.dataset[ref_time].values)
        val_dates = ref_times[:, np.newaxis] + val_timedeltas[np.newaxis, :]
        val_dates = np.array(list(map(lambda x: x.tz_localize("UTC"), pd.to_datetime(val_dates.ravel()))))
        self.dataset = (self.dataset
                        .stack(val_datetime=(ref_time, val_time))
                        .reset_index(ref_time).reset_coords(ref_time)
                        .assign_coords(val_datetime=val_dates))

        return self
    
    def get_dataset(self):
        """Returns the processed dataset."""
        return self.dataset
    


class PandasPreprocessing:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def remove_nans(self, how, subset):
        """Removes missing values."""
        self.dataframe = self.dataframe.dropna(how=how, subset=subset)

        return self

    def fill_nans(self, columns, mean=False, mean_over=None):
        """Fills missing values."""
        for column in columns:
            if mean:
                self.dataframe[column] = self.dataframe[column].fillna(value=self.dataframe[column].mean())
            else:
                self.dataframe[column] = self.dataframe.groupby(mean_over)[column].transform(func=lambda x: x.fillna(x.mean()))
        
        return self
    
    def remove_outliers (self, columns, drop=False, mean_over=None):
        """Removes or replaces outliers for one or more columns."""
        for column in columns:
            q_1 = self.dataframe[column].quantile(0.25)
            q_3 = self.dataframe[column].quantile(0.75)
            iqr = q_3 - q_1

            lower_bound = q_1 - 1.5*iqr
            upper_bound = q_3 + 1.5*iqr
            
            if drop:
                self.dataframe = self.dataframe[(self.dataframe[column] >= lower_bound) & (self.dataframe[column] <= upper_bound)]
            else:
                self.dataframe[column] = self.dataframe[column].where((self.dataframe[column] >= lower_bound) | (self.dataframe[column] <= upper_bound),
                                            other=np.nan)
                self.dataframe[column] = self.dataframe.groupby(mean_over)[column].transform(lambda x: x.fillna(x.mean()))
        
        return self
    
    def get_dataframe(self):
        """Returns the processed dataframe."""
        
        return self.dataframe
    