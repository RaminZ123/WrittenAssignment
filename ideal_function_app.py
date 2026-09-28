"""
Ideal Function Selection and Test Data Mapping Application

"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from bokeh.plotting import figure
from bokeh.plotting import show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import gridplot
import os
import sys

# Create a shared Base 
Base = declarative_base()


class DataLoadError(Exception):
    """Exception raised when data loading fails."""

    pass

class FunctionSelectionError(Exception):
    """Exception raised when function selection fails."""

    pass


class TrainingData(Base):
    """Table for storing training data."""

    __tablename__ = 'training_data'
    
    X = Column(Float, primary_key=True)
    Y1 = Column(Float)
    Y2 = Column(Float)
    Y3 = Column(Float)
    Y4 = Column(Float)


class IdealFunctions(Base):
    """Table for storing ideal functions."""

    __tablename__ = 'ideal_functions'
    
    X = Column(Float, primary_key=True)
    # 50 ideal functions
    for i in range(1, 51):
        locals()['Y{}'.format(i)] = Column(Float)


class TestDataMapping(Base):
    """Table for storing test data mapping results."""

    __tablename__ = 'test_data_mapping'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    X = Column(Float)
    Y = Column(Float)
    No_of_ideal_func = Column(Integer)
    Delta_Y = Column(Float)


class DatabaseManager:
    """Base class for managing database operations."""
    
    def __init__(self, db_path='database.db'):
        """Initialize the database manager."""

        self.db_path = db_path
        self.engine = create_engine('sqlite:///{}'.format(db_path))
        self.Base = Base
        self.Session = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all tables defined in the Base metadata."""

        self.Base.metadata.create_all(self.engine)
        
    def get_session(self):
        """Get a new database session."""

        return self.Session()


class DataProcessor(DatabaseManager):
    """Class for processing data, including loading, selecting ideal functions, and mapping test data."""
    
    def __init__(self, db_path='database.db'):
        """Initialize the data processor."""

        super().__init__(db_path)
        self.TrainingData = TrainingData
        self.IdealFunctions = IdealFunctions
        self.TestDataMapping = TestDataMapping
        self.create_tables()
        
    def load_training_data(self, training_data_file_path):
        """
        Load training data from the CSV file into the database.
        
        Args:
            training_data_file_path (str): Path to the training data CSV file.
            
        Raises:
            DataLoadError: If data loading fails.
        """
        try:
            # Load data from file
            df = pd.read_csv(training_data_file_path)

            df.columns = ['X', 'Y1', 'Y2', 'Y3', 'Y4']
            
            # Load to database
            df.to_sql('training_data', self.engine, if_exists='replace', index=False)
            
        except Exception as e:
            raise DataLoadError("Failed to load training data: {}".format(str(e)))
            
    def load_ideal_functions(self, ideal_functions_file_path):
        """
        Load ideal functions from the CSV file into the database.
        
        Args:
            ideal_functions_file_path (str): Path to the ideal functions CSV file.
            
        Raises:
            DataLoadError: If data loading fails.
        """
        try:
            # Load data from file
            df = pd.read_csv(ideal_functions_file_path)
            
            df.columns = ['X'] + ['Y{}'.format(i) for i in range(1, 51)]
            
            # Load to database
            df.to_sql('ideal_functions', self.engine, if_exists='replace', index=False)
            
        except Exception as e:
            raise DataLoadError("Failed to load ideal functions: {}".format(str(e)))
            
    def select_best_ideal_functions(self):
        """
        Select the best ideal functions for each training dataset.
        
        Returns:
            list: List of indices of the selected ideal functions.
            
        Raises:
            FunctionSelectionError: If function selection fails.
        """
        try:
            
            session = self.get_session()

            # Get training data
            training_data = pd.read_sql_table('training_data', self.engine)
            
            # Get ideal functions
            ideal_functions = pd.read_sql_table('ideal_functions', self.engine)
            
            session.close()
            
            selected_function_indices = []
            
            # For each training dataset initialize minimum sum of squares to infinity
            for training_dataset_index in range(1, 5):
                min_sum_squares = float('inf')
                best_function_index = -1
                
                # For each ideal function calculate sum of squared deviations
                for ideal_function_index in range(1, 51):
                    sum_of_squared_deviations = np.sum( (training_data['Y{}'.format(training_dataset_index)] - ideal_functions['Y{}'.format(ideal_function_index)])**2)
                    
                    # Update if this function has a smaller sum of squares
                    if sum_of_squared_deviations < min_sum_squares:
                        min_sum_squares = sum_of_squared_deviations
                        best_function_index = ideal_function_index
                
                # Add the best function index to the results
                selected_function_indices.append(best_function_index)
                
            return selected_function_indices
            
        except Exception as e:
            raise FunctionSelectionError("Failed to select ideal functions: {}".format(str(e)))
            
    def calculate_maximum_deviations(self, selected_function_indices):
        """
        Calculate maximum deviations between training data and selected ideal functions.
        
        Args:
            selected_function_indices (list): List of indices of the selected ideal functions.
            
        Returns:
            list: List of maximum deviations for each training dataset.
            
        Raises:
            DataLoadError: If calculation fails.
        """
        try:
            # Get training data
            session = self.get_session()
            training_data = pd.read_sql_table('training_data', self.engine)
            
            # Get ideal functions
            ideal_functions = pd.read_sql_table('ideal_functions', self.engine)
            
            # Close session
            session.close()
            
            # Calculate maximum deviation for each training dataset and its ideal function
            maximum_deviations = []
            
            # For each training dataset and its selected ideal function calculate the deviations
            for training_dataset_index, ideal_function_index in enumerate(selected_function_indices):
                deviations = np.abs(training_data['Y{}'.format(training_dataset_index + 1)] - ideal_functions['Y{}'.format(ideal_function_index)])
                # Get maximum deviation
                maximum_deviation = np.max(deviations)
                # Add to results
                maximum_deviations.append(maximum_deviation)
                
            return maximum_deviations
            
        except Exception as e:
            raise DataLoadError("Failed to calculate maximum deviations: {}".format(str(e)))
            
    def map_test_data(self, test_data_file_path, selected_function_indices, maximum_deviations):
        """
        Map test data to the selected ideal functions.
        
        Args:
            test_data_file_path (str): Path to the test data CSV file.
            selected_function_indices (list): List of indices of the selected ideal functions.
            maximum_deviations (list): List of maximum deviations for each training dataset.
            
        Returns:
            DataFrame: DataFrame containing the mapping results.
            
        Raises:
            DataLoadError: If data loading or mapping fails.
        """
        try:
            # Get ideal functions
            session = self.get_session()
            ideal_functions = pd.read_sql_table('ideal_functions', self.engine)
            
            # Load test data
            test_data = pd.read_csv(test_data_file_path)
            
            test_data.columns = ['X', 'Y']
            
            # Map test data
            mapping_results = []
            
            # For each test point
            for _, test_point in test_data.iterrows():
                test_x_value = test_point['X']
                test_y_value = test_point['Y']
                
                # For each selected ideal function find the ideal function value at test_x_value
                for training_dataset_index, ideal_function_index in enumerate(selected_function_indices):
                    
                    ideal_function_row = ideal_functions[ideal_functions['X'] == test_x_value]
                    
                    if not ideal_function_row.empty:
                        ideal_y_value = ideal_function_row.iloc[0, ideal_function_index]
                        deviation = np.abs(test_y_value - ideal_y_value)
                        
                        # Check if deviation is within threshold
                        if deviation <= maximum_deviations[training_dataset_index] * np.sqrt(2):
                            mapping_results.append({
                                'X': test_x_value,
                                'Y': test_y_value,
                                'No_of_ideal_func': ideal_function_index,
                                'Delta_Y': deviation
                            })
                            break
            
            # Save results to database
            results_df = pd.DataFrame(mapping_results)
            results_df.to_sql('test_data_mapping', self.engine, if_exists='replace', index=False)
            
            # Close session
            session.close()
            
            return results_df
            
        except Exception as e:
            raise DataLoadError("Failed to map test data: {}".format(str(e)))


class DataVisualizer(DatabaseManager):
    """Class for visualizing data using Bokeh."""
    
    def __init__(self, db_path='database.db'):
        """Initialize the data visualizer."""

        super().__init__(db_path)
        
    def visualize_data(self, selected_function_indices):
        """
        Visualize training data, test data, and selected ideal functions.
        
        Args:
            selected_function_indices (list): List of indices of the selected ideal functions.
        """
       
        session = self.get_session()

         # Get data from database
        training_data = pd.read_sql_table('training_data', self.engine)
        ideal_functions = pd.read_sql_table('ideal_functions', self.engine)
        test_mapping = pd.read_sql_table('test_data_mapping', self.engine)
       
        session.close()
        
        plot_grid = []
        
        # For each training dataset create a new plot
        for training_dataset_index in range(1, 5):
            plot_title = "Training Dataset {} and Ideal Function {}".format(training_dataset_index, selected_function_indices[training_dataset_index-1])
            current_plot = figure(title=plot_title, x_axis_label='X', y_axis_label='Y')
            
            current_plot.scatter(
                x=training_data['X'], 
                y=training_data['Y{}'.format(training_dataset_index)], size=8, color='blue', alpha=0.7, legend_label='Training Data')
            
            # Plot ideal function
            ideal_function_index = selected_function_indices[training_dataset_index-1]
            current_plot.line(
                ideal_functions['X'], ideal_functions['Y{}'.format(ideal_function_index)], line_width=2, color='red', legend_label='Ideal Function')
            
            # Plot test data mapping 
            test_data_for_function = test_mapping[test_mapping['No_of_ideal_func'] == ideal_function_index]
            if not test_data_for_function.empty:
                test_data_source = ColumnDataSource(data={'x': test_data_for_function['X'], 'y': test_data_for_function['Y'], 'Delta_Y': test_data_for_function['Delta_Y']})
                current_plot.scatter(x='x', y='y', source=test_data_source,size=8, color='green', alpha=0.7, legend_label='Test Data')
            
            # Add hover tool for deviation
            hover = HoverTool()
            hover.tooltips = [("X", "@x"), ("Y", "@y"), ("Deviation", "@Delta_Y")]
            current_plot.add_tools(hover)
            
            plot_grid.append(current_plot)
        
        # Create a grid of plots
        grid = gridplot([plot_grid[:2], plot_grid[2:]])
        
        # Show the plot in a browser
        output_file("ideal_functions.html")
        show(grid)


class IdealFunctionApp:
    """Main application class for the ideal function assignment."""
    
    def __init__(self, db_path='database.db'):
        """Initialize the application."""
        self.db_path = db_path
        self.data_processor = DataProcessor(db_path)
        self.data_visualizer = DataVisualizer(db_path)
        
    def run_application(self, training_data_file, ideal_functions_file, test_data_file):
        """
        Run the application.
        
        Args:
            training_data_file (str): Path to the training data CSV file.
            ideal_functions_file (str): Path to the ideal functions CSV file.
            test_data_file (str): Path to the test data CSV file.
        """
        try:
            # Load data
            print("Loading training data...")
            self.data_processor.load_training_data(training_data_file)
            
            print("Loading ideal functions...")
            self.data_processor.load_ideal_functions(ideal_functions_file)
            
            # Select best ideal functions
            print("Selecting best ideal functions...")
            selected_function_indices = self.data_processor.select_best_ideal_functions()
            print("Selected ideal functions: {}".format(selected_function_indices))
            
            # Calculate maximum deviations
            print("Calculating maximum deviations...")
            maximum_deviations = self.data_processor.calculate_maximum_deviations(selected_function_indices)
            
            # Map test data
            print("Mapping test data...")
            mapping_results = self.data_processor.map_test_data(test_data_file, selected_function_indices, maximum_deviations)
            print("Mapped {} test data points.".format(len(mapping_results)))
            
            # Visualize data
            print("Visualizing data...")
            self.data_visualizer.visualize_data(selected_function_indices)
            
            print("Application completed successfully.")
            
        except Exception as e:
            print("Error: {}".format(str(e)))

# Main execution
if __name__ == "__main__":
    training_data_file = "train.csv"
    ideal_functions_file = "ideal.csv"
    test_data_file = "test.csv"
    
    # Create and run the application
    app = IdealFunctionApp()
    app.run_application(training_data_file, ideal_functions_file, test_data_file)