import tempfile
import unittest

from ideal_function_app import DatabaseManager, DataProcessor, DataVisualizer, IdealFunctionApp

class TestIdealFunctionApp(unittest.TestCase):
    """Unit tests for the Ideal Function App."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""

        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
    
    def tearDown(self):
        """Clean up after each test method."""

        import os
        try:
            os.unlink(self.db_path)
        except:
            pass
    
    def test_database_manager_initialization(self):
        """Test DatabaseManager initialization."""

        db_manager = DatabaseManager(self.db_path)
        self.assertEqual(db_manager.db_path, self.db_path)
    
    def test_data_processor_initialization(self):
        """Test DataProcessor initialization."""

        processor = DataProcessor(self.db_path)
        self.assertEqual(processor.db_path, self.db_path)
    
    def test_data_visualizer_initialization(self):
        """Test DataVisualizer initialization."""

        visualizer = DataVisualizer(self.db_path)
        self.assertEqual(visualizer.db_path, self.db_path)
    
    def test_ideal_function_app_initialization(self):
        """Test IdealFunctionApp initialization."""

        app = IdealFunctionApp()
        self.assertIsNotNone(app.data_processor)
        self.assertIsNotNone(app.data_visualizer)
    
    def test_database_connection(self):
        """Test database connection."""

        db_manager = DatabaseManager(self.db_path)
        session = db_manager.get_session()
        self.assertIsNotNone(session)
        session.close()
    
    def test_data_processor_methods_exist(self):
        """Test that DataProcessor has the required methods."""

        processor = DataProcessor(self.db_path)
        
        self.assertTrue(hasattr(processor, 'load_training_data'))
        self.assertTrue(hasattr(processor, 'load_ideal_functions'))
        self.assertTrue(hasattr(processor, 'select_best_ideal_functions'))
        self.assertTrue(hasattr(processor, 'calculate_maximum_deviations'))
        self.assertTrue(hasattr(processor, 'map_test_data'))
    
    def test_data_visualizer_methods_exist(self):
        """Test that DataVisualizer has the required methods."""

        visualizer = DataVisualizer(self.db_path)

        self.assertTrue(hasattr(visualizer, 'visualize_data'))
    
    def test_ideal_function_app_methods_exist(self):
        """Test that IdealFunctionApp has the required methods."""

        app = IdealFunctionApp()
        self.assertTrue(hasattr(app, 'run_application'))


if __name__ == '__main__':
    unittest.main()