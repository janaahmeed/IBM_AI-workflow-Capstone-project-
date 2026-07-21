from fileinput import filename
import unittest
import json
import pandas as pd
import numpy as np
from app_core_logging import ingest_data, engineer_features, train_production_model_RFR, app ,GLOBAL_MODEL
import app_core_logging

class EnterpriseTimeSeriesTestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Prepares a lightweight mock dataframe to simulate historical data safely."""
        # Create a mock 150-day dataset to accommodate 30-day lags and 90-day forward horizons
        np.random.seed(42)
        cls.mock_csv_path = "mock_test_data.csv"
        
        dates = pd.date_range(start="2026-01-01", periods=365, freq="D")
        df = pd.DataFrame({
            'country': ['United Kingdom'] * 120 + ['United States'] * 60 + ['Canada'] * 60 + ['Germany']*100 + ['France'] * 25,
            'invoice': range(1000, 1365),
            'price': np.random.uniform(5.0,70.0,365),
            'StreamID': ['85048'] * 365,
            'stream_id': ['85048'] * 365,
            'times_viewed': np.random.randint(1, 10, 365),
            'year': dates.year,
            'month': dates.month,
            'day': dates.day
        })
        df.to_csv(cls.mock_csv_path, index=False)
        
        # Initialize the Flask testing client
        
        cls.api_client = app.test_client()

    #  Test 1: Data Ingestion 
    def test_data_ingestion(self):
        df = ingest_data(self.mock_csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('price', df.columns)
        self.assertIn('times_viewed', df.columns)
        # Verify the unified columns cleanup dropped duplicates
        self.assertNotIn('StreamID', df.columns)
        self.assertIn('stream_id', df.columns)

    #  Test 2: Feature Engineering Matrix Construction 
    def test_feature_engineering(self):
        df = ingest_data(self.mock_csv_path)
        X, Y, full_df = engineer_features(df)
        
        # Verify columns exist
        self.assertIn('rolling_mn_1Mon', X.columns)
        self.assertIn('lag_90', X.columns)
        # Verify target array shape spans the requested 90-day vector array layout
        self.assertEqual(Y.shape[1], 90)

    # Test 3: API Endpoint Error Boundaries & Assertions 
    def test_flask_prediction_api(self):
        # 1. Force setup data, model training and registration for integration coverage
        df = ingest_data(self.mock_csv_path)
        X, Y, _ = engineer_features(df)
        app_core_logging.GLOBAL_MODEL= train_production_model_RFR(df, X, Y)
        
         
        
        valid_payload = {
            "rolling_mn_1Mon": 25.5,
            "rolling_std_1Mon": 4.2,
            "lag_30": 22.1,
            "lag_60": 19.8,
            "lag_90": 30.4,
            "month": 7,
            "quarter": 3
        }
        
        # Send POST request to our app instance route
        response = self.api_client.post('/predict', 
                                       data=json.dumps(valid_payload),
                                       content_type='application/json')
        
        res_data = json.loads(response.data)
        filename = f"pred_results_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(filename, 'w') as f:
           json.dump(res_data, f, indent=4)
        
        app_core_logging.logging.info(f"results saved locally to file: {filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(res_data['status'], 'success')
        self.assertEqual(len(res_data['predictions']), 90)

        self.assertIn('executive_summary', res_data)
        self.assertIsInstance(res_data['executive_summary'], str)
        self.assertGreater(len(res_data['executive_summary']), 20)

    @classmethod
    def tearDownClass(cls):
        """Cleans up temporary disk assets generated for testing boundaries."""
        import os
        if os.path.exists(cls.mock_csv_path):
            os.remove(cls.mock_csv_path)


if __name__ == "__main__":
    unittest.main()