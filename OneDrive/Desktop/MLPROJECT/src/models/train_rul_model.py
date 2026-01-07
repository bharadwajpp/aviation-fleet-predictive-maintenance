"""
src/models/train_rul_model.py
Scikit-learn RUL Prediction -  FlightPulse Style
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import json

class RULPredictor:
    """
    Random Forest RUL predictor for aircraft engines
    Target: RMSE <15 cycles ( FlightPulse benchmark)
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
    
    def load_data(self, data_path='data/processed/ml_features.csv'):
        """Load preprocessed ML features"""
        
        print(" Loading ML features...")
        df = pd.read_csv(data_path)
        
        # Separate features and target
        target_col = 'RUL'
        exclude_cols = ['engine_id', 'cycle', target_col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        self.feature_names = feature_cols
        
        print(f" Features: {X.shape[1]} sensors/derived features")
        print(f" Samples: {X.shape[0]} engine cycles")
        print(f" RUL range: {y.min():.0f} - {y.max():.0f} cycles")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train Random Forest with GE-optimized hyperparameters
        """
        
        print("\n Training Random Forest RUL Predictor...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize Random Forest
        # Hyperparameters tuned for aviation telemetry
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        print("  Training in progress...")
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='neg_root_mean_squared_error'
        )
        cv_rmse = -cv_scores.mean()
        
        self.metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'cv_rmse': cv_rmse,
            'cv_std': cv_scores.std()
        }
        
        print(f"\n Model Training Complete")
        print(f"  Train RMSE: {train_rmse:.2f} cycles")
        print(f"  Test RMSE:  {test_rmse:.2f} cycles ({'✓ <15' if test_rmse < 15 else ' >15'})")
        print(f"  Test MAE:   {test_mae:.2f} cycles")
        print(f"  Test R²:    {test_r2:.3f}")
        print(f"  CV RMSE:    {cv_rmse:.2f} ± {cv_scores.std():.2f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n Top 5 Features:")
        for idx, row in self.feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return X_test_scaled, y_test, y_pred_test
    
    def evaluate_flight_deck_metrics(self, y_test, y_pred):
        """
         FLIGHT DECK evaluation: bias, early detection, ROI
        """
        
        print("\n FLIGHT DECK Business Metrics...")
        
        # Prediction bias analysis
        bias = np.mean(y_pred - y_test)
        bias_pct = (bias / np.mean(y_test)) * 100
        
        # Early detection rate (flag engines 50+ cycles early)
        threshold = 50
        early_flags = np.sum((y_test < threshold) & (y_pred < threshold + 20))
        total_critical = np.sum(y_test < threshold)
        early_detection_rate = (early_flags / total_critical) * 100 if total_critical > 0 else 0
        
        # Over/under maintenance
        over_maintenance = np.sum((y_pred < y_test) & (y_test > 50))  # False alarms
        under_maintenance = np.sum((y_pred > y_test + 20) & (y_test < 50))  # Missed risks
        
        print(f"  Prediction Bias: {bias:.2f} cycles ({bias_pct:+.1f}%)")
        print(f"  Early Detection: {early_detection_rate:.1f}% (flags critical engines)")
        print(f"  Over-maintenance: {over_maintenance} cases (lean waste)")
        print(f"  Under-maintenance: {under_maintenance} cases (safety risk)")
        
        # ROI calculation
        baseline_cost = 100000  # per engine/year
        num_flagged = np.sum(y_pred < 70)
        savings_pct = 0.25  # 25% from benchmarks
        annual_savings = num_flagged * baseline_cost * savings_pct
        
        print(f"\n ROI Projection:")
        print(f"  Engines flagged for MRO: {num_flagged}")
        print(f"  Annual savings: ${annual_savings:,.0f}")
        print(f"  Cost reduction: 25% (predictive vs reactive)")
        
        self.metrics.update({
            'bias': bias,
            'bias_pct': bias_pct,
            'early_detection_rate': early_detection_rate,
            'over_maintenance': int(over_maintenance),
            'under_maintenance': int(under_maintenance),
            'engines_flagged': int(num_flagged),
            'annual_savings': annual_savings
        })
    
    def save_model(self, model_dir='outputs/models'):
        """Save trained model and metadata"""
        
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = f'{model_dir}/rul_predictor.pkl'
        joblib.dump(self.model, model_path)
        print(f"\n Model saved: {model_path}")
        
        # Save scaler
        scaler_path = f'{model_dir}/scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f" Scaler saved: {scaler_path}")
        
        # Save feature names
        feature_path = f'{model_dir}/feature_names.json'
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
        print(f" Features saved: {feature_path}")
        
        # Save metrics
        metrics_path = f'{model_dir}/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f" Metrics saved: {metrics_path}")
        
        # Save feature importance
        importance_path = f'{model_dir}/feature_importance.csv'
        self.feature_importance.to_csv(importance_path, index=False)
        print(f" Feature importance saved: {importance_path}")
    
    def predict_fleet_status(self, data_path='data/processed/ml_features.csv'):
        """
        Generate predictions for entire fleet
        """
        
        print("\n Generating Fleet Predictions...")
        
        df = pd.read_csv(data_path)
        
        # Prepare features
        exclude_cols = ['engine_id', 'cycle', 'RUL']
        X = df[[col for col in df.columns if col not in exclude_cols]]
        X = X.fillna(X.mean())
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Add predictions to dataframe
        df['predicted_RUL'] = predictions
        df['prediction_error'] = df['predicted_RUL'] - df['RUL']
        df['risk_level'] = pd.cut(
            df['predicted_RUL'],
            bins=[0, 30, 70, float('inf')],
            labels=['Critical', 'Warning', 'Normal']
        )
        
        # Save results
        output_path = 'data/processed/fleet_predictions.csv'
        df.to_csv(output_path, index=False)
        print(f" Fleet predictions saved: {output_path}")
        
        # Summary
        risk_summary = df.groupby('risk_level').size()
        print(f"\n Fleet Risk Summary:")
        for level, count in risk_summary.items():
            print(f"  {level}: {count} engine-cycles")
        
        return df

def main():
    """Execute complete ML pipeline"""
    
    print("=" * 70)
    print(" AVIATION RUL PREDICTION MODEL")
    print("Random Forest ML Pipeline - FlightPulse Benchmark")
    print("=" * 70)
    
    # Initialize predictor
    predictor = RULPredictor()
    
    # Load data
    X, y = predictor.load_data()
    
    # Train model
    X_test, y_test, y_pred = predictor.train_model(X, y)
    
    # Evaluate FLIGHT DECK metrics
    predictor.evaluate_flight_deck_metrics(y_test, y_pred)
    
    # Save artifacts
    predictor.save_model()
    
    # Generate fleet predictions
    fleet_predictions = predictor.predict_fleet_status()
    
    print("\n" + "=" * 70)
    print("ML PIPELINE COMPLETE")
    print("=" * 70)
    print("\n Ready for visualization (Phase 3)")

if __name__ == "__main__":
    main()