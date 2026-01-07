"""
src/data_processing/ingest_data.py
Heavy Fleet Data Pipeline - Aviation Style
"""

import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path


class FleetDataPipeline:
    """
    GE-style data ingestion with SQL analytics
    Processes NASA CMAPSS turbofan degradation data
    """

    def __init__(self, db_path='data/processed/fleet_analytics.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        print(f" Database initialized: {db_path}")

    def load_cmapss_data(self, data_path='data/raw/train_FD001.txt'):
        """
        Load NASA CMAPSS dataset and calculate RUL
        """

        print("\n Loading CMAPSS Fleet Data...")

        columns = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
        columns += [f'sensor_{i}' for i in range(1, 22)]

        # FIX: raw string to avoid SyntaxWarning
        df = pd.read_csv(data_path, sep=r'\s+', header=None, names=columns)

        # Calculate RUL
        max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
        max_cycles.columns = ['engine_id', 'max_cycle']

        df = df.merge(max_cycles, on='engine_id')
        df['RUL'] = df['max_cycle'] - df['cycle']

        print(f" Loaded {len(df)} records from {df['engine_id'].nunique()} engines")
        print(f" Cycle range: {df['cycle'].min()}-{df['cycle'].max()}")

        return df

    def create_sql_tables(self, df):
        """
        Create SQL tables with GE-style schema
        """

        print("\n  Creating SQL Tables...")

        df.to_sql('engine_telemetry', self.conn, if_exists='replace', index=False)
        print(" Created: engine_telemetry")

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_engine_cycle 
            ON engine_telemetry(engine_id, cycle)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rul 
            ON engine_telemetry(RUL)
        """)

        self.conn.commit()
        print(" Indexes created")

    def run_fleet_analytics(self):
        """
        Execute GE-style SQL analytics
        """

        print("\n Running Fleet Risk Analytics (SQL)...")

        fleet_risk_query = """
        SELECT 
            engine_id,
            AVG(sensor_2) AS avg_degradation_temp,
            AVG(sensor_4) AS avg_pressure,
            MAX(cycle) AS total_cycles,
            AVG(RUL) AS mean_rul,
            MIN(RUL) AS min_rul,
            COUNT(*) AS op_cycles,
            CASE 
                WHEN AVG(RUL) < 50 THEN 'High Risk - Schedule MRO'
                WHEN AVG(RUL) < 100 THEN 'Medium - Monitor'
                ELSE 'Low - Extend Interval'
            END AS lean_priority,
            CASE 
                WHEN AVG(RUL) < 50 THEN 1
                WHEN AVG(RUL) < 100 THEN 2
                ELSE 3
            END AS priority_rank
        FROM engine_telemetry 
        GROUP BY engine_id 
        HAVING total_cycles > 50
        ORDER BY mean_rul ASC
        """

        fleet_risk = pd.read_sql_query(fleet_risk_query, self.conn)
        fleet_risk.to_csv('data/processed/fleet_risk_summary.csv', index=False)

        print(f" Fleet Risk Analysis: {len(fleet_risk)} engines profiled")
        print(f"  - High Risk: {len(fleet_risk[fleet_risk['lean_priority'].str.contains('High')])}")
        print(f"  - Medium Risk: {len(fleet_risk[fleet_risk['lean_priority'].str.contains('Medium')])}")
        print(f"  - Low Risk: {len(fleet_risk[fleet_risk['lean_priority'].str.contains('Low')])}")

        degradation_query = """
        SELECT 
            engine_id,
            cycle,
            sensor_2 AS temp_degradation,
            sensor_4 AS pressure_degradation,
            sensor_7 AS pressure_ratio,
            sensor_11 AS fan_speed,
            RUL,
            CASE 
                WHEN RUL < 30 THEN 'Critical'
                WHEN RUL < 70 THEN 'Warning'
                ELSE 'Normal'
            END AS health_status
        FROM engine_telemetry
        WHERE engine_id IN (
            SELECT engine_id
            FROM engine_telemetry
            GROUP BY engine_id
            HAVING AVG(RUL) < 80
        )
        ORDER BY engine_id, cycle
        """

        degradation_trends = pd.read_sql_query(degradation_query, self.conn)
        degradation_trends.to_csv('data/processed/degradation_trends.csv', index=False)

        print(f" Trends: {len(degradation_trends)} records")

        # -------------------- ROI (SAFE) --------------------

        roi_query = """
        SELECT 
            COUNT(DISTINCT engine_id) AS high_risk_engines,
            AVG(mean_rul) AS avg_rul,
            MIN(mean_rul) AS worst_rul
        FROM (
            SELECT engine_id, AVG(RUL) AS mean_rul
            FROM engine_telemetry
            GROUP BY engine_id
            HAVING mean_rul < 50
        )
        """

        roi_data = pd.read_sql_query(roi_query, self.conn)

        num_flagged = int(roi_data['high_risk_engines'].iloc[0] or 0)

        avg_rul = roi_data['avg_rul'].iloc[0]
        worst_rul = roi_data['worst_rul'].iloc[0]

        avg_rul = float(avg_rul) if avg_rul is not None else np.nan
        worst_rul = float(worst_rul) if worst_rul is not None else np.nan

        baseline_cost = 100000
        pred_savings_pct = 0.25
        annual_savings = num_flagged * baseline_cost * pred_savings_pct

        print(f"\n FLIGHT DECK ROI Projection:")
        print(f"  - High-risk engines flagged: {num_flagged}")
        print(f"  - Projected annual savings: ${annual_savings:,.0f}")
        print(f"  - Cost avoidance: 25% reduction in AOG events")

        roi_metrics = {
            'high_risk_engines': num_flagged,
            'avg_rul': avg_rul,
            'worst_rul': worst_rul,
            'projected_savings': annual_savings,
            'baseline_cost_per_engine': baseline_cost,
            'savings_percentage': pred_savings_pct
        }

        pd.DataFrame([roi_metrics]).to_csv('data/processed/roi_metrics.csv', index=False)

        return fleet_risk, degradation_trends, roi_metrics

    def generate_ml_features(self):
        """
        Feature engineering for ML models
        """

        print("\n Generating ML Features...")

        ml_data = pd.read_sql_query("""
        SELECT 
            engine_id, cycle,
            op_setting_1, op_setting_2, op_setting_3,
            sensor_2, sensor_3, sensor_4, sensor_7,
            sensor_8, sensor_9, sensor_11, sensor_12,
            sensor_13, sensor_14, sensor_15, sensor_17,
            sensor_20, sensor_21,
            RUL
        FROM engine_telemetry
        ORDER BY engine_id, cycle
        """, self.conn)

        for col in ['sensor_2', 'sensor_4', 'sensor_7', 'sensor_11']:
            ml_data[f'{col}_rolling_mean'] = (
                ml_data.groupby('engine_id')[col]
                .transform(lambda x: x.rolling(5, min_periods=1).mean())
            )

        ml_data.to_csv('data/processed/ml_features.csv', index=False)
        print(f" ML Features: {ml_data.shape}")

        return ml_data

    def close(self):
        self.conn.close()
        print("\n Database connection closed")


def main():
    print("=" * 70)
    print(" AVIATION FLEET DATA PIPELINE")
    print("Heavy Analytics for Predictive Maintenance")
    print("=" * 70)

    pipeline = FleetDataPipeline()

    try:
        df = pipeline.load_cmapss_data()
        pipeline.create_sql_tables(df)
        pipeline.run_fleet_analytics()
        pipeline.generate_ml_features()

        print("\n DATA PIPELINE COMPLETE — READY FOR ML MODELS")

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
