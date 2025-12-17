"""
Macroparameter Analysis Script
Analyzes the relationships between MakeHuman2 macroparameters and body measurements.

This script performs:
1. Exploratory Data Analysis (EDA)
2. Correlation Analysis
3. Regression Modeling (Linear, Polynomial, Random Forest)
4. Feature Importance Analysis
5. Interaction Effects Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
MACROPARAMETERS = ['age', 'muscle', 'weight', 'height', 'proportions']
MEASUREMENTS = [
    'height_cm', 'shoulder_width_cm', 'hip_width_cm', 'head_width_cm',
    'neck_length_cm', 'upper_arm_length_cm', 'forearm_length_cm', 'hand_length_cm'
]

class MacroparameterAnalyzer:
    """Analyzes relationships between macroparameters and body measurements."""

    def __init__(self, csv_path):
        """
        Initialize analyzer with data from CSV file.

        Args:
            csv_path: Path to the lookup table CSV file
        """
        self.df = pd.read_csv(csv_path)
        self.X = self.df[MACROPARAMETERS]
        self.y = self.df[MEASUREMENTS]
        self.models = {}
        self.results = {}

        print(f"Loaded data: {len(self.df)} samples")
        print(f"Macroparameters: {MACROPARAMETERS}")
        print(f"Measurements: {MEASUREMENTS}")
        print("-" * 80)

    def exploratory_analysis(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)

        # Basic statistics
        print("\n1. MACROPARAMETER STATISTICS:")
        print(self.X.describe())

        print("\n2. MEASUREMENT STATISTICS:")
        print(self.y.describe())

        # Check for missing values
        print("\n3. MISSING VALUES:")
        print(f"Macroparameters: {self.X.isnull().sum().sum()}")
        print(f"Measurements: {self.y.isnull().sum().sum()}")

        # Distribution plots
        self._plot_distributions()

    def _plot_distributions(self):
        """Plot distributions of macroparameters and measurements."""
        # Macroparameters distribution
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Distribution of Macroparameters', fontsize=16)
        axes = axes.flatten()

        for i, param in enumerate(MACROPARAMETERS):
            axes[i].hist(self.X[param], bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(param)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)

        axes[-1].axis('off')
        plt.tight_layout()
        plt.savefig('analysis_output/01_macroparameter_distributions.png', dpi=300, bbox_inches='tight')
        print("\n Saved: analysis_output/01_macroparameter_distributions.png")
        plt.close()

        # Measurements distribution
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Distribution of Body Measurements', fontsize=16)
        axes = axes.flatten()

        for i, measure in enumerate(MEASUREMENTS):
            axes[i].hist(self.y[measure], bins=30, edgecolor='black', alpha=0.7, color='coral')
            axes[i].set_title(measure.replace('_', ' ').title())
            axes[i].set_xlabel('cm')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_output/02_measurement_distributions.png', dpi=300, bbox_inches='tight')
        print(" Saved: analysis_output/02_measurement_distributions.png")
        plt.close()

    def correlation_analysis(self):
        """Analyze correlations between macroparameters and measurements."""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)

        # Correlation matrix: Macroparameters vs Measurements
        correlation_matrix = pd.DataFrame(
            index=MACROPARAMETERS,
            columns=MEASUREMENTS
        )

        for param in MACROPARAMETERS:
            for measure in MEASUREMENTS:
                corr, _ = stats.pearsonr(self.X[param], self.y[measure])
                correlation_matrix.loc[param, measure] = corr

        correlation_matrix = correlation_matrix.astype(float)

        print("\nCorrelation Matrix (Pearson's r):")
        print(correlation_matrix.round(3))

        # Heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, cbar_kws={'label': "Pearson's r"})
        plt.title('Correlation: Macroparameters vs Measurements', fontsize=14, pad=20)
        plt.xlabel('Body Measurements', fontsize=12)
        plt.ylabel('Macroparameters', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('analysis_output/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("\nSaved: analysis_output/03_correlation_heatmap.png")
        plt.close()

        # Find strongest correlations
        print("\nStrongest Correlations (|r| > 0.7):")
        for param in MACROPARAMETERS:
            for measure in MEASUREMENTS:
                corr_val = correlation_matrix.loc[param, measure]
                if abs(corr_val) > 0.7:
                    print(f"  {param:12s} <-> {measure:20s}: r = {corr_val:+.3f}")

        return correlation_matrix

    def linear_regression_analysis(self):
        """Perform linear regression for each measurement."""
        print("\n" + "="*80)
        print("LINEAR REGRESSION ANALYSIS")
        print("="*80)

        results = []

        for measure in MEASUREMENTS:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y[measure], test_size=0.2, random_state=42
            )

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Store results
            self.models[f'linear_{measure}'] = model
            results.append({
                'Measurement': measure,
                'R²': r2,
                'MAE': mae,
                'RMSE': rmse
            })

            # Print coefficients
            print(f"\n{measure}:")
            print(f"  R² = {r2:.4f}, MAE = {mae:.4f} cm, RMSE = {rmse:.4f} cm")
            print("  Coefficients:")
            for param, coef in zip(MACROPARAMETERS, model.coef_):
                print(f"    {param:12s}: {coef:+.4f}")
            print(f"    {'Intercept':12s}: {model.intercept_:+.4f}")

        # Summary table
        results_df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("LINEAR REGRESSION SUMMARY:")
        print(results_df.to_string(index=False))

        self._plot_regression_performance(results_df, 'Linear Regression')

        return results_df

    def polynomial_regression_analysis(self, degree=2):
        """Perform polynomial regression for each measurement."""
        print("\n" + "="*80)
        print(f"POLYNOMIAL REGRESSION ANALYSIS (Degree={degree})")
        print("="*80)

        results = []

        for measure in MEASUREMENTS:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y[measure], test_size=0.2, random_state=42
            )

            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            # Train model with regularization to avoid overfitting
            model = Ridge(alpha=1.0)
            model.fit(X_train_poly, y_train)

            # Predictions
            y_pred = model.predict(X_test_poly)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Store results
            self.models[f'poly_{measure}'] = (poly, model)
            results.append({
                'Measurement': measure,
                'R²': r2,
                'MAE': mae,
                'RMSE': rmse
            })

            print(f"{measure}: R² = {r2:.4f}, MAE = {mae:.4f} cm, RMSE = {rmse:.4f} cm")

        # Summary table
        results_df = pd.DataFrame(results)
        print("\n" + "="*80)
        print(f"POLYNOMIAL REGRESSION (degree={degree}) SUMMARY:")
        print(results_df.to_string(index=False))

        self._plot_regression_performance(results_df, f'Polynomial Regression (degree={degree})')

        return results_df

    def random_forest_analysis(self):
        """Perform Random Forest regression and feature importance analysis."""
        print("\n" + "="*80)
        print("RANDOM FOREST ANALYSIS")
        print("="*80)

        results = []
        feature_importances = pd.DataFrame(index=MACROPARAMETERS)

        for measure in MEASUREMENTS:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y[measure], test_size=0.2, random_state=42
            )

            # Train model
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Store results
            self.models[f'rf_{measure}'] = model
            results.append({
                'Measurement': measure,
                'R²': r2,
                'MAE': mae,
                'RMSE': rmse
            })

            # Feature importance
            feature_importances[measure] = model.feature_importances_

            print(f"{measure}: R² = {r2:.4f}, MAE = {mae:.4f} cm, RMSE = {rmse:.4f} cm")

        # Summary table
        results_df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("RANDOM FOREST SUMMARY:")
        print(results_df.to_string(index=False))

        self._plot_regression_performance(results_df, 'Random Forest')
        self._plot_feature_importance(feature_importances)

        return results_df, feature_importances

    def _plot_regression_performance(self, results_df, model_name):
        """Plot regression performance metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{model_name} Performance', fontsize=16)

        measurements_short = [m.replace('_cm', '').replace('_', ' ') for m in results_df['Measurement']]

        # R²
        axes[0].bar(range(len(results_df)), results_df['R²'], color='steelblue', edgecolor='black')
        axes[0].set_title('R² Score (Goodness of Fit)')
        axes[0].set_ylabel('R²')
        axes[0].set_xticks(range(len(results_df)))
        axes[0].set_xticklabels(measurements_short, rotation=45, ha='right')
        axes[0].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='0.9')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].legend()

        # MAE
        axes[1].bar(range(len(results_df)), results_df['MAE'], color='coral', edgecolor='black')
        axes[1].set_title('Mean Absolute Error')
        axes[1].set_ylabel('MAE (cm)')
        axes[1].set_xticks(range(len(results_df)))
        axes[1].set_xticklabels(measurements_short, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')

        # RMSE
        axes[2].bar(range(len(results_df)), results_df['RMSE'], color='lightgreen', edgecolor='black')
        axes[2].set_title('Root Mean Squared Error')
        axes[2].set_ylabel('RMSE (cm)')
        axes[2].set_xticks(range(len(results_df)))
        axes[2].set_xticklabels(measurements_short, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        plt.savefig(f'analysis_output/04_{filename}_performance.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: analysis_output/04_{filename}_performance.png")
        plt.close()

    def _plot_feature_importance(self, feature_importances):
        """Plot feature importance from Random Forest."""
        plt.figure(figsize=(14, 6))

        # Heatmap
        sns.heatmap(feature_importances, annot=True, fmt='.3f', cmap='YlOrRd',
                    cbar_kws={'label': 'Importance'})
        plt.title('Feature Importance (Random Forest)', fontsize=14, pad=20)
        plt.xlabel('Body Measurements', fontsize=12)
        plt.ylabel('Macroparameters', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('analysis_output/05_feature_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: analysis_output/05_feature_importance.png")
        plt.close()

        print("\nFEATURE IMPORTANCE ANALYSIS:")
        print(feature_importances.round(3))

        # Top contributors for each measurement
        print("\nTop 3 Macroparameters for Each Measurement:")
        for measure in MEASUREMENTS:
            top_3 = feature_importances[measure].nlargest(3)
            print(f"\n{measure}:")
            for param, importance in top_3.items():
                print(f"  {param:12s}: {importance:.3f}")

    def interaction_effects_analysis(self):
        """Analyze interaction effects between macroparameters."""
        print("\n" + "="*80)
        print("INTERACTION EFFECTS ANALYSIS")
        print("="*80)

        # Create interaction terms
        from itertools import combinations

        interactions = list(combinations(MACROPARAMETERS, 2))
        print(f"\nAnalyzing {len(interactions)} pairwise interactions...")

        # Test a few key interactions with scatter plots
        key_interactions = [
            ('height', 'proportions'),
            ('weight', 'muscle'),
            ('age', 'height'),
        ]

        for measure in ['height_cm', 'shoulder_width_cm', 'hip_width_cm']:
            self._plot_interaction_effects(key_interactions, measure)

    def _plot_interaction_effects(self, interactions, measurement):
        """Plot interaction effects for a specific measurement."""
        fig, axes = plt.subplots(1, len(interactions), figsize=(15, 4))
        fig.suptitle(f'Interaction Effects on {measurement.replace("_", " ").title()}', fontsize=14)

        for idx, (param1, param2) in enumerate(interactions):
            # Create scatter plot colored by measurement value
            scatter = axes[idx].scatter(
                self.X[param1],
                self.X[param2],
                c=self.y[measurement],
                cmap='viridis',
                alpha=0.6,
                s=20
            )
            axes[idx].set_xlabel(param1)
            axes[idx].set_ylabel(param2)
            axes[idx].set_title(f'{param1} x {param2}')
            plt.colorbar(scatter, ax=axes[idx], label='cm')

        plt.tight_layout()
        filename = measurement.replace('_cm', '')
        plt.savefig(f'analysis_output/06_interactions_{filename}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: analysis_output/06_interactions_{filename}.png")
        plt.close()

    def compare_models(self):
        """Compare all models side by side."""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        if not hasattr(self, 'linear_results'):
            print("Run all analyses first!")
            return

        comparison = pd.DataFrame({
            'Measurement': self.linear_results['Measurement'],
            'Linear R²': self.linear_results['R²'],
            'Polynomial R²': self.poly_results['R²'],
            'Random Forest R²': self.rf_results['R²']
        })

        print("\nR² Comparison:")
        print(comparison.to_string(index=False))

        # Plot comparison
        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(comparison))
        width = 0.25

        ax.bar(x - width, comparison['Linear R²'], width, label='Linear', color='steelblue')
        ax.bar(x, comparison['Polynomial R²'], width, label='Polynomial (deg=2)', color='coral')
        ax.bar(x + width, comparison['Random Forest R²'], width, label='Random Forest', color='lightgreen')

        ax.set_ylabel('R² Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        measurements_short = [m.replace('_cm', '').replace('_', ' ') for m in comparison['Measurement']]
        ax.set_xticklabels(measurements_short, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('analysis_output/07_model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nSaved: analysis_output/07_model_comparison.png")
        plt.close()

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)

        report = []
        report.append("MACROPARAMETER TO MEASUREMENT RELATIONSHIP ANALYSIS")
        report.append("="*80)
        report.append(f"\nDataset: {len(self.df)} samples")
        report.append(f"Macroparameters: {', '.join(MACROPARAMETERS)}")
        report.append(f"Measurements: {', '.join(MEASUREMENTS)}")

        report.append("\n\nKEY FINDINGS:")
        report.append("-" * 80)

        # Best model for each measurement
        report.append("\n1. BEST MODEL FOR EACH MEASUREMENT:")
        for _, row in self.linear_results.iterrows():
            measure = row['Measurement']
            linear_r2 = row['R²']
            poly_r2 = self.poly_results[self.poly_results['Measurement'] == measure]['R²'].values[0]
            rf_r2 = self.rf_results[self.rf_results['Measurement'] == measure]['R²'].values[0]

            best_r2 = max(linear_r2, poly_r2, rf_r2)
            if best_r2 == linear_r2:
                best_model = "Linear"
            elif best_r2 == poly_r2:
                best_model = "Polynomial"
            else:
                best_model = "Random Forest"

            report.append(f"  {measure:25s}: {best_model:15s} (R² = {best_r2:.4f})")

        # Relationship complexity
        report.append("\n2. RELATIONSHIP COMPLEXITY:")
        avg_linear_r2 = self.linear_results['R²'].mean()
        avg_poly_r2 = self.poly_results['R²'].mean()
        avg_rf_r2 = self.rf_results['R²'].mean()

        report.append(f"  Average R² - Linear: {avg_linear_r2:.4f}")
        report.append(f"  Average R² - Polynomial: {avg_poly_r2:.4f}")
        report.append(f"  Average R² - Random Forest: {avg_rf_r2:.4f}")

        if avg_rf_r2 - avg_linear_r2 < 0.05:
            report.append("  -> Relationships are primarily LINEAR")
        elif avg_poly_r2 > avg_linear_r2 + 0.05:
            report.append("  -> Relationships have POLYNOMIAL components")
        else:
            report.append("  -> Relationships have COMPLEX NON-LINEAR components")

        # Most important macroparameters
        report.append("\n3. MOST INFLUENTIAL MACROPARAMETERS:")
        avg_importance = self.feature_importances.mean(axis=1).sort_values(ascending=False)
        for param, importance in avg_importance.items():
            report.append(f"  {param:12s}: {importance:.3f}")

        report_text = "\n".join(report)
        print(report_text)

        # Save report
        with open('analysis_output/00_SUMMARY_REPORT.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        print("\nSaved: analysis_output/00_SUMMARY_REPORT.txt")

    def run_full_analysis(self):
        """Run all analyses in sequence."""
        import os
        os.makedirs('analysis_output', exist_ok=True)

        print("\n" + "="*80)
        print(" MACROPARAMETER ANALYSIS SUITE")
        print("="*80)

        # Run all analyses
        self.exploratory_analysis()
        self.correlation_analysis()
        self.linear_results = self.linear_regression_analysis()
        self.poly_results = self.polynomial_regression_analysis(degree=2)
        self.rf_results, self.feature_importances = self.random_forest_analysis()
        self.interaction_effects_analysis()
        self.compare_models()
        self.generate_summary_report()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nAll results saved to 'analysis_output/' directory")
        print("\nGenerated files:")
        print("  00_SUMMARY_REPORT.txt")
        print("  01_macroparameter_distributions.png")
        print("  02_measurement_distributions.png")
        print("  03_correlation_heatmap.png")
        print("  04_linear_regression_performance.png")
        print("  04_polynomial_regression_degree2_performance.png")
        print("  04_random_forest_performance.png")
        print("  05_feature_importance.png")
        print("  06_interactions_*.png")
        print("  07_model_comparison.png")


if __name__ == "__main__":
    import sys

    # Default to female Asian lookup table
    csv_path = "lookup_tables/lookup_table_female_asian.csv"

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    print(f"Analyzing: {csv_path}")

    # Run analysis
    analyzer = MacroparameterAnalyzer(csv_path)
    analyzer.run_full_analysis()
