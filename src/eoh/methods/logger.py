import pandas as pd
import os


class EvolLogger:
    def __init__(self, output_path):
        self.output_path = output_path
        self.individual_logs = []  # DataFrame 1: per-operator metrics
        self.best_logs = []  # DataFrame 2: best-so-far tracker

    def log_individual(self, generation, operator, individual_id, offspring):
        """Log metrics for each individual offspring"""
        if offspring["objective"] is None:
            return  # Skip failed evaluations

        metrics = offspring.get("eval_metrics", {})

        self.individual_logs.append(
            {
                "generation": generation,
                "operator": operator,
                "individual_id": individual_id,
                "avg_aoi": metrics.get("avg_aoi", None),
                "avg_dropped_ratio": metrics.get("avg_dropped_ratio", None),
                "fitness": offspring["objective"],
            }
        )

    def log_best(self, generation, best_individual):
        """Log best individual so far"""
        if best_individual["objective"] is None:
            return

        metrics = best_individual.get("eval_metrics", {})

        self.best_logs.append(
            {
                "generation": generation,
                "best_fitness": best_individual["objective"],
                "best_avg_aoi": metrics.get("avg_aoi", None),
                "best_dropped_ratio": metrics.get("avg_dropped_ratio", None),
            }
        )

    def save_to_csv(self):
        """Save all logs to CSV files"""
        # Create logs directory
        log_dir = os.path.join(self.output_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Save individual logs
        if self.individual_logs:
            df_individuals = pd.DataFrame(self.individual_logs)
            csv_path = os.path.join(log_dir, "individual_metrics.csv")
            df_individuals.to_csv(csv_path, index=False)
            print(f"Saved individual metrics to {csv_path}")

        # Save best logs
        if self.best_logs:
            df_best = pd.DataFrame(self.best_logs)
            csv_path = os.path.join(log_dir, "best_so_far.csv")
            df_best.to_csv(csv_path, index=False)
            print(f"Saved best-so-far to {csv_path}")
