from typing import List
import logging
from scipy import stats
from colorama import Fore, Style
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_correlations(
    groundtruth: List[float],
    predictions: List[float],
):
    # Step one: collect nan indices 
    nan_indices = []
    for idx, pred in enumerate(predictions):
        try:
            pred = float(pred)
        except:
            nan_indices.append(idx)
    
    original_len = len(predictions)
    if len(nan_indices) > 0:
        logger.info(f"Original length of predictions: {Fore.GREEN}{original_len}{Style.RESET_ALL}, but {Fore.RED}{len(nan_indices)}{Style.RESET_ALL} predictions are NaN. Will remove these predictions and use the remaining {Fore.GREEN}{original_len - len(nan_indices)}{Style.RESET_ALL} predictions for evaluation.")

    predictions = [float(pred) for idx, pred in enumerate(predictions) if idx not in nan_indices]
    groundtruth = [float(gt) for idx, gt in enumerate(groundtruth) if idx not in nan_indices]

    # Pearson correlation
    pearson_correlation = stats.pearsonr(groundtruth, predictions)
    logger.info(f"Pearson correlation: {Fore.GREEN}{pearson_correlation[0]:.4f}{Style.RESET_ALL} (p-value: {Fore.GREEN}{pearson_correlation[1]:.4f}{Style.RESET_ALL})")

    # Spearman correlation
    spearman_correlation = stats.spearmanr(groundtruth, predictions)
    logger.info(f"Spearman correlation: {Fore.GREEN}{spearman_correlation[0]:.4f}{Style.RESET_ALL} (p-value: {Fore.GREEN}{spearman_correlation[1]:.4f}{Style.RESET_ALL})")

    # Kendall correlation
    kendall_correlation = stats.kendalltau(groundtruth, predictions)
    logger.info(f"Kendall correlation: {Fore.GREEN}{kendall_correlation[0]:.4f}{Style.RESET_ALL} (p-value: {Fore.GREEN}{kendall_correlation[1]:.4f}{Style.RESET_ALL})")

    # Calculate the RMSE
    rmse = np.sqrt(np.mean((np.array(groundtruth) - np.array(predictions))**2))
    logger.info(f"RMSE: {Fore.GREEN}{rmse:.4f}{Style.RESET_ALL}")

    return {
        'pearson_r': pearson_correlation[0],
        'pearson_p_value': pearson_correlation[1],
        'spearman_r': spearman_correlation[0],
        'spearman_p_value': spearman_correlation[1],
        'kendall_tau': kendall_correlation[0],
        'kenall_p_value': kendall_correlation[1],
        'rmse': rmse,
    }