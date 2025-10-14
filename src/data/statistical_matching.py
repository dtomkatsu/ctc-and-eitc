"""
Statistical Matching Framework for Tax Unit Enhancement.

This module provides a comprehensive framework for statistical matching between
PUMS data and multiple public data sources to improve income estimation accuracy.

Key Features:
- Propensity score matching
- Hot deck imputation
- Distance-based matching
- Quality metrics and validation
- Support for multiple donor datasets

Data Sources Supported:
- BLS Occupational Employment Statistics (OES)
- Consumer Expenditure Survey (CEX)
- National SOI Public Use File (PUF)
- Survey of Consumer Finances (SCF)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MatchingMethod(Enum):
    """Statistical matching methods."""
    PROPENSITY_SCORE = "propensity_score"
    HOT_DECK = "hot_deck"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    MAHALANOBIS = "mahalanobis"
    EXACT = "exact"


class MatchingQuality(Enum):
    """Quality levels for matched data."""
    EXCELLENT = "excellent"  # >90% confidence
    GOOD = "good"  # 70-90% confidence
    FAIR = "fair"  # 50-70% confidence
    POOR = "poor"  # <50% confidence


@dataclass
class MatchingConfig:
    """Configuration for statistical matching."""
    method: MatchingMethod = MatchingMethod.PROPENSITY_SCORE
    matching_variables: List[str] = field(default_factory=list)
    exact_match_vars: List[str] = field(default_factory=list)
    caliper: Optional[float] = 0.25  # For propensity score matching
    n_neighbors: int = 5  # For nearest neighbor matching
    distance_metric: str = "euclidean"
    min_quality_threshold: float = 0.5
    use_weights: bool = True
    random_seed: int = 42


@dataclass
class MatchResult:
    """Result of a statistical matching operation."""
    recipient_id: Union[str, int]
    donor_id: Union[str, int]
    match_score: float
    quality: MatchingQuality
    method: MatchingMethod
    matched_variables: Dict[str, any] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    @property
    def confidence(self) -> float:
        """Return confidence score (0-100)."""
        return min(100, self.match_score * 100)


class StatisticalMatcher:
    """
    Core statistical matching engine.
    
    Implements multiple matching algorithms to link recipient data (PUMS)
    with donor data (BLS, CEX, SOI PUF, etc.).
    """
    
    def __init__(self, config: Optional[MatchingConfig] = None):
        """
        Initialize the statistical matcher.
        
        Args:
            config: Matching configuration (uses defaults if not provided)
        """
        self.config = config or MatchingConfig()
        self.match_history: List[MatchResult] = []
        np.random.seed(self.config.random_seed)
        
    def match(
        self,
        recipients: pd.DataFrame,
        donors: pd.DataFrame,
        recipient_id_col: str = 'id',
        donor_id_col: str = 'id'
    ) -> Tuple[pd.DataFrame, List[MatchResult]]:
        """
        Perform statistical matching between recipients and donors.
        
        Args:
            recipients: DataFrame of recipient records (e.g., PUMS tax units)
            donors: DataFrame of donor records (e.g., BLS, CEX, SOI)
            recipient_id_col: Column name for recipient IDs
            donor_id_col: Column name for donor IDs
            
        Returns:
            Tuple of (matched_data, match_results)
        """
        logger.info(f"Starting statistical matching using {self.config.method.value}")
        logger.info(f"Recipients: {len(recipients):,} | Donors: {len(donors):,}")
        
        # Validate matching variables exist in both datasets
        self._validate_matching_variables(recipients, donors)
        
        # Select matching method
        if self.config.method == MatchingMethod.PROPENSITY_SCORE:
            matches = self._propensity_score_matching(recipients, donors, recipient_id_col, donor_id_col)
        elif self.config.method == MatchingMethod.HOT_DECK:
            matches = self._hot_deck_matching(recipients, donors, recipient_id_col, donor_id_col)
        elif self.config.method == MatchingMethod.NEAREST_NEIGHBOR:
            matches = self._nearest_neighbor_matching(recipients, donors, recipient_id_col, donor_id_col)
        elif self.config.method == MatchingMethod.MAHALANOBIS:
            matches = self._mahalanobis_matching(recipients, donors, recipient_id_col, donor_id_col)
        elif self.config.method == MatchingMethod.EXACT:
            matches = self._exact_matching(recipients, donors, recipient_id_col, donor_id_col)
        else:
            raise ValueError(f"Unknown matching method: {self.config.method}")
        
        # Store match history
        self.match_history.extend(matches)
        
        # Merge matched data
        matched_data = self._merge_matched_data(recipients, donors, matches, recipient_id_col, donor_id_col)
        
        # Log quality metrics
        self._log_match_quality(matches)
        
        return matched_data, matches
    
    def _validate_matching_variables(self, recipients: pd.DataFrame, donors: pd.DataFrame):
        """Validate that matching variables exist in both datasets."""
        recipient_cols = set(recipients.columns)
        donor_cols = set(donors.columns)
        
        missing_in_recipients = set(self.config.matching_variables) - recipient_cols
        missing_in_donors = set(self.config.matching_variables) - donor_cols
        
        if missing_in_recipients:
            raise ValueError(f"Matching variables missing in recipients: {missing_in_recipients}")
        if missing_in_donors:
            raise ValueError(f"Matching variables missing in donors: {missing_in_donors}")
    
    def _propensity_score_matching(
        self,
        recipients: pd.DataFrame,
        donors: pd.DataFrame,
        recipient_id_col: str,
        donor_id_col: str
    ) -> List[MatchResult]:
        """
        Propensity score matching using logistic regression.
        
        This method:
        1. Combines recipients and donors with treatment indicator
        2. Estimates propensity scores using logistic regression
        3. Matches recipients to donors with similar propensity scores
        4. Applies caliper to ensure good matches
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data for propensity score estimation
        recipients_copy = recipients.copy()
        donors_copy = donors.copy()
        
        recipients_copy['_treatment'] = 1
        donors_copy['_treatment'] = 0
        
        # Combine datasets
        combined = pd.concat([
            recipients_copy[[recipient_id_col] + self.config.matching_variables + ['_treatment']],
            donors_copy[[donor_id_col] + self.config.matching_variables + ['_treatment']]
        ], ignore_index=True)
        
        # Handle missing values
        combined = combined.fillna(combined.median(numeric_only=True))
        
        # Standardize matching variables
        scaler = StandardScaler()
        X = scaler.fit_transform(combined[self.config.matching_variables])
        y = combined['_treatment']
        
        # Estimate propensity scores
        model = LogisticRegression(random_state=self.config.random_seed, max_iter=1000)
        model.fit(X, y)
        propensity_scores = model.predict_proba(X)[:, 1]
        
        # Split back into recipients and donors
        n_recipients = len(recipients)
        recipient_scores = propensity_scores[:n_recipients]
        donor_scores = propensity_scores[n_recipients:]
        
        # Match recipients to donors
        matches = []
        for i, (recipient_idx, recipient_score) in enumerate(zip(recipients.index, recipient_scores)):
            # Find donors within caliper
            score_diff = np.abs(donor_scores - recipient_score)
            within_caliper = score_diff <= self.config.caliper
            
            if not np.any(within_caliper):
                # No match within caliper - use closest
                donor_idx = np.argmin(score_diff)
                quality = MatchingQuality.POOR
            else:
                # Use closest within caliper
                valid_diffs = score_diff.copy()
                valid_diffs[~within_caliper] = np.inf
                donor_idx = np.argmin(valid_diffs)
                
                # Determine quality based on score difference
                min_diff = score_diff[donor_idx]
                if min_diff < 0.05:
                    quality = MatchingQuality.EXCELLENT
                elif min_diff < 0.15:
                    quality = MatchingQuality.GOOD
                elif min_diff < 0.25:
                    quality = MatchingQuality.FAIR
                else:
                    quality = MatchingQuality.POOR
            
            match_score = 1.0 - score_diff[donor_idx]
            
            matches.append(MatchResult(
                recipient_id=recipients.iloc[i][recipient_id_col],
                donor_id=donors.iloc[donor_idx][donor_id_col],
                match_score=match_score,
                quality=quality,
                method=MatchingMethod.PROPENSITY_SCORE,
                metadata={
                    'propensity_score_recipient': recipient_score,
                    'propensity_score_donor': donor_scores[donor_idx],
                    'score_difference': score_diff[donor_idx]
                }
            ))
        
        return matches
    
    def _hot_deck_matching(
        self,
        recipients: pd.DataFrame,
        donors: pd.DataFrame,
        recipient_id_col: str,
        donor_id_col: str
    ) -> List[MatchResult]:
        """
        Hot deck imputation matching.
        
        Matches recipients to donors within the same "deck" (exact match on
        categorical variables) and uses nearest neighbor on continuous variables.
        """
        matches = []
        
        # Create decks based on exact match variables
        if self.config.exact_match_vars:
            recipient_decks = recipients.groupby(self.config.exact_match_vars)
            donor_decks = donors.groupby(self.config.exact_match_vars)
            
            for deck_key, recipient_group in recipient_decks:
                if deck_key in donor_decks.groups:
                    donor_group = donors.loc[donor_decks.groups[deck_key]]
                    
                    # Within deck, use nearest neighbor on continuous variables
                    continuous_vars = [v for v in self.config.matching_variables 
                                     if v not in self.config.exact_match_vars]
                    
                    if continuous_vars:
                        deck_matches = self._nearest_neighbor_within_group(
                            recipient_group, donor_group, continuous_vars,
                            recipient_id_col, donor_id_col
                        )
                    else:
                        # Random selection within deck
                        deck_matches = self._random_selection_within_group(
                            recipient_group, donor_group,
                            recipient_id_col, donor_id_col
                        )
                    
                    matches.extend(deck_matches)
                else:
                    # No matching deck - fall back to nearest neighbor
                    logger.warning(f"No matching deck for {deck_key}, using fallback")
                    fallback_matches = self._nearest_neighbor_within_group(
                        recipient_group, donors, self.config.matching_variables,
                        recipient_id_col, donor_id_col, quality=MatchingQuality.POOR
                    )
                    matches.extend(fallback_matches)
        else:
            # No exact match variables - use nearest neighbor
            matches = self._nearest_neighbor_matching(recipients, donors, recipient_id_col, donor_id_col)
        
        return matches
    
    def _nearest_neighbor_matching(
        self,
        recipients: pd.DataFrame,
        donors: pd.DataFrame,
        recipient_id_col: str,
        donor_id_col: str
    ) -> List[MatchResult]:
        """Nearest neighbor matching using distance metrics."""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        recipient_data = recipients[self.config.matching_variables].fillna(0)
        donor_data = donors[self.config.matching_variables].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        recipient_scaled = scaler.fit_transform(recipient_data)
        donor_scaled = scaler.transform(donor_data)
        
        # Fit nearest neighbors
        nn = NearestNeighbors(
            n_neighbors=min(self.config.n_neighbors, len(donors)),
            metric=self.config.distance_metric
        )
        nn.fit(donor_scaled)
        
        # Find matches
        distances, indices = nn.kneighbors(recipient_scaled)
        
        matches = []
        for i, (recipient_idx, donor_indices, dists) in enumerate(zip(recipients.index, indices, distances)):
            # Use closest neighbor
            donor_idx = donor_indices[0]
            distance = dists[0]
            
            # Determine quality based on distance
            if distance < 0.5:
                quality = MatchingQuality.EXCELLENT
            elif distance < 1.0:
                quality = MatchingQuality.GOOD
            elif distance < 2.0:
                quality = MatchingQuality.FAIR
            else:
                quality = MatchingQuality.POOR
            
            # Convert distance to similarity score
            match_score = 1.0 / (1.0 + distance)
            
            matches.append(MatchResult(
                recipient_id=recipients.iloc[i][recipient_id_col],
                donor_id=donors.iloc[donor_idx][donor_id_col],
                match_score=match_score,
                quality=quality,
                method=MatchingMethod.NEAREST_NEIGHBOR,
                metadata={
                    'distance': distance,
                    'n_neighbors_available': len(donor_indices)
                }
            ))
        
        return matches
    
    def _mahalanobis_matching(
        self,
        recipients: pd.DataFrame,
        donors: pd.DataFrame,
        recipient_id_col: str,
        donor_id_col: str
    ) -> List[MatchResult]:
        """Mahalanobis distance matching (accounts for correlation structure)."""
        from scipy.spatial.distance import mahalanobis
        from scipy.linalg import inv
        
        # Prepare data
        recipient_data = recipients[self.config.matching_variables].fillna(0)
        donor_data = donors[self.config.matching_variables].fillna(0)
        
        # Calculate covariance matrix from combined data
        combined_data = pd.concat([recipient_data, donor_data])
        cov_matrix = combined_data.cov()
        
        # Handle singular matrix
        try:
            inv_cov_matrix = inv(cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, adding small regularization")
            cov_matrix += np.eye(len(cov_matrix)) * 1e-6
            inv_cov_matrix = inv(cov_matrix)
        
        matches = []
        for i, recipient_idx in enumerate(recipients.index):
            recipient_vec = recipient_data.iloc[i].values
            
            # Calculate Mahalanobis distance to all donors
            distances = []
            for donor_idx in range(len(donors)):
                donor_vec = donor_data.iloc[donor_idx].values
                dist = mahalanobis(recipient_vec, donor_vec, inv_cov_matrix)
                distances.append(dist)
            
            # Find closest donor
            distances = np.array(distances)
            donor_idx = np.argmin(distances)
            distance = distances[donor_idx]
            
            # Determine quality
            if distance < 1.0:
                quality = MatchingQuality.EXCELLENT
            elif distance < 2.0:
                quality = MatchingQuality.GOOD
            elif distance < 3.0:
                quality = MatchingQuality.FAIR
            else:
                quality = MatchingQuality.POOR
            
            match_score = 1.0 / (1.0 + distance)
            
            matches.append(MatchResult(
                recipient_id=recipients.iloc[i][recipient_id_col],
                donor_id=donors.iloc[donor_idx][donor_id_col],
                match_score=match_score,
                quality=quality,
                method=MatchingMethod.MAHALANOBIS,
                metadata={'mahalanobis_distance': distance}
            ))
        
        return matches
    
    def _exact_matching(
        self,
        recipients: pd.DataFrame,
        donors: pd.DataFrame,
        recipient_id_col: str,
        donor_id_col: str
    ) -> List[MatchResult]:
        """Exact matching on all matching variables."""
        matches = []
        
        # Create matching keys
        recipient_keys = recipients[self.config.matching_variables].apply(
            lambda x: tuple(x), axis=1
        )
        donor_keys = donors[self.config.matching_variables].apply(
            lambda x: tuple(x), axis=1
        )
        
        # Create donor lookup
        donor_lookup = {}
        for idx, key in zip(donors.index, donor_keys):
            if key not in donor_lookup:
                donor_lookup[key] = []
            donor_lookup[key].append(idx)
        
        # Match recipients
        for i, (recipient_idx, key) in enumerate(zip(recipients.index, recipient_keys)):
            if key in donor_lookup:
                # Exact match found - randomly select one donor
                donor_idx = np.random.choice(donor_lookup[key])
                quality = MatchingQuality.EXCELLENT
                match_score = 1.0
            else:
                # No exact match - use fallback (nearest neighbor)
                logger.warning(f"No exact match for recipient {recipients.iloc[i][recipient_id_col]}")
                # Find nearest neighbor as fallback
                donor_idx = 0  # Placeholder
                quality = MatchingQuality.POOR
                match_score = 0.0
            
            matches.append(MatchResult(
                recipient_id=recipients.iloc[i][recipient_id_col],
                donor_id=donors.iloc[donor_idx][donor_id_col],
                match_score=match_score,
                quality=quality,
                method=MatchingMethod.EXACT,
                metadata={'exact_match': match_score == 1.0}
            ))
        
        return matches
    
    def _nearest_neighbor_within_group(
        self,
        recipients: pd.DataFrame,
        donors: pd.DataFrame,
        variables: List[str],
        recipient_id_col: str,
        donor_id_col: str,
        quality: Optional[MatchingQuality] = None
    ) -> List[MatchResult]:
        """Helper for nearest neighbor matching within a group."""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        
        if len(donors) == 0:
            return []
        
        # Prepare data
        recipient_data = recipients[variables].fillna(0)
        donor_data = donors[variables].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        recipient_scaled = scaler.fit_transform(recipient_data)
        donor_scaled = scaler.transform(donor_data)
        
        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=1, metric=self.config.distance_metric)
        nn.fit(donor_scaled)
        
        # Find matches
        distances, indices = nn.kneighbors(recipient_scaled)
        
        matches = []
        for i, (recipient_idx, donor_idx, distance) in enumerate(zip(recipients.index, indices[:, 0], distances[:, 0])):
            if quality is None:
                # Determine quality based on distance
                if distance < 0.5:
                    match_quality = MatchingQuality.EXCELLENT
                elif distance < 1.0:
                    match_quality = MatchingQuality.GOOD
                elif distance < 2.0:
                    match_quality = MatchingQuality.FAIR
                else:
                    match_quality = MatchingQuality.POOR
            else:
                match_quality = quality
            
            match_score = 1.0 / (1.0 + distance)
            
            matches.append(MatchResult(
                recipient_id=recipients.loc[recipient_idx, recipient_id_col],
                donor_id=donors.iloc[donor_idx][donor_id_col],
                match_score=match_score,
                quality=match_quality,
                method=MatchingMethod.HOT_DECK,
                metadata={'distance': distance}
            ))
        
        return matches
    
    def _random_selection_within_group(
        self,
        recipients: pd.DataFrame,
        donors: pd.DataFrame,
        recipient_id_col: str,
        donor_id_col: str
    ) -> List[MatchResult]:
        """Helper for random selection within a group."""
        matches = []
        
        for i, recipient_idx in enumerate(recipients.index):
            donor_idx = np.random.choice(donors.index)
            
            matches.append(MatchResult(
                recipient_id=recipients.loc[recipient_idx, recipient_id_col],
                donor_id=donors.loc[donor_idx, donor_id_col],
                match_score=0.7,  # Moderate score for random selection
                quality=MatchingQuality.FAIR,
                method=MatchingMethod.HOT_DECK,
                metadata={'random_selection': True}
            ))
        
        return matches
    
    def _merge_matched_data(
        self,
        recipients: pd.DataFrame,
        donors: pd.DataFrame,
        matches: List[MatchResult],
        recipient_id_col: str,
        donor_id_col: str
    ) -> pd.DataFrame:
        """Merge matched donor data into recipient records."""
        # Create match lookup
        match_lookup = {m.recipient_id: m for m in matches}
        
        # Add match metadata to recipients
        recipients_copy = recipients.copy()
        recipients_copy['_match_score'] = recipients_copy[recipient_id_col].map(
            lambda x: match_lookup[x].match_score if x in match_lookup else 0.0
        )
        recipients_copy['_match_quality'] = recipients_copy[recipient_id_col].map(
            lambda x: match_lookup[x].quality.value if x in match_lookup else 'poor'
        )
        recipients_copy['_donor_id'] = recipients_copy[recipient_id_col].map(
            lambda x: match_lookup[x].donor_id if x in match_lookup else None
        )
        
        # Merge donor data
        donor_cols = [c for c in donors.columns if c not in recipients.columns or c == donor_id_col]
        matched_data = recipients_copy.merge(
            donors[donor_cols],
            left_on='_donor_id',
            right_on=donor_id_col,
            how='left',
            suffixes=('', '_donor')
        )
        
        return matched_data
    
    def _log_match_quality(self, matches: List[MatchResult]):
        """Log quality metrics for matches."""
        if not matches:
            logger.warning("No matches to evaluate")
            return
        
        quality_counts = {}
        for match in matches:
            quality_counts[match.quality.value] = quality_counts.get(match.quality.value, 0) + 1
        
        total = len(matches)
        avg_score = np.mean([m.match_score for m in matches])
        
        logger.info(f"Match Quality Summary:")
        logger.info(f"  Total matches: {total:,}")
        logger.info(f"  Average match score: {avg_score:.3f}")
        logger.info(f"  Quality distribution:")
        for quality, count in sorted(quality_counts.items()):
            pct = count / total * 100
            logger.info(f"    {quality}: {count:,} ({pct:.1f}%)")
    
    def get_quality_report(self) -> pd.DataFrame:
        """Generate a detailed quality report for all matches."""
        if not self.match_history:
            return pd.DataFrame()
        
        report_data = []
        for match in self.match_history:
            report_data.append({
                'recipient_id': match.recipient_id,
                'donor_id': match.donor_id,
                'match_score': match.match_score,
                'confidence': match.confidence,
                'quality': match.quality.value,
                'method': match.method.value,
                **match.metadata
            })
        
        return pd.DataFrame(report_data)


def calculate_match_balance(
    recipients: pd.DataFrame,
    matched_data: pd.DataFrame,
    variables: List[str]
) -> pd.DataFrame:
    """
    Calculate balance statistics for matched data.
    
    Compares distributions of matching variables before and after matching
    to assess quality of the match.
    
    Args:
        recipients: Original recipient data
        matched_data: Data after matching
        variables: Variables to check balance on
        
    Returns:
        DataFrame with balance statistics
    """
    balance_stats = []
    
    for var in variables:
        if var not in recipients.columns or var not in matched_data.columns:
            continue
        
        # Calculate means
        recipient_mean = recipients[var].mean()
        matched_mean = matched_data[var].mean()
        
        # Calculate standard deviations
        recipient_std = recipients[var].std()
        matched_std = matched_data[var].std()
        
        # Calculate standardized difference
        pooled_std = np.sqrt((recipient_std**2 + matched_std**2) / 2)
        std_diff = (recipient_mean - matched_mean) / pooled_std if pooled_std > 0 else 0
        
        balance_stats.append({
            'variable': var,
            'recipient_mean': recipient_mean,
            'matched_mean': matched_mean,
            'recipient_std': recipient_std,
            'matched_std': matched_std,
            'standardized_diff': std_diff,
            'balanced': abs(std_diff) < 0.1  # Common threshold
        })
    
    return pd.DataFrame(balance_stats)
