"""
Task Configuration Module for Distributed Data Valuation

This module provides unified configuration for all FolkTables prediction tasks:
- ACSIncome: Predict income > $50,000
- ACSEmployment: Predict employment status
- ACSPublicCoverage: Predict public health insurance coverage
- ACSMobility: Predict same residential address one year ago
- ACSTravelTime: Predict commute > 20 minutes

Each task includes:
- Feature definitions and binning specifications
- Sensitive variable configuration
- Task-specific preprocessing requirements
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from folktables import (
    ACSDataSource, 
    ACSIncome, 
    ACSEmployment, 
    ACSPublicCoverage, 
    ACSMobility, 
    ACSTravelTime
)


# =============================================================================
# Common Binning Specifications (shared across tasks)
# =============================================================================

# Age binning
AGE_BINS = [0, 18, 30, 45, 65, 100]
AGE_LABELS = ['0-18', '19-30', '31-45', '46-65', '65+']

# Education (SCHL) binning
SCHL_BINS = [0, 16, 17, 21, 22, 23, 25]
SCHL_LABELS = ['<HS', 'HS_Grad', 'Some_College_Assoc', 'Bachelor', 'Master', 'Doctorate_Prof']

# Work hours per week (WKHP) binning
WKHP_BINS = [-1, 0, 10, 20, 30, 39, 41, 50, 100]
WKHP_LABELS = ['0', '1-9', '10-19', '20-29', '30-38', '39-40', '41-49', '50+']

# Occupation (OCCP) binning
OCCP_MIN, OCCP_MAX = 0, 9999
NUM_OCCP_BINS = 10
OCCP_BINS = np.linspace(OCCP_MIN, OCCP_MAX + 1, NUM_OCCP_BINS + 1).tolist()
OCCP_LABELS = [f'Occ_Grp_{i+1}' for i in range(NUM_OCCP_BINS)]

# Place of Birth (POBP) binning
POBP_BINS = [0, 60, 100, 200, 300, 310, 400, 500, 555]
POBP_LABELS = ['US_States', 'PR_US_Islands', 'Europe', 'Asia', 
               'Northern_America', 'Latin_America', 'Africa', 'Oceania_At_Sea']

# Marital Status (MAR) - already categorical but can group
MAR_BINS = [0, 1, 2, 3, 4, 5, 6]
MAR_LABELS = ['Married', 'Widowed', 'Divorced', 'Separated', 'Never_Married']

# Class of Worker (COW) - already categorical
COW_BINS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
COW_LABELS = ['Private_For_Profit', 'Private_NonProfit', 'Local_Gov', 'State_Gov', 
              'Federal_Gov', 'Self_Emp_NotInc', 'Self_Emp_Inc', 'Family_Work', 'Unemployed']

# Relationship (RELP) binning - simplify the 18 categories
RELP_BINS = [0, 1, 2, 6, 11, 18]
RELP_LABELS = ['Householder', 'Spouse', 'Child_Related', 'Other_Relative', 'Non_Relative']

# Disability status (DIS) - binary
DIS_BINS = [0, 1, 3]
DIS_LABELS = ['No_Disability', 'With_Disability']

# Citizenship (CIT)
CIT_BINS = [0, 1, 2, 3, 4, 6]
CIT_LABELS = ['Born_US', 'Born_PR_US_Island', 'Born_Abroad_US_Parents', 'Naturalized', 'Not_Citizen']

# Means of Transportation (JWTR) 
JWTR_BINS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 13]
JWTR_LABELS = ['Car_Truck_Van', 'Bus', 'Streetcar', 'Subway', 'Railroad', 
               'Ferryboat', 'Taxicab', 'Motorcycle', 'Other', 'Work_Home']

# Travel time bins (JWMNP)
JWMNP_BINS = [-1, 0, 10, 20, 30, 45, 60, 90, 200]
JWMNP_LABELS = ['0', '1-10', '11-20', '21-30', '31-45', '46-60', '61-90', '90+']

# Income bins (for auxiliary analysis, not for ACSIncome target)
PINCP_BINS = [-100000, 0, 15000, 30000, 50000, 75000, 100000, 200000, 1000000]
PINCP_LABELS = ['Negative', '0-15k', '15k-30k', '30k-50k', '50k-75k', '75k-100k', '100k-200k', '200k+']

# Public Assistance Income (PAP)
PAP_BINS = [-1, 0, 1, 5000, 15000, 50000]
PAP_LABELS = ['None', '1-4999', '5k-15k', '15k+']

# Health Insurance (HINS) - can be constructed
# Military service (MIL)
MIL_BINS = [0, 1, 2, 3, 4, 5]
MIL_LABELS = ['Now_Active', 'Active_Past', 'Only_Training', 'Never_Served']

# English ability (ENG)
ENG_BINS = [0, 1, 2, 3, 4, 5]
ENG_LABELS = ['Very_Well', 'Well', 'Not_Well', 'Not_At_All']

# Ancestry (ANC)
ANC_BINS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ANC_LABELS = ['Single', 'Multiple', 'Unclassified', 'Not_Reported']

# Nativity (NATIVITY)
NATIVITY_BINS = [0, 1, 3]
NATIVITY_LABELS = ['Native', 'Foreign_Born']

# Educational Attainment for those enrolled (SCHG)
SCHG_BINS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
SCHG_LABELS = ['Nursery', 'Kindergarten', 'Grade_1-4', 'Grade_5-8', 
               'Grade_9-12', 'Undergrad', 'Graduate']


# =============================================================================
# Task Configuration Dataclass
# =============================================================================

@dataclass
class TaskConfig:
    """Configuration for a FolkTables prediction task."""
    
    name: str
    task_object: Any  # FolkTables task object (ACSIncome, ACSEmployment, etc.)
    description: str
    
    # Features configuration
    features: List[str]
    sensitive_var: str = 'SEX'
    target_var: str = 'label'
    group_var: str = 'RAC1P'
    
    # Binning specifications for continuous/ordinal features
    bin_specs: Dict[str, Dict] = field(default_factory=dict)
    
    # Features to use for MI calculation (after binning)
    mi_features: List[str] = field(default_factory=list)
    
    # Default optimization weights (optimized via differential evolution)
    default_weights: Dict[str, float] = field(default_factory=lambda: {
        'alpha_SN': 0.8889,  # Penalize Sensitive-NonSensitive correlation
        'alpha_ST': 2.0,     # Penalize Sensitive-Target correlation
        'beta_NN': 0.1111,   # Penalize NonSensitive-NonSensitive redundancy
        'delta_NT': 1.3333   # Reward NonSensitive-Target utility
    })
    
    # Privacy parameters (RDP composition for all contingency tables)
    default_epsilon: float = 1.0   # Total budget for full MI matrix release
    default_delta: float = 1e-5
    
    def get_binned_feature_name(self, feature: str) -> str:
        """Get the binned column name for a feature."""
        if feature in self.bin_specs:
            return f"{feature}_BINNED"
        return feature
    
    def get_mi_features_after_binning(self) -> List[str]:
        """Get feature names after binning transformation."""
        result = []
        for f in self.mi_features:
            result.append(self.get_binned_feature_name(f))
        return result
    
    def get_non_sensitive_vars(self) -> List[str]:
        """Get non-sensitive variable names (excluding sensitive and target)."""
        mi_features = self.get_mi_features_after_binning()
        return [v for v in mi_features if v != self.sensitive_var and v != self.target_var]


# =============================================================================
# Pre-defined Task Configurations
# =============================================================================

# ACSIncome Task Configuration
ACS_INCOME_CONFIG = TaskConfig(
    name="ACSIncome",
    task_object=ACSIncome,
    description="Predict whether income exceeds $50,000. Filters: age>16, hours>=1/week, income>=$100.",
    features=['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P'],
    sensitive_var='SEX',
    target_var='label',
    group_var='RAC1P',
    bin_specs={
        'AGEP': {'bins': AGE_BINS, 'labels': AGE_LABELS},
        'SCHL': {'bins': SCHL_BINS, 'labels': SCHL_LABELS},
        'WKHP': {'bins': WKHP_BINS, 'labels': WKHP_LABELS},
        'OCCP': {'bins': OCCP_BINS, 'labels': OCCP_LABELS},
        'POBP': {'bins': POBP_BINS, 'labels': POBP_LABELS}
    },
    mi_features=['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P', 'label']
)

# ACSEmployment Task Configuration
ACS_EMPLOYMENT_CONFIG = TaskConfig(
    name="ACSEmployment",
    task_object=ACSEmployment,
    description="Predict employment status. Filters: age 16-90.",
    features=['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 
              'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P'],
    sensitive_var='SEX',
    target_var='label',
    group_var='RAC1P',
    bin_specs={
        'AGEP': {'bins': AGE_BINS, 'labels': AGE_LABELS},
        'SCHL': {'bins': SCHL_BINS, 'labels': SCHL_LABELS}
    },
    mi_features=['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'CIT', 'MIL', 'NATIVITY', 'SEX', 'RAC1P', 'label']
)

# ACSPublicCoverage Task Configuration  
ACS_PUBLIC_COVERAGE_CONFIG = TaskConfig(
    name="ACSPublicCoverage",
    task_object=ACSPublicCoverage,
    description="Predict public health insurance coverage. Filters: age<65, income<$30,000.",
    features=['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC',
              'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'ST', 'FER', 'RAC1P'],
    sensitive_var='SEX',
    target_var='label', 
    group_var='RAC1P',
    bin_specs={
        'AGEP': {'bins': AGE_BINS, 'labels': AGE_LABELS},
        'SCHL': {'bins': SCHL_BINS, 'labels': SCHL_LABELS},
        'PINCP': {'bins': PINCP_BINS, 'labels': PINCP_LABELS}
    },
    mi_features=['AGEP', 'SCHL', 'MAR', 'DIS', 'CIT', 'MIL', 'NATIVITY', 'PINCP', 'SEX', 'RAC1P', 'label']
)

# ACSMobility Task Configuration
ACS_MOBILITY_CONFIG = TaskConfig(
    name="ACSMobility", 
    task_object=ACSMobility,
    description="Predict same residential address one year ago. Filters: age 18-35.",
    features=['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC',
              'NATIVITY', 'RELP', 'DEAR', 'DEYE', 'DREM', 'RAC1P', 'GCL', 'COW', 'ESR', 'WKHP', 'JWMNP', 'PINCP'],
    sensitive_var='SEX',
    target_var='label',
    group_var='RAC1P',
    bin_specs={
        'AGEP': {'bins': [0, 18, 22, 26, 30, 36], 'labels': ['0-17', '18-21', '22-25', '26-29', '30-35']},
        'SCHL': {'bins': SCHL_BINS, 'labels': SCHL_LABELS},
        'WKHP': {'bins': WKHP_BINS, 'labels': WKHP_LABELS},
        'JWMNP': {'bins': JWMNP_BINS, 'labels': JWMNP_LABELS},
        'PINCP': {'bins': PINCP_BINS, 'labels': PINCP_LABELS}
    },
    mi_features=['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'CIT', 'COW', 'WKHP', 'SEX', 'RAC1P', 'label']
)

# ACSTravelTime Task Configuration
ACS_TRAVEL_TIME_CONFIG = TaskConfig(
    name="ACSTravelTime",
    task_object=ACSTravelTime,
    description="Predict commute > 20 minutes. Filters: employed, age>16.",
    features=['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'MIG', 'RELP', 'RAC1P', 
              'PUMA', 'ST', 'CIT', 'OCCP', 'JWTR', 'POWPUMA', 'POVPIP'],
    sensitive_var='SEX',
    target_var='label',
    group_var='RAC1P',
    bin_specs={
        'AGEP': {'bins': AGE_BINS, 'labels': AGE_LABELS},
        'SCHL': {'bins': SCHL_BINS, 'labels': SCHL_LABELS},
        'OCCP': {'bins': OCCP_BINS, 'labels': OCCP_LABELS},
        'JWTR': {'bins': JWTR_BINS, 'labels': JWTR_LABELS}
    },
    mi_features=['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'CIT', 'OCCP', 'JWTR', 'SEX', 'RAC1P', 'label']
)


# =============================================================================
# Task Registry
# =============================================================================

AVAILABLE_TASKS = {
    'ACSIncome': ACS_INCOME_CONFIG,
    'ACSEmployment': ACS_EMPLOYMENT_CONFIG,
    'ACSPublicCoverage': ACS_PUBLIC_COVERAGE_CONFIG,
    'ACSMobility': ACS_MOBILITY_CONFIG,
    'ACSTravelTime': ACS_TRAVEL_TIME_CONFIG
}


def get_task_config(task_name: str) -> TaskConfig:
    """
    Get task configuration by name.
    
    Args:
        task_name: Name of the task (e.g., 'ACSIncome', 'ACSEmployment')
        
    Returns:
        TaskConfig object for the specified task
        
    Raises:
        ValueError: If task_name is not recognized
    """
    if task_name not in AVAILABLE_TASKS:
        available = ', '.join(AVAILABLE_TASKS.keys())
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}")
    return AVAILABLE_TASKS[task_name]


def list_available_tasks() -> List[str]:
    """Return list of available task names."""
    return list(AVAILABLE_TASKS.keys())


def print_task_info(task_name: Optional[str] = None):
    """Print information about available tasks."""
    if task_name:
        config = get_task_config(task_name)
        print(f"\n{'='*60}")
        print(f"Task: {config.name}")
        print(f"{'='*60}")
        print(f"Description: {config.description}")
        print(f"Features ({len(config.features)}): {config.features}")
        print(f"Sensitive Variable: {config.sensitive_var}")
        print(f"Target Variable: {config.target_var}")
        print(f"Group Variable: {config.group_var}")
        print(f"Binned Features: {list(config.bin_specs.keys())}")
        print(f"MI Features: {config.mi_features}")
    else:
        print("\n" + "="*60)
        print("Available FolkTables Tasks for Optimal Federation Selection")
        print("="*60)
        for name, config in AVAILABLE_TASKS.items():
            print(f"\n• {name}")
            print(f"  {config.description}")
            print(f"  Features: {len(config.features)}, Sensitive: {config.sensitive_var}")


# =============================================================================
# Common Constants
# =============================================================================

ALL_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]


def get_data_source(survey_year: str = '2018', horizon: str = '1-Year') -> ACSDataSource:
    """Create an ACS data source with specified parameters."""
    return ACSDataSource(survey_year=survey_year, horizon=horizon, survey='person')


if __name__ == "__main__":
    # Demo: Print all available tasks
    print_task_info()
    
    print("\n\nDetailed info for ACSIncome:")
    print_task_info('ACSIncome')
