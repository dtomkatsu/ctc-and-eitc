"""Project configuration settings."""

# Census API Key (register at https://api.census.gov/data/key_signup.html)
CENSUS_API_KEY = "YOUR_CENSUS_API_KEY"

# Data years to analyze
YEARS = [2020, 2021, 2022]  # Update with relevant years

# FIPS codes for Hawaii
STATE_FIPS = "15"  # Hawaii's FIPS code
COUNTY_FIPS = {
    "Hawaii": "001",
    "Honolulu": "003",
    "Kalawao": "005",
    "Kauai": "007",
    "Maui": "009"
}

# PUMS data settings
PUMS_YEARS = {
    2020: {
        "1-year": "2020",
        "5-year": "2016-2020"
    },
    # Add other years as needed
}

# Output settings
OUTPUT_DIR = "../data/processed"

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
