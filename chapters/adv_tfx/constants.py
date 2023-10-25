#!/usr/bin/env python3
# encoding: utf-8
"""Provide some constants."""

from typing import Text

def transformed_name(key: Text) -> Text:
  """Generate the name of the transformed feature from original name."""
  return key + '_xf'

# Keys
LABEL_KEY = 'label'
INPUT_KEY = 'image/raw'

# Feature keys
RAW_FEATURE_KEYS = [INPUT_KEY]

# Constants
IMG_SIZE = 160