import unittest
from unittest.mock import patch, MagicMock
import json
import os
from exploring.file_cache import get_from_cache, save_to_cache

def test_cache():
	
	# Call the function
	old_result = get_from_cache({"param1": "v1", "param2":"v2"})
	save_to_cache({"param1": "v1", "param2":"v2"}, "Hello World")
	new_result = get_from_cache({"param1": "v1", "param2":"v2"})
	
	# Assertions
	assert new_result == "Hello World"
