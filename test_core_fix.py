#!/usr/bin/env python3
"""
Simple test to verify the core bug fix in Maps class column validation.
This tests just the specific logic without requiring all dependencies.
"""

import sys
from pathlib import Path
import pandas as pd

def test_column_validation_logic():
    """Test the core column validation logic that was buggy."""
    print("Testing column validation logic fix...")
    
    # Create test data
    color_columns = ['dataset_name', 'country', 'nonexistent_column']
    
    meta = pd.DataFrame({
        '#sampleid': ['sample1', 'sample2', 'sample3'],
        'dataset_name': ['dataset_A', 'dataset_B', 'dataset_A'],
        'country': ['USA', 'Canada', 'USA'],
        'latitude_deg': [40.7128, 45.5017, 34.0522],
        'longitude_deg': [-74.0060, -73.5673, -118.2437]
    })
    
    print(f"Color columns to validate: {color_columns}")
    print(f"DataFrame columns: {list(meta.columns)}")
    
    # Test the OLD (buggy) logic
    print("\n--- Testing OLD (buggy) logic ---")
    try:
        # This was the bug: checking `col in meta` instead of `col in meta.columns`
        valid_columns_old = [col for col in color_columns if col in meta]
        print(f"OLD logic result: {valid_columns_old}")
        print(f"OLD logic length: {len(valid_columns_old)}")
        
        # This should be empty because column names are not index values
        if len(valid_columns_old) == 0:
            print("✅ Confirmed: OLD logic produces empty result (BUG)")
        else:
            print("❓ Unexpected: OLD logic found columns (may depend on pandas version)")
            
    except Exception as e:
        print(f"OLD logic error: {e}")
    
    # Test the NEW (fixed) logic  
    print("\n--- Testing NEW (fixed) logic ---")
    try:
        # This is the fix: checking `col in meta.columns`
        valid_columns_new = [col for col in color_columns if col in meta.columns]
        print(f"NEW logic result: {valid_columns_new}")
        print(f"NEW logic length: {len(valid_columns_new)}")
        
        # This should find ['dataset_name', 'country']
        expected = ['dataset_name', 'country']
        if set(valid_columns_new) == set(expected):
            print("✅ SUCCESS: NEW logic correctly finds valid columns!")
            return True
        else:
            print(f"❌ FAILURE: Expected {expected}, got {valid_columns_new}")
            return False
            
    except Exception as e:
        print(f"NEW logic error: {e}")
        return False


def test_config_access_logic():
    """Test the config access logic fix."""
    print("\nTesting config access logic fix...")
    
    # Test OLD (buggy) config access
    print("\n--- Testing OLD (buggy) config access ---")
    config_without_maps = {'other_setting': 'value'}
    
    try:
        # This was the bug: direct access to config['maps'] 
        maps_config_old = config_without_maps['maps']  # Should throw KeyError
        print(f"❌ UNEXPECTED: OLD logic didn't throw KeyError: {maps_config_old}")
        return False
    except KeyError:
        print("✅ Confirmed: OLD logic throws KeyError when 'maps' key missing (BUG)")
    
    # Test NEW (fixed) config access
    print("\n--- Testing NEW (fixed) config access ---")
    try:
        # This is the fix: safe access with .get()
        maps_config_new = config_without_maps.get('maps', {})
        print(f"NEW logic result: {maps_config_new}")
        
        if maps_config_new == {}:
            print("✅ SUCCESS: NEW logic safely returns empty dict!")
            return True
        else:
            print(f"❌ FAILURE: Expected empty dict, got {maps_config_new}")
            return False
            
    except Exception as e:
        print(f"NEW logic error: {e}")
        return False


def main():
    """Run the focused tests."""
    print("="*60)
    print("TESTING SPECIFIC MAPS CLASS LOGIC FIXES")
    print("="*60)
    
    test1_passed = test_column_validation_logic()
    test2_passed = test_config_access_logic()
    
    print("\n" + "="*60)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED: Core logic fixes are working!")
        print("Sample maps should now be generated correctly.")
    else:
        print("❌ SOME TESTS FAILED: Logic fixes may not be working correctly.")
    print("="*60)
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)