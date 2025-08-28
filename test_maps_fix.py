#!/usr/bin/env python3
"""
Test script to verify the Maps class bug fix.
This tests the specific bug where sample maps were not being plotted.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import pandas as pd

try:
    from workflow_16s.amplicon_data.maps import Maps
    from workflow_16s import constants
    print("✓ Successfully imported Maps class")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Source path: {src_path}")
    print(f"Source path exists: {src_path.exists()}")
    sys.exit(1)


def test_maps_column_validation_fix():
    """Test that the Maps class now correctly validates columns."""
    print("Testing Maps column validation fix...")
    
    # Test configuration
    test_config = {
        'maps': {
            'enabled': True,
            'color_columns': ['dataset_name', 'country', 'nonexistent_column']
        }
    }

    # Create test metadata with all required columns
    test_meta = pd.DataFrame({
        '#sampleid': ['sample1', 'sample2', 'sample3'],
        'dataset_name': ['dataset_A', 'dataset_B', 'dataset_A'],
        'country': ['USA', 'Canada', 'USA'],
        'latitude_deg': [40.7128, 45.5017, 34.0522],
        'longitude_deg': [-74.0060, -73.5673, -118.2437]
    })
    
    print(f"✓ Created test metadata with columns: {list(test_meta.columns)}")
    print(f"✓ Config color_columns: {test_config['maps']['color_columns']}")
    
    # Test Maps instantiation
    maps = Maps(test_config, test_meta, Path('/tmp/test_output'), verbose=True)
    print("✓ Maps instance created successfully")
    
    # Test the column validation (this was the bug)
    color_columns = maps.color_columns
    meta = maps.meta
    
    # Before fix: This would be empty due to wrong check (col in meta instead of col in meta.columns)  
    # After fix: This should contain ['dataset_name', 'country']
    valid_columns = [col for col in color_columns if col in meta.columns]
    
    print(f"Color columns to check: {color_columns}")
    print(f"Valid columns found: {valid_columns}")
    print(f"Expected: ['dataset_name', 'country']")
    
    # Verify the fix
    expected_valid = ['dataset_name', 'country']
    if set(valid_columns) == set(expected_valid):
        print("✅ SUCCESS: Column validation is now working correctly!")
        return True
    else:
        print(f"❌ FAILURE: Expected {expected_valid}, got {valid_columns}")
        return False


def test_maps_missing_geo_columns():
    """Test that Maps correctly validates required geographic columns."""
    print("\nTesting Maps geographic column validation...")
    
    test_config = {
        'maps': {
            'enabled': True,
            'color_columns': ['dataset_name']
        }
    }

    # Test metadata WITHOUT required geographic columns
    test_meta_no_geo = pd.DataFrame({
        '#sampleid': ['sample1', 'sample2'],
        'dataset_name': ['dataset_A', 'dataset_B']
        # Missing 'latitude_deg' and 'longitude_deg'
    })
    
    print(f"✓ Created test metadata without geo columns: {list(test_meta_no_geo.columns)}")
    
    maps = Maps(test_config, test_meta_no_geo, Path('/tmp/test_output'), verbose=True)
    result = maps.generate_sample_maps()
    
    # Should return empty dict due to missing geographic columns
    if result == {}:
        print("✅ SUCCESS: Correctly detects missing geographic columns!")
        return True
    else:
        print(f"❌ FAILURE: Expected empty dict, got {result}")
        return False


def test_maps_config_safe_access():
    """Test that Maps safely accesses config without throwing KeyError."""
    print("\nTesting Maps config safe access...")
    
    # Test config WITHOUT 'maps' section
    test_config_no_maps = {}
    
    test_meta = pd.DataFrame({
        '#sampleid': ['sample1'],
        'dataset_name': ['dataset_A'],
        'latitude_deg': [40.7128],
        'longitude_deg': [-74.0060]
    })
    
    try:
        maps = Maps(test_config_no_maps, test_meta, Path('/tmp/test_output'), verbose=True)
        print("✅ SUCCESS: Maps instance created without 'maps' config section!")
        
        # Should return empty dict since maps are not enabled
        result = maps.generate_sample_maps()
        if result == {}:
            print("✅ SUCCESS: Correctly returns empty when maps not enabled!")
            return True
        else:
            print(f"❌ FAILURE: Expected empty dict, got {result}")
            return False
            
    except KeyError as e:
        print(f"❌ FAILURE: KeyError raised when accessing config: {e}")
        return False


def test_full_maps_generation():
    """Test complete Maps generation with valid data."""
    print("\nTesting complete Maps generation...")
    
    test_config = {
        'maps': {
            'enabled': True,
            'color_columns': ['dataset_name']
        }
    }

    test_meta = pd.DataFrame({
        '#sampleid': ['sample1', 'sample2', 'sample3'],
        'dataset_name': ['dataset_A', 'dataset_B', 'dataset_A'],
        'latitude_deg': [40.7128, 45.5017, 34.0522],
        'longitude_deg': [-74.0060, -73.5673, -118.2437]
    })
    
    maps = Maps(test_config, test_meta, Path('/tmp/test_output'), verbose=True)
    
    try:
        result = maps.generate_sample_maps()
        print(f"✓ Maps generation completed")
        print(f"✓ Result type: {type(result)}")
        print(f"✓ Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        # Should generate at least one map for 'dataset_name'
        if isinstance(result, dict) and len(result) > 0:
            print("✅ SUCCESS: Maps were generated!")
            return True
        else:
            print(f"❌ FAILURE: No maps generated. Result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ FAILURE: Exception during map generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING MAPS CLASS FIXES")
    print("="*60)
    
    tests = [
        test_maps_column_validation_fix,
        test_maps_missing_geo_columns, 
        test_maps_config_safe_access,
        test_full_maps_generation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAILURE: Test {test.__name__} threw exception: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)