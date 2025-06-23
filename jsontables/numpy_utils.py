#!/usr/bin/env python3
"""
Automatic numpy type and NaN handling for JSON-Tables.

Provides seamless conversion between numpy/pandas types and JSON-compatible types,
with intelligent handling of NaN values and type preservation.
"""

import json
from typing import Any, Dict, List, Union, Optional
import sys

def is_numpy_available():
    """Check if numpy is available."""
    try:
        import numpy as np
        return True
    except ImportError:
        return False

def is_pandas_available():
    """Check if pandas is available."""
    try:
        import pandas as pd
        return True
    except ImportError:
        return False

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that automatically handles numpy types and NaN values."""
    
    def default(self, obj):
        if not is_numpy_available():
            return super().default(obj)
        
        import numpy as np
        
        # Handle numpy scalars
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None  # Convert NaN to JSON null
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.str_, np.unicode_)):
            return str(obj)
        
        # Handle pandas types if available
        if is_pandas_available():
            import pandas as pd
            if pd.isna(obj):
                return None  # Convert any pandas NA to JSON null
        
        return super().default(obj)

def convert_numpy_types(data: Any) -> Any:
    """
    Recursively convert numpy types to JSON-compatible Python types.
    
    Args:
        data: Data that may contain numpy types
        
    Returns:
        Data with numpy types converted to standard Python types
    """
    if not is_numpy_available():
        return data
    
    import numpy as np
    
    # Handle scalars
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        if np.isnan(data):
            return None
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, (np.str_, np.unicode_)):
        return str(data)
    elif isinstance(data, np.ndarray):
        return [convert_numpy_types(item) for item in data.tolist()]
    
    # Handle pandas NA if available
    if is_pandas_available():
        import pandas as pd
        if pd.isna(data):
            return None
    
    # Handle containers
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_numpy_types(item) for item in data]
    
    return data

def handle_nan_values(data: Any, nan_strategy: str = "null") -> Any:
    """
    Handle NaN values according to specified strategy.
    
    Args:
        data: Data that may contain NaN values
        nan_strategy: How to handle NaNs ("null", "string", "skip")
        
    Returns:
        Data with NaN values handled according to strategy
    """
    if not is_numpy_available():
        return data
    
    import numpy as np
    
    if is_pandas_available():
        import pandas as pd
        is_na_func = pd.isna
    else:
        def is_na_func(x):
            try:
                return np.isnan(x)
            except (TypeError, ValueError):
                return False
    
    def process_value(value):
        if is_na_func(value):
            if nan_strategy == "null":
                return None
            elif nan_strategy == "string":
                return "NaN"
            elif nan_strategy == "skip":
                return "__SKIP__"
        return convert_numpy_types(value)
    
    # Handle different data structures
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            processed = process_value(value)
            if processed != "__SKIP__":
                result[key] = processed
        return result
    elif isinstance(data, (list, tuple)):
        result = []
        for item in data:
            if isinstance(item, dict):
                processed = handle_nan_values(item, nan_strategy)
            else:
                processed = process_value(item)
            if processed != "__SKIP__":
                result.append(processed)
        return result
    else:
        return process_value(data)

def detect_numpy_dtypes(data: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Detect numpy/pandas dtypes for each column in the data.
    
    Args:
        data: List of records
        
    Returns:
        Dictionary mapping column names to detected dtypes
    """
    if not data or not is_pandas_available():
        return {}
    
    import pandas as pd
    
    try:
        df = pd.DataFrame(data)
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
    except Exception:
        return {}

def restore_numpy_dtypes(data: List[Dict[str, Any]], dtypes: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Restore numpy dtypes to decoded data.
    
    Args:
        data: Decoded data with standard Python types
        dtypes: Dictionary of column name to dtype mappings
        
    Returns:
        Data with numpy dtypes restored where possible
    """
    if not data or not dtypes or not is_pandas_available():
        return data
    
    import pandas as pd
    import numpy as np
    
    try:
        df = pd.DataFrame(data)
        
        # Restore dtypes where possible
        for col, dtype_str in dtypes.items():
            if col in df.columns:
                try:
                    # Special handling for different dtype families
                    if 'int' in dtype_str.lower():
                        df[col] = pd.to_numeric(df[col], errors='ignore').astype('Int64')
                    elif 'float' in dtype_str.lower():
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    elif 'bool' in dtype_str.lower():
                        df[col] = df[col].astype('boolean')
                    elif 'object' in dtype_str.lower() or 'string' in dtype_str.lower():
                        df[col] = df[col].astype('string')
                except Exception:
                    # If conversion fails, keep original
                    pass
        
        return df.to_dict('records')
    except Exception:
        return data

def smart_json_dumps(data: Any, **kwargs) -> str:
    """
    JSON dumps with automatic numpy type handling.
    
    Args:
        data: Data to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string with numpy types automatically converted
    """
    # Set default encoder
    if 'cls' not in kwargs:
        kwargs['cls'] = NumpyJSONEncoder
    
    return json.dumps(data, **kwargs)

def preprocess_for_json_tables(data: Union[List[Dict], Any]) -> tuple:
    """
    Preprocess data for JSON-Tables encoding with numpy/pandas awareness.
    
    Args:
        data: Input data (records or DataFrame)
        
    Returns:
        Tuple of (processed_data, metadata) where metadata contains dtype info
    """
    metadata = {}
    
    # If it's a pandas DataFrame, extract records and dtypes
    if is_pandas_available():
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            metadata['dtypes'] = {col: str(dtype) for col, dtype in data.dtypes.items()}
            metadata['index_name'] = data.index.name
            data = data.to_dict('records')
    
    # Handle numpy types and NaNs in records
    if isinstance(data, list) and data:
        # Detect dtypes if not already available
        if 'dtypes' not in metadata:
            metadata['dtypes'] = detect_numpy_dtypes(data)
        
        # Convert numpy types and handle NaNs
        processed_data = handle_nan_values(data, nan_strategy="null")
    else:
        processed_data = convert_numpy_types(data)
    
    return processed_data, metadata

def postprocess_from_json_tables(data: List[Dict], metadata: Dict) -> Union[List[Dict], Any]:
    """
    Postprocess data after JSON-Tables decoding with numpy/pandas restoration.
    
    Args:
        data: Decoded records
        metadata: Metadata from preprocessing
        
    Returns:
        Data with numpy/pandas types restored if applicable
    """
    if not data:
        return data
    
    # Restore dtypes if available
    if metadata.get('dtypes'):
        data = restore_numpy_dtypes(data, metadata['dtypes'])
    
    return data

def demo_numpy_handling():
    """Demonstrate automatic numpy type handling."""
    print("🔢 Numpy Type Handling Demo")
    print("=" * 35)
    
    if not is_numpy_available():
        print("❌ Numpy not available - install with: pip install numpy")
        return
    
    import numpy as np
    
    # Create data with various numpy types
    original_data = [
        {
            "int64_field": np.int64(42),
            "float64_field": np.float64(3.14159),
            "bool_field": np.bool_(True),
            "nan_field": np.nan,
            "str_field": np.str_("numpy string"),
            "normal_field": "regular python string"
        },
        {
            "int64_field": np.int64(100),
            "float64_field": np.float64(2.71828),
            "bool_field": np.bool_(False),
            "nan_field": np.float64(42.0),
            "str_field": np.str_("another string"), 
            "normal_field": "another regular string"
        }
    ]
    
    print("📄 Original data with numpy types:")
    for i, row in enumerate(original_data, 1):
        print(f"  Row {i}:")
        for key, value in row.items():
            print(f"    {key}: {value} ({type(value).__name__})")
    
    print(f"\n🔄 Converting numpy types...")
    processed_data = handle_nan_values(original_data)
    
    print(f"📄 After numpy conversion:")
    for i, row in enumerate(processed_data, 1):
        print(f"  Row {i}: {row}")
    
    print(f"\n🔍 JSON serialization test:")
    json_str = smart_json_dumps(processed_data, indent=2)
    print(json_str)
    
    # Test round-trip
    decoded = json.loads(json_str)
    print(f"\n✅ Round-trip successful!")
    print(f"  Original records: {len(original_data)}")
    print(f"  Decoded records: {len(decoded)}")

if __name__ == "__main__":
    demo_numpy_handling() 